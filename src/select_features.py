import pandas as pd
from openai import OpenAI
import json
from tqdm import tqdm
import time

def llm_feature_selection(input_csv, output_csv, api_key, target_features=50):
    """Use an LLM to reduce dimensionality by removing highly similar features."""

    # Read data
    df = pd.read_csv(input_csv)

    print(f"Original feature count: {len(df.columns)}")
    print(f"Target feature count: {target_features}")

    # Keep the first column (ID columns)
    id_column = df.columns[0]
    print(f"Retain ID columns: {id_column}")

    # Get all column names except ID
    all_columns = df.columns[1:].tolist()  # Starting from the second column

    # Compute basic statistics for each column
    feature_stats = []
    for col in all_columns:
        stats = {
            'name': col,
            'mean': float(df[col].mean()) if df[col].dtype in ['float64', 'int64'] else None,
            'std': float(df[col].std()) if df[col].dtype in ['float64', 'int64'] else None,
            'missing_rate': float(df[col].isnull().sum() / len(df)),
            'unique_rate': float(df[col].nunique() / len(df))
        }
        feature_stats.append(stats)

    # Initialize DeepSeek client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Process features in batches (Each batch: 20 items)
    batch_size = 20
    selected_features = []

    print("\n Start LLM feature selection...")

    for i in tqdm(range(0, len(all_columns), batch_size), desc="Processing batches"):
        batch_columns = all_columns[i:i + batch_size]
        batch_stats = feature_stats[i:i + batch_size]

        prompt = f"""
        You are a feature engineering expert. Given the statistical information of the following {len(batch_columns)} features, please select the most valuable features with low similarity.

        Feature Statistics:
        {json.dumps(batch_stats, ensure_ascii=False, indent=2)}
        
        Selection Criteria:
        1. Remove highly similar features (e.g., similar names, similar statistical properties)
        2. Retain features with high information content (high unique_rate, low missing rate)
        3. Consider feature diversity and avoid redundancy
        4. Remove features with highly similar data distributions
        5. Select {min(target_features // (len(all_columns) // batch_size + 1), len(batch_columns))} most important features per batch
        
        Please output strictly in the following JSON format with no other text:
        {{
          "selected_features": ["feature1", "feature2", ...],
          "reason": "brief explanation for selection"
        }}"""

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            result_text = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            selected = result.get('selected_features', [])
            selected_features.extend(selected)

            print(f"\n Batch {i // batch_size + 1}: Selected {len(selected)}  features")
            print(f"Reason: {result.get('reason', '')}")

            time.sleep(0.5)

        except Exception as e:
            print(f"\n Batch {i // batch_size + 1} Processing failed: {e}")
            # Keep all features in the batch if processing fails
            selected_features.extend(batch_columns)

    # Ensure the target feature count is not exceeded
    if len(selected_features) > target_features:
        print(f"\n Feature count exceeds the target, Perform a second selection pass...")
        selected_features = llm_final_selection(
            selected_features, feature_stats, target_features, client
        )

    print(f"\n✓ Final selection {len(selected_features)}  features")

    # Save the reduced data
    selected_features_with_id = [id_column] + selected_features  # ID column goes first
    selected_features.pop(-1)
    df_reduced = df[selected_features_with_id]
    df_reduced.to_csv(output_csv, index=False)

    print(f"✓ Reduced data saved: {output_csv}")
    print(f"  Original dimensions: {len(all_columns)}")
    print(f"  After reduction: {len(selected_features)}")
    print(f"  Reduction ratio: {(1 - len(selected_features) / len(all_columns)) * 100:.1f}%")

    # Save the feature list
    with open(output_csv.replace('.csv', '_features.txt'), 'w') as f:
        f.write('\n'.join(selected_features))

    return df_reduced, selected_features


def llm_final_selection(features, feature_stats, target_n, client):
    """Second selection pass, Ensure the feature count meets the target"""

    stats_dict = {s['name']: s for s in feature_stats if s['name'] in features}
    filtered_stats = [stats_dict[f] for f in features if f in stats_dict]

    prompt = f"""
    You are a feature engineering expert. From the following {len(features)} features, select the {target_n} most important features.

    Feature Statistics:
    {json.dumps(filtered_stats, ensure_ascii=False, indent=2)}
    
    Selection Criteria:
    1. Maximum information content (high unique_rate)
    2. Good data quality (low missing rate)
    3. Low similarity between features
    
    Please output strictly in the following JSON format with {target_n} feature names:
    {{
      "selected_features": ["feature1", "feature2", ...]
    }}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()

        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        return result.get('selected_features', features[:target_n])

    except Exception as e:
        print(f"Second selection failed: {e}, Use the first {target_n} items")
        return features[:target_n]


if __name__ == "__main__":

    # Configuration

    # Path
    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    # data_file = ['2023_SH', '2023_SZ', '2022_SH', '2022_SZ', '2022_all', '2023_all', '2022_USA']
    data_file = ['2022_BSS', '2023_BSS']

    for data in data_file:
        print("Process : ", data)
        INPUT_CSV = data_dir + data + "/csv_data/enterprise_features.csv"  # Input file
        OUTPUT_CSV = data_dir + data + "/csv_data/reduced_features.csv"  # Output file
        DEEPSEEK_API_KEY = "deepseek-apikey"  # DeepSeek API Key
        TARGET_FEATURES = 50  # Target feature count

        # Run dimensionality reduction
        df_reduced, selected_features = llm_feature_selection(
            input_csv=INPUT_CSV,
            output_csv=OUTPUT_CSV,
            api_key=DEEPSEEK_API_KEY,
            target_features=TARGET_FEATURES
        )

        print("\n Selected features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"{i}. {feat}")
        print("\n")