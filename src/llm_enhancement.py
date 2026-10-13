# llm_enhancement_v6_industry_standard.py

"""
Industry-standard-based LLM feature enhancement - v6
Core improvement: Use objective financial-health thresholds, Avoid data leakage

Key features: 
1. Explicit industry-standard thresholds (no data leakage)
2. Enforce 10%-80%-10% distribution
3. Temperature=0.7 maintains diversity
"""

import pandas as pd
import numpy as np
import json
from openai import OpenAI
import re
import time
from tqdm import tqdm
import warnings
import os
from datetime import datetime


warnings.filterwarnings('ignore')


class IndustryStandardLLMEnhancer:
    """Industry-standard-based LLM enhancer"""

    def __init__(self, api_key, base_url="https://api.deepseek.com", model="deepseek-chat"):
    # def __init__(self, api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", model="qwen-plus"):

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.feature_stats = None

    def compute_feature_statistics(self, features_df, feature_columns):
        """Compute feature statistics (Unsupervised)"""
        stats = {}
        for col in feature_columns:
            values = features_df[col].dropna()
            stats[col] = {
                'q05': float(values.quantile(0.05)),
                'q10': float(values.quantile(0.10)),
                'q25': float(values.quantile(0.25)),
                'q50': float(values.quantile(0.50)),
                'q75': float(values.quantile(0.75)),
                'q90': float(values.quantile(0.90)),
                'q95': float(values.quantile(0.95)),
                'mean': float(values.mean()),
                'std': float(values.std())
            }
        return stats

    def get_percentile_rank(self, features_df, feature_columns):
        """Compute percentile ranks"""
        return features_df[feature_columns].rank(pct=True)

    def generate_context_description(self, feature_values, feature_stats, percentile_ranks):
        """Describe the enterprise's relative position"""
        descriptions = []
        extreme_features = []

        for col, pct_rank in percentile_ranks.items():
            if pct_rank >= 0.90:
                extreme_features.append((col, pct_rank, 'high'))
            elif pct_rank <= 0.10:
                extreme_features.append((col, pct_rank, 'low'))

        if len(extreme_features) < 3:
            sorted_features = sorted(percentile_ranks.items(),
                                     key=lambda x: abs(x[1] - 0.5),
                                     reverse=True)
            for col, pct_rank in sorted_features[:5]:
                if (col, pct_rank, 'high' if pct_rank > 0.5 else 'low') not in extreme_features:
                    extreme_features.append((col, pct_rank, 'high' if pct_rank > 0.5 else 'low'))

        for col, pct_rank, direction in extreme_features[:6]:
            stat = feature_stats[col]
            value = feature_values[col]

            if direction == 'high':
                if pct_rank >= 0.95:
                    desc = f"- {col}: {value:.2f} (TOP {int((1 - pct_rank) * 100)}% ⚠️ EXTREMELY HIGH, median: {stat['q50']:.2f})"
                else:
                    desc = f"- {col}: {value:.2f} (Top {int((1 - pct_rank) * 100)}%, median: {stat['q50']:.2f})"
            else:
                if pct_rank <= 0.05:
                    desc = f"- {col}: {value:.2f} (BOTTOM {int(pct_rank * 100)}% ⚠️ EXTREMELY LOW, median: {stat['q50']:.2f})"
                else:
                    desc = f"- {col}: {value:.2f} (Bottom {int(pct_rank * 100)}%, median: {stat['q50']:.2f})"

            descriptions.append(desc)

        return "\n".join(descriptions)

    def design_industry_standard_prompt(self, enterprise_id, feature_values, context_description, feature_stats):
        """
        ✅ Core improvement: Industry-standard-based prompt (no data leakage)
        """
        key_features = list(feature_values.keys())[:5]
        features_display = {k: round(feature_values[k], 4) for k in key_features}

        # ========================================
        # ✅ Industry-standard System Message
        # ========================================
        system_message = """
You are a Senior Financial Risk Analyst using OBJECTIVE industry standards.

INDUSTRY-STANDARD RISK CRITERIA (use these objective thresholds):

🟢 LOW RISK (Scores 1-3) - Only ~10-15% of companies:
Financial Health:
  - ROE > 15%, ROA > 8%
  - Profit Margin > 10%
  - Revenue Growth > 15% annually
Liquidity:
  - Current Ratio > 2.0
  - Quick Ratio > 1.5
  - Cash/Assets > 0.20
Leverage:
  - Debt-to-Asset < 0.30
  - Debt-to-Equity < 0.50
  - Interest Coverage > 5x
Operational Efficiency:
  - Asset Turnover > 1.0
  - Inventory Turnover > 6x
  - Low operating costs
Market/Systemic:
  - Low beta (< 0.8)
  - Stable earnings
  - Diversified revenue streams

🟡 MEDIUM RISK (Scores 4-7) - ~70-80% of companies (MOST COMMON):
Financial Health:
  - ROE 5-15%, ROA 3-8%
  - Profit Margin 3-10%
  - Modest growth or stable
Liquidity:
  - Current Ratio 1.0-2.0
  - Quick Ratio 0.8-1.5
  - Cash/Assets 0.10-0.20
Leverage:
  - Debt-to-Asset 0.30-0.70
  - Debt-to-Equity 0.50-2.0
  - Interest Coverage 2-5x
Operational Efficiency:
  - Asset Turnover 0.5-1.0
  - Average efficiency
  - Moderate costs
Market/Systemic:
  - Beta 0.8-1.5
  - Some volatility
  - Normal market exposure

🔴 HIGH RISK (Scores 8-10) - Only ~10-15% of companies:
Financial Health:
  - Negative ROE/ROA
  - Operating losses
  - Declining revenue
Liquidity:
  - Current Ratio < 1.0 (insolvency risk!)
  - Quick Ratio < 0.5
  - Cash/Assets < 0.10 (cash crisis!)
Leverage:
  - Debt-to-Asset > 0.70 (overleveraged!)
  - Debt-to-Equity > 2.0
  - Interest Coverage < 2x (default risk!)
Operational Efficiency:
  - Asset Turnover < 0.5
  - Poor management
  - High cost structure
Market/Systemic:
  - Beta > 1.5 (high volatility)
  - Distress signals
  - Industry downturn

SCORING PROCESS:
Step 1: Count how many metrics fall in each category (Low/Medium/High risk zone)
Step 2: Apply decision rules:
  - If 3+ metrics in LOW risk zone → Score 1-3
  - If 3+ metrics in HIGH risk zone → Score 8-10
  - Otherwise → Score 4-7 based on balance
Step 3: Fine-tune within range based on severity

MANDATORY DISTRIBUTION (remember across ALL companies):
- 10-15% receive 1-3 (exceptional performers)
- 70-80% receive 4-7 (normal companies)
- 10-15% receive 8-10 (distressed companies)

CRITICAL RULES:
1. DO NOT give extreme scores unless metrics CLEARLY meet thresholds
2. When in doubt, default to 4-7 range (most companies are average)
3. Use objective thresholds, not subjective judgment
4. Compare metrics to industry standards, not just to dataset percentiles

OUTPUT: JSON only, no explanations.
        """

        # ========================================
        # User Message
        # ========================================
        user_message = f"""
Assess Enterprise {enterprise_id}'s financial risk using the objective industry standards provided.

## RELATIVE POSITION IN DATASET
{context_description}

## KEY FINANCIAL METRICS
{json.dumps(features_display, indent=2)}

## ASSESSMENT INSTRUCTIONS

Step 1: For EACH risk dimension, compare the enterprise's metrics to the objective thresholds:

**Financial Health Risk**:
- Check: ROE, ROA, Profit Margin, Revenue Growth
- Apply thresholds: >15% ROE = Low Risk, <0% = High Risk

**Liquidity Risk**:
- Check: Current Ratio, Quick Ratio, Cash/Assets
- Apply thresholds: CR>2.0 = Low Risk, CR<1.0 = High Risk

**Leverage Risk**:
- Check: Debt-to-Asset, Debt-to-Equity, Interest Coverage
- Apply thresholds: D/A<0.3 = Low Risk, D/A>0.7 = High Risk

**Operational Efficiency Risk**:
- Check: Asset Turnover, Inventory Turnover, Cost Ratios
- Apply thresholds: AT>1.0 = Low Risk, AT<0.5 = High Risk

**Market/Systemic Risk**:
- Check: Beta, Volatility, Market Exposure
- Apply thresholds: Beta<0.8 = Low Risk, Beta>1.5 = High Risk

Step 2: Determine risk type based on highest scoring dimension

Step 3: Provide brief reason citing specific metrics vs thresholds

## CRITICAL REMINDERS
- Most companies score 4-7 (this is NORMAL)
- Only give 1-3 if metrics clearly exceed low-risk thresholds
- Only give 8-10 if metrics clearly fall below high-risk thresholds
- Use OBJECTIVE thresholds, not percentile comparisons alone

## OUTPUT FORMAT (JSON only)
{{
  "financial_health_risk": <integer 1-10>,
  "liquidity_risk": <integer 1-10>,
  "leverage_risk": <integer 1-10>,
  "operational_efficiency_risk": <integer 1-10>,
  "market_systemic_risk": <integer 1-10>,
  "risk_type": "<string>",
  "risk_reason": "<brief explanation citing specific metrics vs thresholds>"
}}
        """
        return system_message, user_message

    def call_llm(self, system_msg, user_msg, temperature=0.7, max_tokens=500):
        """Call LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"\n❌ API call failed: {e}")
            return None

    def parse_output(self, llm_output):
        """Parse LLM Output"""
        try:
            cleaned = llm_output.replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                raise ValueError("No matching JSON")

            required_fields = [
                'financial_health_risk', 'liquidity_risk', 'leverage_risk',
                'operational_efficiency_risk', 'market_systemic_risk', 'risk_type'
            ]

            for field in required_fields:
                if field not in result:
                    if 'risk' in field and field != 'risk_type':
                        result[field] = 5
                    else:
                        result[field] = 'Unknown'

            # Limit the score range
            for field in required_fields:
                if field != 'risk_type' and 'risk_reason' not in field:
                    result[field] = max(1, min(10, int(result[field])))

            return result

        except Exception as e:
            return {
                'financial_health_risk': 5,
                'liquidity_risk': 5,
                'leverage_risk': 5,
                'operational_efficiency_risk': 5,
                'market_systemic_risk': 5,
                'risk_type': 'Parse Error',
                'risk_reason': 'JSON parsing failed'
            }

    def enhance_dataset(self, features_df, save_path, batch_size=10, temperature=0.7):
        """Enhance the entire dataset"""

        print(f"  → Initialize LLM client...")

        id_column = features_df.columns[0]
        feature_columns = [c for c in features_df.columns[1:]
                           if features_df[c].dtype in ['float64', 'int64']]

        # Step 1: Compute statistics
        print("  [1/3] Compute feature statistics...")
        self.feature_stats = self.compute_feature_statistics(features_df, feature_columns)
        percentile_df = self.get_percentile_rank(features_df, feature_columns)
        print(f"  ✓ Statistics computed")

        # Step 2: Call for each sample: LLM
        print(f"  [2/3] Call LLM to generate risk scores (Industry Standard, Temperature={temperature})...")
        print(f"  → The first API call may take 10-20 seconds to establish a connection; please wait...")
        results = []

        for idx, row in tqdm(features_df.iterrows(), total=len(features_df),
                             desc="  Processing", unit="enterprise"):
            enterprise_id = row[id_column]
            feature_values = {col: float(row[col]) for col in feature_columns}
            percentile_ranks = percentile_df.iloc[idx].to_dict()

            context_desc = self.generate_context_description(
                feature_values, self.feature_stats, percentile_ranks
            )

            sys_msg, user_msg = self.design_industry_standard_prompt(
                enterprise_id, feature_values, context_desc, self.feature_stats
            )

            output = self.call_llm(sys_msg, user_msg, temperature=temperature)

            if output:
                parsed = self.parse_output(output)
                parsed['enterprise_id'] = enterprise_id
                results.append(parsed)

                # Notify after the first successful call
                if idx == 0:
                    print(f"\n  ✓ First API call succeeded!Starting batch processing...\n")
            else:
                results.append({
                    'enterprise_id': enterprise_id,
                    'financial_health_risk': 5,
                    'liquidity_risk': 5,
                    'leverage_risk': 5,
                    'operational_efficiency_risk': 5,
                    'market_systemic_risk': 5,
                    'risk_type': 'API Error',
                    'risk_reason': 'API call failed'
                })

            # Rate limiting
            if (idx + 1) % batch_size == 0:
                time.sleep(0.5)

            # Save periodically
            if (idx + 1) % 50 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(save_path, index=False)

        # Final save
        print("  [3/3] Save results...")
        final_df = pd.DataFrame(results)
        final_df.to_csv(save_path, index=False)

        return final_df


def print_quality_stats(df, dataset_name):
    """Print quality statistics"""
    risk_cols = ['financial_health_risk', 'liquidity_risk', 'leverage_risk',
                 'operational_efficiency_risk', 'market_systemic_risk']

    print(f"\n  📊 {dataset_name} Quality statistics:")
    for col in risk_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()

        # Assess quality
        if std_val >= 2.0 and min_val <= 3 and max_val >= 8:
            status = "✅ Excellent"
        elif std_val >= 1.5:
            status = "⚠️ Fair"
        else:
            status = "❌ Poor"

        print(f"    {col[:30]:<32}: Mean={mean_val:.2f}, Standard deviation={std_val:.2f}, Range=[{min_val}-{max_val}] {status}")


def main():
    """Main function - Automatic batch processing"""

    # deepseek API Key
    API_KEY = "deepseek-apikey"

    # qianwen API Key
    # API_KEY = "deepseek-apikey"

    # 🔧 Select paths automatically based on the runtime environment
    local_path = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    server_path = '/home/user/SQH/project/GNN4/data/'

    if os.path.exists(local_path):
        data_dir = local_path
        print(f"\n✓ Local environment detected, Using path: {data_dir}")
    elif os.path.exists(server_path):
        data_dir = server_path
        print(f"\n✓ Server environment detected, Using path: {data_dir}")
    else:
        print("\n📁 Preset paths not found, Enter the path manually:")
        print("  Tip: This directory should contain  2022_SH, 2022_SZ  and similar subdirectories")
        data_dir = input("Data directory path: ").strip()
        if not data_dir.endswith('/'):
            data_dir += '/'

        if not os.path.exists(data_dir):
            print(f"❌ Error: Path does not exist: {data_dir}")
            return

    # datasets = ['2022_SH', '2022_SZ', '2022_USA', '2023_SH', '2023_SZ', '2022_all', '2023_all']
    datasets = ['2022_BSS']

    print("\n" + "=" * 100)
    print("Industry-standard-based LLM feature enhancement - v6")
    print("=" * 100)
    print("\n Core improvement:")
    print("  ✅ Use objective financial-health thresholds (no data leakage)")
    print("  ✅ Explicit 10%-80%-10% distribution requirements")
    print("  ✅ Temperature=0.7 maintains diversity")
    print("\n Expected effects:")
    print("  📈 Discriminability: 0.5-0.7+")
    print("  📈 Correlation: 0.2-0.3+")
    print("  📈 Score distribution: More realistic (most scores 4-7 points)")
    print("=" * 100)

    # Select datasets
    print(f"\n Available datasets:")
    for i, ds in enumerate(datasets):
        print(f"  {i}: {ds}")

    print("\n Options:")
    print("  - Enter indices (for example, '0,3') to select multiple datasets")
    print("  - Enter 'all' to process all 7 datasets")

    choice = input("\n Your selection: ").strip()

    if choice.lower() == 'all':
        selected_datasets = datasets
    else:
        try:
            choice = choice.replace('\uff0c', ',').replace(' ', '')
            indices = [int(x.strip()) for x in choice.split(',')]
            selected_datasets = [datasets[i] for i in indices]
        except Exception as e:
            print(f"❌ Invalid selection: {e}")
            print("Tip: Separate values with ASCII commas, for example, '0,1,2' or enter 'all'")
            return

    # Estimate cost and time
    total_samples = 0
    dataset_info = []

    for dataset in selected_datasets:
        input_csv = f'{data_dir}{dataset}/csv_data/reduced_features.csv'
        try:
            df = pd.read_csv(input_csv)
            total_samples += len(df)
            dataset_info.append({
                'name': dataset,
                'samples': len(df),
                'path': input_csv
            })
        except:
            print(f"⚠️ Unable to load {dataset}, Skip")

    # Confirm
    print(f"\n{'=' * 100}")
    print("Processing plan:")
    print(f"{'=' * 100}")
    for info in dataset_info:
        print(f"  - {info['name']:<12}: {info['samples']:>5,}  samples")
    print(f"\n Total:")
    print(f"  - Number of datasets: {len(dataset_info)}  items")
    print(f"  - Total samples: {total_samples:,}  items")
    print(f"  - Estimated cost: ${total_samples * 0.002:.2f} (DeepSeek)")
    print(f"  - Estimated time: ~{total_samples * 2 / 60:.0f}  minutes")
    print(f"{'=' * 100}")

    confirm = input("\n Confirm batch processing?(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Cancelled")
        return

    # Batch processing
    print(f"\n{'=' * 100}")
    print(f"Starting batch processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    enhancer = IndustryStandardLLMEnhancer(api_key=API_KEY)

    success_count = 0
    failed_datasets = []

    for idx, info in enumerate(dataset_info):
        dataset = info['name']

        print(f"\n{'#' * 100}")
        print(f"# [{idx + 1}/{len(dataset_info)}] Processing dataset: {dataset}")
        print(f"# Sample count: {info['samples']:,}")
        print(f"{'#' * 100}\n")

        try:
            # Load data
            print(f"\n  [1/4] Load data...")
            df = pd.read_csv(info['path'])
            print(f"  ✓ Loading completed: {len(df):,}  rows")

            # Output path (v6 version)
            output_csv = f"{data_dir}{dataset}/data_process/llm_enhanced_features_v6_industry_qianwen.csv"

            # Run enhancement
            print(f"  [2/4] Start calling LLM (This may take a few minutes, Please wait...)")
            result_df = enhancer.enhance_dataset(
                df, output_csv,
                batch_size=10,
                temperature=0.7
            )

            # Print quality statistics
            print_quality_stats(result_df, dataset)

            print(f"\n✅ {dataset} Completed!Saved to:")
            print(f"   {output_csv}")
            success_count += 1

        except Exception as e:
            print(f"\n❌ {dataset} Processing failed: {e}")
            import traceback
            traceback.print_exc()
            failed_datasets.append(dataset)

        print(f"\n{'-' * 100}\n")

    # Final summary
    print(f"\n{'=' * 100}")
    print("Batch processing completed")
    print(f"{'=' * 100}")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n Results:")
    print(f"  ✅ Success: {success_count}/{len(dataset_info)}")
    if failed_datasets:
        print(f"  ❌ Failure: {', '.join(failed_datasets)}")
    print(f"\n💡 Tip:")
    print(f"  - Generated files use the suffix  '_v6_industry.csv'")
    print(f"  - Run the quality-check script to verify the results")
    print(f"  - Update the check script's path to the v6 version")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()