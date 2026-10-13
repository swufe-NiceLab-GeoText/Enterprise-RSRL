# # llm_enhancement_v6_no_standard.py
#
# """
# Without industry standards: LLM feature enhancement - Control-group version
# Ablation experiment for label-permutation testing
#
# Key features: 
# 1. ❌ Explicit industry-standard thresholds removed
# 2. ❌ Removed 10%-80%-10% distribution requirements
# 3. ⚠️ Scores based only on relative positions within the dataset (Prone to leakage)
# """
#
# import pandas as pd
# import numpy as np
# import json
# from openai import OpenAI
# import re
# import time
# from tqdm import tqdm
# import warnings
# import os
# from datetime import datetime
#
#
# warnings.filterwarnings('ignore')
#
#
# class NoStandardLLMEnhancer:
#     """Without industry standards: LLM enhancer (Control group)"""
#
#     def __init__(self, api_key, base_url="https://api.deepseek.com", model="deepseek-chat"):
#         self.client = OpenAI(api_key=api_key, base_url=base_url)
#         self.model = model
#         self.feature_stats = None
#
#     def compute_feature_statistics(self, features_df, feature_columns):
#         """Compute feature statistics (Unsupervised)"""
#         stats = {}
#         for col in feature_columns:
#             values = features_df[col].dropna()
#             stats[col] = {
#                 'q05': float(values.quantile(0.05)),
#                 'q10': float(values.quantile(0.10)),
#                 'q25': float(values.quantile(0.25)),
#                 'q50': float(values.quantile(0.50)),
#                 'q75': float(values.quantile(0.75)),
#                 'q90': float(values.quantile(0.90)),
#                 'q95': float(values.quantile(0.95)),
#                 'mean': float(values.mean()),
#                 'std': float(values.std())
#             }
#         return stats
#
#     def get_percentile_rank(self, features_df, feature_columns):
#         """Compute percentile ranks"""
#         return features_df[feature_columns].rank(pct=True)
#
#     def generate_context_description(self, feature_values, feature_stats, percentile_ranks):
#         """Describe the enterprise's relative position"""
#         descriptions = []
#         extreme_features = []
#
#         for col, pct_rank in percentile_ranks.items():
#             if pct_rank >= 0.90:
#                 extreme_features.append((col, pct_rank, 'high'))
#             elif pct_rank <= 0.10:
#                 extreme_features.append((col, pct_rank, 'low'))
#
#         if len(extreme_features) < 3:
#             sorted_features = sorted(percentile_ranks.items(),
#                                      key=lambda x: abs(x[1] - 0.5),
#                                      reverse=True)
#             for col, pct_rank in sorted_features[:5]:
#                 if (col, pct_rank, 'high' if pct_rank > 0.5 else 'low') not in extreme_features:
#                     extreme_features.append((col, pct_rank, 'high' if pct_rank > 0.5 else 'low'))
#
#         for col, pct_rank, direction in extreme_features[:6]:
#             stat = feature_stats[col]
#             value = feature_values[col]
#
#             if direction == 'high':
#                 if pct_rank >= 0.95:
#                     desc = f"- {col}: {value:.2f} (TOP {int((1 - pct_rank) * 100)}% ⚠️ EXTREMELY HIGH, median: {stat['q50']:.2f})"
#                 else:
#                     desc = f"- {col}: {value:.2f} (Top {int((1 - pct_rank) * 100)}%, median: {stat['q50']:.2f})"
#             else:
#                 if pct_rank <= 0.05:
#                     desc = f"- {col}: {value:.2f} (BOTTOM {int(pct_rank * 100)}% ⚠️ EXTREMELY LOW, median: {stat['q50']:.2f})"
#                 else:
#                     desc = f"- {col}: {value:.2f} (Bottom {int(pct_rank * 100)}%, median: {stat['q50']:.2f})"
#
#             descriptions.append(desc)
#
#         return "\n".join(descriptions)
#
#     def design_no_standard_prompt(self, enterprise_id, feature_values, context_description):
#         """
#         ⚠️ Core change: Remove industry standards from the prompt (Prone to leakage)
#         """
#         key_features = list(feature_values.keys())[:5]
#         features_display = {k: round(feature_values[k], 4) for k in key_features}
#
#         # ========================================
#         # ⚠️ Without industry standards System Message
#         # ========================================
#         system_message = """
# You are a Senior Financial Risk Analyst.
#
# Your task is to assess enterprise financial risk based on the provided metrics.
#
# ASSESSMENT DIMENSIONS:
# 1. Financial Health Risk (profitability, growth, efficiency)
# 2. Liquidity Risk (short-term solvency, cash management)
# 3. Leverage Risk (debt levels, financial structure)
# 4. Operational Efficiency Risk (asset utilization, cost control)
# 5. Market/Systemic Risk (volatility, market exposure)
#
# SCORING GUIDELINES:
# - Score each dimension from 1 to 10
# - 1 = Very Low Risk (strong performance)
# - 5-6 = Moderate Risk (average performance)
# - 10 = Very High Risk (poor performance)
#
# IMPORTANT:
# - Consider the enterprise's position relative to other companies in the dataset
# - Use the percentile rankings and median values provided
# - Assess risk holistically based on multiple metrics
# - Provide a risk type classification and brief reasoning
#
# OUTPUT: JSON only, no explanations.
#         """
#
#         # ========================================
#         # User Message
#         # ========================================
#         user_message = f"""
# Assess Enterprise {enterprise_id}'s financial risk.
#
# ## RELATIVE POSITION IN DATASET
# {context_description}
#
# ## KEY FINANCIAL METRICS
# {json.dumps(features_display, indent=2)}
#
# ## ASSESSMENT INSTRUCTIONS
#
# Step 1: For EACH risk dimension, evaluate the enterprise based on:
# - Its relative position in the dataset (percentile rankings)
# - Comparison to median values
# - Overall financial health indicators
#
# **Financial Health Risk**:
# - Evaluate: ROE, ROA, Profit Margin, Revenue Growth
# - Consider: Is profitability strong or weak relative to peers?
#
# **Liquidity Risk**:
# - Evaluate: Current Ratio, Quick Ratio, Cash/Assets
# - Consider: Can the company meet short-term obligations?
#
# **Leverage Risk**:
# - Evaluate: Debt-to-Asset, Debt-to-Equity, Interest Coverage
# - Consider: Is the debt burden manageable?
#
# **Operational Efficiency Risk**:
# - Evaluate: Asset Turnover, Inventory Turnover, Cost Ratios
# - Consider: How efficiently are resources being used?
#
# **Market/Systemic Risk**:
# - Evaluate: Beta, Volatility, Market Exposure
# - Consider: How vulnerable is the company to market shocks?
#
# Step 2: Determine the primary risk type based on the highest scoring dimension
#
# Step 3: Provide brief reasoning based on the key metrics
#
# ## OUTPUT FORMAT (JSON only)
# {{
#   "financial_health_risk": <integer 1-10>,
#   "liquidity_risk": <integer 1-10>,
#   "leverage_risk": <integer 1-10>,
#   "operational_efficiency_risk": <integer 1-10>,
#   "market_systemic_risk": <integer 1-10>,
#   "risk_type": "<string>",
#   "risk_reason": "<brief explanation>"
# }}
#         """
#         return system_message, user_message
#
#     def call_llm(self, system_msg, user_msg, temperature=0.7, max_tokens=500):
#         """Call LLM API"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": user_msg}
#                 ],
#                 temperature=temperature,
#                 max_tokens=max_tokens
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"\n❌ API call failed: {e}")
#             return None
#
#     def parse_output(self, llm_output):
#         """Parse LLM Output"""
#         try:
#             cleaned = llm_output.replace('```json', '').replace('```', '').strip()
#             json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
#
#             if json_match:
#                 json_str = json_match.group(0)
#                 result = json.loads(json_str)
#             else:
#                 raise ValueError("No matching JSON")
#
#             required_fields = [
#                 'financial_health_risk', 'liquidity_risk', 'leverage_risk',
#                 'operational_efficiency_risk', 'market_systemic_risk', 'risk_type'
#             ]
#
#             for field in required_fields:
#                 if field not in result:
#                     if 'risk' in field and field != 'risk_type':
#                         result[field] = 5
#                     else:
#                         result[field] = 'Unknown'
#
#             # Limit the score range
#             for field in required_fields:
#                 if field != 'risk_type' and 'risk_reason' not in field:
#                     result[field] = max(1, min(10, int(result[field])))
#
#             return result
#
#         except Exception as e:
#             return {
#                 'financial_health_risk': 5,
#                 'liquidity_risk': 5,
#                 'leverage_risk': 5,
#                 'operational_efficiency_risk': 5,
#                 'market_systemic_risk': 5,
#                 'risk_type': 'Parse Error',
#                 'risk_reason': 'JSON parsing failed'
#             }
#
#     def enhance_dataset(self, features_df, save_path, batch_size=10, temperature=0.7):
#         """Enhance the entire dataset"""
#
#         print(f"  → Initialize LLM client...")
#
#         id_column = features_df.columns[0]
#         feature_columns = [c for c in features_df.columns[1:]
#                            if features_df[c].dtype in ['float64', 'int64']]
#
#         # Step 1: Compute statistics
#         print("  [1/3] Compute feature statistics...")
#         self.feature_stats = self.compute_feature_statistics(features_df, feature_columns)
#         percentile_df = self.get_percentile_rank(features_df, feature_columns)
#         print(f"  ✓ Statistics computed")
#
#         # Step 2: Call for each sample: LLM
#         print(f"  [2/3] Call LLM to generate risk scores (No Industry Standard, Temperature={temperature})...")
#         print(f"  → The first API call may take 10-20 seconds to establish a connection; please wait...")
#         results = []
#
#         for idx, row in tqdm(features_df.iterrows(), total=len(features_df),
#                              desc="  Processing", unit="enterprise"):
#             enterprise_id = row[id_column]
#             feature_values = {col: float(row[col]) for col in feature_columns}
#             percentile_ranks = percentile_df.iloc[idx].to_dict()
#
#             context_desc = self.generate_context_description(
#                 feature_values, self.feature_stats, percentile_ranks
#             )
#
#             sys_msg, user_msg = self.design_no_standard_prompt(
#                 enterprise_id, feature_values, context_desc
#             )
#
#             output = self.call_llm(sys_msg, user_msg, temperature=temperature)
#
#             if output:
#                 parsed = self.parse_output(output)
#                 parsed['enterprise_id'] = enterprise_id
#                 results.append(parsed)
#
#                 # Notify after the first successful call
#                 if idx == 0:
#                     print(f"\n  ✓ First API call succeeded!Starting batch processing...\n")
#             else:
#                 results.append({
#                     'enterprise_id': enterprise_id,
#                     'financial_health_risk': 5,
#                     'liquidity_risk': 5,
#                     'leverage_risk': 5,
#                     'operational_efficiency_risk': 5,
#                     'market_systemic_risk': 5,
#                     'risk_type': 'API Error',
#                     'risk_reason': 'API call failed'
#                 })
#
#             # Rate limiting
#             if (idx + 1) % batch_size == 0:
#                 time.sleep(0.5)
#
#             # Save periodically
#             if (idx + 1) % 50 == 0:
#                 temp_df = pd.DataFrame(results)
#                 temp_df.to_csv(save_path, index=False)
#
#         # Final save
#         print("  [3/3] Save results...")
#         final_df = pd.DataFrame(results)
#         final_df.to_csv(save_path, index=False)
#
#         return final_df
#
#
# def print_quality_stats(df, dataset_name):
#     """Print quality statistics"""
#     risk_cols = ['financial_health_risk', 'liquidity_risk', 'leverage_risk',
#                  'operational_efficiency_risk', 'market_systemic_risk']
#
#     print(f"\n  📊 {dataset_name} Quality statistics:")
#     for col in risk_cols:
#         mean_val = df[col].mean()
#         std_val = df[col].std()
#         min_val = df[col].min()
#         max_val = df[col].max()
#
#         print(f"    {col[:30]:<32}: Mean={mean_val:.2f}, Standard deviation={std_val:.2f}, Range=[{min_val}-{max_val}]")
#
#
# def main():
#     """Main function - Automatic batch processing (Version without industry standards)"""
#
#     # deepseek API Key
#     API_KEY = "deepseek-apikey"
#
#     # 🔧 Select paths automatically based on the runtime environment
#     local_path = '/data/'
#     server_path = '/home/user/SQH/project/GNN4/data/'
#
#     if os.path.exists(local_path):
#         data_dir = local_path
#         print(f"\n✓ Local environment detected, Using path: {data_dir}")
#     elif os.path.exists(server_path):
#         data_dir = server_path
#         print(f"\n✓ Server environment detected, Using path: {data_dir}")
#     else:
#         print("\n📁 Preset paths not found, Enter the path manually:")
#         print("  Tip: This directory should contain  2022_SH, 2022_SZ  and similar subdirectories")
#         data_dir = input("Data directory path: ").strip()
#         if not data_dir.endswith('/'):
#             data_dir += '/'
#
#         if not os.path.exists(data_dir):
#             print(f"❌ Error: Path does not exist: {data_dir}")
#             return
#
#     # Choose a few datasets for ablation experiments
#     datasets = ['2022_SZ']  # Select only 2 representative datasets
#
#     print("\n" + "=" * 100)
#     print("⚠️  Without industry standards: LLM feature enhancement - Ablation control group")
#     print("=" * 100)
#     print("\n⚠️  Warning:")
#     print("  ❌ Objective financial-health thresholds removed")
#     print("  ❌ Removed 10%-80%-10% distribution requirements")
#     print("  ⚠️  Scores based only on relative positions within the dataset (Prone to leakage)")
#     print("\n Purpose:")
#     print("  📊 Ablation experiment for label-permutation testing")
#     print("  📊 Demonstrate how industry standards prevent leakage")
#     print("=" * 100)
#
#     # Estimate cost and time
#     total_samples = 0
#     dataset_info = []
#
#     for dataset in datasets:
#         input_csv = f'{data_dir}{dataset}/csv_data/reduced_features.csv'
#         try:
#             df = pd.read_csv(input_csv)
#             total_samples += len(df)
#             dataset_info.append({
#                 'name': dataset,
#                 'samples': len(df),
#                 'path': input_csv
#             })
#         except:
#             print(f"⚠️ Unable to load {dataset}, Skip")
#
#     # Confirm
#     print(f"\n{'=' * 100}")
#     print("Processing plan:")
#     print(f"{'=' * 100}")
#     for info in dataset_info:
#         print(f"  - {info['name']:<12}: {info['samples']:>5,}  samples")
#     print(f"\n Total:")
#     print(f"  - Number of datasets: {len(dataset_info)}  items")
#     print(f"  - Total samples: {total_samples:,}  items")
#     print(f"  - Estimated cost: ${total_samples * 0.002:.2f} (DeepSeek)")
#     print(f"  - Estimated time: ~{total_samples * 2 / 60:.0f}  minutes")
#     print(f"{'=' * 100}")
#
#     confirm = input("\n⚠️  Confirm feature generation without industry standards?(yes/no): ").strip().lower()
#     if confirm != 'yes':
#         print("❌ Cancelled")
#         return
#
#     # Batch processing
#     print(f"\n{'=' * 100}")
#     print(f"Starting batch processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"{'=' * 100}\n")
#
#     enhancer = NoStandardLLMEnhancer(api_key=API_KEY)
#
#     success_count = 0
#     failed_datasets = []
#
#     for idx, info in enumerate(dataset_info):
#         dataset = info['name']
#
#         print(f"\n{'#' * 100}")
#         print(f"# [{idx + 1}/{len(dataset_info)}] Processing dataset: {dataset}")
#         print(f"# Sample count: {info['samples']:,}")
#         print(f"{'#' * 100}\n")
#
#         try:
#             # Load data
#             print(f"\n  [1/4] Load data...")
#             df = pd.read_csv(info['path'])
#             print(f"  ✓ Loading completed: {len(df):,}  rows")
#
#             # Output path (v6_no_standard version)
#             output_csv = f"{data_dir}{dataset}/data_process/llm_enhanced_features_v6_no_standard.csv"
#
#             # Run enhancement
#             print(f"  [2/4] Start calling LLM (This may take a few minutes, Please wait...)")
#             result_df = enhancer.enhance_dataset(
#                 df, output_csv,
#                 batch_size=10,
#                 temperature=0.7
#             )
#
#             # Print quality statistics
#             print_quality_stats(result_df, dataset)
#
#             print(f"\n✅ {dataset} Completed!Saved to:")
#             print(f"   {output_csv}")
#             success_count += 1
#
#         except Exception as e:
#             print(f"\n❌ {dataset} Processing failed: {e}")
#             import traceback
#             traceback.print_exc()
#             failed_datasets.append(dataset)
#
#         print(f"\n{'-' * 100}\n")
#
#     # Final summary
#     print(f"\n{'=' * 100}")
#     print("Batch processing completed")
#     print(f"{'=' * 100}")
#     print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"\n Results:")
#     print(f"  ✅ Success: {success_count}/{len(dataset_info)}")
#     if failed_datasets:
#         print(f"  ❌ Failure: {', '.join(failed_datasets)}")
#     print(f"\n💡 Next steps:")
#     print(f"  1. Use  v6_industry  and  v6_no_standard Train the models separately")
#     print(f"  2. Run label-permutation tests on both versions")
#     print(f"  3. Compare results, Demonstrate the leakage-prevention effect of industry standards")
#     print(f"{'=' * 100}\n")
#
#
# if __name__ == "__main__":
#     main()




















# llm_enhancement_v6_no_standard.py

"""
Without industry standards: LLM feature enhancement - Control-group version
Ablation experiment for label-permutation testing

Key features: 
1. ❌ Explicit industry-standard thresholds removed
2. ❌ Removed 10%-80%-10% distribution requirements
3. ⚠️ Scores based only on relative positions within the dataset (Prone to leakage)
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


class NoStandardLLMEnhancer:
    """Without industry standards: LLM enhancer (Control group)"""

    def __init__(self, api_key, base_url="https://api.deepseek.com", model="deepseek-chat"):
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

    def design_no_standard_prompt(self, enterprise_id, feature_values, context_description):
        """
        ⚠️ Core change: Remove industry standards from the prompt
        """
        key_features = list(feature_values.keys())[:5]
        features_display = {k: round(feature_values[k], 4) for k in key_features}

        # ========================================
        # ⚠️ Without industry standards System Message
        # ========================================
        system_message = """
        You are a financial risk analyst.

        Score enterprise financial risk from 1 to 10 across five dimensions.

        GENERAL GUIDELINES:
        - Lower scores indicate lower risk (better financial health)
        - Higher scores indicate higher risk (financial distress)
        - Consider standard financial analysis principles
        - Evaluate each dimension independently

        OUTPUT: JSON only.
        """

        user_message = f"""
        Assess Enterprise {enterprise_id}'s financial risk.

        ## FINANCIAL METRICS
        {json.dumps(features_display, indent=2)}

        ## RELATIVE POSITION
        {context_description}

        Evaluate based on:
        - Financial Health: profitability, growth, efficiency
        - Liquidity: short-term solvency, cash management
        - Leverage: debt levels, financial structure
        - Operational Efficiency: asset utilization, cost control
        - Market/Systemic: volatility, market exposure

        Score each dimension (1-10) and provide risk type.
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
        print(f"  [2/3] Call LLM to generate risk scores (No Industry Standard, Temperature={temperature})...")
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

            sys_msg, user_msg = self.design_no_standard_prompt(
                enterprise_id, feature_values, context_desc
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

        print(f"    {col[:30]:<32}: Mean={mean_val:.2f}, Standard deviation={std_val:.2f}, Range=[{min_val}-{max_val}]")


def main():
    """Main function - Automatic batch processing (Version without industry standards)"""

    # deepseek API Key
    API_KEY = "deepseek-apikey"

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

    # Choose a few datasets for ablation experiments
    datasets = ['2022_SH']  # Select only 2 representative datasets

    print("\n" + "=" * 100)
    print("⚠️  Without industry standards: LLM feature enhancement - Ablation control group")
    print("=" * 100)
    print("\n⚠️  Warning:")
    print("  ❌ Objective financial-health thresholds removed")
    print("  ❌ Removed 10%-80%-10% distribution requirements")
    print("  ⚠️  Scores based only on relative positions within the dataset (Prone to leakage)")
    print("\n Purpose:")
    print("  📊 Ablation experiment for label-permutation testing")
    print("  📊 Demonstrate how industry standards prevent leakage")
    print("=" * 100)

    # Estimate cost and time
    total_samples = 0
    dataset_info = []

    for dataset in datasets:
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

    confirm = input("\n⚠️  Confirm feature generation without industry standards?(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Cancelled")
        return

    # Batch processing
    print(f"\n{'=' * 100}")
    print(f"Starting batch processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    enhancer = NoStandardLLMEnhancer(api_key=API_KEY)

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

            # Output path (v6_no_standard version)
            output_csv = f"{data_dir}{dataset}/data_process/llm_enhanced_features_v6_No_standard.csv"

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
    print(f"\n💡 Next steps:")
    print(f"  1. Use  v6_industry  and  v6_no_standard Train the models separately")
    print(f"  2. Run label-permutation tests on both versions")
    print(f"  3. Compare results, Demonstrate the leakage-prevention effect of industry standards")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()