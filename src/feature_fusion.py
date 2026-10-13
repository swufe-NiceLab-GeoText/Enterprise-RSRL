# src/feature_fusion.py

import os
import pandas as pd
import numpy as np

class FeatureFusion:
    """Feature fusion class"""

    def __init__(self, original_features_path, llm_features_path, output_path):
        self.original_path = original_features_path
        self.llm_path = llm_features_path
        self.output_path = output_path

    def load_data(self):
        """Load original and LLM features"""
        print("Load data...")

        # Check whether files exist
        if not os.path.exists(self.original_path):
            raise FileNotFoundError(f"Original feature file does not exist: {self.original_path}")
        if not os.path.exists(self.llm_path):
            raise FileNotFoundError(f"LLM feature file does not exist: {self.llm_path}")

        # Load data
        self.original_df = pd.read_csv(self.original_path)
        self.llm_df = pd.read_csv(self.llm_path)

        print(f"✓ Original features: {self.original_df.shape}")
        print(f"   columns: {list(self.original_df.columns[:5])}... ( total {len(self.original_df.columns)} columns)")

        print(f"✓ LLM features: {self.llm_df.shape}")
        print(f"   columns: {list(self.llm_df.columns)}")

        # Check enterprise ID consistency
        original_ids = set(self.original_df['\u4f01\u4e1aID'])
        llm_ids = set(self.llm_df['enterprise_id'])

        if original_ids != llm_ids:
            print(f"--Warning:  enterprise IDs do not fully match")
            print(f"  Original features contain: {len(original_ids)} enterprises")
            print(f"  LLM features contain: {len(llm_ids)} enterprises")
            print(f"  Intersection: {len(original_ids & llm_ids)} enterprises")
            print(f"  Only in the original data: {len(original_ids - llm_ids)} enterprises")
            print(f"  Only in LLM: {len(llm_ids - original_ids)} enterprises")

    def concat_fusion(self):
        """Concatenation fusion"""

        # Merge by enterprise ID (left join; keep all enterprises from the original features)
        fused_df = pd.merge(
            self.original_df,
            self.llm_df,
            left_on='\u4f01\u4e1aID',
            right_on='enterprise_id',
            how='inner',
            suffixes=('', '_llm')
        )

        print(f"✓ After merging shape: {fused_df.shape}")

        # 🔥 Dynamically detect columns that need filling
        llm_cols = ['enterprise_id', 'financial_health_risk', 'liquidity_risk',
                    'leverage_risk', 'operational_efficiency_risk',
                    'market_systemic_risk', 'risk_type', 'risk_reason']

        # Find existing LLM columns
        existing_llm_cols = [c for c in llm_cols if c in fused_df.columns]

        # Check missing values
        missing_count = fused_df[existing_llm_cols].isnull().sum()

        if missing_count.sum() > 0:
            print(f"\n--Missing values found:")
            for col, count in missing_count.items():
                if count > 0:
                    print(f"  {col}: {count} items")

            # Fill missing values
            print("\n Fill missing values...")

            # Risk-score columns (Automatically detect all columns containing 'risk' that are numeric)
            risk_score_cols = [c for c in existing_llm_cols
                               if 'risk' in c.lower()
                               and c != 'risk_type'
                               and c != 'risk_reason'
                               and fused_df[c].dtype in ['float64', 'int64']]

            for col in risk_score_cols:
                fused_df[col] = fused_df[col].fillna(5)
                print(f"  ✓ {col}: Fill with 5")

            # Risk type
            if 'risk_type' in fused_df.columns:
                fused_df['risk_type'] = fused_df['risk_type'].fillna('Healthy Enterprise')
                print(f"  ✓ risk_type: Fill with 'Healthy Enterprise'")

            # Risk reason
            if 'risk_reason' in fused_df.columns:
                fused_df['risk_reason'] = fused_df['risk_reason'].fillna('')
                print(f"  ✓ risk_reason: Fill with an empty string")

            print("✓ Missing values filled")

        self.fused_df = fused_df

        # Statistics
        print("--Fused feature statistics:")
        print(f"- Total enterprises: {len(fused_df)}")
        print(f"- Total features: {len(fused_df.columns)}")
        print(f"  ·  enterprise ID: 1 columns")
        print(f"  · Original features: {len(self.original_df.columns) - 1} columns")
        print(f"  · LLM features: {len([c for c in existing_llm_cols if c != 'enterprise_id'])} columns")

        return fused_df

    # def concat_fusion(self):
    #     """Concatenation fusion"""
    #
    #     # Merge by enterprise ID (left join; keep all enterprises from the original features)
    #     fused_df = pd.merge(
    #         self.original_df,
    #         self.llm_df,
    #         left_on=' enterprise ID',
    #         right_on='enterprise_id',
    #         how='inner',
    #         suffixes=('', '_llm')  # Avoid column-name collisions
    #     )
    #
    #     print(f"✓ After merging shape: {fused_df.shape}")
    #
    #     # Check missing values
    #     llm_cols = ['enterprise_id', 'financial_health_risk', 'liquidity_risk', 'leverage_risk', 'operational_efficiency_risk', 'market_systemic_risk', 'risk_type', 'risk_reason']
    #     missing_count = fused_df[llm_cols].isnull().sum()
    #
    #     if missing_count.sum() > 0:
    #         print(f"\n--Missing values found:")
    #         for col, count in missing_count.items():
    #             if count > 0:
    #                 print(f"  {col}: {count} items")
    #
    #         # Fill missing values
    #         print("\n Fill missing values...")
    #         # Use medium risk for risk scores(5)Fill
    #         fused_df['financial_health_risk'] = fused_df['financial_health_risk'].fillna(5)
    #         fused_df['data_quality_risk'] = fused_df['data_quality_risk'].fillna(5)
    #         fused_df['contagion_risk'] = fused_df['contagion_risk'].fillna(5)
    #         # Use for risk type "Healthy enterprise"Fill
    #         fused_df['risk_type'] = fused_df['risk_type'].fillna('Healthy Enterprise')
    #         print("✓ Missing values filled")
    #
    #     self.fused_df = fused_df
    #
    #     # Statistics
    #     print("--Fused feature statistics:")
    #     print(f"- Total enterprises: {len(fused_df)}")
    #     print(f"- Total features: {len(fused_df.columns)}")
    #     print(f"  ·  enterprise ID: 1 columns")
    #     print(f"  · Original features: {len(self.original_df.columns) - 1} columns")
    #     print(f"  · LLM features: {len(self.llm_df.columns) - 1} columns")
    #
    #     return fused_df

    def save_fused_features(self):
        """Save fused features"""
        print("\n Saving fused features...")

        self.fused_df.to_csv(self.output_path, index=False)
        print(f"--Saved to: {self.output_path}")

        # File size
        file_size = os.path.getsize(self.output_path) / 1024 / 1024
        print(f"--File size: {file_size:.2f} MB")

    def run(self):
        """Run the complete fusion pipeline"""
        # 1. Load data
        self.load_data()

        # 2. Concatenation fusion
        fused_df = self.concat_fusion()

        # 3. Save results
        self.save_fused_features()
        print("Fusion completed!\n\n")

        return fused_df


# ========== Usage example ==========
if __name__ == "__main__":

    # Create the fusion object
    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    data_file = ['2023_SH', '2023_SZ', '2022_SH', '2022_SZ', '2022_all', '2023_all', '2022_USA']    # for data in data_file:
    # for data in data_file:
    for data in ['2023_BSS', '2022_BSS']:

        print("Process : " + data)
        fusion = FeatureFusion(
            original_features_path=data_dir + data + '/csv_data/reduced_features.csv',
            llm_features_path=data_dir + data + '/data_process/llm_enhanced_features_v6_industry.csv',
            output_path=data_dir + data + '/data_process/fused_features.csv'
        )
        fusion.run()

    print("Completed!")

