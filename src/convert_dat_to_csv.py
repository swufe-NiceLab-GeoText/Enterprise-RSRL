import pandas as pd
import numpy as np
import os
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')


class DatToCSVConverter:
    """
    High-performance data converter: Optimized for tens of millions of records, Includes row-count auditing
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir.rstrip('/') + '/'

    def count_file_lines(self, file_path):
        """Count source-file rows quickly (Without loading the file into memory)"""
        count = 0
        with open(file_path, 'rb') as f:
            for _ in f:
                count += 1
        return count

    def load_features(self, npy_path='features_norm.npy'):
        print(f"--- Loading feature matrix: {npy_path} ---")
        self.features = np.load(self.data_dir + npy_path)
        print(f"    [Check] Feature-matrix shape: {self.features.shape}")
        return self.features

    def load_label_dat(self, label_path='label.dat'):
        full_path = self.data_dir + label_path
        raw_count = self.count_file_lines(full_path)
        print(f"--- Reading label file: {label_path} (Actual file row count: {raw_count}) ---")

        self.labels_df = pd.read_csv(
            full_path,
            sep='\t',
            header=None,
            usecols=[0, 3],
            names=['\u4f01\u4e1aID', '\u6807\u7b7e'],
            dtype={'\u4f01\u4e1aID': 'int32', '\u6807\u7b7e': 'int8'},
            engine='python',  # Use python engine to tolerate unusual characters
            on_bad_lines='warn'
        )

        print(f"    [Check] Rows read successfully: {len(self.labels_df)}")
        return self.labels_df

    def load_link_dat(self, link_path='link.dat'):
        full_path = self.data_dir + link_path
        # 1. Count physical lines first
        raw_count = self.count_file_lines(full_path)
        print(f"--- Reading relation file: {link_path} ---")
        print(f"    [Physical count] Actual lines in the file: {raw_count}")

        # 2. Attempt high-performance parsing
        # If parsed and physical line counts differ, the file usually contains  Null  bytes(\x00)
        self.relations_df = pd.read_csv(
            full_path,
            sep='\t',
            header=None,
            names=['\u4f01\u4e1aA', '\u4f01\u4e1aB', '\u5173\u7cfb\u7c7b\u578b', '\u6743\u91cd'],
            dtype={'\u4f01\u4e1aA': 'int32', '\u4f01\u4e1aB': 'int32', '\u5173\u7cfb\u7c7b\u578b': 'int8', '\u6743\u91cd': 'float32'},
            engine='python',  # Switch to python  engine, Better compatibility with complex formats
            on_bad_lines='skip',  # Skip individual malformed lines
            encoding_errors='ignore'  # Ignore decoding errors to prevent interruption
        )

        print(f"    [Check] Rows successfully parsed and loaded: {len(self.relations_df)}")

        if len(self.relations_df) < raw_count:
            print(f"    ⚠️ Warning: Fewer parsed rows than physical file lines!(Shortfall: {raw_count - len(self.relations_df)}  rows)")
            print("    Check the file for unusual control characters or null bytes.")

        return self.relations_df

    def generate_features_csv(self):
        print("--- Generating the enterprise feature table ---")
        num_nodes, _ = self.features.shape
        id_col = pd.DataFrame({'\u4f01\u4e1aID': np.arange(num_nodes, dtype='int32')})
        feat_cols = pd.DataFrame(self.features).astype('float32')
        feat_cols.columns = [f'feature_{i}' for i in range(feat_cols.shape[1])]
        self.features_df = pd.concat([id_col, feat_cols], axis=1)
        return self.features_df

    def save_all(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Exporting data to: {output_dir} ---")

        # Save and verify
        paths = {
            '\u7279\u5f81\u8868': f'{output_dir}/enterprise_features.csv',
            '\u6807\u7b7e\u8868': f'{output_dir}/enterprise_labels.csv',
            '\u5173\u7cfb\u8868': f'{output_dir}/enterprise_relations.csv'
        }

        self.features_df.to_csv(paths['\u7279\u5f81\u8868'], index=False)
        self.labels_df.to_csv(paths['\u6807\u7b7e\u8868'], index=False)
        self.relations_df.to_csv(paths['\u5173\u7cfb\u8868'], index=False)

        print("\n" + "=" * 30)
        print("Final conversion row-count report:")
        print(f"1. Feature records: {len(self.features_df)}  entries")
        print(f"2. Label data: {len(self.labels_df)}  entries")
        print(f"3. Relation records: {len(self.relations_df)}  entries")
        print("=" * 30)
        print("✅ All conversion tasks completed!")

    def run(self, output_dir):
        self.load_features()
        self.load_label_dat()
        self.load_link_dat()
        self.generate_features_csv()
        self.save_all(output_dir)


if __name__ == "__main__":
    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    # Target subdirectory
    target_folders = ['2023_BSS', '2022_BSS']

    for folder in target_folders:
        print(f"\n🚀 Start processing directory: {folder}")
        path = os.path.join(data_dir, folder)
        if os.path.exists(path):
            converter = DatToCSVConverter(data_dir=path)
            converter.run(output_dir=os.path.join(path, 'csv_data'))
        else:
            print(f"❌ Error: Path not found {path}")
