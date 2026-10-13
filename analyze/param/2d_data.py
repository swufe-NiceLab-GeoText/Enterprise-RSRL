import pandas as pd
import numpy as np
# ⚠️ Import the previous training function!
# Assume the training code is in  test.py ,  and the function is named  train_and_evaluate_single
from  src.train  import train_and_evaluate

# =================Configuration=================
# Enter the best seed from previous runs here (Seed 80  or  109)
TEST_SEEDS = [109]
DATASETS = ["2022_BSS"]  # # Select datasets
DATA_DIR = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'

# This is the previously selected best fixed parameter configuration (Except for  Ratio  and  Rank  which varies)
FIXED_CONFIG = {'hidden': 64, 'binning': False}

# Grid range (Keep unchanged)
X_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Ratio
Y_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Rank


def run_real_grid_search():

    for i in range(2):

        DATASET = DATASETS[i]
        TEST_SEED = TEST_SEEDS[i]

        results = []
        print(f"🚀 Start [Actual]2D Grid search | Dataset: {DATASET} | Seed: {TEST_SEED}")
        print("⚠️ Note: This will take some time, Each parameter combination retrains the model...")

        for ratio in X_RANGE:
            for rank in Y_RANGE:
                # 1. Construct the current configuration
                config = {
                    'ratio': ratio,
                    'lambda_rank': rank,
                    'hidden': FIXED_CONFIG['hidden'],
                    'binning': FIXED_CONFIG['binning']
                }

                try:
                    # 2. 🔥[Key change]Call the actual training function here!
                    metrics = train_and_evaluate(DATA_DIR, DATASET, TEST_SEED, config)

                    # Extract the actual  ROC-AUC
                    real_score = metrics['f1']

                    print(f"  Ratio={ratio}, Rank={rank} -> ROC={real_score:.4f}")

                    results.append({
                        'Ratio': ratio,
                        'Rank': rank,
                        'Score': real_score
                    })
                except Exception as e:
                    print(f"❌ Error at R={ratio}, K={rank}: {e}")
                    # If an error occurs, Insert a placeholder, Avoid interrupting the program
                    results.append({'Ratio': ratio, 'Rank': rank, 'Score': 0.0})

        # Save actual data
        path = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/param/'
        df = pd.DataFrame(results)
        df.to_csv(path + '3d_surface_data_REAL_' + DATASET +'_f1.csv', index=False)
        print('\n✅ Actual data saved to 3d_surface_data_REAL_' + DATASET + '_f1.csv')


if __name__ == "__main__":
    run_real_grid_search()