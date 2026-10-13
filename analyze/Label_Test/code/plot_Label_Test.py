# -*- coding: utf-8 -*-
"""
Label-permutation test visualization (Ablation version)
- Compare industry-standard-based scores with  vs  shuffled-label results without industry standards
- Use dark blue and + light blue colors
- Do not display numeric labels
- Do not display the random baseline
- Optimize the legend layout, Avoid covering the bars
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Global style settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# =========================
# Color configuration (Use the provided figure as a reference)
# =========================
BAR_COLOR_ORIGINAL = "#9AC9DB"  # Light blue (Original labels)
BAR_COLOR_WITH_STD = "#205AA7"  # Dark blue (With industry standards+Shuffled)
BAR_COLOR_NO_STD = "#6B7C93"  # Gray-blue (Without industry standards+Shuffled)
AXIS_GREY = "#B5B5B5"
TEXT_GREY = "#333333"
GRID_GREY = "#E6E6E6"


def plot_ablation_comparison(csv_path, output_dir='./permutation_results_ablation'):
    """
    Read CSV file and generate ablation comparison bar charts
    - Three bars: Original labels, With industry standards+Shuffled, Without industry standards+Shuffled
    - Do not display numeric labels
    - Do not display the random baseline
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract the dataset name
    dataset_name = Path(csv_path).stem.replace('_Ablation_Test', '')

    # Arrange metrics in the specified order: F1, ROC-AUC, PR-AUC, KS
    metric_order = ['F1', 'ROC-AUC', 'PR-AUC', 'KS']
    df_sorted = df.set_index('Metric').loc[metric_order].reset_index()

    # Extract data
    metrics = df_sorted['Metric'].values
    original = df_sorted['Original_WithStd'].values
    permuted_with_std = df_sorted['Permuted_WithStd'].values
    permuted_no_std = df_sorted['Permuted_NoStd'].values

    # Create the figure
    fig, ax = plt.subplots(figsize=(7.5, 4))

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0, color=GRID_GREY)
    ax.set_axisbelow(True)

    # Borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_GREY)
        spine.set_linewidth(0.8)

    # Draw bar charts (Three bars)
    x = np.arange(len(metrics))
    bar_width = 0.25

    bars1 = ax.bar(
        x - bar_width, original, bar_width,
        label='Original Labels',
        color=BAR_COLOR_ORIGINAL,
        edgecolor=AXIS_GREY,
        linewidth=0.6,
        alpha=0.90,
        zorder=2
    )

    bars2 = ax.bar(
        x, permuted_with_std, bar_width,
        label='Permuted (With Standard)',
        color=BAR_COLOR_WITH_STD,
        edgecolor=AXIS_GREY,
        linewidth=0.6,
        alpha=0.90,
        zorder=2
    )

    bars3 = ax.bar(
        x + bar_width, permuted_no_std, bar_width,
        label='Permuted (No Standard)',
        color=BAR_COLOR_NO_STD,
        edgecolor=AXIS_GREY,
        linewidth=0.6,
        alpha=0.90,
        zorder=2
    )

    # Set labels
    ax.set_xlabel('Metrics', color=TEXT_GREY, fontweight='bold')
    ax.set_ylabel('Score', color=TEXT_GREY, fontweight='bold')
    ax.set_title(
        f'Label Permutation Test (Ablation): {dataset_name}',
        color=TEXT_GREY, fontweight='bold', pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.tick_params(axis='both', colors=TEXT_GREY)

    # y axis range
    all_values = np.concatenate([original, permuted_with_std, permuted_no_std])
    y_max = all_values.max()
    ax.set_ylim(0, y_max * 1.15)

    # Legend (Place at the upper right, Avoid covering the bars)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),  # Near the upper right
        frameon=True,
        facecolor='white',
        edgecolor=GRID_GREY,
        framealpha=0.95,
        borderpad=0.5,
        labelspacing=0.3
    )

    # Adjust layout
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_Ablation_Result.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(
        output_path, dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.1
    )

    print(f"✅ Image saved: {output_path}")

    plt.close()

    return output_path


def batch_plot_all_ablation_results(csv_dir='./permutation_results_ablation',
                                    output_dir='./permutation_results_ablation'):
    """
    Batch process all CSV files and generate images
    """
    # Find all CSV file
    csv_files = list(Path(csv_dir).glob('*_Ablation_Test.csv'))

    if not csv_files:
        print(f"⚠️  In  {csv_dir}  contains no matching  *_Ablation_Test.csv  file")
        return []

    print("=" * 80)
    print(f"📊 Start generating ablation comparison plots")
    print(f"Found {len(csv_files)}  items CSV file")
    print("=" * 80)

    saved_files = []
    for csv_path in csv_files:
        print(f"\n Process : {csv_path.name}")
        try:
            output_path = plot_ablation_comparison(str(csv_path), output_dir)
            saved_files.append(output_path)
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"✅ Completed!Generated a total of  {len(saved_files)}  images")
    print("=" * 80)
    for f in saved_files:
        print(f"  📄 {f}")
    print("=" * 80)

    return saved_files


if __name__ == '__main__':
    # Batch process all CSV file
    batch_plot_all_ablation_results(
        csv_dir='/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/Label_Test/results/',
        output_dir='/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/Label_Test/results/'
    )