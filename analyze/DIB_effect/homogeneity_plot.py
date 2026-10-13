# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================
# Global Style Settings (Match the provided style exactly)
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12  # Slightly smaller, Match the provided code
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# =========================
# Colors & Palettes
# =========================
BAR_COLOR = "#9AC9DB"  # Light blue (Retained edge count)
LINE_COLOR = "#C62828"  # Dark red (Homophily line)
AXIS_GREY = "#B5B5B5"
TEXT_GREY = "#333333"  # Black/Dark gray text
GRID_GREY = "#E6E6E6"


def plot_sensitivity_analysis_academic(input_csv, save_img_path, dataset_name):
    # 1. Load data
    if not os.path.exists(input_csv):
        print(f"❌ Error: File not found {input_csv}")
        return

    df = pd.read_csv(input_csv)
    target_degrees = df['target_degree'].values
    edge_counts = df['edge_count'].values
    homophily_values = df['homophily'].values

    # 2. Create the figure (Use the provided size as a reference 6 x 2.7 Or keep  7 x 4.5)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Grid and background
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0, color=GRID_GREY)
    ax1.set_axisbelow(True)

    # Border settings
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_GREY)

    # 3. Left axis (Axis 1): Retained edge count - Bar chart
    ax1.bar(target_degrees, edge_counts, color=BAR_COLOR,
            edgecolor=AXIS_GREY, linewidth=0.4, alpha=0.90,
            width=1.5, zorder=2)

    ax1.set_xlabel('Target Average Degree ($d_{target}$)', color=TEXT_GREY)
    ax1.set_ylabel('Number of Retained Edges', color=TEXT_GREY)
    ax1.tick_params(axis='y', labelcolor=TEXT_GREY)

    # 4. Right axis (Axis 2): Homophily - Line chart
    ax2 = ax1.twinx()
    # Ensure the right-axis border is visible and uses the same color
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_GREY)

    ax2.plot(target_degrees, homophily_values, color=LINE_COLOR,
             marker='D', markersize=6, linewidth=1.6,
             markeredgecolor="white", markeredgewidth=0.8,
             label='Homophily', zorder=3)

    # --- Key change: Right-side  Y  axis text changed to black ---
    ax2.set_ylabel('Label Homophily Score', color=TEXT_GREY)
    ax2.tick_params(axis='y', labelcolor=TEXT_GREY)

    # 5. Title and legend
    plt.title(f'Graph Denoising Sensitivity: {dataset_name}', pad=12)

    legend_handles = [
        Patch(facecolor=BAR_COLOR, edgecolor=AXIS_GREY, linewidth=0.4, alpha=0.90, label='Retained Edges'),
        Line2D([0], [0], color=LINE_COLOR, marker='D', markersize=6,
               linewidth=1.6, markeredgecolor="white", label='Homophily Score')
    ]

    ax1.legend(handles=legend_handles, loc='upper left',
               bbox_to_anchor=(0.02, 0.98),  # Adjust positions slightly
               frameon=True, facecolor='white', edgecolor=GRID_GREY,
               framealpha=0.95, fontsize=9)

    # 6. Save as an image (PNG)
    # Removed  PDF  saving logic, Keep only high-quality  PNG
    out_path = Path(save_img_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor="white")

    plt.show()
    plt.close(fig)
    print(f"✅ Image exported successfully: {out_path}")


if __name__ == "__main__":

    data_name = '2023_SH'

    CSV_PATH = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/DIB_effect/' + data_name + '_sensitivity_results.csv'
    # This path does not need an extension, The code adds it automatically .png
    IMG_PATH = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/DIB_effect/' + data_name + '_denoising_sensitivity_analysis'

    plot_sensitivity_analysis_academic(CSV_PATH, IMG_PATH, data_name)