import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==============================================================================
# 📌 Font-size configuration - Configure centrally here
# ==============================================================================
FONT_SIZE_LEGEND = 16          # Legend font size
FONT_SIZE_TICK = 24            # Axis tick-number font size
FONT_SIZE_LABEL = 24           # Axis labels (Y Axis name)Font size
FONT_SIZE_XTICK = 24           # X Axis tick font size

# ==============================================================================
# 📌 Global style configuration (Use consistent settings)
# ==============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = FONT_SIZE_LABEL  # Use variables
plt.rcParams["xtick.labelsize"] = FONT_SIZE_XTICK  # Use variables
plt.rcParams["ytick.labelsize"] = FONT_SIZE_TICK   # Use variables
plt.rcParams["legend.fontsize"] = FONT_SIZE_LEGEND # Use variables
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# Standard colors
BAR_COLOR_NORMAL = "#9AC9DB"  # Light blue
BAR_COLOR_HIGHLIGHT = "#205AA7"  # Dark blue (HGT)
LINE_COLOR_NORMAL = "#D08985"  # Muted pink line
LINE_COLOR_HIGHLIGHT = "#C62828"  # Red star (HGT)
AXIS_GREY = "#B5B5B5"
TEXT_GREY = "#333333"
GRID_GREY = "#E6E6E6"

CONFIG = {
    'input_csv': '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/time/2022_BSS_model_time_statistics_final.csv',
    'output_png': '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/time/2022_BSS_efficiency_analysis_paper_style.png',
    'highlight_model': 'RSRL'
}


def plot_efficiency_paper_style(csv_path, output_path):
    # Read and sort (Sort by inference time, Give bars a stepped appearance)
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('infer_time', ascending=True).reset_index(drop=True)

    models = df_sorted['model'].tolist()
    train_times = df_sorted['train_time'].tolist()
    infer_times = df_sorted['infer_time'].tolist()
    x = np.arange(len(models))

    fig, ax1 = plt.subplots(figsize=(12, 5), facecolor='white')

    # Grid-line settings
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0, color=GRID_GREY)
    ax1.set_axisbelow(True)

    # Set border colors
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_GREY)
        spine.set_linewidth(0.8)

    # ========== Left axis (ax1):  inference time (Bar chart) ==========
    colors = [BAR_COLOR_HIGHLIGHT if m == CONFIG['highlight_model'] else BAR_COLOR_NORMAL for m in models]
    bars = ax1.bar(x, infer_times, width=0.6, color=colors,
                   edgecolor=AXIS_GREY, linewidth=0.6, alpha=0.9, zorder=2)

    # ✅ Y Axis labels use variables
    ax1.set_ylabel("Inference Time (s)", color=TEXT_GREY, fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis='y', labelcolor=TEXT_GREY, labelsize=FONT_SIZE_TICK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=35, ha='right', color=TEXT_GREY, fontsize=FONT_SIZE_XTICK)

    # ========== Right axis (ax2): Training time (Line chart) ==========
    ax2 = ax1.twinx()

    # Set the right-axis border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_GREY)
        spine.set_linewidth(0.8)
    ax2.spines['top'].set_visible(False)  # Hide the top border

    ax2.plot(x, train_times, color=LINE_COLOR_NORMAL, linewidth=2,
             marker='o', markersize=6, markerfacecolor='white',
             markeredgewidth=1.5, alpha=0.9, zorder=3, label='Training Time')

    # For  RSRL  add a red star highlight
    if CONFIG['highlight_model'] in models:
        idx = models.index(CONFIG['highlight_model'])
        ax2.scatter(idx, train_times[idx], s=180, c=LINE_COLOR_HIGHLIGHT,
                    marker="*", zorder=5, edgecolors="white", linewidths=0.8)

    # ✅ Y Axis labels use variables
    ax2.set_ylabel("Total Training Time (s)", color=TEXT_GREY, fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis='y', labelcolor=TEXT_GREY, labelsize=FONT_SIZE_TICK)

    # ========== ✅ Customize the legend - Upper left inside the plot, Display in two rows ==========
    legend_handles = [
        Patch(facecolor=BAR_COLOR_NORMAL, edgecolor=AXIS_GREY, alpha=0.9, label="Inference Time (Baseline)"),
        Patch(facecolor=BAR_COLOR_HIGHLIGHT, edgecolor=AXIS_GREY, alpha=0.9,
              label=f"Inference Time ({CONFIG['highlight_model']})"),
        Line2D([0], [0], color=LINE_COLOR_NORMAL, marker='o', markerfacecolor='white',
               linewidth=2, label="Training Time (Baseline)"),
        Line2D([0], [0], color='none', marker='*', markerfacecolor=LINE_COLOR_HIGHLIGHT,
               markersize=12, markeredgecolor="white", label=f"Training Time ({CONFIG['highlight_model']})")
    ]

    # ✅ Legend at the upper left inside the plot, ncol=2 Two rows, Font size uses a variable
    ax1.legend(handles=legend_handles,
               loc="upper left",
               bbox_to_anchor=(0.02, 0.98),
               ncol=2,                          # 2 columns, Two rows
               frameon=True,
               facecolor="white",
               edgecolor=AXIS_GREY,
               framealpha=0.95,
               borderpad=0.6,
               labelspacing=0.4,
               columnspacing=1.0,
               fontsize=FONT_SIZE_LEGEND)       # ✅ Use variables

    # Adjust layout automatically and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)

    # Save both  PDF  version
    output_pdf = output_path.replace('.png', '.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)

    print(f"✅ Paper-style chart saved to: {output_path}")
    print(f"✅ PDF  version saved to: {output_pdf}")
    plt.show()


if __name__ == "__main__":
    plot_efficiency_paper_style(CONFIG['input_csv'], CONFIG['output_png'])