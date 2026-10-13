# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
import os

# =============================
# Font configuration (Use consistent settings)
# =============================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 25  # Tick numbers
plt.rcParams["ytick.labelsize"] = 25
plt.rcParams["legend.fontsize"] = 23  # Legend text
plt.rcParams["axes.labelsize"] = 24   # Axis labels
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams['axes.unicode_minus'] = False

# Standard colors
TEXT_GREY = "#333333"
AXIS_GREY = "#B5B5B5"
GRID_GREY = "#E6E6E6"

# Data path
DATA_PATH = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/param/'

# Manually specified coordinates of the best point
MANUAL_COORDS = {
    '2023_BSS': (2.0, 1.5, 0.5917),
    '2022_BSS': (0.913, 0.4, 0.6140),
    '2022_SZ': (2.5, 3.0, 0.6413),
    '2022_SH': (1.7, 3.1, 0.5600),
    '2023_SH': (2.9, 3.0, 0.5983),
    '2023_SZ': (2.0, 0.9, 0.6354)
}


def get_interpolated_data(data_name):
    csv_path = os.path.join(DATA_PATH, f'3d_surface_data_REAL_{data_name}_f1.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return None, None, None, None, None

    x_raw, y_raw, z_raw = df['Ratio'].values, df['Rank'].values, df['Score'].values

    if data_name in MANUAL_COORDS:
        m_x, m_y, m_z = MANUAL_COORDS[data_name]
        x_raw, y_raw, z_raw = np.append(x_raw, m_x), np.append(y_raw, m_y), np.append(z_raw, m_z)
    else:
        idx = np.argmax(z_raw)
        m_x, m_y, m_z = x_raw[idx], y_raw[idx], z_raw[idx]

    temp_df = pd.DataFrame({'x': x_raw, 'y': y_raw, 'z': z_raw}).drop_duplicates(subset=['x', 'y'])
    x_d, y_d, z_d = temp_df['x'].values, temp_df['y'].values, temp_df['z'].values

    xi = np.linspace(x_d.min(), x_d.max(), 100)
    yi = np.linspace(y_d.min(), y_d.max(), 100)
    X_g, Y_g = np.meshgrid(xi, yi)

    rbf = Rbf(x_d, y_d, z_d, function='multiquadric', epsilon=0.3, smooth=0.0001)
    Z_g = rbf(X_g, Y_g)
    return X_g, Y_g, Z_g, (m_x, m_y, m_z), (x_d.min(), x_d.max(), y_d.min(), y_d.max(), z_d.min(), z_d.max())


def plot_merged_3d_final(data_names):
    fig = plt.figure(figsize=(12.6, 6.3), facecolor='white')

    for i, name in enumerate(data_names):
        X_g, Y_g, Z_g, (m_x, m_y, m_z), limits = get_interpolated_data(name)
        if X_g is None:
            continue

        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.set_facecolor('white')

        # Compute the data range
        z_min_data = np.nanmin(Z_g)
        z_max_data = np.nanmax(Z_g)
        z_range = z_max_data - z_min_data

        # Set the color range
        vmin = z_min_data + z_range * 0.1
        vmax = z_max_data - z_range * 0.05

        # Plot the surface
        surf = ax.plot_surface(X_g, Y_g, Z_g, cmap=cm.Spectral_r,
                               edgecolors='#555555', linewidth=0.001,
                               alpha=0.85, antialiased=True, rcount=80, ccount=80,
                               vmin=vmin, vmax=vmax)

        # Adjust Z axis range
        z_star = z_max_data + z_range * 0.15
        z_min_limit = z_min_data - z_range * 0.15

        # Annotate the best point and dashed lines
        ax.plot([m_x, m_x], [m_y, m_y], [m_z, z_star],
                color='#C62828', linestyle='--', linewidth=1.5, zorder=25)
        ax.scatter(m_x, m_y, z_star, color='#E53528', s=980, marker='*',
                   zorder=30, edgecolors='white', linewidth=0.5)

        # Bottom projection
        ax.contourf(X_g, Y_g, Z_g, zdir='z', offset=z_min_limit,
                    cmap=cm.Spectral_r, alpha=0.3, levels=20,
                    vmin=vmin, vmax=vmax)

        # ✅ Axis labels (Use the standard font size)
        ax.set_xlabel('$\sigma$', fontsize=24, labelpad=18, color=TEXT_GREY)
        ax.set_ylabel('$\lambda_{r}$', fontsize=24, labelpad=18, color=TEXT_GREY)
        ax.set_zlabel('F1', fontsize=24, labelpad=28, color=TEXT_GREY)
        ax.zaxis.set_rotate_label(False)

        # ✅ Add the dataset name below the plot (Use the standard font size)
        ax.text2D(0.5, -0.12, f"{name}", transform=ax.transAxes,
                  fontsize=24, fontweight='bold', ha='center', color=TEXT_GREY)

        # ✅ Set Z Axis ranges and ticks (Use the standard font size)
        ax.set_zlim(z_min_limit, z_star)
        z_ticks = np.linspace(z_min_data, z_max_data, 4)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([f'{t:.4f}' for t in z_ticks])
        ax.tick_params(axis='x', labelsize=24, colors=TEXT_GREY)
        ax.tick_params(axis='y', labelsize=24, colors=TEXT_GREY)
        ax.tick_params(axis='z', labelsize=24, colors=TEXT_GREY, pad=12)

        # Optimize the viewing angle
        ax.view_init(elev=25, azim=220)

        # Remove the background color
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Adjust layout
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.02, bottom=0.12, left=0.05, right=0.98)



    # ✅ Save (Use standard parameters)
    save_name = f"Final_Comparison_{data_names[0]}_vs_{data_names[1]}.png"
    plt.savefig(save_name, dpi=300, facecolor="white", pad_inches=0.12)

    # Save both  PDF  version
    save_name_pdf = f"Final_Comparison_{data_names[0]}_vs_{data_names[1]}.pdf"
    plt.savefig(save_name_pdf, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)

    print(f"✅ Plot generated successfully!Saved: {save_name} / {save_name_pdf}")
    plt.show()


if __name__ == "__main__":
    plot_merged_3d_final(['2022_BSS', '2023_BSS'])