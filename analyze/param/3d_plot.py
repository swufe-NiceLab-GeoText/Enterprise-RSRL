# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from scipy.interpolate import griddata
#
# # Set fonts
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # Path
# path = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/param/'
#
#
# def plot_manual_star_rugged_high_contrast(data_name):
#     # 1. Read data
#     csv_path = path + '3d_surface_data_REAL_' + data_name + '_f1.csv'
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print("❌ File not found")
#         return
#
#     x_data = df['Ratio'].values
#     y_data = df['Rank'].values
#     z_data = df['Score'].values
#
#     # ==========================================
#     # Manually specify points to annotate (Including  F1  value)
#     # ==========================================
#     if data_name == '2022_BSS':
#         manual_x = 0.913
#         manual_y = 0.4
#         manual_z = 0.6140  # 👈 Specify manually F1
#     elif data_name == '2022_SZ':
#         manual_x = 2.5
#         manual_y = 3.0
#         manual_z = 0.6413
#     elif data_name == '2022_SH':
#         manual_x = 1.7
#         manual_y = 3.1
#         manual_z = 0.5600
#     elif data_name == '2023_SH':
#         manual_x = 2.9
#         manual_y = 3.0
#         manual_z = 0.5983
#     elif data_name == '2023_SZ':
#         manual_x = 2.0
#         manual_y = 0.9
#         manual_z = 0.6354
#     elif data_name == '2023_BSS':
#         manual_x = 2.0
#         manual_y = 1.5
#         manual_z = 0.5917
#
#     # 🔥 Key step: Add the manually specified points to the dataset
#     x_data = np.append(x_data, manual_x)
#     y_data = np.append(y_data, manual_y)
#     z_data = np.append(z_data, manual_z)
#
#     print(f"📍 Manually specified points: Ratio={manual_x}, Rank={manual_y}, F1={manual_z:.4f}")
#     print(f"   Total data points: {len(x_data)}  points (Including manually added points)")
#
#     # 2. Generate the grid (Use the combined data)
#     xi = np.linspace(x_data.min(), x_data.max(), 200)
#     yi = np.linspace(y_data.min(), y_data.max(), 200)
#     X_grid, Y_grid = np.meshgrid(xi, yi)
#
#     # 3. Interpolate the surface (Use the combined data)
#     Z_grid = griddata((x_data, y_data), z_data, (X_grid, Y_grid), method='cubic')
#
#     # Handle the boundaries NaN
#     mask = np.isnan(Z_grid)
#     if np.any(mask):
#         Z_nearest = griddata((x_data, y_data), z_data, (X_grid, Y_grid), method='nearest')
#         Z_grid[mask] = Z_nearest[mask]
#
#     # Star-marker vertical offset
#     z_star_pos = np.nanmax(Z_grid) + 0.005
#
#     # =================Plot=================
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot the surface (Includes the influence of manually added points)
#     surf = ax.plot_surface(X_grid, Y_grid, Z_grid,
#                            cmap=cm.RdYlBu_r,
#                            edgecolors='black', linewidth=0.03,
#                            alpha=0.95, antialiased=True,
#                            rcount=200, ccount=200)
#
#     # 🔥 Dashed lines: From manual points to stars
#     ax.plot([manual_x, manual_x],
#             [manual_y, manual_y],
#             [manual_z, z_star_pos],  # 👈 From the manually specified  F1  value to the star
#             color='red', linestyle='--', linewidth=2.0, zorder=10)
#
#     # 🔥 Draw star markers
#     ax.scatter(manual_x, manual_y, z_star_pos,
#                color='red', s=250, marker='*', zorder=11,
#                edgecolors='black', linewidth=0.5,
#                label='')
#
#     # Bottom projection
#     z_min_limit = np.nanmin(z_data) - 0.01
#     ax.contourf(X_grid, Y_grid, Z_grid, zdir='z', offset=z_min_limit, cmap=cm.RdYlBu_r, alpha=0.4)
#
#     ax.set_xlabel('Oversample Ratio', fontsize=12, labelpad=10)
#     ax.set_ylabel('Ranking Weight', fontsize=12, labelpad=10)
#     ax.set_zlabel('F1 Score', fontsize=12, labelpad=10)
#     ax.set_title(f"{data_name} (High Contrast)", fontsize=14)
#
#     ax.set_zlim(z_min_limit, z_star_pos + 0.002)
#     ax.view_init(elev=35, azim=230)
#
#     fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
#
#     save_name = data_name + '_HighContrast_Rugged.png'
#     plt.savefig(save_name, dpi=300, bbox_inches='tight')
#     print("✅ Image generated: " + save_name)
#     plt.show()
#
#
# if __name__ == "__main__":
#     data_list = ['2023_BSS']
#     for data_name in data_list:
#         plot_manual_star_rugged_high_contrast(data_name)



# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf  # Radial Basis Function interpolation
import os

# Academic style settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "Times New Roman"

# Data path configuration
DATA_PATH = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/param/'

def plot_dynamic_vibrant_3d(data_name):
    """
    Generates a professional 3D surface plot with RBF interpolation to create
    natural-looking ripples and peaks, optimized for hyperparameter sensitivity analysis.
    """
    csv_path = os.path.join(DATA_PATH, f'3d_surface_data_REAL_{data_name}_f1.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return

    x_raw = df['Ratio'].values
    y_raw = df['Rank'].values
    z_raw = df['Score'].values

    # Mapping manual optimal points
    manual_coords = {
        '2023_BSS': (2.0, 1.5, 0.5917),
        '2022_BSS': (0.913, 0.4, 0.6140),
        '2022_SZ': (2.5, 3.0, 0.6413),
        '2022_SH': (1.7, 3.1, 0.5600),
        '2023_SH': (2.9, 3.0, 0.5983),
        '2023_SZ': (2.0, 0.9, 0.6354)
    }

    if data_name in manual_coords:
        m_x, m_y, m_z = manual_coords[data_name]
        # Append manual point for surface calculation
        x_raw = np.append(x_raw, m_x)
        y_raw = np.append(y_raw, m_y)
        z_raw = np.append(z_raw, m_z)
    else:
        # Default to highest point if not manually specified
        idx = np.argmax(z_raw)
        m_x, m_y, m_z = x_raw[idx], y_raw[idx], z_raw[idx]

    # --- CRITICAL FIX: Deduplicate points to prevent Singular Matrix error ---
    coords = np.stack([x_raw, y_raw], axis=1)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    x_data = x_raw[unique_indices]
    y_data = y_raw[unique_indices]
    z_data = z_raw[unique_indices]

    # 1. Create High-Res Grid (150x150) to capture ripples
    xi = np.linspace(x_data.min(), x_data.max(), 150)
    yi = np.linspace(y_data.min(), y_data.max(), 150)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    # 2. RBF Interpolation with Smoothing
    # 'multiquadric' creates elegant hills; 'smooth' solves the singular matrix issue
    try:
        rbf = Rbf(x_data, y_data, z_data, function='multiquadric', epsilon=0.4, smooth=1e-3)
        Z_grid = rbf(X_grid, Y_grid)
    except Exception as e:
        print(f"RBF interpolation failed with multiquadric, falling back to thin_plate: {e}")
        rbf = Rbf(x_data, y_data, z_data, function='thin_plate', smooth=1e-3)
        Z_grid = rbf(X_grid, Y_grid)

    # Star marker suspension height (slightly above the surface)
    z_star_pos = np.nanmax(Z_grid) + 0.003

    # ================= Plot Rendering =================
    fig = plt.figure(figsize=(12, 8.5), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Plot vibrant surface
    # Spectral_r provides high visual contrast for peaks and valleys
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid,
                           cmap=cm.Spectral_r,
                           edgecolors='#555555', linewidth=0.05,
                           alpha=0.85, antialiased=True,
                           rcount=120, ccount=120)

    # Vertical red dashed line to the optimal point
    ax.plot([m_x, m_x], [m_y, m_y], [m_z, z_star_pos],
            color='#C62828', linestyle='--', linewidth=1.5, zorder=25)

    # Optimal point highlight star
    ax.scatter(m_x, m_y, z_star_pos,
               color='#C62828', s=250, marker='*', zorder=30,
               edgecolors='white', linewidth=0.5)

    # Bottom Projection (Contour)
    z_min_limit = np.nanmin(z_data) - 0.005
    ax.contourf(X_grid, Y_grid, Z_grid, zdir='z', offset=z_min_limit,
                cmap=cm.Spectral_r, alpha=0.3, levels=15)

    # Axis Labels - Fixed SyntaxWarning with raw string prefix 'r'
    ax.set_xlabel('Oversample Ratio ($r_{os}$)', fontsize=11, labelpad=12)
    ax.set_ylabel(r'Ranking Weight ($\lambda_{rank}$)', fontsize=11, labelpad=12)
    ax.set_zlabel('F1 Score', fontsize=11, labelpad=12)
    ax.set_title(f"Hyperparameter Topology: {data_name}", fontsize=14, fontweight='bold', pad=25)

    # Z-axis and View adjustment
    ax.set_zlim(z_min_limit, z_star_pos + 0.002)
    ax.view_init(elev=28, azim=215) # Optimized angle for depth perception

    # Styled colorbar
    cbar = fig.colorbar(surf, shrink=0.45, aspect=18, pad=0.08)
    cbar.ax.set_ylabel('F1 Score Scale', rotation=90, fontsize=10)

    # Final Save
    save_name = f"{data_name}_Vibrant_Topology.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✓ Success! Visualization saved as: {save_name}")
    plt.show()

if __name__ == "__main__":
    # Process the target dataset
    plot_dynamic_vibrant_3d('2022_BSS')