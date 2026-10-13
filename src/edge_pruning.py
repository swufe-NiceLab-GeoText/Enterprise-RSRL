
# edge_pruning_kdd.py
"""
Heterogeneous graph edge pruning using a risk-contagion-oriented information bottleneck

Paper title (Suggested): 
"Risk-Oriented Deterministic Information Bottleneck for
 Heterogeneous Graph Pruning in Enterprise Risk Prediction"

Contributions: 
1. Risk-contagion-oriented self-supervised targets (Feature differences + Structural bridges + Relation strength)
2. Dynamic retention-ratio strategy (Adapt to graph size and density)
3. Graph homophily invariance constraint (Ensure fairness, Avoid label leakage)
4. Multi-metric evaluation framework (F1, ROC-AUC, PR-AUC, KS)

Applications: 
- Financial risk management: Enterprise risk contagion prediction
- Social networks: Fraud detection
- Recommender systems: Noisy-edge filtering

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from collections import defaultdict
import warnings

# Keep existing dataset column names compatible with stored CSV files.
_ENTERPRISE_A = '\u4f01\u4e1aA'
_ENTERPRISE_B = '\u4f01\u4e1aB'


warnings.filterwarnings('ignore')


# ==========================================
# 1. Deterministic information bottleneck model (Optimized version)
# ==========================================
# edge_pruning_kdd.py

class RiskOrientedDIB(nn.Module):
    """
    Risk-Oriented DIB (Optimized version v2)

    Fix: 
    1. Remove Tanh activation (Allow Z a wider range)
    2. Adjust the compression-term scale
    3. Numerical stability
    """

    def __init__(self, input_dim, z_dim=32, beta=0.1):
        super().__init__()
        self.beta = beta
        self.z_dim = z_dim

        # ✅ Encoder: Linear output
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, z_dim)  # ✅ No activation function
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def compute_ib_loss(self, x, target):
        """
        DIB loss = Compression term + β × Reconstruction term

        Compression term: Allow Z to contain less information
        Reconstruction term: Allow Z to retain contagion information
        """
        # Encode
        z = self.encode(x)

        # Decode
        t_pred = self.decode(z)

        # ✅ Compression term: Z mean L2 norm
        z_norm = torch.norm(z, p=2, dim=1).mean()
        compression_term = z_norm * 0.1  # Scale to a reasonable range

        # Reconstruction term: Accuracy of contagion prediction
        reconstruction_loss = F.binary_cross_entropy(
            t_pred.squeeze(),
            target
        )

        # Total loss
        loss = compression_term + self.beta * reconstruction_loss

        return loss, compression_term, reconstruction_loss

    def get_edge_importance(self, x):
        """
        Edge importance = Compressed information content

        Important edges: Z have large norms (Information-rich)
        Unimportant edges: Z have small norms (Redundant)
        """
        z = self.encode(x)
        importance = torch.norm(z, p=2, dim=1)

        return importance


# ==========================================
# 2. ✨ Risk-contagion-oriented self-supervised target
# ==========================================
def build_risk_oriented_targets(edges_df, node_to_idx, x):
    """
    Build risk-contagion-oriented self-supervised targets

    Core contribution: From "similarity" to "contagion"

    Properties of risk-contagion edges: 
    1. Connect nodes with different features (Heterogeneous contagion)
    2. Structural bridge edges (Cross-community propagation)
    3. Strong relationships (High influence)
    4. Connect high-centrality nodes (Hub contagion)

    Mathematical form: 
    T_risk = α × Feature_Diff + β × Bridge_Score + γ × Strength + δ × Centrality

    Where: 
    - Feature_Diff = 1 - cosine(u, v)
    - Bridge_Score = 1 - jaccard(u, v)
    - Strength = normalized_weight
    - Centrality = degree[u] × degree[v]
    """
    print("  Building risk-contagion-oriented self-supervised targets...")

    # 1. Build adjacency lists (For computing Jaccard and degrees)
    adj_dict = defaultdict(set)
    degree_dict = defaultdict(int)

    for _, row in edges_df.iterrows():
        u, v = row['\u4f01\u4e1aA'], row['\u4f01\u4e1aB']
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            adj_dict[u_idx].add(v_idx)
            adj_dict[v_idx].add(u_idx)
            degree_dict[u_idx] += 1
            degree_dict[v_idx] += 1

    # 2. Compute each component
    feature_diffs = []
    bridge_scores = []
    strengths = []
    centralities = []

    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df),
                       desc="  Build targets"):
        u, v = row['\u4f01\u4e1aA'], row['\u4f01\u4e1aB']

        if u not in node_to_idx or v not in node_to_idx:
            feature_diffs.append(0.5)
            bridge_scores.append(0.5)
            strengths.append(0.5)
            centralities.append(0.5)
            continue

        u_idx, v_idx = node_to_idx[u], node_to_idx[v]

        # Dimension 1: Feature differences (Heterogeneity)
        cosine = (x[u_idx] * x[v_idx]).sum().item()
        feature_diff = 1 - cosine  # Larger differences,  may imply stronger contagion
        feature_diffs.append(feature_diff)

        # Dimension 2: Bridge score (Structure)
        neighbors_u = adj_dict[u_idx]
        neighbors_v = adj_dict[v_idx]

        if len(neighbors_u) == 0 or len(neighbors_v) == 0:
            jaccard = 0.0
        else:
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            jaccard = intersection / union if union > 0 else 0.0

        bridge_score = 1 - jaccard  # Jaccard Low  =  bridge edges
        bridge_scores.append(bridge_score)

        # Dimension 3: Relation strength
        strength = row['\u6743\u91cd']
        strengths.append(strength)

        # Dimension 4: Centrality (Connect important nodes)
        degree_u = degree_dict[u_idx]
        degree_v = degree_dict[v_idx]
        centrality = np.sqrt(degree_u * degree_v)  # Geometric mean
        centralities.append(centrality)

    # 3. Normalize
    feature_diffs = np.array(feature_diffs)
    bridge_scores = np.array(bridge_scores)
    strengths = np.array(strengths)
    centralities = np.array(centralities)

    feature_diffs = (feature_diffs - feature_diffs.min()) / (feature_diffs.max() - feature_diffs.min() + 1e-8)
    bridge_scores = (bridge_scores - bridge_scores.min()) / (bridge_scores.max() - bridge_scores.min() + 1e-8)
    strengths = (strengths - strengths.min()) / (strengths.max() - strengths.min() + 1e-8)
    centralities = (centralities - centralities.min()) / (centralities.max() - centralities.min() + 1e-8)

    # 4. Fusion (Adjustable weights)
    alpha, beta, gamma, delta = 0.35, 0.30, 0.25, 0.10  # Tunable parameters

    targets = (
            alpha * feature_diffs +
            beta * bridge_scores +
            gamma * strengths +
            delta * centralities
    )

    # Normalize again to [0, 1]
    targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)

    print(f"    ✓ Target statistics:")
    print(f"      Mean feature difference: {feature_diffs.mean():.3f}")
    print(f"      Mean bridge score: {bridge_scores.mean():.3f}")
    print(f"      Mean relation strength: {strengths.mean():.3f}")
    print(f"      Mean centrality: {centralities.mean():.3f}")
    print(f"      Combined target range: [{targets.min():.3f}, {targets.max():.3f}]")

    return torch.tensor(targets, dtype=torch.float32)


# ==========================================
# 3. ✨ Fast DIB training (Optimized version)
# ==========================================
def compute_dib_scores_fast(edges_df, features_df, node_to_idx,
                            z_dim=32, beta=0.1, epochs=100):
    """
    Fast DIB training (KDD Optimized version)

    Optimizations: 
    1. epochs: 300 → 100
    2. More aggressive Early Stopping (patience=20)
    3. Learning-rate scheduling (Cosine Annealing)
    4. Gradient clipping (Prevent exploding gradients)
    """
    print(f"  Using Risk-Oriented DIB to compute edge importance...")
    print(f"    Hyperparameters: z_dim={z_dim}, beta={beta}, max_epochs={epochs}")

    # 1. Prepare node features
    exclude_cols = ['\u4f01\u4e1aID', 'risk_type', 'risk_reason', 'cluster', 'label', '\u6807\u7b7e']
    feat_cols = [c for c in features_df.columns if
                 c not in exclude_cols and features_df[c].dtype in ['float64', 'int64']]

    feat_values = features_df[feat_cols].fillna(0).values
    x = torch.tensor(feat_values, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)

    print(f"    Node features: {x.shape}")

    # 2. Build edge features
    valid_edges = []
    src_indices = []
    dst_indices = []

    for idx, row in enumerate(edges_df.itertuples()):
        u, v = getattr(row, _ENTERPRISE_A), getattr(row, _ENTERPRISE_B)
        if u in node_to_idx and v in node_to_idx:
            src_indices.append(node_to_idx[u])
            dst_indices.append(node_to_idx[v])
            valid_edges.append(idx)

    src = torch.tensor(src_indices, dtype=torch.long)
    dst = torch.tensor(dst_indices, dtype=torch.long)

    edge_features = torch.cat([x[src], x[dst]], dim=1)
    print(f"    Valid edge features: {edge_features.shape}")

    # 3. ✨ Build risk-contagion-oriented self-supervised targets
    targets = build_risk_oriented_targets(edges_df, node_to_idx, x)
    targets = targets[valid_edges]  # Keep targets for valid edges only

    # 4. Initialize DIB Model
    model = RiskOrientedDIB(
        input_dim=x.shape[1],
        z_dim=z_dim,
        beta=beta
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5.  training DIB
    print("     training Risk-Oriented DIB...")
    model.train()

    best_loss = float('inf')
    patience = 0
    max_patience = 20  # ✨ More aggressive Early Stopping

    for epoch in range(epochs):
        optimizer.zero_grad()

        # DIB loss
        loss, mi_xz, recon_loss = model.compute_ib_loss(edge_features, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"      ✓ Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 1 == 0:
            print(f"      Epoch {epoch + 1}/{epochs}")
            print(f"        Loss: {loss.item():.4f} | I(X;Z): {mi_xz.item():.4f} | Recon: {recon_loss.item():.4f}")
            print(f"        LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 6. Extract edge importance
    model.eval()
    with torch.no_grad():
        dib_scores_valid = model.get_edge_importance(edge_features).numpy()

    # 7. Map back to original edge indices
    dib_scores = np.zeros(len(edges_df))
    dib_scores[valid_edges] = dib_scores_valid

    # Normalize
    dib_scores = (dib_scores - dib_scores.min()) / \
                 (dib_scores.max() - dib_scores.min() + 1e-8)

    print(f"    ✓ DIB Score: [{dib_scores.min():.4f}, {dib_scores.max():.4f}], Mean: {dib_scores.mean():.4f}")

    return dib_scores


# ==========================================
# 4. ✨ Dynamic retention-ratio strategy
# ==========================================
def compute_dynamic_retention_ratio(num_nodes, num_edges):
    """
    Adjust the retention ratio dynamically based on graph size

    Core idea: 
    - Denser graphs,  lower retention ratios (Greater denoising needs)
    - Sparser graphs,  higher retention ratios (Preserve connectivity)
    - Keep the average degree after pruning within a reasonable range

    Target average degree: 200-300 (Empirical value, Suitable for GNN)

    Formula: 
    retention_ratio = min(max((target_degree * N) / (2 * M), min_ratio), max_ratio)
    """
    avg_degree = 2 * num_edges / num_nodes
    target_avg_degree = 30  # Adjust target average degree with graph size

    # Compute the retention ratio
    retention_ratio = (target_avg_degree * num_nodes) / (2 * num_edges)

    # Bounds
    min_ratio = 0.0  # Retain at least 5%
    max_ratio = 0.3  # Retain at most 30%

    retention_ratio = min(max(retention_ratio, min_ratio), max_ratio)

    print(f"\n  ✨ Dynamic retention-ratio calculation:")
    print(f"    Original average degree: {avg_degree:.1f}")
    print(f"    Target average degree: {target_avg_degree}")
    print(f"    Suggested retention ratio: {retention_ratio:.2%}")

    return retention_ratio


# ==========================================
# 5. ✨ Graph homophily check
# ==========================================
def check_homophily_invariance(edges_before, edges_after, labels_df):
    """
    Check changes in graph homophily before and after pruning

    Purpose: Ensure pruning does not substantially increase label homophily (Avoid indirect leakage)

    Homophily (Homophily) = P(Adjacent nodes have the same label)

    Criteria: 
    - Δh < 0.02: ✅ Safe
    - 0.02 ≤ Δh < 0.05: ⚠️ Gray area
    - Δh ≥ 0.05: ❌ Possible indirect leakage
    """
    print("\n  ✨ Graph homophily check (Homophily Invariance Check):")

    label_dict = dict(zip(labels_df['\u4f01\u4e1aID'], labels_df['\u6807\u7b7e']))

    def compute_homophily(edges_df):
        same_label = 0
        total = 0

        for _, row in edges_df.iterrows():
            u, v = row['\u4f01\u4e1aA'], row['\u4f01\u4e1aB']
            if u in label_dict and v in label_dict:
                if label_dict[u] == label_dict[v]:
                    same_label += 1
                total += 1

        return same_label / total if total > 0 else 0

    h_before = compute_homophily(edges_before)
    h_after = compute_homophily(edges_after)
    delta_h = h_after - h_before

    print(f"    Original graph homophily: {h_before:.4f}")
    print(f"    Homophily after pruning: {h_after:.4f}")
    print(f"    Change: {delta_h:+.4f}")

    if delta_h < 0.02:
        print(f"    ✅ Conclusion: Small homophily change, No label leakage risk")
    elif delta_h < 0.05:
        print(f"    ⚠️  Conclusion: Slight increase in homophily, Review recommended")
    else:
        print(f"    ❌ Warning: Significant increase in homophily, Possible indirect leakage!")

    return h_before, h_after, delta_h


# ==========================================
# 6. Main pruning function
# ==========================================
def prune_edges_risk_oriented_dib(edges_df, features_df, labels_df=None,
                                  retention_ratio=None, use_dynamic=True):
    """
    Based on Risk-Oriented DIB edge pruning (KDD version)

    Args:
        edges_df: Edge data
        features_df: Node features
        labels_df: Label data (Used only for homophily checks, Not used for pruning)
        retention_ratio: Fixed retention ratio (If specified)
        use_dynamic: Whether to use a dynamic retention ratio

    Returns:
        edges_pruned: Pruned edges
        stats: Statistics (For paper figures and tables)
    """
    print(f"=" * 80)
    print(f"Risk-Oriented DIB Edge Pruning (KDD 2025)")
    print(f"=" * 80)

    # 1. Prepare data
    node_ids = sorted(features_df['\u4f01\u4e1aID'].unique())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    num_edges = len(edges_df)

    print(f"\n Original graph statistics:")
    print(f"  Node count: {num_nodes:,}")
    print(f"  Edge count: {num_edges:,}")
    print(f"  Average degree: {2 * num_edges / num_nodes:.1f}")

    # 2. ✨ Dynamic retention ratio
    if use_dynamic:
        retention_ratio = compute_dynamic_retention_ratio(num_nodes, num_edges)

    print(f"\n Retention ratio used: {retention_ratio:.2%}")

    # 3. Filter valid edges
    valid_mask = edges_df['\u4f01\u4e1aA'].isin(node_ids) & edges_df['\u4f01\u4e1aB'].isin(node_ids)
    edges_df = edges_df[valid_mask].copy()
    print(f"  Valid edge count: {len(edges_df):,}")

    # 4. ✨ DIB scoring
    dib_scores = compute_dib_scores_fast(
        edges_df, features_df, node_to_idx,
        z_dim=32,
        beta=0.1,
        epochs=100
    )

    # 5.  to fuse weights (DIB dominant)
    weights = edges_df['\u6743\u91cd'].values
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

    # Fusion strategy: DIB is dominant (70%)
    final_scores = 0.70 * dib_scores + 0.30 * weights_norm

    edges_df['pruning_score'] = final_scores

    # 6. Top-K Retain 
    target_count = int(len(edges_df) * retention_ratio)
    min_edges = num_nodes * 3  # Ensure basic connectivity
    target_count = max(target_count, min_edges)
    target_count = min(target_count, len(edges_df))

    edges_pruned = edges_df.nlargest(target_count, 'pruning_score')

    # 7. Statistics
    print(f"\n Pruning results:")
    print(f"  Retained edge count: {len(edges_pruned):,} ({len(edges_pruned) / len(edges_df) * 100:.1f}%)")
    print(f"  Average degree after pruning: {(len(edges_pruned) * 2) / num_nodes:.1f}")
    print(f"  DIB Mean score: {edges_pruned['pruning_score'].mean():.4f}")

    # 8. ✨ Homophily check
    if labels_df is not None:
        h_before, h_after, delta_h = check_homophily_invariance(
            edges_df, edges_pruned, labels_df
        )
    else:
        h_before, h_after, delta_h = None, None, None

    # 9. Prepare statistics (For the paper)
    stats = {
        'num_nodes': num_nodes,
        'num_edges_before': len(edges_df),
        'num_edges_after': len(edges_pruned),
        'retention_ratio': retention_ratio,
        'avg_degree_before': 2 * len(edges_df) / num_nodes,
        'avg_degree_after': 2 * len(edges_pruned) / num_nodes,
        'homophily_before': h_before,
        'homophily_after': h_after,
        'homophily_delta': delta_h
    }

    # Remove temporary columns
    edges_pruned = edges_pruned.drop(columns=['pruning_score'])

    print(f"\n" + "=" * 80)
    print(f"Pruning completed!")
    print(f"=" * 80)

    return edges_pruned, stats


# ==========================================
# 7. Main program
# ==========================================
def main():
    """
    Main function: Process all datasets

    Output: 
    1. Pruned edge files
    2. Statistics (JSON format, For paper figures and tables)
    """
    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    # data_file = [ '2022_all', '2023_all', '2023_SH', '2023_SZ', '2022_SH', '2022_SZ', '2022_USA']
    data_file = [ '2023_BSS']

    # Store all statistics
    all_stats = {}

    for data in data_file:
        print("\n" + "=" * 80)
        print(f"Processing dataset: {data}")
        print("=" * 80)

        # Path
        edges_path = data_dir + data + '/csv_data/enterprise_relations.csv'
        feat_path = data_dir + data + '/data_process/fused_features.csv'
        # feat_path = data_dir + data + '/csv_data/reduced_features.csv'
        labels_path = data_dir + data + '/csv_data/enterprise_labels.csv'

        # Check files
        if not all(os.path.exists(p) for p in [edges_path, feat_path]):
            print(f"  ⚠️  Missing files, Skip")
            continue

        try:
            # Load data
            edges_df = pd.read_csv(edges_path)
            features_df = pd.read_csv(feat_path)

            # Load labels (Used only for homophily checks)
            labels_df = None
            if os.path.exists(labels_path):
                labels_df = pd.read_csv(labels_path)
                print(f"  ✓ Load label data (Used only for homophily checks)")

            # ✨ Risk-Oriented DIB Prune
            edges_pruned, stats = prune_edges_risk_oriented_dib(
                edges_df, features_df, labels_df,
                use_dynamic=True  # Use a dynamic retention ratio
            )

            # Save pruned edges
            output_path = data_dir + data + '/data_process/enterprise_relations_dib_pruned_GJ_e4.csv'
            edges_pruned.to_csv(output_path, index=False)
            print(f"\n  ✅ Saved to: {output_path}")

            # Save statistics (For the paper)
            all_stats[data] = stats

        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

    # Save all statistics
    import json
    stats_path = data_dir + 'pruning_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✅ Statistics saved: {stats_path}")

    print("\n" + "=" * 80)
    print("✅ All datasets processed!")
    print("=" * 80)

    return all_stats


if __name__ == "__main__":
    stats = main()

#
# # edge_pruning_kdd.py
# """
# Heterogeneous graph edge pruning using a risk-contagion-oriented information bottleneck
#
# Paper title (Suggested): 
# "Risk-Oriented Deterministic Information Bottleneck for
#  Heterogeneous Graph Pruning in Enterprise Risk Prediction"
#
# Contributions: 
# 1. Risk-contagion-oriented self-supervised targets (Feature differences + Structural bridges + Relation strength)
# 2. Dynamic retention-ratio strategy (Adapt to graph size and density)
# 3. Graph homophily invariance constraint (Ensure fairness, Avoid label leakage)
# 4. Multi-metric evaluation framework (F1, ROC-AUC, PR-AUC, KS)
#
# Applications: 
# - Financial risk management: Enterprise risk contagion prediction
# - Social networks: Fraud detection
# - Recommender systems: Noisy-edge filtering
#
# """
#
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# from tqdm import tqdm
# from collections import defaultdict
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # ==========================================
# # 1. Deterministic information bottleneck model (Optimized version)
# # ==========================================
# # edge_pruning_kdd.py
#
# class RiskOrientedDIB(nn.Module):
#     """
#     Risk-Oriented DIB (Optimized version v2)
#
#     Fix: 
#     1. Remove Tanh activation (Allow Z a wider range)
#     2. Adjust the compression-term scale
#     3. Numerical stability
#     """
#
#     def __init__(self, input_dim, z_dim=32, beta=0.1):
#         super().__init__()
#         self.beta = beta
#         self.z_dim = z_dim
#
#         # ✅ Encoder: Linear output
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim * 2, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, z_dim)  # ✅ No activation function
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#
#     def encode(self, x):
#         return self.encoder(x)
#
#     def decode(self, z):
#         return self.decoder(z)
#
#     def compute_ib_loss(self, x, target):
#         """
#         DIB loss = Compression term + β × Reconstruction term
#
#         Compression term: Allow Z to contain less information
#         Reconstruction term: Allow Z to retain contagion information
#         """
#         # Encode
#         z = self.encode(x)
#
#         # Decode
#         t_pred = self.decode(z)
#
#         # ✅ Compression term: Z mean L2 norm
#         z_norm = torch.norm(z, p=2, dim=1).mean()
#         compression_term = z_norm * 0.1  # Scale to a reasonable range
#
#         # Reconstruction term: Accuracy of contagion prediction
#         reconstruction_loss = F.binary_cross_entropy(
#             t_pred.squeeze(),
#             target
#         )
#
#         # Total loss
#         loss = compression_term + self.beta * reconstruction_loss
#
#         return loss, compression_term, reconstruction_loss
#
#     def get_edge_importance(self, x):
#         """
#         Edge importance = Compressed information content
#
#         Important edges: Z have large norms (Information-rich)
#         Unimportant edges: Z have small norms (Redundant)
#         """
#         z = self.encode(x)
#         importance = torch.norm(z, p=2, dim=1)
#
#         return importance
#
#
# # ==========================================
# # 2. ✨ Risk-contagion-oriented self-supervised target
# # ==========================================
# def build_risk_oriented_targets(edges_df, node_to_idx, x):
#     """
#     Build risk-contagion-oriented self-supervised targets
#
#     Core contribution: From "similarity" to "contagion"
#
#     Properties of risk-contagion edges: 
#     1. Connect nodes with different features (Heterogeneous contagion)
#     2. Structural bridge edges (Cross-community propagation)
#     3. Strong relationships (High influence)
#     4. Connect high-centrality nodes (Hub contagion)
#
#     Mathematical form: 
#     T_risk = α × Feature_Diff + β × Bridge_Score + γ × Strength + δ × Centrality
#
#     Where: 
#     - Feature_Diff = 1 - cosine(u, v)
#     - Bridge_Score = 1 - jaccard(u, v)
#     - Strength = normalized_weight
#     - Centrality = degree[u] × degree[v]
#     """
#     print("  Building risk-contagion-oriented self-supervised targets...")
#
#     # 1. Build adjacency lists (For computing Jaccard and degrees)
#     adj_dict = defaultdict(set)
#     degree_dict = defaultdict(int)
#
#     for _, row in edges_df.iterrows():
#         u, v = row[' enterprise A'], row[' enterprise B']
#         if u in node_to_idx and v in node_to_idx:
#             u_idx, v_idx = node_to_idx[u], node_to_idx[v]
#             adj_dict[u_idx].add(v_idx)
#             adj_dict[v_idx].add(u_idx)
#             degree_dict[u_idx] += 1
#             degree_dict[v_idx] += 1
#
#     # 2. Compute each component
#     feature_diffs = []
#     bridge_scores = []
#     strengths = []
#     centralities = []
#
#     for _, row in tqdm(edges_df.iterrows(), total=len(edges_df),
#                        desc="  Build targets"):
#         u, v = row[' enterprise A'], row[' enterprise B']
#
#         if u not in node_to_idx or v not in node_to_idx:
#             feature_diffs.append(0.5)
#             bridge_scores.append(0.5)
#             strengths.append(0.5)
#             centralities.append(0.5)
#             continue
#
#         u_idx, v_idx = node_to_idx[u], node_to_idx[v]
#
#         # Dimension 1: Feature differences (Heterogeneity)
#         cosine = (x[u_idx] * x[v_idx]).sum().item()
#         feature_diff = 1 - cosine  # Larger differences,  may imply stronger contagion
#         feature_diffs.append(feature_diff)
#
#         # Dimension 2: Bridge score (Structure)
#         neighbors_u = adj_dict[u_idx]
#         neighbors_v = adj_dict[v_idx]
#
#         if len(neighbors_u) == 0 or len(neighbors_v) == 0:
#             jaccard = 0.0
#         else:
#             intersection = len(neighbors_u & neighbors_v)
#             union = len(neighbors_u | neighbors_v)
#             jaccard = intersection / union if union > 0 else 0.0
#
#         bridge_score = 1 - jaccard  # Jaccard Low  =  bridge edges
#         bridge_scores.append(bridge_score)
#
#         # Dimension 3: Relation strength
#         strength = row['Weight']
#         strengths.append(strength)
#
#         # Dimension 4: Centrality (Connect important nodes)
#         degree_u = degree_dict[u_idx]
#         degree_v = degree_dict[v_idx]
#         centrality = np.sqrt(degree_u * degree_v)  # Geometric mean
#         centralities.append(centrality)
#
#     # 3. Normalize
#     feature_diffs = np.array(feature_diffs)
#     bridge_scores = np.array(bridge_scores)
#     strengths = np.array(strengths)
#     centralities = np.array(centralities)
#
#     feature_diffs = (feature_diffs - feature_diffs.min()) / (feature_diffs.max() - feature_diffs.min() + 1e-8)
#     bridge_scores = (bridge_scores - bridge_scores.min()) / (bridge_scores.max() - bridge_scores.min() + 1e-8)
#     strengths = (strengths - strengths.min()) / (strengths.max() - strengths.min() + 1e-8)
#     centralities = (centralities - centralities.min()) / (centralities.max() - centralities.min() + 1e-8)
#
#     # 4. Fusion (Adjustable weights)
#     alpha, beta, gamma, delta = 0.35, 0.30, 0.25, 0.10  # Tunable parameters
#
#     targets = (
#             alpha * feature_diffs +
#             beta * bridge_scores +
#             gamma * strengths +
#             delta * centralities
#     )
#
#     # Normalize again to [0, 1]
#     targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
#
#     print(f"    ✓ Target statistics:")
#     print(f"      Mean feature difference: {feature_diffs.mean():.3f}")
#     print(f"      Mean bridge score: {bridge_scores.mean():.3f}")
#     print(f"      Mean relation strength: {strengths.mean():.3f}")
#     print(f"      Mean centrality: {centralities.mean():.3f}")
#     print(f"      Combined target range: [{targets.min():.3f}, {targets.max():.3f}]")
#
#     return torch.tensor(targets, dtype=torch.float32)
#
#
# # ==========================================
# # 3. ✨ Fast DIB training (Optimized version)
# # ==========================================
# def compute_dib_scores_fast(edges_df, features_df, node_to_idx,
#                             z_dim=32, beta=0.1, epochs=100):
#     """
#     Fast DIB training (KDD Optimized version)
#
#     Optimizations: 
#     1. epochs: 300 → 100
#     2. More aggressive Early Stopping (patience=20)
#     3. Learning-rate scheduling (Cosine Annealing)
#     4. Gradient clipping (Prevent exploding gradients)
#     """
#     print(f"  Using Risk-Oriented DIB to compute edge importance...")
#     print(f"    Hyperparameters: z_dim={z_dim}, beta={beta}, max_epochs={epochs}")
#
#     # 1. Prepare node features
#     exclude_cols = [' enterprise ID', 'risk_type', 'risk_reason', 'cluster', 'label', 'Label']
#     feat_cols = [c for c in features_df.columns if
#                  c not in exclude_cols and features_df[c].dtype in ['float64', 'int64']]
#
#     feat_values = features_df[feat_cols].fillna(0).values
#     x = torch.tensor(feat_values, dtype=torch.float32)
#     x = F.normalize(x, p=2, dim=1)
#
#     print(f"    Node features: {x.shape}")
#
#     # 2. Build edge features
#     valid_edges = []
#     src_indices = []
#     dst_indices = []
#
#     for idx, row in enumerate(edges_df.itertuples()):
#         u, v = row. enterprise A, row. enterprise B
#         if u in node_to_idx and v in node_to_idx:
#             src_indices.append(node_to_idx[u])
#             dst_indices.append(node_to_idx[v])
#             valid_edges.append(idx)
#
#     src = torch.tensor(src_indices, dtype=torch.long)
#     dst = torch.tensor(dst_indices, dtype=torch.long)
#
#     edge_features = torch.cat([x[src], x[dst]], dim=1)
#     print(f"    Valid edge features: {edge_features.shape}")
#
#     # 3. ✨ Build risk-contagion-oriented self-supervised targets
#     targets = build_risk_oriented_targets(edges_df, node_to_idx, x)
#     targets = targets[valid_edges]  # Keep targets for valid edges only
#
#     # 4. Initialize DIB Model
#     model = RiskOrientedDIB(
#         input_dim=x.shape[1],
#         z_dim=z_dim,
#         beta=beta
#     )
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=epochs, eta_min=1e-6
#     )
#
#     # 5.  training DIB
#     print("     training Risk-Oriented DIB...")
#     model.train()
#
#     best_loss = float('inf')
#     patience = 0
#     max_patience = 20  # ✨ More aggressive Early Stopping
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#
#         # DIB loss
#         loss, mi_xz, recon_loss = model.compute_ib_loss(edge_features, targets)
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()
#
#         # Early stopping
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             patience = 0
#         else:
#             patience += 1
#
#         if patience >= max_patience:
#             print(f"      ✓ Early stopping at epoch {epoch + 1}")
#             break
#
#         if (epoch + 1) % 1 == 0:
#             print(f"      Epoch {epoch + 1}/{epochs}")
#             print(f"        Loss: {loss.item():.4f} | I(X;Z): {mi_xz.item():.4f} | Recon: {recon_loss.item():.4f}")
#             print(f"        LR: {optimizer.param_groups[0]['lr']:.6f}")
#
#     # 6. Extract edge importance
#     model.eval()
#     with torch.no_grad():
#         dib_scores_valid = model.get_edge_importance(edge_features).numpy()
#
#     # 7. Map back to original edge indices
#     dib_scores = np.zeros(len(edges_df))
#     dib_scores[valid_edges] = dib_scores_valid
#
#     # Normalize
#     dib_scores = (dib_scores - dib_scores.min()) / \
#                  (dib_scores.max() - dib_scores.min() + 1e-8)
#
#     print(f"    ✓ DIB Score: [{dib_scores.min():.4f}, {dib_scores.max():.4f}], Mean: {dib_scores.mean():.4f}")
#
#     return dib_scores
#
#
# # ==========================================
# # 4. ✨ Dynamic retention-ratio strategy
# # ==========================================
# def compute_dynamic_retention_ratio(num_nodes, num_edges):
#     """
#     Adjust the retention ratio dynamically based on graph size
#
#     Core idea: 
#     - Denser graphs,  lower retention ratios (Greater denoising needs)
#     - Sparser graphs,  higher retention ratios (Preserve connectivity)
#     - Keep the average degree after pruning within a reasonable range
#
#     Target average degree: 200-300 (Empirical value, Suitable for GNN)
#
#     Formula: 
#     retention_ratio = min(max((target_degree * N) / (2 * M), min_ratio), max_ratio)
#     """
#     avg_degree = 2 * num_edges / num_nodes
#     target_avg_degree = 30  # Adjust target average degree with graph size
#
#     # Compute the retention ratio
#     retention_ratio = (target_avg_degree * num_nodes) / (2 * num_edges)
#
#     # Bounds
#     min_ratio = 0.0  # Retain at least 5%
#     max_ratio = 0.3  # Retain at most 30%
#
#     retention_ratio = min(max(retention_ratio, min_ratio), max_ratio)
#
#     print(f"\n  ✨ Dynamic retention-ratio calculation:")
#     print(f"    Original average degree: {avg_degree:.1f}")
#     print(f"    Target average degree: {target_avg_degree}")
#     print(f"    Suggested retention ratio: {retention_ratio:.2%}")
#
#     return retention_ratio
#
#
# # ==========================================
# # 5. ✨ Graph homophily check
# # ==========================================
# def check_homophily_invariance(edges_before, edges_after, labels_df):
#     """
#     Check changes in graph homophily before and after pruning
#
#     Purpose: Ensure pruning does not substantially increase label homophily (Avoid indirect leakage)
#
#     Homophily (Homophily) = P(Adjacent nodes have the same label)
#
#     Criteria: 
#     - Δh < 0.02: ✅ Safe
#     - 0.02 ≤ Δh < 0.05: ⚠️ Gray area
#     - Δh ≥ 0.05: ❌ Possible indirect leakage
#     """
#     print("\n  ✨ Graph homophily check (Homophily Invariance Check):")
#
#     label_dict = dict(zip(labels_df[' enterprise ID'], labels_df['Label']))
#
#     def compute_homophily(edges_df):
#         same_label = 0
#         total = 0
#
#         for _, row in edges_df.iterrows():
#             u, v = row[' enterprise A'], row[' enterprise B']
#             if u in label_dict and v in label_dict:
#                 if label_dict[u] == label_dict[v]:
#                     same_label += 1
#                 total += 1
#
#         return same_label / total if total > 0 else 0
#
#     h_before = compute_homophily(edges_before)
#     h_after = compute_homophily(edges_after)
#     delta_h = h_after - h_before
#
#     print(f"    Original graph homophily: {h_before:.4f}")
#     print(f"    Homophily after pruning: {h_after:.4f}")
#     print(f"    Change: {delta_h:+.4f}")
#
#     if delta_h < 0.02:
#         print(f"    ✅ Conclusion: Small homophily change, No label leakage risk")
#     elif delta_h < 0.05:
#         print(f"    ⚠️  Conclusion: Slight increase in homophily, Review recommended")
#     else:
#         print(f"    ❌ Warning: Significant increase in homophily, Possible indirect leakage!")
#
#     return h_before, h_after, delta_h
#
#
# # ==========================================
# # 6. Main pruning function
# # ==========================================
# def prune_edges_risk_oriented_dib(edges_df, features_df, labels_df=None,
#                                   retention_ratio=None, target_edge_count=None, use_dynamic=True):
#     """
#     Based on Risk-Oriented DIB edge pruning (KDD version)
#
#     Args:
#         edges_df: Edge data
#         features_df: Node features
#         labels_df: Label data (Used only for homophily checks, Not used for pruning)
#         retention_ratio: Fixed retention ratio (If specified)
#         target_edge_count: ✨ Target number of retained edges (If specified, Highest priority)
#         use_dynamic: Whether to use a dynamic retention ratio
#
#     Returns:
#         edges_pruned: Pruned edges
#         stats: Statistics (For paper figures and tables)
#     """
#     print(f"=" * 80)
#     print(f"Risk-Oriented DIB Edge Pruning (KDD 2025)")
#     print(f"=" * 80)
#
#     # 1. Prepare data
#     node_ids = sorted(features_df[' enterprise ID'].unique())
#     node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
#     num_nodes = len(node_ids)
#     num_edges = len(edges_df)
#
#     print(f"\n Original graph statistics:")
#     print(f"  Node count: {num_nodes:,}")
#     print(f"  Edge count: {num_edges:,}")
#     print(f"  Average degree: {2 * num_edges / num_nodes:.1f}")
#
#     # 2. ✨ Determine the retention strategy (Priority: target_edge_count > retention_ratio > Dynamic)
#     if target_edge_count is not None:
#         # Prioritize the specified edge count
#         print(f"\n✨ Use the specified target edge count: {target_edge_count:,}")
#         retention_ratio = target_edge_count / num_edges
#         print(f"   Corresponding retention ratio: {retention_ratio:.2%}")
#     elif retention_ratio is not None:
#         # Otherwise use the specified retention ratio
#         print(f"\n Use the specified retention ratio: {retention_ratio:.2%}")
#     elif use_dynamic:
#         # Finally use a dynamic retention ratio
#         retention_ratio = compute_dynamic_retention_ratio(num_nodes, num_edges)
#     else:
#         raise ValueError("Must specify  target_edge_count, retention_ratio  or enable  use_dynamic")
#
#     # 3. Filter valid edges
#     valid_mask = edges_df[' enterprise A'].isin(node_ids) & edges_df[' enterprise B'].isin(node_ids)
#     edges_df = edges_df[valid_mask].copy()
#     print(f"  Valid edge count: {len(edges_df):,}")
#
#     # 4. ✨ DIB scoring
#     dib_scores = compute_dib_scores_fast(
#         edges_df, features_df, node_to_idx,
#         z_dim=32,
#         beta=0.1,
#         epochs=100
#     )
#
#     # 5.  to fuse weights (DIB dominant)
#     weights = edges_df['Weight'].values
#     weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
#
#     # Fusion strategy: DIB is dominant (70%)
#     final_scores = 0.70 * dib_scores + 0.30 * weights_norm
#
#     edges_df['pruning_score'] = final_scores
#
#     # 6. Top-K Retain 
#     if target_edge_count is not None:
#         target_count = target_edge_count
#     else:
#         target_count = int(len(edges_df) * retention_ratio)
#
#     min_edges = num_nodes * 3  # Ensure basic connectivity
#     target_count = max(target_count, min_edges)
#     target_count = min(target_count, len(edges_df))
#
#     edges_pruned = edges_df.nlargest(target_count, 'pruning_score')
#
#     # 7. Statistics
#     print(f"\n Pruning results:")
#     print(f"  Retained edge count: {len(edges_pruned):,} ({len(edges_pruned) / len(edges_df) * 100:.1f}%)")
#     print(f"  Average degree after pruning: {(len(edges_pruned) * 2) / num_nodes:.1f}")
#     print(f"  DIB Mean score: {edges_pruned['pruning_score'].mean():.4f}")
#
#     # 8. ✨ Homophily check
#     if labels_df is not None:
#         h_before, h_after, delta_h = check_homophily_invariance(
#             edges_df, edges_pruned, labels_df
#         )
#     else:
#         h_before, h_after, delta_h = None, None, None
#
#     # 9. Prepare statistics (For the paper)
#     stats = {
#         'num_nodes': num_nodes,
#         'num_edges_before': len(edges_df),
#         'num_edges_after': len(edges_pruned),
#         'retention_ratio': len(edges_pruned) / len(edges_df),
#         'avg_degree_before': 2 * len(edges_df) / num_nodes,
#         'avg_degree_after': 2 * len(edges_pruned) / num_nodes,
#         'homophily_before': h_before,
#         'homophily_after': h_after,
#         'homophily_delta': delta_h
#     }
#
#     # Remove temporary columns
#     edges_pruned = edges_pruned.drop(columns=['pruning_score'])
#
#     print(f"\n" + "=" * 80)
#     print(f"Pruning completed!")
#     print(f"=" * 80)
#
#     return edges_pruned, stats
#
#
# # ==========================================
# # 7. Main program
# # ==========================================
# def main():
#     """
#     Main function: Process all datasets
#
#     Output: 
#     1. Pruned edge files
#     2. Statistics (JSON format, For paper figures and tables)
#     """
#     data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
#     data_file = ['2022_all', '2023_all', '2023_SH', '2023_SZ', '2022_SH', '2022_SZ', '2022_USA']
#
#     # Store all statistics
#     all_stats = {}
#
#     for data in data_file:
#         print("\n" + "=" * 80)
#         print(f"Processing dataset: {data}")
#         print("=" * 80)
#
#         # Path
#         edges_path = data_dir + data + '/csv_data/enterprise_relations.csv'
#         feat_path = data_dir + data + '/data_process/fused_features.csv'
#         # feat_path = data_dir + data + '/csv_data/reduced_features.csv'
#         labels_path = data_dir + data + '/csv_data/enterprise_labels.csv'
#
#         # Check files
#         if not all(os.path.exists(p) for p in [edges_path, feat_path]):
#             print(f"  ⚠️  Missing files, Skip")
#             continue
#
#         try:
#             # Load data
#             edges_df = pd.read_csv(edges_path)
#             features_df = pd.read_csv(feat_path)
#
#             # Load labels (Used only for homophily checks)
#             labels_df = None
#             if os.path.exists(labels_path):
#                 labels_df = pd.read_csv(labels_path)
#                 print(f"  ✓ Load label data (Used only for homophily checks)")
#
#             # ✨ Risk-Oriented DIB Prune
#             # Usage example: 
#             # # Approach 1: Use a dynamic retention ratio
#             # edges_pruned, stats = prune_edges_risk_oriented_dib(
#             #     edges_df, features_df, labels_df,
#             #     use_dynamic=True
#             # )
#
#             # Approach 2: Specify the retention ratio (For example, retain 20%)
#             # edges_pruned, stats = prune_edges_risk_oriented_dib(
#             #     edges_df, features_df, labels_df,
#             #     retention_ratio=0.2,
#             #     use_dynamic=False
#             # )
#
#             # Approach 3: ✨ Specify the retained edge count (For example, retain 50000 edges)
#             edges_pruned, stats = prune_edges_risk_oriented_dib(
#                 edges_df, features_df, labels_df,
#                 target_edge_count=8000
#             )
#
#             # Save pruned edges
#             output_path = data_dir + data + '/data_process/enterprise_relations_dib_pruned_GJ_8000.csv'
#             edges_pruned.to_csv(output_path, index=False)
#             print(f"\n  ✅ Saved to: {output_path}")
#
#             # Save statistics (For the paper)
#             all_stats[data] = stats
#
#         except Exception as e:
#             print(f"  ❌ Processing failed: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # Save all statistics
#     import json
#     stats_path = data_dir + 'pruning_stats.json'
#     with open(stats_path, 'w') as f:
#         json.dump(all_stats, f, indent=2)
#     print(f"\n✅ Statistics saved: {stats_path}")
#
#     print("\n" + "=" * 80)
#     print("✅ All datasets processed!")
#     print("=" * 80)
#
#     return all_stats
#
#
# if __name__ == "__main__":
#     stats = main()