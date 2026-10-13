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
# 1. Deterministic information bottleneck model
# ==========================================
class RiskOrientedDIB(nn.Module):
    def __init__(self, input_dim, z_dim=32, beta=0.1):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def encode(self, x): return self.encoder(x)

    def decode(self, z): return self.decoder(z)

    def compute_ib_loss(self, x, target):
        z = self.encode(x)
        t_pred = self.decode(z)
        z_norm = torch.norm(z, p=2, dim=1).mean()
        loss = z_norm * 0.1 + self.beta * F.binary_cross_entropy(t_pred.squeeze(), target)
        return loss

    def get_edge_importance(self, x):
        z = self.encode(x)
        return torch.norm(z, p=2, dim=1)


# ==========================================
# 2. Self-supervised targets and score computation
# ==========================================
def build_risk_oriented_targets(edges_df, node_to_idx, x):
    adj_dict = defaultdict(set)
    degree_dict = defaultdict(int)
    for _, row in edges_df.iterrows():
        u, v = row['\u4f01\u4e1aA'], row['\u4f01\u4e1aB']
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            adj_dict[u_idx].add(v_idx);
            adj_dict[v_idx].add(u_idx)
            degree_dict[u_idx] += 1;
            degree_dict[v_idx] += 1

    f_diffs, b_scores, strengths, cents = [], [], [], []
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Build targets"):
        u, v = row['\u4f01\u4e1aA'], row['\u4f01\u4e1aB']
        if u not in node_to_idx or v not in node_to_idx:
            for l in [f_diffs, b_scores, strengths, cents]: l.append(0.5)
            continue
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        f_diffs.append(1 - (x[u_idx] * x[v_idx]).sum().item())
        n_u, n_v = adj_dict[u_idx], adj_dict[v_idx]
        jaccard = len(n_u & n_v) / len(n_u | n_v) if (n_u | n_v) else 0
        b_scores.append(1 - jaccard)
        strengths.append(row['\u6743\u91cd'])
        cents.append(np.sqrt(degree_dict[u_idx] * degree_dict[v_idx]))

    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    targets = 0.35 * norm(np.array(f_diffs)) + 0.30 * norm(np.array(b_scores)) + 0.25 * norm(
        np.array(strengths)) + 0.10 * norm(np.array(cents))
    return torch.tensor((targets - targets.min()) / (targets.max() - targets.min() + 1e-8), dtype=torch.float32)


def compute_dib_scores_internal(edges_df, features_df, node_to_idx):
    exclude = ['\u4f01\u4e1aID', 'risk_type', 'risk_reason', 'cluster', 'label', '\u6807\u7b7e']
    feat_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64']]
    x = F.normalize(torch.tensor(features_df[feat_cols].fillna(0).values, dtype=torch.float32), p=2, dim=1)
    v_idx, s_idx, d_idx = [], [], []
    for i, r in enumerate(edges_df.itertuples()):
        if getattr(r, _ENTERPRISE_A) in node_to_idx and getattr(r, _ENTERPRISE_B) in node_to_idx:
            s_idx.append(node_to_idx[getattr(r, _ENTERPRISE_A)]);
            d_idx.append(node_to_idx[getattr(r, _ENTERPRISE_B)]);
            v_idx.append(i)
    e_feat = torch.cat([x[torch.tensor(s_idx)], x[torch.tensor(d_idx)]], dim=1)
    targets = build_risk_oriented_targets(edges_df, node_to_idx, x)[v_idx]
    model = RiskOrientedDIB(input_dim=x.shape[1]);
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Training  DIB Model...");
    [(opt.zero_grad(), model.compute_ib_loss(e_feat, targets).backward(), opt.step()) for _ in range(100)]
    model.eval();
    with torch.no_grad():
        v_scores = model.get_edge_importance(e_feat).numpy()
    full_scores = np.zeros(len(edges_df));
    full_scores[v_idx] = v_scores
    return (full_scores - full_scores.min()) / (full_scores.max() - full_scores.min() + 1e-8)


# ==========================================
# 3. Main experiment-data saving logic
# ==========================================
def save_sensitivity_data(data_dir, dataset, output_csv):
    path = os.path.join(data_dir, dataset)
    edges_df = pd.read_csv(f"{path}/csv_data/enterprise_relations.csv")
    features_df = pd.read_csv(f"{path}/data_process/fused_features.csv")
    labels_df = pd.read_csv(f"{path}/csv_data/enterprise_labels.csv")
    label_dict = dict(zip(labels_df['\u4f01\u4e1aID'], labels_df['\u6807\u7b7e']))
    node_to_idx = {nid: i for i, nid in enumerate(sorted(features_df['\u4f01\u4e1aID'].unique()))}
    num_nodes = len(node_to_idx)

    dib_scores = compute_dib_scores_internal(edges_df, features_df, node_to_idx)
    weights_norm = (edges_df['\u6743\u91cd'] - edges_df['\u6743\u91cd'].min()) / (
                edges_df['\u6743\u91cd'].max() - edges_df['\u6743\u91cd'].min() + 1e-6)
    edges_df['final_score'] = 0.70 * dib_scores + 0.30 * weights_norm

    target_degrees = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = []

    for d in target_degrees:
        target_count = min(int((d * num_nodes) / 2), len(edges_df))
        edges_pruned = edges_df.nlargest(target_count, 'final_score')

        # Compute homophily
        same, total = 0, 0
        for _, row in edges_pruned.iterrows():
            if row['\u4f01\u4e1aA'] in label_dict and row['\u4f01\u4e1aB'] in label_dict:
                if label_dict[row['\u4f01\u4e1aA']] == label_dict[row['\u4f01\u4e1aB']]: same += 1
                total += 1
        h_val = same / total if total > 0 else 0
        results.append({'target_degree': d, 'edge_count': len(edges_pruned), 'homophily': h_val})
        print(f"[Degrees {d}] Edge count: {len(edges_pruned)}, Homophily: {h_val:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"✅ Experiment data saved to: {output_csv}")


if __name__ == "__main__":
    BASE_DIR = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    SAVE_PATH = '/analyze/DIB_effect/sensitivity_results.csv'
    save_sensitivity_data(BASE_DIR, '2022_SZ', SAVE_PATH)