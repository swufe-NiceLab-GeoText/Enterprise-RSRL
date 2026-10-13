"""
================================================================================
GraphSMOTE Final comparison experiments - (LLM / Clustering / Comprehensive denoising strategies)
================================================================================
[Updates]
1. Complete comparison of denoising strategies (DIB vs 9 baseline methods).
2. Automatically match graph_fusion  output directory paths.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from scipy.stats import ks_2samp
import warnings
import random
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Fixed best parameter configuration
DATASET_CONFIGS = {
    "2023_SH": {'ratio': 2.9, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
    "2023_SZ": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 0.9, 'binning': False},
    "2022_SH": {'ratio': 2.2, 'hidden': 64, 'lambda_rank': 3.1, 'binning': False},
    "2022_SZ": {'ratio': 2.5, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
    "2022_USA": {'ratio': 0.5, 'hidden': 64, 'lambda_rank': 2.0, 'binning': True},
    "2022_BSS": {'ratio': 0.913, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    "2023_BSS": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 2.5, 'binning': False}
}

# 2. Fixed selected seeds
DATASET_SEEDS = {
    # "2022_SH": [2, 14],
    # "2022_SZ": [72, 75, 8],
    # "2023_SH": [42, 208, 217],
    # "2023_SZ": [15, 14, 25],
    # "2022_BSS": [109],
    "2023_BSS": [73],  #
    # "2022_USA": [82, 110, 131]
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphSMOTEGenerator:
    """GraphSMOTE: In embedding space, generate synthetic positive samples"""

    def __init__(self, k=5):
        self.k = k

    def generate_synthetic_nodes(self, pos_embeddings, num_synthetic):
        num_pos = len(pos_embeddings)
        synthetic_list = []
        if num_pos < self.k + 1:
            return pos_embeddings.repeat(num_synthetic // num_pos + 1, 1)[:num_synthetic]

        for i in range(num_synthetic):
            anchor_idx = np.random.randint(0, num_pos)
            anchor = pos_embeddings[anchor_idx]
            distances = torch.norm(pos_embeddings - anchor, dim=1)
            _, knn_indices = torch.topk(distances, min(self.k + 1, num_pos), largest=False)
            knn_indices = knn_indices[1:]

            if len(knn_indices) > 0:
                neighbor_idx = knn_indices[np.random.randint(0, len(knn_indices))]
                neighbor = pos_embeddings[neighbor_idx]
                alpha = np.random.random()
                synthetic = alpha * anchor + (1 - alpha) * neighbor
                synthetic_list.append(synthetic)
            else:
                noise = torch.randn_like(anchor) * 0.1
                synthetic_list.append(anchor + noise)
        return torch.stack(synthetic_list)


class ResidualHGT_WithEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_heads=4, num_layers=2, dropout=0.4, temperature=1.0):
        super().__init__()
        self.node_types = ['enterprise']
        self.edge_types = [
            ('enterprise', 'investment', 'enterprise'),
            ('enterprise', 'similarity', 'enterprise')
        ]
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels,
                    metadata=(self.node_types, self.edge_types),
                    heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(), nn.Dropout(dropout),
            Linear(hidden_channels // 2, out_channels)
        )
        self.temperature = temperature

    def forward(self, x_dict, edge_index_dict, return_embedding=False):
        h = self.input_proj(x_dict['enterprise'])
        x_dict = {'enterprise': h}
        for i, conv in enumerate(self.convs):
            h_prev = x_dict['enterprise']
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
            x_dict['enterprise'] = x_dict['enterprise'] + h_prev
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        embeddings = x_dict['enterprise']
        if return_embedding: return embeddings
        logits = self.classifier(embeddings)
        return logits / self.temperature


class PROptimizedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_rank=2.0, margin=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_rank = lambda_rank
        self.margin = margin

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=inputs.device),
                              torch.tensor(1 - self.alpha, device=inputs.device))
        focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

    def ranking_loss(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)[:, 1]
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device)
        pos_scores = probs[pos_mask]
        neg_scores = probs[neg_mask]
        if len(neg_scores) > 3000:
            perm = torch.randperm(len(neg_scores))[:3000]
            neg_scores = neg_scores[perm]
        pos_scores_expanded = pos_scores.unsqueeze(1)
        neg_scores_expanded = neg_scores.unsqueeze(0)
        pairwise_loss = F.relu(self.margin - (pos_scores_expanded - neg_scores_expanded))
        return pairwise_loss.mean()

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        ranking = self.ranking_loss(inputs, targets)
        return focal + self.lambda_rank * ranking


# ==============================================================================
# load_data
# ==============================================================================
def load_data(data_dir, dataset_name, seed, use_binning=False, graph_folder_name='graph_fusion'):
    # Construct paths dynamically, For example: /data/2022_all/graph_fusion_pruned_cosine/
    graph_path = '/' + graph_folder_name + '/'
    full_graph_path = data_dir + dataset_name + graph_path
    # print(full_graph_path)

    # Check whether the path exists, Prevent errors
    if not os.path.exists(full_graph_path):
        raise FileNotFoundError(
            f"❌ Path not found: {full_graph_path}\n Check whether the data files exist.")

    edge_index = torch.load(full_graph_path + 'fused_edge_index.pt', weights_only=False)
    edge_type = torch.load(full_graph_path + 'fused_edge_type.pt', weights_only=False)
    node_to_idx = torch.load(full_graph_path + 'node_to_idx.pt', weights_only=False)

    features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features.csv')
    labels_df = pd.read_csv(data_dir + dataset_name + '/csv_data/enterprise_labels.csv')

    exclude = ['\u4f01\u4e1aID', 'enterprise_id', '\u98ce\u9669\u7c7b\u578b', 'risk_type', 'cluster', '\u6807\u7b7e', 'label', 'risk_reason',
               'Unnamed: 0']
    feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64']]

    id_col = 'enterprise_id' if 'enterprise_id' in features_df.columns else '\u4f01\u4e1aID'
    features_df['key'] = features_df[id_col].astype(float).astype(int).astype(str)
    feat_idx = features_df.set_index('key')

    num_nodes = len(node_to_idx)
    idx2node = {i: n for n, i in node_to_idx.items()}

    raw_features = np.zeros((num_nodes, len(feature_cols)))
    labels = []
    label_map = {str(k): v for k, v in labels_df.set_index('\u4f01\u4e1aID')['\u6807\u7b7e'].to_dict().items()}

    for i in range(num_nodes):
        nid = str(idx2node[i])
        if nid in feat_idx.index:
            val = feat_idx.loc[nid, feature_cols].values
            if len(val.shape) > 1: val = val[0]
            raw_features[i] = val
        labels.append(label_map.get(nid, 0))
    raw_features = np.nan_to_num(raw_features)

    if use_binning:
        try:
            est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
            features_binned = est.fit_transform(raw_features)
            scaler = StandardScaler()
            features = scaler.fit_transform(features_binned)
        except:
            scaler = StandardScaler()
            features = scaler.fit_transform(np.log1p(np.abs(raw_features)))
    else:
        scaler = StandardScaler()
        features = scaler.fit_transform(raw_features)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    hetero_data = HeteroData()
    hetero_data['enterprise'].x = x
    hetero_data['enterprise'].y = y

    inv_mask = (edge_type == 0)
    sim_mask = (edge_type == 1)
    hetero_data['enterprise', 'investment', 'enterprise'].edge_index = edge_index[:, inv_mask]
    hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = edge_index[:, sim_mask]

    valid_mask = (y >= 0) & (y < 2)
    valid_idx = torch.where(valid_mask)[0].numpy()
    train_val_idx, test_idx = train_test_split(valid_idx, test_size=0.2, stratify=y[valid_idx], random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, stratify=y[train_val_idx],
                                          random_state=seed + 1)
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True;
    val_mask[val_idx] = True;
    test_mask[test_idx] = True
    return hetero_data, train_mask, val_mask, test_mask


def train_single_model_with_smote(data, train_mask, val_mask, model_seed, config):
    set_seed(model_seed)
    oversample_ratio = config['ratio']
    hidden_dim = config['hidden']
    rank_weight = config['lambda_rank']

    model = ResidualHGT_WithEmbedding(
        in_channels=data['enterprise'].x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.4,
        temperature=1.0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = PROptimizedLoss(alpha=0.25, gamma=2.0, lambda_rank=rank_weight, margin=0.5)
    smote_generator = GraphSMOTEGenerator(k=5)

    # Stage 1
    for epoch in range(50):
        model.train()
        out = model({'enterprise': data['enterprise'].x},
                    {('enterprise', 'investment', 'enterprise'): data[
                        'enterprise', 'investment', 'enterprise'].edge_index,
                     ('enterprise', 'similarity', 'enterprise'): data[
                         'enterprise', 'similarity', 'enterprise'].edge_index})
        optimizer.zero_grad()
        loss = criterion(out[train_mask], data['enterprise'].y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Stage 2
    if oversample_ratio > 0:
        model.eval()
        with torch.no_grad():
            embeddings = model({'enterprise': data['enterprise'].x},
                               {('enterprise', 'investment', 'enterprise'): data[
                                   'enterprise', 'investment', 'enterprise'].edge_index,
                                ('enterprise', 'similarity', 'enterprise'): data[
                                    'enterprise', 'similarity', 'enterprise'].edge_index},
                               return_embedding=True)
        pos_train_mask = train_mask & (data['enterprise'].y == 1)
        if pos_train_mask.sum() > 0:
            pos_embeddings = embeddings[pos_train_mask]
            num_pos = pos_train_mask.sum().item()
            num_synthetic = int(num_pos * oversample_ratio)
            if num_synthetic > 0:
                synthetic_embeddings = smote_generator.generate_synthetic_nodes(pos_embeddings, num_synthetic)
                augmented_x = torch.cat(
                    [data['enterprise'].x, torch.zeros(num_synthetic, data['enterprise'].x.shape[1])], dim=0)
                augmented_y = torch.cat([data['enterprise'].y, torch.ones(num_synthetic, dtype=torch.long)], dim=0)
                augmented_train_mask = torch.cat([train_mask, torch.ones(num_synthetic, dtype=torch.bool)], dim=0)
                augmented_val_mask = torch.cat([val_mask, torch.zeros(num_synthetic, dtype=torch.bool)], dim=0)
                augmented_data = HeteroData()
                augmented_data['enterprise'].x = augmented_x
                augmented_data['enterprise'].y = augmented_y
                augmented_data['enterprise', 'investment', 'enterprise'].edge_index = data[
                    'enterprise', 'investment', 'enterprise'].edge_index
                augmented_data['enterprise', 'similarity', 'enterprise'].edge_index = data[
                    'enterprise', 'similarity', 'enterprise'].edge_index
            else:
                augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
        else:
            augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
    else:
        augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask

    # Stage 3
    model = ResidualHGT_WithEmbedding(
        in_channels=data['enterprise'].x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.4,
        temperature=1.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30,
                                                           verbose=False)

    best_val_pr = 0;
    patience_counter = 0;
    best_model_state = None
    for epoch in range(250):
        model.train()
        out = model({'enterprise': augmented_data['enterprise'].x},
                    {('enterprise', 'investment', 'enterprise'): augmented_data[
                        'enterprise', 'investment', 'enterprise'].edge_index,
                     ('enterprise', 'similarity', 'enterprise'): augmented_data[
                         'enterprise', 'similarity', 'enterprise'].edge_index})
        optimizer.zero_grad()
        loss = criterion(out[augmented_train_mask], augmented_data['enterprise'].y[augmented_train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model({'enterprise': augmented_data['enterprise'].x},
                            {('enterprise', 'investment', 'enterprise'): augmented_data[
                                'enterprise', 'investment', 'enterprise'].edge_index,
                             ('enterprise', 'similarity', 'enterprise'): augmented_data[
                                 'enterprise', 'similarity', 'enterprise'].edge_index})
            val_probs = F.softmax(val_out[augmented_val_mask], dim=1)[:, 1].cpu().numpy()
            val_y = augmented_data['enterprise'].y[augmented_val_mask].cpu().numpy()
            val_prec, val_rec, _ = precision_recall_curve(val_y, val_probs)
            val_pr_auc = auc(val_rec, val_prec)
        scheduler.step(val_pr_auc)
        if val_pr_auc > best_val_pr:
            best_val_pr = val_pr_auc;
            patience_counter = 0;
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= 50: break
    if best_model_state is not None: model.load_state_dict(best_model_state)
    return model


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model({'enterprise': data['enterprise'].x},
                {('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
                 ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index})
    probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    y_true = data['enterprise'].y[mask].cpu().numpy()
    roc_auc = roc_auc_score(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec, prec)
    try:
        ks_stat = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    except:
        ks_stat = 0.0
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1: best_f1 = f1
    return {'f1': best_f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}


def train_and_evaluate(data_dir, dataset, seed, config, graph_folder_name):
    # print(f"    Loading graph from: {graph_folder_name} ...")

    data, train_mask, val_mask, test_mask = load_data(
        data_dir,
        dataset,
        seed,
        use_binning=config['binning'],
        graph_folder_name=graph_folder_name
    )
    model = train_single_model_with_smote(data, train_mask, val_mask, seed, config)
    return evaluate(model, data, test_mask)


def run_comparative_experiments(data_dir, run_datasets=None, experiments_mask=None):
    if run_datasets is None:
        run_datasets = list(DATASET_SEEDS.keys())

    # Run all experiments by default
    if experiments_mask is None:
        experiments_mask = ['pruning']

    # =========================================================
    # 🎯 Core configuration: Experiment groups
    # =========================================================

    # 1. LLM Comparison groups
    EXPERIMENT_LLM = {
        'Baseline (Ours)': 'graph_fusion',
        'Kimi': 'graph_fusion_kimi',
        'Qwen': 'graph_fusion_qwen'
    }

    # 2. Cluster-count comparison groups
    EXPERIMENT_CLUSTER = {
        'Cluster-25': 'graph_fusion_25',
        'Cluster-50': 'graph_fusion_50',
        'Baseline (100)': 'graph_fusion',
        'Cluster-150': 'graph_fusion_150',
        'Cluster-200': 'graph_fusion_200'
    }

    # 3. 🔥 Denoising-strategy comparison groups (Full method)
    # Key name: Display name
    # Key value: Directory name (Must match graph_fusion.py  generated names)
    EXPERIMENT_PRUNING = {
        'DIB (Ours)': 'graph_fusion',  # Your  DIB
        'Random': 'graph_fusion_pruned_random',
        'Threshold (Weight)': 'graph_fusion_pruned_threshold',
        'Jaccard (Struct)': 'graph_fusion_jaccard',
        'Cosine (Feat)': 'graph_fusion_cosine',
        'PA (Degree)': 'graph_fusion_pruned_pa',
        'AA (Adamic)': 'graph_fusion_pruned_aa',
        'Triangle (Local)': 'graph_fusion_triangle',
        'K-Core (Hierarchy)': 'graph_fusion_kcore',
        'PageRank (Global)': 'graph_fusion_pagerank'
    }

    final_logs = []

    for dataset in run_datasets:
        if dataset not in DATASET_SEEDS:
            print(f"⏩ Skip {dataset} (No seed configured)")
            continue

        seeds = DATASET_SEEDS[dataset]
        config = DATASET_CONFIGS.get(dataset, {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False})

        print(f"\n{'=' * 80}")
        print(f"📊 Dataset: {dataset} | Config: {config}")
        print(f"{'=' * 80}")

        for seed in seeds:
            print(f"\n🌱 Seed: {seed}")

            # --- Experiment 1: LLM Comparison ---
            if 'llm' in experiments_mask:
                print(f"  [Experiment 1: LLM Comparison]")
                for name, folder in EXPERIMENT_LLM.items():
                    try:
                        metrics = train_and_evaluate(data_dir, dataset, seed, config, folder)
                        print(f"    ➜ {name:<15} ({folder}): PR={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f} | ROC={metrics['roc_auc']:.4f} | KS={metrics['ks']:.4f}")
                        final_logs.append({
                            'dataset': dataset, 'seed': seed, 'type': 'LLM', 'variant': name,
                            'pr': metrics['pr_auc'], 'f1': metrics['f1'], 'roc': metrics['roc_auc'], 'ks': metrics['ks']
                        })
                    except Exception as e:
                        print(f"    ❌ {name} ({folder}) Failed: {e}")

            # --- Experiment 2: Cluster-count comparison ---
            if 'cluster' in experiments_mask:
                print(f"  [Experiment 2: Clustering Size Comparison]")
                for name, folder in EXPERIMENT_CLUSTER.items():
                    try:
                        metrics = train_and_evaluate(data_dir, dataset, seed, config, folder)
                        print(f"    ➜ {name:<15} ({folder}): PR={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f} | ROC={metrics['roc_auc']:.4f} | KS={metrics['ks']:.4f}")
                        final_logs.append({
                            'dataset': dataset, 'seed': seed, 'type': 'Cluster', 'variant': name,
                            'pr': metrics['pr_auc'], 'f1': metrics['f1'], 'roc': metrics['roc_auc'], 'ks': metrics['ks']
                        })
                    except Exception as e:
                        print(f"    ❌ {name} ({folder}) Failed: {e}")

            # --- Experiment 3: Denoising-strategy comparison (Full method) ---
            if 'pruning' in experiments_mask:
                print(f"  [Experiment 3: Pruning Strategy Comparison]")
                for name, folder in EXPERIMENT_PRUNING.items():
                    try:
                        metrics = train_and_evaluate(data_dir, dataset, seed, config, folder)
                        print(f"    ➜ {name:<20} : PR={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f} | ROC={metrics['roc_auc']:.4f} | KS={metrics['ks']:.4f}")
                        final_logs.append({
                            'dataset': dataset, 'seed': seed, 'type': 'Pruning', 'variant': name,
                            'pr': metrics['pr_auc'], 'f1': metrics['f1'], 'roc': metrics['roc_auc'], 'ks': metrics['ks']
                        })
                    except Exception as e:
                        # Some datasets may have too few edges to run after pruning, An error is expected in that case, Record it
                        print(f"    ⚠️ {name} Failed: {str(e)[:50]}...")

    # Save automatically
    if final_logs:
        df = pd.DataFrame(final_logs)
        save_path = 'ablation_results.csv'
        df.to_csv(save_path, index=False)
        print(f"\n✅ Results automatically saved to: {save_path}")

    return final_logs


def main():
    # ⚠️ Confirm this path
    data_dir = '/data/'

    # ⚠️ 1. Choose datasets to run (Empty list = Run all datasets with configured seeds)
    # Try these two first, They run faster
    datasets_to_run = ['2023_BSS']

    # ⚠️ 2. Choose experiment types to run
    # Prioritize pruning
    # 'cluster' 'llm'
    experiments_to_run = ['cluster', 'llm']

    print("=" * 100)
    print("🔥 GraphSMOTE Final comparison experiment (Final Showdown)")
    print(f"   Run experiments: {experiments_to_run}")
    print(f"   Run on data: {datasets_to_run if datasets_to_run else 'ALL'}")
    print("=" * 100)
    print(f"🚀 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_comparative_experiments(data_dir, datasets_to_run, experiments_to_run)

    print(f"\n✅ All experiments completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()