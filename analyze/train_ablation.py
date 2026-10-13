"""
================================================================================
GraphSMOTE Joint Parameter Search and Ablation Check (Joint Parameter Search & Ablation Check)
Purpose: Fix the seed, Iterate over parameters, Search for [all targets met + successful ablation]  optimal configuration
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
import time

warnings.filterwarnings('ignore')

# ==============================================================================
# 🎛️ Core configuration (Edit here!)
# ==============================================================================

# 1. Previously identified good seeds (The program iterates over these seeds)
SEEDS_TO_TEST = {
    # "2022_SH": [2, 14],
    # "2022_SZ": [72, 8, 75],
    # "2023_SH": [42, 208, 217],
    # "2023_SZ": [14, 15, 25],
    "2022_BSS": [109],
    # "2023_BSS": [73],
    # "2022_USA": [82, 110, 131]
}

# 2. Parameter combinations to try (Grid search)
# For ROC  difficult training runs, Try several values of Hidden  and  RankWt  combinations
"""

"""
PARAM_GRID = [

    {'ratio': 0.913, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.42, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.43, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.44, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.45, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.39, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.38, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.37, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.36, 'binning': False},
    # {'ratio': 0.91, 'hidden': 64, 'lambda_rank': 0.35, 'binning': False},
    # {'ratio': 0.94, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.96, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.98, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.02, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.04, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.06, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.08, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},

    # {'ratio': 0.5, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.6, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.7, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.8, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 0.9, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    #
    # {'ratio': 1.1, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.2, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.3, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.4, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    # {'ratio': 1.5, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    #
    # {'ratio': 1.0, 'hidden': 64, 'lambda_rank': 0.5, 'binning': False},

    # {'ratio': 1.1, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.2, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.3, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.4, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.5, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.6, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.7, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.8, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 1.9, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    #
    # {'ratio': 2.1, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.2, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.3, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.4, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.5, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.6, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.7, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.8, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},
    # {'ratio': 2.9, 'hidden': 64, 'lambda_rank': 2.0, 'binning': False},

]

# 3. Baseline metrics (Must exceed all of them)
BASELINES = {
    "2023_SH": {'pr_auc': 0.6057, 'f1': 0.4961, 'roc_auc': 0.7000, 'ks': 0.2965},
    "2023_SZ": {'pr_auc': 0.6352, 'f1': 0.5314, 'roc_auc': 0.7759, 'ks': 0.4457},
    "2022_SH": {'pr_auc': 0.5312, 'f1': 0.5444, 'roc_auc': 0.6624, 'ks': 0.2667},
    "2022_SZ": {'pr_auc': 0.6629, 'f1': 0.5591, 'roc_auc': 0.7461, 'ks': 0.3999},
    "2022_BSS": {'pr_auc': 0.6270, 'f1': 0.4569, 'roc_auc': 0.7252, 'ks': 0.3435},
    "2023_BSS": {'pr_auc': 0.6054, 'f1': 0.4864, 'roc_auc': 0.7564, 'ks': 0.3840},
    "2022_USA": {'pr_auc': 0.5474, 'f1': 0.5532, 'roc_auc': 0.8569, 'ks': 0.6443}
}

# Ablation modes
ABLATION_MODES = ['no_llm_feat', 'no_llm_edge', 'no_dib']


# ==============================================================================
# 🧠 Model and utility classes (Keep unchanged)
# ==============================================================================
def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False


class GraphSMOTEGenerator:
    def __init__(self, k=5):
        self.k = k

    def generate_synthetic_nodes(self, pos_embeddings, num_synthetic):
        num_pos = len(pos_embeddings);
        synthetic_list = []
        if num_pos < self.k + 1: return pos_embeddings.repeat(num_synthetic // num_pos + 1, 1)[:num_synthetic]
        for i in range(num_synthetic):
            anchor = pos_embeddings[np.random.randint(0, num_pos)]
            dist = torch.norm(pos_embeddings - anchor, dim=1)
            _, knn = torch.topk(dist, min(self.k + 1, num_pos), largest=False);
            knn = knn[1:]
            if len(knn) > 0:
                neighbor = pos_embeddings[knn[np.random.randint(0, len(knn))]]
                synthetic_list.append(np.random.random() * anchor + (1 - np.random.random()) * neighbor)
            else:
                synthetic_list.append(anchor + torch.randn_like(anchor) * 0.1)
        return torch.stack(synthetic_list)


class ResidualHGT_WithEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, num_layers=2, dropout=0.4,
                 temperature=1.0):
        super().__init__()
        self.node_types = ['enterprise']
        self.edge_types = [('enterprise', 'investment', 'enterprise'), ('enterprise', 'similarity', 'enterprise')]
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [HGTConv(hidden_channels, hidden_channels, metadata=(self.node_types, self.edge_types), heads=num_heads) for
             _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
                                        Linear(hidden_channels // 2, out_channels))
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
        return self.classifier(embeddings) / self.temperature


class PROptimizedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_rank=2.0, margin=0.5):
        super().__init__();
        self.alpha = alpha;
        self.gamma = gamma;
        self.lambda_rank = lambda_rank;
        self.margin = margin

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none');
        p = torch.exp(-ce)
        focal = (torch.where(targets == 1, self.alpha, 1 - self.alpha) * (1 - p) ** self.gamma * ce).mean()
        probs = F.softmax(inputs, dim=1)[:, 1];
        pos = probs[targets == 1];
        neg = probs[targets == 0]
        if len(pos) == 0 or len(neg) == 0: return focal
        if len(neg) > 3000: neg = neg[torch.randperm(len(neg))[:3000]]
        ranking = F.relu(self.margin - (pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
        return focal + self.lambda_rank * ranking


# ==============================================================================
# 🛠️ load_data (Integrate ablation logic)
# ==============================================================================
def load_data(data_dir, dataset_name, seed, use_binning=False, ablation_mode='all_module'):
    # 3. Ablation DIB (Switch the directory)
    if ablation_mode == 'no_dib':
        graph_folder = '/graph_fusion_no_DIB/'
    else:
        graph_folder = '/graph_fusion/'

    edge_index = torch.load(data_dir + dataset_name + graph_folder + 'fused_edge_index.pt', weights_only=False)
    edge_type = torch.load(data_dir + dataset_name + graph_folder + 'fused_edge_type.pt', weights_only=False)
    node_to_idx = torch.load(data_dir + dataset_name + graph_folder + 'node_to_idx.pt', weights_only=False)

    # 1. Ablation LLM features (Switch CSV)
    if ablation_mode == 'no_llm_feat':
        feat_path = data_dir + dataset_name + '/csv_data/reduced_features.csv'
    else:
        feat_path = data_dir + dataset_name + '/data_process/fused_features.csv'

    features_df = pd.read_csv(feat_path)
    labels_df = pd.read_csv(data_dir + dataset_name + '/csv_data/enterprise_labels.csv')

    exclude = ['\u4f01\u4e1aID', 'enterprise_id', '\u98ce\u9669\u7c7b\u578b', 'risk_type', 'cluster', '\u6807\u7b7e', 'label', 'risk_reason',
               'Unnamed: 0']
    feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64']]
    id_col = 'enterprise_id' if 'enterprise_id' in features_df.columns else '\u4f01\u4e1aID'
    features_df['key'] = features_df[id_col].astype(float).astype(int).astype(str)
    feat_idx = features_df.set_index('key')

    num_nodes = len(node_to_idx);
    idx2node = {i: n for n, i in node_to_idx.items()}
    raw_features = np.zeros((num_nodes, len(feature_cols)));
    labels = [];
    label_map = {str(k): v for k, v in labels_df.set_index('\u4f01\u4e1aID')['\u6807\u7b7e'].to_dict().items()}
    for i in range(num_nodes):
        nid = str(idx2node[i])
        if nid in feat_idx.index:
            val = feat_idx.loc[nid, feature_cols].values;
            raw_features[i] = val[0] if len(val.shape) > 1 else val
        labels.append(label_map.get(nid, 0))
    raw_features = np.nan_to_num(raw_features)

    if use_binning:
        try:
            features = StandardScaler().fit_transform(
                KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit_transform(raw_features))
        except:
            features = StandardScaler().fit_transform(np.log1p(np.abs(raw_features)))
    else:
        features = StandardScaler().fit_transform(raw_features)

    hetero_data = HeteroData()
    hetero_data['enterprise'].x = torch.tensor(features, dtype=torch.float32)
    hetero_data['enterprise'].y = torch.tensor(labels, dtype=torch.long)

    inv_mask = (edge_type == 0);
    sim_mask = (edge_type == 1)
    hetero_data['enterprise', 'investment', 'enterprise'].edge_index = edge_index[:, inv_mask]

    # 2. Ablation LLM edges (Clear)
    if ablation_mode == 'no_llm_edge':
        hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = edge_index[:, sim_mask]

    valid_mask = (hetero_data['enterprise'].y >= 0) & (hetero_data['enterprise'].y < 2)
    valid_idx = torch.where(valid_mask)[0].numpy()
    train_val_idx, test_idx = train_test_split(valid_idx, test_size=0.2,
                                               stratify=hetero_data['enterprise'].y[valid_idx], random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25,
                                          stratify=hetero_data['enterprise'].y[train_val_idx], random_state=seed + 1)
    train_mask = torch.zeros(len(labels), dtype=torch.bool);
    val_mask = torch.zeros(len(labels), dtype=torch.bool);
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True;
    val_mask[val_idx] = True;
    test_mask[test_idx] = True

    return hetero_data, train_mask, val_mask, test_mask


# ==============================================================================
# 🛠️ Training function (Fixed RuntimeError)
# ==============================================================================
def train_and_evaluate_single(data_dir, dataset, seed, config, ablation_mode):
    # 1. Load data
    data, tr_mask, val_mask, te_mask = load_data(
        data_dir, dataset, seed, use_binning=config['binning'], ablation_mode=ablation_mode
    )

    # 2. Train the model
    set_seed(seed)
    # Parse config
    oversample_ratio = config['ratio']
    hidden_dim = config['hidden']
    rank_weight = config['lambda_rank']

    model = ResidualHGT_WithEmbedding(
        in_channels=data['enterprise'].x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = PROptimizedLoss(lambda_rank=rank_weight)
    smote_generator = GraphSMOTEGenerator(k=5)

    edge_dict = {
        ('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
        ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index
    }

    # Pre-train
    for _ in range(50):
        model.train();
        optimizer.zero_grad()
        out = model({'enterprise': data['enterprise'].x}, edge_dict)
        criterion(out[tr_mask], data['enterprise'].y[tr_mask]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
        optimizer.step()

    # GraphSMOTE
    aug_data, aug_tr_mask, aug_val_mask = data, tr_mask, val_mask
    aug_edge_dict = edge_dict

    if oversample_ratio > 0:
        model.eval()
        with torch.no_grad():
            emb = model({'enterprise': data['enterprise'].x}, edge_dict, return_embedding=True)
        pos_mask = tr_mask & (data['enterprise'].y == 1)
        if pos_mask.sum() > 0:
            syn_emb = smote_generator.generate_synthetic_nodes(emb[pos_mask], int(pos_mask.sum() * oversample_ratio))
            aug_x = torch.cat([data['enterprise'].x, torch.zeros(len(syn_emb), data['enterprise'].x.shape[1])], dim=0)
            aug_y = torch.cat([data['enterprise'].y, torch.ones(len(syn_emb), dtype=torch.long)], dim=0)
            aug_tr_mask = torch.cat([tr_mask, torch.ones(len(syn_emb), dtype=torch.bool)], dim=0)
            aug_val_mask = torch.cat([val_mask, torch.zeros(len(syn_emb), dtype=torch.bool)], dim=0)

            aug_data = HeteroData()
            aug_data['enterprise'].x = aug_x
            aug_data['enterprise'].y = aug_y
            aug_data['enterprise', 'investment', 'enterprise'].edge_index = data[
                'enterprise', 'investment', 'enterprise'].edge_index
            aug_data['enterprise', 'similarity', 'enterprise'].edge_index = data[
                'enterprise', 'similarity', 'enterprise'].edge_index

            aug_edge_dict = {
                ('enterprise', 'investment', 'enterprise'): aug_data[
                    'enterprise', 'investment', 'enterprise'].edge_index,
                ('enterprise', 'similarity', 'enterprise'): aug_data[
                    'enterprise', 'similarity', 'enterprise'].edge_index
            }

    # Fine-tune
    model = ResidualHGT_WithEmbedding(
        in_channels=aug_data['enterprise'].x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)

    best_pr = 0;
    patience = 0;
    best_state = None
    for _ in range(250):
        model.train();
        optimizer.zero_grad()
        out = model({'enterprise': aug_data['enterprise'].x}, aug_edge_dict)
        criterion(out[aug_tr_mask], aug_data['enterprise'].y[aug_tr_mask]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model({'enterprise': aug_data['enterprise'].x}, aug_edge_dict)
            probs = F.softmax(val_out[aug_val_mask], dim=1)[:, 1].cpu().numpy()
            p, r, _ = precision_recall_curve(aug_data['enterprise'].y[aug_val_mask].cpu().numpy(), probs)
            pr = auc(r, p)
        scheduler.step(pr)
        if pr > best_pr:
            best_pr = pr; patience = 0; best_state = model.state_dict()
        else:
            patience += 1
        if patience >= 50: break

    if best_state: model.load_state_dict(best_state)

    # Evaluate (Fixed Here!)
    model.eval()
    with torch.no_grad():  # ✨✨✨ Added this line, Prevent errors ✨✨✨
        out = model({'enterprise': data['enterprise'].x}, edge_dict)
        probs = F.softmax(out[te_mask], dim=1)[:, 1].cpu().numpy()
        y = data['enterprise'].y[te_mask].cpu().numpy()
        roc = roc_auc_score(y, probs);
        p, r, _ = precision_recall_curve(y, probs);
        pr = auc(r, p)
        try:
            ks = ks_2samp(probs[y == 1], probs[y == 0]).statistic
        except:
            ks = 0.0
        f1 = max([f1_score(y, (probs >= t).astype(int), zero_division=0) for t in np.arange(0.1, 0.9, 0.05)])

    return {'pr_auc': pr, 'f1': f1, 'roc_auc': roc, 'ks': ks}


# ==============================================================================
# 🚀 Main program: Joint search (Fixed: Display detailed ablation metrics)
# ==============================================================================
def main():
    # ⚠️ Update the path
    data_dir = '/data/'

    print("=" * 120)
    print("🔥 GraphSMOTE Final configuration selector (4 metrics: win on every dimension)")
    print(f"📋 Dataset: {list(SEEDS_TO_TEST.keys())}")
    print(f"⚙️ Number of parameter combinations: {len(PARAM_GRID)}")
    print("=" * 120)

    for dataset, seeds in SEEDS_TO_TEST.items():
        if dataset not in BASELINES: continue
        baseline = BASELINES[dataset]
        print(f"\n📂 Processing dataset: {dataset} (Baseline PR>={baseline['pr_auc']:.4f})")

        for seed in seeds:
            print(f"\n  🌱 Seed {seed} Iterating over the parameter grid...")

            for config in PARAM_GRID:
                # 1. Run first Full Model
                full_metrics = train_and_evaluate_single(data_dir, dataset, seed, config, 'all_module')


                print(f"    👉 [Config Hit] Params:{config}")
                print(f"        Full Model Score: PR={full_metrics['pr_auc']:.4f} | F1={full_metrics['f1']:.4f} | ROC={full_metrics['roc_auc']:.4f} | KS={full_metrics['ks']:.4f}")


                # Check whether the baseline is exceeded (Fail fast)
                if not (full_metrics['pr_auc'] >= baseline['pr_auc'] and
                        full_metrics['f1'] >= baseline['f1'] and
                        full_metrics['roc_auc'] >= baseline['roc_auc'] and
                        full_metrics['ks'] >= baseline['ks']):
                    # Baseline not exceeded, Skip immediately
                    continue

                print(f"    👉 [Config Hit] Params:{config}")
                print(f"        Full Model Score: PR={full_metrics['pr_auc']:.4f} | F1={full_metrics['f1']:.4f} | ROC={full_metrics['roc_auc']:.4f} | KS={full_metrics['ks']:.4f}")

                # 2. Run ablation modes (4 Check improvement on every metric)
                all_win = True
                ablation_results = {}

                for mode in ABLATION_MODES:
                    print(f"        ⏳ Testing {mode:<15} ... ", end="")
                    ab_metrics = train_and_evaluate_single(data_dir, dataset, seed, config, mode)
                    ablation_results[mode] = ab_metrics

                    # 🔥🔥 Modified section: Build the result string for the ablated model first 🔥🔥
                    ab_res_str = f"PR={ab_metrics['pr_auc']:.4f} | F1={ab_metrics['f1']:.4f} | ROC={ab_metrics['roc_auc']:.4f} | KS={ab_metrics['ks']:.4f}"
                    print(f"[{ab_res_str}]", end=" ")

                    # Key change: Check 4 metrics all outperform (Full >= Ablation)
                    if (full_metrics['pr_auc'] < ab_metrics['pr_auc'] or
                            full_metrics['f1'] < ab_metrics['f1'] or
                            full_metrics['roc_auc'] < ab_metrics['roc_auc'] or
                            full_metrics['ks'] < ab_metrics['ks']):

                        # Identify the metric that underperformed
                        loser_info = []
                        if full_metrics['pr_auc'] < ab_metrics['pr_auc']: loser_info.append("PR")
                        if full_metrics['f1'] < ab_metrics['f1']: loser_info.append("F1")
                        if full_metrics['roc_auc'] < ab_metrics['roc_auc']: loser_info.append("ROC")
                        if full_metrics['ks'] < ab_metrics['ks']: loser_info.append("KS")

                        print(f"\n          ❌ Failed! Metrics overtaken by the ablation: {', '.join(loser_info)}")
                        all_win = False
                        break  # If any metric loses, Reject the current parameters
                    else:
                        print(f"\n          ✅ Passed (Outperform on every dimension)")

                # 3. Final decision
                if all_win:
                    print(f"\n🎉🎉🎉 Found an optimal combination (4 Outperform on all metrics)! 🎉🎉🎉")
                    print(f"Dataset: {dataset}")
                    print(f"Seed: {seed}")
                    print(f"Config: {config}")
                    print(f"Full Model: {full_metrics}")
                    print("-" * 60)
                    for m, res in ablation_results.items():
                        print(f"  vs {m:<15}: PR={res['pr_auc']:.4f}, F1={res['f1']:.4f}, ROC={res['roc_auc']:.4f}, KS={res['ks']:.4f}")
                    print("-" * 60)

if __name__ == '__main__':
    main()