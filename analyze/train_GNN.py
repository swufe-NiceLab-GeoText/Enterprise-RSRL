"""
GNN Architecture ablation experiments - Based on the original train.py logic
Create a separate training function for each model
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HGTConv, GCNConv, GATConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import pandas as pd, numpy as np, warnings, random, gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from scipy.stats import ks_2samp
from datetime import datetime
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class GraphSMOTEGenerator:
    def __init__(self, k=5): self.k = k
    def generate_synthetic_nodes(self, pos_embeddings, num_synthetic):
        num_pos = len(pos_embeddings); synthetic_list = []
        if num_pos < self.k + 1: return pos_embeddings.repeat(num_synthetic // num_pos + 1, 1)[:num_synthetic]
        for i in range(num_synthetic):
            anchor_idx = np.random.randint(0, num_pos); anchor = pos_embeddings[anchor_idx]
            distances = torch.norm(pos_embeddings - anchor, dim=1)
            _, knn_indices = torch.topk(distances, min(self.k + 1, num_pos), largest=False)
            knn_indices = knn_indices[1:]
            if len(knn_indices) > 0:
                neighbor_idx = knn_indices[np.random.randint(0, len(knn_indices))]
                neighbor = pos_embeddings[neighbor_idx]; alpha = np.random.random()
                synthetic = alpha * anchor + (1 - alpha) * neighbor; synthetic_list.append(synthetic)
            else: noise = torch.randn_like(anchor) * 0.1; synthetic_list.append(anchor + noise)
        return torch.stack(synthetic_list)

class PROptimizedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_rank=2.8, margin=0.5):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.lambda_rank = lambda_rank; self.margin = margin
    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none'); p = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, torch.tensor(self.alpha, device=inputs.device), torch.tensor(1 - self.alpha, device=inputs.device))
        return (alpha_t * (1 - p) ** self.gamma * ce_loss).mean()
    def ranking_loss(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)[:, 1]; pos_mask = (targets == 1); neg_mask = (targets == 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0: return torch.tensor(0.0, device=inputs.device)
        pos_scores = probs[pos_mask]; neg_scores = probs[neg_mask]
        if len(neg_scores) > 3000: perm = torch.randperm(len(neg_scores))[:3000]; neg_scores = neg_scores[perm]
        pos_scores_expanded = pos_scores.unsqueeze(1); neg_scores_expanded = neg_scores.unsqueeze(0)
        return F.relu(self.margin - (pos_scores_expanded - neg_scores_expanded)).mean()
    def forward(self, inputs, targets): return self.focal_loss(inputs, targets) + self.lambda_rank * self.ranking_loss(inputs, targets)

class ResidualHGT_WithEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, num_layers=2, dropout=0.4, temperature=1.0):
        super().__init__()
        self.node_types = ['enterprise']; self.edge_types = [('enterprise', 'investment', 'enterprise'), ('enterprise', 'similarity', 'enterprise')]
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([HGTConv(hidden_channels, hidden_channels, metadata=(self.node_types, self.edge_types), heads=num_heads) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
        self.temperature = temperature
    def forward(self, x_dict, edge_index_dict, return_embedding=False):
        h = self.input_proj(x_dict['enterprise']); x_dict = {'enterprise': h}
        for i, conv in enumerate(self.convs):
            h_prev = x_dict['enterprise']; x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
            x_dict['enterprise'] = x_dict['enterprise'] + h_prev
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        embeddings = x_dict['enterprise']
        if return_embedding: return embeddings
        return self.classifier(embeddings) / self.temperature

class GCN_Homogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        edge_index = torch.cat([edge_index_invest, edge_index_sim], dim=1); h = self.input_proj(x)
        for i, conv in enumerate(self.convs):
            h_prev = h; h = conv(h, edge_index); h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

class GAT_Homogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        edge_index = torch.cat([edge_index_invest, edge_index_sim], dim=1); h = self.input_proj(x)
        for i, conv in enumerate(self.convs):
            h_prev = h; h = conv(h, edge_index); h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

class GraphSAGE_Homogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        edge_index = torch.cat([edge_index_invest, edge_index_sim], dim=1); h = self.input_proj(x)
        for i, conv in enumerate(self.convs):
            h_prev = h; h = conv(h, edge_index); h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

class GCN_Heterogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs_invest = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.convs_sim = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        h = self.input_proj(x)
        for i in range(len(self.convs_invest)):
            h_prev = h; h_invest = self.convs_invest[i](h, edge_index_invest); h_sim = self.convs_sim[i](h, edge_index_sim)
            h = h_invest*0.4+h_sim*0.6; h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

class GAT_Heterogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs_invest = nn.ModuleList([GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.convs_sim = nn.ModuleList([GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        h = self.input_proj(x)
        for i in range(len(self.convs_invest)):
            h_prev = h; h_invest = self.convs_invest[i](h, edge_index_invest); h_sim = self.convs_sim[i](h, edge_index_sim)
            h = h_invest*0.4+h_sim*0.6; h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

class GraphSAGE_Heterogeneous(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs_invest = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.convs_sim = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_channels // 2, out_channels))
    def forward(self, x, edge_index_invest, edge_index_sim, return_embedding=False):
        h = self.input_proj(x)
        for i in range(len(self.convs_invest)):
            h_prev = h; h_invest = self.convs_invest[i](h, edge_index_invest); h_sim = self.convs_sim[i](h, edge_index_sim)
            h = h_invest*0.4+h_sim*0.6; h = self.norms[i](h); h = h + h_prev; h = F.relu(h); h = self.dropout(h)
        if return_embedding: return h
        return self.classifier(h)

def load_data(data_dir, dataset_name, seed):
    graph_path = '/graph_fusion/'
    edge_index = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_index.pt', weights_only=False)
    edge_type = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_type.pt', weights_only=False)
    node_to_idx = torch.load(data_dir + dataset_name + graph_path + 'node_to_idx.pt', weights_only=False)
    features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features.csv')
    labels_df = pd.read_csv(data_dir + dataset_name + '/csv_data/enterprise_labels.csv')
    exclude = ['\u4f01\u4e1aID', 'enterprise_id', '\u98ce\u9669\u7c7b\u578b', 'risk_type', 'cluster', '\u6807\u7b7e', 'label', 'risk_reason', 'Unnamed: 0']
    feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64']]
    id_col = 'enterprise_id' if 'enterprise_id' in features_df.columns else '\u4f01\u4e1aID'
    features_df['key'] = features_df[id_col].astype(float).astype(int).astype(str)
    feat_idx = features_df.set_index('key'); num_nodes = len(node_to_idx); idx2node = {i: n for n, i in node_to_idx.items()}
    raw_features = np.zeros((num_nodes, len(feature_cols))); labels = []
    label_map = {str(k): v for k, v in labels_df.set_index('\u4f01\u4e1aID')['\u6807\u7b7e'].to_dict().items()}
    for i in range(num_nodes):
        nid = str(idx2node[i])
        if nid in feat_idx.index:
            val = feat_idx.loc[nid, feature_cols].values
            if len(val.shape) > 1: val = val[0]
            raw_features[i] = val
        labels.append(label_map.get(nid, 0))
    raw_features = np.nan_to_num(raw_features); scaler = StandardScaler(); features = scaler.fit_transform(raw_features)
    x = torch.tensor(features, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.long)
    hetero_data = HeteroData(); hetero_data['enterprise'].x = x; hetero_data['enterprise'].y = y
    inv_mask = (edge_type == 0); sim_mask = (edge_type == 1)
    hetero_data['enterprise', 'investment', 'enterprise'].edge_index = edge_index[:, inv_mask]
    hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = edge_index[:, sim_mask]
    valid_mask = (y >= 0) & (y < 2); valid_idx = torch.where(valid_mask)[0].numpy()
    train_val_idx, test_idx = train_test_split(valid_idx, test_size=0.2, stratify=y[valid_idx], random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, stratify=y[train_val_idx], random_state=seed + 1)
    train_mask = torch.zeros(len(y), dtype=torch.bool); val_mask = torch.zeros(len(y), dtype=torch.bool); test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True; val_mask[val_idx] = True; test_mask[test_idx] = True
    return hetero_data, train_mask, val_mask, test_mask

def create_train_function(ModelClass, model_name, is_hgt=False):
    def train_fn(data, train_mask, val_mask, model_seed, config):
        set_seed(model_seed); oversample_ratio = config['ratio']; hidden_dim = config['hidden']; rank_weight = config['lambda_rank']
        if is_hgt: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 4, 2, 0.4, 1.0)
        elif 'GAT' in model_name: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 4, 2, 0.4)
        else: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 2, 0.4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
        criterion = PROptimizedLoss(alpha=0.25, gamma=2.0, lambda_rank=rank_weight, margin=0.5)
        smote_generator = GraphSMOTEGenerator(k=5)
        for epoch in range(50):
            model.train()
            if is_hgt: out = model({'enterprise': data['enterprise'].x}, {('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index, ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index})
            else: out = model(data['enterprise'].x, data['enterprise', 'investment', 'enterprise'].edge_index, data['enterprise', 'similarity', 'enterprise'].edge_index)
            optimizer.zero_grad(); loss = criterion(out[train_mask], data['enterprise'].y[train_mask]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
        if oversample_ratio > 0:
            model.eval()
            with torch.no_grad():
                if is_hgt: embeddings = model({'enterprise': data['enterprise'].x}, {('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index, ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index}, return_embedding=True)
                else: embeddings = model(data['enterprise'].x, data['enterprise', 'investment', 'enterprise'].edge_index, data['enterprise', 'similarity', 'enterprise'].edge_index, return_embedding=True)
            pos_train_mask = train_mask & (data['enterprise'].y == 1)
            if pos_train_mask.sum() > 0:
                pos_embeddings = embeddings[pos_train_mask]; num_pos = pos_train_mask.sum().item(); num_synthetic = int(num_pos * oversample_ratio)
                if num_synthetic > 0:
                    synthetic_embeddings = smote_generator.generate_synthetic_nodes(pos_embeddings, num_synthetic)
                    augmented_data = HeteroData()
                    augmented_data['enterprise'].x = torch.cat([data['enterprise'].x, torch.zeros(num_synthetic, data['enterprise'].x.shape[1])], dim=0)
                    augmented_data['enterprise'].y = torch.cat([data['enterprise'].y, torch.ones(num_synthetic, dtype=torch.long)], dim=0)
                    augmented_data['enterprise', 'investment', 'enterprise'].edge_index = data['enterprise', 'investment', 'enterprise'].edge_index
                    augmented_data['enterprise', 'similarity', 'enterprise'].edge_index = data['enterprise', 'similarity', 'enterprise'].edge_index
                    augmented_train_mask = torch.cat([train_mask, torch.ones(num_synthetic, dtype=torch.bool)], dim=0)
                    augmented_val_mask = torch.cat([val_mask, torch.zeros(num_synthetic, dtype=torch.bool)], dim=0)
        if is_hgt: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 4, 2, 0.4, 1.0)
        elif 'GAT' in model_name: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 4, 2, 0.4)
        else: model = ModelClass(data['enterprise'].x.shape[1], hidden_dim, 2, 2, 0.4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30, verbose=False)
        best_val_pr = 0; patience_counter = 0; best_model_state = None
        for epoch in range(250):
            model.train()
            if is_hgt: out = model({'enterprise': augmented_data['enterprise'].x}, {('enterprise', 'investment', 'enterprise'): augmented_data['enterprise', 'investment', 'enterprise'].edge_index, ('enterprise', 'similarity', 'enterprise'): augmented_data['enterprise', 'similarity', 'enterprise'].edge_index})
            else: out = model(augmented_data['enterprise'].x, augmented_data['enterprise', 'investment', 'enterprise'].edge_index, augmented_data['enterprise', 'similarity', 'enterprise'].edge_index)
            optimizer.zero_grad(); loss = criterion(out[augmented_train_mask], augmented_data['enterprise'].y[augmented_train_mask]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            model.eval()
            with torch.no_grad():
                if is_hgt: val_out = model({'enterprise': augmented_data['enterprise'].x}, {('enterprise', 'investment', 'enterprise'): augmented_data['enterprise', 'investment', 'enterprise'].edge_index, ('enterprise', 'similarity', 'enterprise'): augmented_data['enterprise', 'similarity', 'enterprise'].edge_index})
                else: val_out = model(augmented_data['enterprise'].x, augmented_data['enterprise', 'investment', 'enterprise'].edge_index, augmented_data['enterprise', 'similarity', 'enterprise'].edge_index)
                val_probs = F.softmax(val_out[augmented_val_mask], dim=1)[:, 1].cpu().numpy()
                val_y = augmented_data['enterprise'].y[augmented_val_mask].cpu().numpy()
                val_prec, val_rec, _ = precision_recall_curve(val_y, val_probs)
                val_pr_auc = auc(val_rec, val_prec)
            scheduler.step(val_pr_auc)
            if val_pr_auc > best_val_pr:
                best_val_pr = val_pr_auc; patience_counter = 0; best_model_state = model.state_dict()
            else:
                patience_counter += 1
            if patience_counter >= 50:
                break
        if best_model_state is not None: model.load_state_dict(best_model_state)
        return model
    return train_fn

train_GCN = create_train_function(GCN_Homogeneous, 'GCN')
train_GAT = create_train_function(GAT_Homogeneous, 'GAT')
train_GraphSAGE = create_train_function(GraphSAGE_Homogeneous, 'GraphSAGE')
train_GCN_Het = create_train_function(GCN_Heterogeneous, 'GCN_Het')
train_GAT_Het = create_train_function(GAT_Heterogeneous, 'GAT_Het')
train_GraphSAGE_Het = create_train_function(GraphSAGE_Heterogeneous, 'GraphSAGE_Het')
train_HGT = create_train_function(ResidualHGT_WithEmbedding, 'HGT', is_hgt=True)

@torch.no_grad()
def evaluate(model, data, mask, is_hgt):
    model.eval()
    if is_hgt: out = model({'enterprise': data['enterprise'].x}, {('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index, ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index})
    else: out = model(data['enterprise'].x, data['enterprise', 'investment', 'enterprise'].edge_index, data['enterprise', 'similarity', 'enterprise'].edge_index)
    probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy(); y_true = data['enterprise'].y[mask].cpu().numpy()
    roc_auc = roc_auc_score(y_true, probs); prec, rec, _ = precision_recall_curve(y_true, probs); pr_auc = auc(rec, prec)
    try: ks_stat = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    except: ks_stat = 0.0
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= threshold).astype(int); f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1: best_f1 = f1
    return {'f1': best_f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}

def main():
    data_dir = '/data/';
    dataset = '2023_BSS'
    seed = 73
    config = {'hidden': 64, 'ratio': 2.0, 'lambda_rank': 2.5}

    print("="*120); print("🔥 GNN Architecture ablation and comparison experiments"); print("="*120)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120)
    print(f"\n📊 Dataset: {dataset}"); print(f"🎲 Random seed: {seed}")
    print(f"⚙️  Configuration: Hidden={config['hidden']}, Ratio={config['ratio']}, Lambda_rank={config['lambda_rank']}")
    print("="*120)

    models_to_train = {'GCN': (train_GCN, False), 'GAT': (train_GAT, False), 'GraphSAGE': (train_GraphSAGE, False), 'GCN+Ours': (train_GCN_Het, False), 'GAT+Ours': (train_GAT_Het, False), 'GraphSAGE+Ours': (train_GraphSAGE_Het, False), 'Ours (HGT)': (train_HGT, True)}
    results = {};
    print("\n"+"="*120); print("🚀 Start training 7 methods...")
    print("="*120)
    for i, (method_name, (train_func, is_hgt)) in enumerate(models_to_train.items(), 1):
        print(f"\n{'─'*120}")
        print(f"🔥 Method {i}/7: {method_name}")
        print(f"{'─'*120}")
        print(f"  ⏳ Training (Three stages: Pretraining 50 epochs → GraphSMOTE → Retraining 250 epochs)...")
        data, train_mask, val_mask, test_mask = load_data(data_dir, dataset, seed)
        model = train_func(data, train_mask, val_mask, seed, config)
        metrics = evaluate(model, data, test_mask, is_hgt); results[method_name] = metrics
        print(f"  ✓ PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  ✓ F1:     {metrics['f1']:.4f}")
        print(f"  ✓ ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  ✓ KS:     {metrics['ks']:.4f}")
        del model, data, train_mask, val_mask, test_mask; torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    print("\n"+"="*120); print("📊 Experiment results"); print("="*120); print(f"\n{'Method':<20} {'PR-AUC':<12} {'F1':<12} {'ROC-AUC':<12} {'KS':<12}"); print("─"*120)
    for method_name, metrics in results.items(): print(f"{method_name:<20} {metrics['pr_auc']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f} {metrics['ks']:<12.4f}")
    print("\n"+"="*120); print("📈 Improvements from heterogeneous routing"); print("="*120); pairs = [('GCN', 'GCN+Ours'), ('GAT', 'GAT+Ours'), ('GraphSAGE', 'GraphSAGE+Ours')]; print(f"\n{'Comparison':<30} {'PR-AUC improvement':<15} {'F1 improvement':<15} {'ROC-AUC improvement':<15} {'KS improvement':<15}"); print("─"*120)
    for base, improved in pairs:
        base_metrics = results[base]; improved_metrics = results[improved]
        pr_improve = ((improved_metrics['pr_auc'] - base_metrics['pr_auc']) / base_metrics['pr_auc']) * 100; f1_improve = ((improved_metrics['f1'] - base_metrics['f1']) / base_metrics['f1']) * 100; roc_improve = ((improved_metrics['roc_auc'] - base_metrics['roc_auc']) / base_metrics['roc_auc']) * 100; ks_improve = ((improved_metrics['ks'] - base_metrics['ks']) / base_metrics['ks']) * 100
        print(f"{base} → {improved:<15} {pr_improve:>+6.2f}%        {f1_improve:>+6.2f}%        {roc_improve:>+6.2f}%        {ks_improve:>+6.2f}%")
    print("\n"+"="*120); print("🏆 Compared with the full method (Ours HGT)  comparison"); print("="*120); hgt_metrics = results['Ours (HGT)']; print(f"\n{'Method':<20} {'PR-AUC gap':<15} {'F1 gap':<15} {'ROC-AUC gap':<15} {'KS gap':<15}"); print("─"*120)
    for method_name, metrics in results.items():
        if method_name == 'Ours (HGT)': continue
        pr_gap = hgt_metrics['pr_auc'] - metrics['pr_auc']; f1_gap = hgt_metrics['f1'] - metrics['f1']; roc_gap = hgt_metrics['roc_auc'] - metrics['roc_auc']; ks_gap = hgt_metrics['ks'] - metrics['ks']
        print(f"{method_name:<20} {pr_gap:>+7.4f}       {f1_gap:>+7.4f}       {roc_gap:>+7.4f}       {ks_gap:>+7.4f}")
    print("\n"+"="*120); print(f"✅ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); print("="*120)

if __name__ == '__main__': main()