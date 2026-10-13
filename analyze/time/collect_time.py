"""
================================================================================
GNN Model efficiency analysis - Native PyTorch/PyG version (Without DGL)
================================================================================
Statistics 11  models' training and inference times, Save to  CSV  file

Model list (None use DGL): 
1. LR (sklearn)
2. XGBoost (xgboost)
3. MLP (Native PyTorch)
4. GCN (PyG)
5. GAT (PyG)
6. GraphSAGE (PyG)
7. DAGNN (PyTorch Implemented manually)
8. PinSAGE (PyG Simplified version)
9. PRAGA (PyTorch Implemented manually)
10. UnGSL (PyTorch Implemented manually)
11. HGT (PyG)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HGTConv
from torch_geometric.nn import Linear as PyGLinear
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import warnings
import random
import gc
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from xgboost import XGBClassifier  # Re-enable

warnings.filterwarnings('ignore')


# ==============================================================================
# 📌 Configuration section - Update these paths
# ==============================================================================

CONFIG = {
    # Data path - Edit here
    'data_dir': '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/',
    'dataset': '2023_BSS',

    # Output path - Edit here
    'output_csv': '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/time/2023_BSS_model_time_statistics_final.csv',

    # Model parameters
    'seed': 2,
    'hidden_dim': 64,
    'dropout': 0.5,
    'num_heads': 4,

    # Training parameters
    'lr': 0.001,
    'weight_decay': 5e-4,
    'max_epochs': 100,
    'patience': 20,
}


# ==============================================================================
# Utility functions
# ==============================================================================

def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_data(data_dir, dataset_name, seed):
    """Load base data"""
    graph_path = '/graph_fusion/'

    edge_index = torch.load(
        data_dir + dataset_name + graph_path + 'fused_edge_index.pt',
        weights_only=False
    )
    edge_type = torch.load(
        data_dir + dataset_name + graph_path + 'fused_edge_type.pt',
        weights_only=False
    )
    node_to_idx = torch.load(
        data_dir + dataset_name + graph_path + 'node_to_idx.pt',
        weights_only=False
    )

    features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features.csv')
    labels_df = pd.read_csv(data_dir + dataset_name + '/csv_data/enterprise_labels.csv')

    exclude = ['\u4f01\u4e1aID', 'enterprise_id', '\u98ce\u9669\u7c7b\u578b', 'risk_type',
               'cluster', '\u6807\u7b7e', 'label', 'risk_reason', 'Unnamed: 0']
    feature_cols = [c for c in features_df.columns
                   if c not in exclude and features_df[c].dtype in ['float64', 'int64']]

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
            if len(val.shape) > 1:
                val = val[0]
            raw_features[i] = val
        labels.append(label_map.get(nid, 0))

    raw_features = np.nan_to_num(raw_features)
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    valid_mask = (y >= 0) & (y < 2)
    valid_idx = torch.where(valid_mask)[0].numpy()

    train_val_idx, test_idx = train_test_split(
        valid_idx, test_size=0.2, stratify=y[valid_idx], random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, stratify=y[train_val_idx], random_state=seed + 1
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return x, y, edge_index, edge_type, train_mask, val_mask, test_mask


# ==============================================================================
# Model definitions
# ==============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_classes, dropout=0.7):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_feats, n_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_classes, num_heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_feats * num_heads, n_classes, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_classes, dropout=0.5):
        super().__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats)
        self.sage2 = SAGEConv(hidden_feats, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return x


class DAGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, num_layers=10, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, n_classes)
        self.alpha = nn.Parameter(torch.ones(num_layers + 1))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        h = F.relu(self.linear_in(x))
        outputs = [h]

        for _ in range(self.num_layers):
            h = torch.mm(adj_norm, h)
            outputs.append(h)

        H = torch.stack(outputs, dim=1)
        coeff = F.softmax(self.alpha, dim=0)
        H_agg = torch.sum(H * coeff.view(1, -1, 1), dim=1)
        H_agg = F.dropout(H_agg, p=self.dropout, training=self.training)
        out = self.linear_out(H_agg)
        return out


class PinSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage1 = SAGEConv(in_features, hidden_features)
        self.sage2 = SAGEConv(hidden_features, out_features)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        return x


class UnGSL(nn.Module):
    def __init__(self, inp_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        self.linear = nn.Linear(inp_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1).clamp(min=1)
        deg_inv = deg.pow(-1)
        adj_norm = deg_inv.view(-1, 1) * adj

        node_feat = torch.mm(adj_norm, x)
        node_feat = self.linear(node_feat)
        node_feat = F.relu(node_feat)
        node_feat = F.dropout(node_feat, p=self.dropout, training=self.training)
        node_feat = self.linear_out(node_feat)
        return node_feat


class PRAGA(nn.Module):
    """PRAGA Model - Simplified version (Based on GitHub code)"""
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Encoder
        feat_embedding = self.encoder(x)

        # Graph convolution (Simplified version)
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        emb_latent = torch.mm(adj_norm, feat_embedding)

        # Decoder
        emb_recon = torch.mm(adj_norm, torch.mm(emb_latent, self.decoder.weight.t()))

        # MLP classifier
        out = self.mlp(emb_latent)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class HGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, num_layers=2, dropout=0.4):
        super().__init__()
        self.node_types = ['enterprise']
        self.edge_types = [
            ('enterprise', 'investment', 'enterprise'),
            ('enterprise', 'similarity', 'enterprise')
        ]

        self.input_proj = PyGLinear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels,
                   metadata=(self.node_types, self.edge_types), heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            PyGLinear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            PyGLinear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        h = self.input_proj(x_dict['enterprise'])
        x_dict = {'enterprise': h}

        for i, conv in enumerate(self.convs):
            h_prev = x_dict['enterprise']
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
            x_dict['enterprise'] = x_dict['enterprise'] + h_prev
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        return self.classifier(x_dict['enterprise'])


# ==============================================================================
# Training and inference
# ==============================================================================

def train_ml(model, x, y, train_mask):
    """ training sklearn Model"""
    model.fit(x[train_mask.numpy()], y[train_mask.numpy()])
    return model

def train_gnn(model, x, y, edge_index, train_mask, val_mask):
    """ training GNN Model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(CONFIG['max_epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = criterion(val_out[val_mask], y[val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                break

    return model

def train_hgt(model, data, train_mask, val_mask):
    """ training HGT"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(CONFIG['max_epochs']):
        model.train()
        out = model(
            {'enterprise': data['enterprise'].x},
            {
                ('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
                ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index
            }
        )

        optimizer.zero_grad()
        loss = criterion(out[train_mask], data['enterprise'].y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(
                {'enterprise': data['enterprise'].x},
                {
                    ('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
                    ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index
                }
            )
            val_loss = criterion(val_out[val_mask], data['enterprise'].y[val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                break

    return model

@torch.no_grad()
def measure_infer_ml(model, x, mask):
    """Measure sklearn inference time - Infer all nodes ( and GNN Same as )"""
    # Infer all nodes, Rather than only the test set
    # Warm up
    for _ in range(5):
        _ = model.predict_proba(x)

    # Measure repeatedly and average
    times = []
    for _ in range(10):
        start = time.time()
        out = model.predict_proba(x)  # Infer all 
        _ = out[mask.numpy()]  # Keep test-set outputs only
        times.append(time.time() - start)

    return np.mean(times)

@torch.no_grad()
def measure_infer_gnn(model, x, edge_index, mask):
    """Measure GNN inference time - Entire graph (Including the test set)"""
    model.eval()

    # Warm up
    for _ in range(5):
        _ = model(x, edge_index)

    # Measure repeatedly and average
    times = []
    for _ in range(10):
        start = time.time()
        out = model(x, edge_index)
        # Keep test-set outputs only (Simulate actual use)
        _ = out[mask]
        times.append(time.time() - start)

    return np.mean(times)

@torch.no_grad()
def measure_infer_hgt(model, data, mask):
    """Measure HGT inference time - Entire graph (Including the test set)"""
    model.eval()

    # Warm up
    for _ in range(5):
        _ = model(
            {'enterprise': data['enterprise'].x},
            {
                ('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
                ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index
            }
        )

    # Measure repeatedly and average
    times = []
    for _ in range(10):
        start = time.time()
        out = model(
            {'enterprise': data['enterprise'].x},
            {
                ('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
                ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index
            }
        )
        # Keep test-set outputs only
        _ = out[mask]
        times.append(time.time() - start)

    return np.mean(times)


# ==============================================================================
# Main function
# ==============================================================================

def main():
    print("=" * 120)
    print("⏱️  GNN Model efficiency analysis - Native PyTorch/PyG version")
    print("=" * 120)
    print(f"📅 Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset: {CONFIG['dataset']}")
    print("=" * 120)

    set_seed(CONFIG['seed'])

    # Load data
    x, y, edge_index, edge_type, train_mask, val_mask, test_mask = load_base_data(
        CONFIG['data_dir'], CONFIG['dataset'], CONFIG['seed']
    )

    results = []

    # 1. LR
    print(f"\n⏱️  1/10: LR")
    start = time.time()
    lr = train_ml(LogisticRegression(max_iter=1000, random_state=CONFIG['seed']),
                  x.numpy(), y.numpy(), train_mask)
    train_time = time.time() - start
    infer_time = measure_infer_ml(lr, x.numpy(), test_mask)
    print(f"   training: {train_time:.2f}s, Inference: {infer_time:.6f}s")
    results.append({'model': 'LR', 'train_time': train_time, 'infer_time': infer_time})
    del lr; gc.collect()

    # 2. XGBoost
    print(f"\n⏱️  2/11: XGBoost")
    start = time.time()
    try:
        xgb = train_ml(XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                      random_state=CONFIG['seed'], eval_metric='logloss'),
                       x.numpy(), y.numpy(), train_mask)
        train_time = time.time() - start
        infer_time = measure_infer_ml(xgb, x.numpy(), test_mask)
        print(f"   training: {train_time:.2f}s, Inference: {infer_time:.6f}s")
        results.append({'model': 'XGBoost', 'train_time': train_time, 'infer_time': infer_time})
        del xgb; gc.collect()
    except Exception as e:
        print(f"  ❌ XGBoost Failure: {e}")
        print(f"  Tip: Run  'brew install libomp' Install OpenMP")

    # 3. MLP
    print(f"\n⏱️  3/11: MLP")  # Change to 3/11
    start = time.time()
    mlp = MLP(x.shape[1], CONFIG['hidden_dim'], 2, CONFIG['dropout'])
    optimizer = torch.optim.Adam(mlp.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    for _ in range(CONFIG['max_epochs']):
        mlp.train()
        optimizer.zero_grad()
        out = mlp(x)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    #  inference time - Entire graph
    mlp.eval()
    for _ in range(5):
        _ = mlp(x)

    times = []
    for _ in range(10):
        start_time = time.time()
        out = mlp(x)
        _ = out[test_mask]  # Keep only the test set
        times.append(time.time() - start_time)
    infer_time = np.mean(times)

    print(f"   training: {train_time:.2f}s, Inference: {infer_time:.6f}s")
    results.append({'model': 'MLP', 'train_time': train_time, 'infer_time': infer_time})
    del mlp; gc.collect()

    # 4-10. GNN Model
    gnn_configs = [
        ('GCN', GCN, {'in_feats': x.shape[1], 'hidden_feats': CONFIG['hidden_dim'], 'n_classes': 2, 'dropout': 0.7}),
        ('GAT', GAT, {'in_feats': x.shape[1], 'hidden_feats': CONFIG['hidden_dim'], 'n_classes': 2, 'num_heads': 4, 'dropout': 0.5}),
        ('GraphSAGE', GraphSAGE, {'in_feats': x.shape[1], 'hidden_feats': CONFIG['hidden_dim'], 'n_classes': 2, 'dropout': 0.5}),
        ('DAGNN', DAGNN, {'in_dim': x.shape[1], 'hidden_dim': CONFIG['hidden_dim'], 'n_classes': 2, 'num_layers': 10, 'dropout': 0.5}),
        ('PinSAGE', PinSAGE, {'in_features': x.shape[1], 'hidden_features': CONFIG['hidden_dim'], 'out_features': 2}),
        ('PRAGA', PRAGA, {'in_dim': x.shape[1], 'hidden_dim': CONFIG['hidden_dim'], 'n_classes': 2, 'dropout': 0.5}),
        ('UnGSL', UnGSL, {'inp_dim': x.shape[1], 'hidden_dim': CONFIG['hidden_dim'], 'n_classes': 2, 'dropout': 0.5}),
    ]

    for i, (name, ModelClass, kwargs) in enumerate(gnn_configs, 4):  # From 4 Start 
        print(f"\n⏱️  {i}/11: {name}")  # Change to /11
        start = time.time()
        model = ModelClass(**kwargs)
        model = train_gnn(model, x, y, edge_index, train_mask, val_mask)
        train_time = time.time() - start
        infer_time = measure_infer_gnn(model, x, edge_index, test_mask)
        print(f"   training: {train_time:.2f}s, Inference: {infer_time:.6f}s")
        results.append({'model': name, 'train_time': train_time, 'infer_time': infer_time})
        del model; gc.collect()

    # 11. HGT
    print(f"\n⏱️  11/11: HGT")  # Change to 11/11
    hetero_data = HeteroData()
    hetero_data['enterprise'].x = x
    hetero_data['enterprise'].y = y
    hetero_data['enterprise', 'investment', 'enterprise'].edge_index = edge_index[:, edge_type == 0]
    hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = edge_index[:, edge_type == 1]

    start = time.time()
    hgt = HGT(x.shape[1], CONFIG['hidden_dim'], 2, num_heads=4, num_layers=2, dropout=0.4)
    hgt = train_hgt(hgt, hetero_data, train_mask, val_mask)
    train_time = time.time() - start
    infer_time = measure_infer_hgt(hgt, hetero_data, test_mask)
    print(f"   training: {train_time:.2f}s, Inference: {infer_time:.6f}s")
    results.append({'model': 'HGT', 'train_time': train_time, 'infer_time': infer_time})

    # Save
    with open(CONFIG['output_csv'], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'train_time', 'infer_time'])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 120)
    print("📊 Results summary")
    print("=" * 120)
    print(f"{'Model':<15} {' training( seconds)':<12} {'Inference( seconds)':<15}")
    print("─" * 120)
    for r in results:
        print(f"{r['model']:<15} {r['train_time']:<12.2f} {r['infer_time']:<15.6f}")
    print(f"\n💾 Saved to: {CONFIG['output_csv']}")
    print(f"✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)


if __name__ == '__main__':
    main()