# """
# ================================================================================
# GraphSMOTE Label-permutation test - Complete ablation version
# ================================================================================
# [Description]
# Complete ablation experiment, Including 5 comparison groups: 
# 1. With industry standards + Original labels
# 2. With industry standards + Shuffled labels
# 3. Without industry standards + Original labels
# 4. Without industry standards + Shuffled labels
# 5. Random-guessing baseline
# ================================================================================
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import HGTConv, Linear
# from torch_geometric.data import HeteroData
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
# from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
# from scipy.stats import ks_2samp
# import warnings
# import random
# import os
# from datetime import datetime
#
# warnings.filterwarnings('ignore')
#
# # ==============================================================================
# # 📌 Configuration: Dataset parameters and seeds
# # ==============================================================================
#
# DATASET_CONFIGS = {
#     "2023_SH": {'ratio': 2.9, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
#     "2023_SZ": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 0.9, 'binning': False},
#     "2022_SH": {'ratio': 1.7, 'hidden': 64, 'lambda_rank': 3.1, 'binning': False},
#     "2022_SZ": {'ratio': 2.5, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
#     "2022_USA": {'ratio': 0.5, 'hidden': 64, 'lambda_rank': 2.0, 'binning': True},
#     "2022_BSS": {'ratio': 0.913, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
#     "2023_BSS": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 1.5, 'binning': False}
# }
#
# VALID_SEEDS = {
#     "2022_SH": [2],
#     "2022_SZ": [72],
#     "2023_SH": [42],
#     "2023_SZ": [15],
#     "2022_BSS": [109],
#     "2023_BSS": [25],
#     "2022_USA": [82]
# }
#
#
# def set_seed(seed):
#     """Set the random seed"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# # ==============================================================================
# # Model definitions
# # ==============================================================================
#
# class GraphSMOTEGenerator:
#     """GraphSMOTE: In embedding space, generate synthetic positive samples"""
#
#     def __init__(self, k=5):
#         self.k = k
#
#     def generate_synthetic_nodes(self, pos_embeddings, num_synthetic):
#         num_pos = len(pos_embeddings)
#         synthetic_list = []
#         if num_pos < self.k + 1:
#             return pos_embeddings.repeat(num_synthetic // num_pos + 1, 1)[:num_synthetic]
#
#         for i in range(num_synthetic):
#             anchor_idx = np.random.randint(0, num_pos)
#             anchor = pos_embeddings[anchor_idx]
#             distances = torch.norm(pos_embeddings - anchor, dim=1)
#             _, knn_indices = torch.topk(distances, min(self.k + 1, num_pos), largest=False)
#             knn_indices = knn_indices[1:]
#
#             if len(knn_indices) > 0:
#                 neighbor_idx = knn_indices[np.random.randint(0, len(knn_indices))]
#                 neighbor = pos_embeddings[neighbor_idx]
#                 alpha = np.random.random()
#                 synthetic = alpha * anchor + (1 - alpha) * neighbor
#                 synthetic_list.append(synthetic)
#             else:
#                 noise = torch.randn_like(anchor) * 0.1
#                 synthetic_list.append(anchor + noise)
#         return torch.stack(synthetic_list)
#
#
# class ResidualHGT_WithEmbedding(nn.Module):
#     """Improved HGT, Return intermediate embedding For GraphSMOTE"""
#
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  num_heads=4, num_layers=2, dropout=0.4, temperature=1.0):
#         super().__init__()
#         self.node_types = ['enterprise']
#         self.edge_types = [
#             ('enterprise', 'investment', 'enterprise'),
#             ('enterprise', 'similarity', 'enterprise')
#         ]
#         self.input_proj = Linear(in_channels, hidden_channels)
#         self.convs = nn.ModuleList([
#             HGTConv(hidden_channels, hidden_channels,
#                     metadata=(self.node_types, self.edge_types),
#                     heads=num_heads)
#             for _ in range(num_layers)
#         ])
#         self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Sequential(
#             Linear(hidden_channels, hidden_channels // 2),
#             nn.ReLU(), nn.Dropout(dropout),
#             Linear(hidden_channels // 2, out_channels)
#         )
#         self.temperature = temperature
#
#     def forward(self, x_dict, edge_index_dict, return_embedding=False):
#         h = self.input_proj(x_dict['enterprise'])
#         x_dict = {'enterprise': h}
#         for i, conv in enumerate(self.convs):
#             h_prev = x_dict['enterprise']
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
#             x_dict['enterprise'] = x_dict['enterprise'] + h_prev
#             x_dict = {k: F.relu(v) for k, v in x_dict.items()}
#             x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
#         embeddings = x_dict['enterprise']
#         if return_embedding:
#             return embeddings
#         logits = self.classifier(embeddings)
#         return logits / self.temperature
#
#
# class PROptimizedLoss(nn.Module):
#     """PR-AUC-optimized loss function"""
#
#     def __init__(self, alpha=0.25, gamma=2.0, lambda_rank=2.0, margin=0.5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.lambda_rank = lambda_rank
#         self.margin = margin
#
#     def focal_loss(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         p = torch.exp(-ce_loss)
#         alpha_t = torch.where(targets == 1,
#                               torch.tensor(self.alpha, device=inputs.device),
#                               torch.tensor(1 - self.alpha, device=inputs.device))
#         focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
#         return focal_loss.mean()
#
#     def ranking_loss(self, inputs, targets):
#         probs = F.softmax(inputs, dim=1)[:, 1]
#         pos_mask = (targets == 1)
#         neg_mask = (targets == 0)
#         if pos_mask.sum() == 0 or neg_mask.sum() == 0:
#             return torch.tensor(0.0, device=inputs.device)
#         pos_scores = probs[pos_mask]
#         neg_scores = probs[neg_mask]
#         if len(neg_scores) > 3000:
#             perm = torch.randperm(len(neg_scores))[:3000]
#             neg_scores = neg_scores[perm]
#         pos_scores_expanded = pos_scores.unsqueeze(1)
#         neg_scores_expanded = neg_scores.unsqueeze(0)
#         pairwise_loss = F.relu(self.margin - (pos_scores_expanded - neg_scores_expanded))
#         return pairwise_loss.mean()
#
#     def forward(self, inputs, targets):
#         focal = self.focal_loss(inputs, targets)
#         ranking = self.ranking_loss(inputs, targets)
#         return focal + self.lambda_rank * ranking
#
#
# # ==============================================================================
# # Data loading (Support features with industry standards and / without industry standards)
# # ==============================================================================
#
# def load_data(data_dir, dataset_name, seed, use_binning=False, use_no_standard=False):
#     """
#     Load data
#
#     Args:
#         use_no_standard: True=Use features without industry standards, False=Use features with industry standards
#     """
#     # Select graph paths based on the parameter
#     if use_no_standard:
#         graph_path = '/graph_fusion_easy/'
#         print(f"  → Use features without industry standards")
#     else:
#         graph_path = '/graph_fusion/'
#         print(f"  → Use features with industry standards")
#
#     edge_index = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_index.pt', weights_only=False)
#     edge_type = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_type.pt', weights_only=False)
#     node_to_idx = torch.load(data_dir + dataset_name + graph_path + 'node_to_idx.pt', weights_only=False)
#
#     # 🔑 Key change: Based on use_no_standard Select different feature files
#     if use_no_standard:
#         # features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features_easy.csv')     # _no_rule
#         features_df = pd.read_csv(data_dir + dataset_name + '/data_process/llm_enhanced_features_easy.csv')     # _no_rule
#
#     else:
#         # features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features.csv')
#         features_df = pd.read_csv(data_dir + dataset_name + '/data_process/llm_enhanced_features_v6_industry.csv')
#
#     labels_df = pd.read_csv(data_dir + dataset_name + '/csv_data/enterprise_labels.csv')
#
#     exclude = [' enterprise ID', 'enterprise_id', 'Risk type', 'risk_type', 'cluster', 'Label', 'label', 'risk_reason',
#                'Unnamed: 0']
#     feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ['float64', 'int64']]
#
#     id_col = 'enterprise_id' if 'enterprise_id' in features_df.columns else ' enterprise ID'
#     features_df['key'] = features_df[id_col].astype(float).astype(int).astype(str)
#     feat_idx = features_df.set_index('key')
#
#     num_nodes = len(node_to_idx)
#     idx2node = {i: n for n, i in node_to_idx.items()}
#
#     raw_features = np.zeros((num_nodes, len(feature_cols)))
#     labels = []
#     label_map = {str(k): v for k, v in labels_df.set_index(' enterprise ID')['Label'].to_dict().items()}
#
#     for i in range(num_nodes):
#         nid = str(idx2node[i])
#         if nid in feat_idx.index:
#             val = feat_idx.loc[nid, feature_cols].values
#             if len(val.shape) > 1:
#                 val = val[0]
#             raw_features[i] = val
#         labels.append(label_map.get(nid, 0))
#     raw_features = np.nan_to_num(raw_features)
#
#     # Feature processing
#     if use_binning:
#         try:
#             est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
#             features_binned = est.fit_transform(raw_features)
#             scaler = StandardScaler()
#             features = scaler.fit_transform(features_binned)
#         except:
#             scaler = StandardScaler()
#             features = scaler.fit_transform(np.log1p(np.abs(raw_features)))
#     else:
#         scaler = StandardScaler()
#         features = scaler.fit_transform(raw_features)
#
#     x = torch.tensor(features, dtype=torch.float32)
#     y = torch.tensor(labels, dtype=torch.long)
#     hetero_data = HeteroData()
#     hetero_data['enterprise'].x = x
#     hetero_data['enterprise'].y = y
#
#     inv_mask = (edge_type == 0)
#     sim_mask = (edge_type == 1)
#     hetero_data['enterprise', 'investment', 'enterprise'].edge_index = edge_index[:, inv_mask]
#     hetero_data['enterprise', 'similarity', 'enterprise'].edge_index = edge_index[:, sim_mask]
#
#     valid_mask = (y >= 0) & (y < 2)
#     valid_idx = torch.where(valid_mask)[0].numpy()
#     train_val_idx, test_idx = train_test_split(valid_idx, test_size=0.2, stratify=y[valid_idx], random_state=seed)
#     train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, stratify=y[train_val_idx],
#                                           random_state=seed + 1)
#
#     train_mask = torch.zeros(len(y), dtype=torch.bool)
#     val_mask = torch.zeros(len(y), dtype=torch.bool)
#     test_mask = torch.zeros(len(y), dtype=torch.bool)
#     train_mask[train_idx] = True
#     val_mask[val_idx] = True
#     test_mask[test_idx] = True
#
#     return hetero_data, train_mask, val_mask, test_mask
#
#
# # ==============================================================================
# # Training function
# # ==============================================================================
#
# def train_single_model_with_smote(data, train_mask, val_mask, model_seed, config):
#     """Train a single model"""
#     set_seed(model_seed)
#
#     oversample_ratio = config['ratio']
#     hidden_dim = config['hidden']
#     rank_weight = config['lambda_rank']
#
#     model = ResidualHGT_WithEmbedding(
#         in_channels=data['enterprise'].x.shape[1],
#         hidden_channels=hidden_dim,
#         out_channels=2,
#         num_heads=4,
#         num_layers=2,
#         dropout=0.4,
#         temperature=1.0
#     )
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
#     criterion = PROptimizedLoss(alpha=0.25, gamma=2.0, lambda_rank=rank_weight, margin=0.5)
#     smote_generator = GraphSMOTEGenerator(k=5)
#
#     # Stage 1: Pretraining
#     for epoch in range(50):
#         model.train()
#         out = model({'enterprise': data['enterprise'].x},
#                     {('enterprise', 'investment', 'enterprise'): data[
#                         'enterprise', 'investment', 'enterprise'].edge_index,
#                      ('enterprise', 'similarity', 'enterprise'): data[
#                          'enterprise', 'similarity', 'enterprise'].edge_index})
#         optimizer.zero_grad()
#         loss = criterion(out[train_mask], data['enterprise'].y[train_mask])
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#
#     # Stage 2: Generate GraphSMOTE Synthetic samples
#     if oversample_ratio > 0:
#         model.eval()
#         with torch.no_grad():
#             embeddings = model({'enterprise': data['enterprise'].x},
#                                {('enterprise', 'investment', 'enterprise'): data[
#                                    'enterprise', 'investment', 'enterprise'].edge_index,
#                                 ('enterprise', 'similarity', 'enterprise'): data[
#                                     'enterprise', 'similarity', 'enterprise'].edge_index},
#                                return_embedding=True)
#         pos_train_mask = train_mask & (data['enterprise'].y == 1)
#         if pos_train_mask.sum() > 0:
#             pos_embeddings = embeddings[pos_train_mask]
#             num_pos = pos_train_mask.sum().item()
#             num_synthetic = int(num_pos * oversample_ratio)
#             if num_synthetic > 0:
#                 synthetic_embeddings = smote_generator.generate_synthetic_nodes(pos_embeddings, num_synthetic)
#                 augmented_x = torch.cat(
#                     [data['enterprise'].x, torch.zeros(num_synthetic, data['enterprise'].x.shape[1])], dim=0)
#                 augmented_y = torch.cat([data['enterprise'].y, torch.ones(num_synthetic, dtype=torch.long)], dim=0)
#                 augmented_train_mask = torch.cat([train_mask, torch.ones(num_synthetic, dtype=torch.bool)], dim=0)
#                 augmented_val_mask = torch.cat([val_mask, torch.zeros(num_synthetic, dtype=torch.bool)], dim=0)
#                 augmented_data = HeteroData()
#                 augmented_data['enterprise'].x = augmented_x
#                 augmented_data['enterprise'].y = augmented_y
#                 augmented_data['enterprise', 'investment', 'enterprise'].edge_index = data[
#                     'enterprise', 'investment', 'enterprise'].edge_index
#                 augmented_data['enterprise', 'similarity', 'enterprise'].edge_index = data[
#                     'enterprise', 'similarity', 'enterprise'].edge_index
#             else:
#                 augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
#         else:
#             augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
#     else:
#         augmented_data, augmented_train_mask, augmented_val_mask = data, train_mask, val_mask
#
#     # Stage 3: Retrain
#     model = ResidualHGT_WithEmbedding(
#         in_channels=data['enterprise'].x.shape[1],
#         hidden_channels=hidden_dim,
#         out_channels=2,
#         num_heads=4,
#         num_layers=2,
#         dropout=0.4,
#         temperature=1.0
#     )
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30,
#                                                            verbose=False)
#
#     best_val_pr = 0
#     patience_counter = 0
#     best_model_state = None
#
#     for epoch in range(250):
#         model.train()
#         out = model({'enterprise': augmented_data['enterprise'].x},
#                     {('enterprise', 'investment', 'enterprise'): augmented_data[
#                         'enterprise', 'investment', 'enterprise'].edge_index,
#                      ('enterprise', 'similarity', 'enterprise'): augmented_data[
#                          'enterprise', 'similarity', 'enterprise'].edge_index})
#         optimizer.zero_grad()
#         loss = criterion(out[augmented_train_mask], augmented_data['enterprise'].y[augmented_train_mask])
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#
#         model.eval()
#         with torch.no_grad():
#             val_out = model({'enterprise': augmented_data['enterprise'].x},
#                             {('enterprise', 'investment', 'enterprise'): augmented_data[
#                                 'enterprise', 'investment', 'enterprise'].edge_index,
#                              ('enterprise', 'similarity', 'enterprise'): augmented_data[
#                                  'enterprise', 'similarity', 'enterprise'].edge_index})
#             val_probs = F.softmax(val_out[augmented_val_mask], dim=1)[:, 1].cpu().numpy()
#             val_y = augmented_data['enterprise'].y[augmented_val_mask].cpu().numpy()
#             val_prec, val_rec, _ = precision_recall_curve(val_y, val_probs)
#             val_pr_auc = auc(val_rec, val_prec)
#         scheduler.step(val_pr_auc)
#
#         if val_pr_auc > best_val_pr:
#             best_val_pr = val_pr_auc
#             patience_counter = 0
#             best_model_state = model.state_dict()
#         else:
#             patience_counter += 1
#
#         if patience_counter >= 50:
#             break
#
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#
#     return model
#
#
# @torch.no_grad()
# def evaluate(model, data, mask):
#     """Evaluate the model"""
#     model.eval()
#     out = model({'enterprise': data['enterprise'].x},
#                 {('enterprise', 'investment', 'enterprise'): data['enterprise', 'investment', 'enterprise'].edge_index,
#                  ('enterprise', 'similarity', 'enterprise'): data['enterprise', 'similarity', 'enterprise'].edge_index})
#     probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
#     y_true = data['enterprise'].y[mask].cpu().numpy()
#
#     roc_auc = roc_auc_score(y_true, probs)
#     prec, rec, _ = precision_recall_curve(y_true, probs)
#     pr_auc = auc(rec, prec)
#
#     try:
#         ks_stat = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
#     except:
#         ks_stat = 0.0
#
#     best_f1 = 0
#     for threshold in np.arange(0.1, 0.9, 0.05):
#         preds = (probs >= threshold).astype(int)
#         f1 = f1_score(y_true, preds, zero_division=0)
#         if f1 > best_f1:
#             best_f1 = f1
#
#     return {'f1': best_f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}
#
#
# def random_guess_baseline(y_true):
#     """
#     Random-guessing baseline: Randomly predict 0 or 1
#
#     Args:
#         y_true: True labels
#
#     Returns:
#         dict: Including 4 metrics in the dictionary
#     """
#     num_samples = len(y_true)
#
#     # Random guessing 0 or 1
#     random_preds = np.random.randint(0, 2, num_samples)
#
#     # Compute F1
#     f1 = f1_score(y_true, random_preds, zero_division=0)
#
#     # To compute ROC-AUC and PR-AUC, Probability scores are required
#     # Random guessing: Randomly assign each sample [0,1] Score
#     random_probs = np.random.rand(num_samples)
#
#     try:
#         roc_auc = roc_auc_score(y_true, random_probs)
#     except:
#         roc_auc = 0.5
#
#     try:
#         prec, rec, _ = precision_recall_curve(y_true, random_probs)
#         pr_auc = auc(rec, prec)
#     except:
#         # PR-AUC random-baseline value equals the positive-class prevalence
#         pr_auc = (y_true == 1).sum() / len(y_true)
#
#     try:
#         ks_stat = ks_2samp(random_probs[y_true == 1], random_probs[y_true == 0]).statistic
#     except:
#         ks_stat = 0.0
#
#     return {'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}
#
#
# def train_and_evaluate(data_dir, dataset, seed, config, use_no_standard=False, permute_labels=False):
#     """
#     Train and evaluate a single seed
#
#     Args:
#         permute_labels: Whether to shuffle training labels
#     """
#     # Load data
#     data, train_mask, val_mask, test_mask = load_data(
#         data_dir, dataset, seed,
#         use_binning=config['binning'],
#         use_no_standard=use_no_standard
#     )
#
#     # If labels need shuffling
#     if permute_labels:
#         original_labels = data['enterprise'].y.clone()
#         train_indices = torch.where(train_mask)[0]
#         permuted_indices = torch.randperm(len(train_indices))
#         permuted_train_labels = original_labels[train_indices][permuted_indices]
#
#         # Create a new label tensor
#         permuted_labels = original_labels.clone()
#         permuted_labels[train_indices] = permuted_train_labels
#
#         # Train using shuffled labels
#         data['enterprise'].y = permuted_labels
#         print(f"  ✅ Training labels shuffled ({len(train_indices)} samples)")
#         print(f"     Original positive count: {(original_labels[train_indices] == 1).sum().item()}")
#         print(f"     Positive count after shuffling: {(permuted_train_labels == 1).sum().item()}")
#
#     # Train the model
#     model = train_single_model_with_smote(data, train_mask, val_mask, seed, config)
#
#     # Use original labels for evaluation
#     if permute_labels:
#         data['enterprise'].y = original_labels
#
#     # Evaluate on the test set
#     metrics = evaluate(model, data, test_mask)
#
#     return metrics
#
#
# # ==============================================================================
# # Save results to CSV
# # ==============================================================================
#
# def save_full_ablation_results(dataset, seed, exp1, exp2, exp3, exp4, exp5,
#                                pos_ratio, output_dir='./permutation_results'):
#     """
#     Save the complete 5 experiment groups' results
#
#     Args:
#         exp1: With industry standards+Original labels
#         exp2: With industry standards+Shuffled labels
#         exp3: Without industry standards+Original labels
#         exp4: Without industry standards+Shuffled labels
#         exp5: Random guessing
#     """
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Build DataFrame
#     results_df = pd.DataFrame({
#         'Experiment': [
#             'WithStd_Original',
#             'WithStd_Permuted',
#             'NoStd_Original',
#             'NoStd_Permuted',
#             'Random_Guess'
#         ],
#         'Features': ['WithStd', 'WithStd', 'NoStd', 'NoStd', 'Random'],
#         'Labels': ['Original', 'Permuted', 'Original', 'Permuted', 'Random'],
#         'PR-AUC': [
#             exp1['pr_auc'],
#             exp2['pr_auc'],
#             exp3['pr_auc'],
#             exp4['pr_auc'],
#             exp5['pr_auc']
#         ],
#         'F1': [
#             exp1['f1'],
#             exp2['f1'],
#             exp3['f1'],
#             exp4['f1'],
#             exp5['f1']
#         ],
#         'ROC-AUC': [
#             exp1['roc_auc'],
#             exp2['roc_auc'],
#             exp3['roc_auc'],
#             exp4['roc_auc'],
#             exp5['roc_auc']
#         ],
#         'KS': [
#             exp1['ks'],
#             exp2['ks'],
#             exp3['ks'],
#             exp4['ks'],
#             exp5['ks']
#         ]
#     })
#
#     # Add metadata
#     results_df['Dataset'] = dataset
#     results_df['Seed'] = seed
#     results_df['Pos_Ratio'] = pos_ratio
#     results_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#     # Save files
#     filename = f"{dataset}_Label_Test.csv"
#     filepath = os.path.join(output_dir, filename)
#     results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
#
#     print(f"  💾 Complete experiment results saved: {filepath}")
#
#     return filepath
#
#
# # ==============================================================================
# # Main function for complete ablation experiments
# # ==============================================================================
#
# def run_full_ablation_experiment(data_dir, datasets_to_test=None,
#                                  output_dir='./permutation_results'):
#     """
#     Run the complete 5 ablation experiment groups
#     """
#     print("=" * 120)
#     print("🔬 Label-permutation test - Complete ablation experiment (5 comparison groups)")
#     print("=" * 120)
#     print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"💾 Output directory: {output_dir}")
#     print("=" * 120)
#
#     if datasets_to_test is None:
#         datasets_to_test = list(VALID_SEEDS.keys())
#
#     all_results = {}
#     saved_files = []
#
#     for dataset in datasets_to_test:
#         if dataset not in VALID_SEEDS:
#             print(f"\n❌ {dataset} No validation seeds, Skip")
#             continue
#
#         seeds = VALID_SEEDS[dataset][:1]
#         config = DATASET_CONFIGS[dataset]
#
#         print(f"\n{'=' * 120}")
#         print(f"📊 Dataset: {dataset}")
#         print(f"⚙️  Parameters: Hidden={config['hidden']}, Ratio={config['ratio']}, Lambda={config['lambda_rank']}")
#         print("=" * 120)
#
#         for seed in seeds:
#             # Compute positive-class prevalence
#             data, train_mask, val_mask, test_mask = load_data(
#                 data_dir, dataset, seed, use_binning=config['binning'], use_no_standard=False
#             )
#             y_test = data['enterprise'].y[test_mask].cpu().numpy()
#             pos_ratio = (y_test == 1).sum() / len(y_test)
#             print(f"  📊 Test-set positive-class prevalence: {pos_ratio:.4f}")
#
#             # Experiment 1: With industry standards+Original labels
#             print(f"\n{'─' * 120}")
#             print(f"📈 [1/5] With industry standards + Original labels")
#             print(f"{'─' * 120}")
#             exp1 = train_and_evaluate(data_dir, dataset, seed, config,
#                                      use_no_standard=False, permute_labels=False)
#             print(f"  Results: PR={exp1['pr_auc']:.4f}, F1={exp1['f1']:.4f}, "
#                   f"ROC={exp1['roc_auc']:.4f}, KS={exp1['ks']:.4f}")
#
#             # Experiment 2: With industry standards+Shuffled labels
#             print(f"\n{'─' * 120}")
#             print(f"🔀 [2/5] With industry standards + Shuffled labels")
#             print(f"{'─' * 120}")
#             exp2 = train_and_evaluate(data_dir, dataset, seed, config,
#                                      use_no_standard=False, permute_labels=True)
#             print(f"  Results: PR={exp2['pr_auc']:.4f}, F1={exp2['f1']:.4f}, "
#                   f"ROC={exp2['roc_auc']:.4f}, KS={exp2['ks']:.4f}")
#
#             # Experiment 3: Without industry standards+Original labels
#             print(f"\n{'─' * 120}")
#             print(f"📈 [3/5] Without industry standards + Original labels")
#             print(f"{'─' * 120}")
#             exp3 = train_and_evaluate(data_dir, dataset, seed, config,
#                                      use_no_standard=True, permute_labels=False)
#             print(f"  Results: PR={exp3['pr_auc']:.4f}, F1={exp3['f1']:.4f}, "
#                   f"ROC={exp3['roc_auc']:.4f}, KS={exp3['ks']:.4f}")
#
#             # Experiment 4: Without industry standards+Shuffled labels
#             print(f"\n{'─' * 120}")
#             print(f"🔀 [4/5] Without industry standards + Shuffled labels")
#             print(f"{'─' * 120}")
#             exp4 = train_and_evaluate(data_dir, dataset, seed, config,
#                                      use_no_standard=True, permute_labels=True)
#             print(f"  Results: PR={exp4['pr_auc']:.4f}, F1={exp4['f1']:.4f}, "
#                   f"ROC={exp4['roc_auc']:.4f}, KS={exp4['ks']:.4f}")
#
#             # Experiment 5: Random guessing
#             print(f"\n{'─' * 120}")
#             print(f"🎲 [5/5] Random-guessing baseline")
#             print(f"{'─' * 120}")
#             exp5 = random_guess_baseline(y_test)
#             print(f"  Results: PR={exp5['pr_auc']:.4f}, F1={exp5['f1']:.4f}, "
#                   f"ROC={exp5['roc_auc']:.4f}, KS={exp5['ks']:.4f}")
#
#             # Print the comparison table
#             print(f"\n{'=' * 120}")
#             print(f"📊 Complete experiment comparison (Seed {seed})")
#             print(f"{'=' * 120}")
#             print(f"{'Experiment':<25} {'PR-AUC':<12} {'F1':<12} {'ROC-AUC':<12} {'KS':<12}")
#             print(f"{'─' * 120}")
#             print(f"{'1. With industry standards+Original labels':<25} {exp1['pr_auc']:<12.4f} {exp1['f1']:<12.4f} "
#                   f"{exp1['roc_auc']:<12.4f} {exp1['ks']:<12.4f}")
#             print(f"{'2. With industry standards+Shuffled labels':<25} {exp2['pr_auc']:<12.4f} {exp2['f1']:<12.4f} "
#                   f"{exp2['roc_auc']:<12.4f} {exp2['ks']:<12.4f}")
#             print(f"{'3. Without industry standards+Original labels':<25} {exp3['pr_auc']:<12.4f} {exp3['f1']:<12.4f} "
#                   f"{exp3['roc_auc']:<12.4f} {exp3['ks']:<12.4f}")
#             print(f"{'4. Without industry standards+Shuffled labels':<25} {exp4['pr_auc']:<12.4f} {exp4['f1']:<12.4f} "
#                   f"{exp4['roc_auc']:<12.4f} {exp4['ks']:<12.4f}")
#             print(f"{'5. Random guessing':<25} {exp5['pr_auc']:<12.4f} {exp5['f1']:<12.4f} "
#                   f"{exp5['roc_auc']:<12.4f} {exp5['ks']:<12.4f}")
#             print(f"{'=' * 120}")
#
#             # Compute performance drops
#             with_std_drop = ((exp1['pr_auc'] - exp2['pr_auc']) / exp1['pr_auc']) * 100
#             no_std_drop = ((exp3['pr_auc'] - exp4['pr_auc']) / exp3['pr_auc']) * 100
#
#             print(f"\n📉 Performance-drop analysis:")
#             print(f"  - With industry standards (1→2): PR-AUC drop {with_std_drop:.1f}%")
#             print(f"  - Without industry standards (3→4): PR-AUC drop {no_std_drop:.1f}%")
#
#             # Determine
#             if with_std_drop > 10 and no_std_drop < 10:
#                 print(f"\n✅ Conclusion: Industry standards effectively prevent data leakage")
#                 print(f"   - With industry standards: Significant performance drop after shuffling (↓{with_std_drop:.1f}%)")
#                 print(f"   - Without industry standards: No clear performance drop after shuffling (↓{no_std_drop:.1f}%)")
#             elif with_std_drop > 10 and no_std_drop > 10:
#                 print(f"\n⚠️  Conclusion: Neither method exhibits leakage")
#                 print(f"   - With industry standards: ↓{with_std_drop:.1f}%")
#                 print(f"   - Without industry standards: ↓{no_std_drop:.1f}%")
#             else:
#                 print(f"\n❌ Conclusion: Further checks are needed")
#
#             # Save results
#             filepath = save_full_ablation_results(
#                 dataset, seed, exp1, exp2, exp3, exp4, exp5, pos_ratio, output_dir
#             )
#             saved_files.append(filepath)
#
#             all_results[f"{dataset}_seed{seed}"] = {
#                 'exp1_with_std_original': exp1,
#                 'exp2_with_std_permuted': exp2,
#                 'exp3_no_std_original': exp3,
#                 'exp4_no_std_permuted': exp4,
#                 'exp5_random_guess': exp5,
#                 'pos_ratio': pos_ratio,
#                 'csv_file': filepath
#             }
#
#     print("\n" + "=" * 120)
#     print(f"📊 All experiment results saved")
#     print("=" * 120)
#     for f in saved_files:
#         print(f"  📄 {f}")
#     print("=" * 120)
#     print(f"✅ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 120)
#
#     return all_results
#
#
# # ==============================================================================
# # Main function
# # ==============================================================================
#
# def main():
#     data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
#
#     # Run the complete ablation experiment
#     datasets_to_test = ['2022_SH']  # Select 2 representative datasets
#     results = run_full_ablation_experiment(
#         data_dir, datasets_to_test,
#         output_dir=data_dir + 'analyze/Label_Test/'
#     )
#
#     return results
#
#
# if __name__ == '__main__':
#     main()


"""
================================================================================
GraphSMOTE Label-permutation test - Complete ablation version (Fix validation-label handling)
================================================================================
[Description]
Complete ablation experiment, Including 5 comparison groups: 
1. With industry standards + Original labels
2. With industry standards + Shuffled labels (Training set+ and validation labels are both shuffled)
3. Without industry standards + Original labels
4. Without industry standards + Shuffled labels (Training set+ and validation labels are both shuffled)
5. Random-guessing baseline

[Key fix]
- ✅ Validation labels are also shuffled (Prevent information leakage)
- ✅ Fix VALID_SEEDS duplicate-key bug
- ✅ Ensure the output directory exists
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
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# 📌 Configuration: Dataset parameters and seeds
# ==============================================================================

DATASET_CONFIGS = {
    "2023_SH": {'ratio': 2.9, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
    "2023_SZ": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 0.9, 'binning': False},
    "2022_SH": {'ratio': 1.7, 'hidden': 64, 'lambda_rank': 3.1, 'binning': False},
    "2022_SZ": {'ratio': 2.5, 'hidden': 64, 'lambda_rank': 3.0, 'binning': False},
    "2022_USA": {'ratio': 0.5, 'hidden': 64, 'lambda_rank': 2.0, 'binning': True},
    "2022_BSS": {'ratio': 0.913, 'hidden': 64, 'lambda_rank': 0.4, 'binning': False},
    "2023_BSS": {'ratio': 2.0, 'hidden': 64, 'lambda_rank': 1.5, 'binning': False}
}

# ✅ Fix: 2022_BSS Duplicate-key issue
VALID_SEEDS = {
    "2022_SH": [2],
    "2022_SZ": [72],
    "2023_SH": [42],
    "2023_SZ": [15],
    "2022_BSS": [109],
    "2023_BSS": [25],  # ✅ Fix: Change to 2023_BSS
    "2022_USA": [82]
}


def set_seed(seed):
    """Set the random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Model definitions
# ==============================================================================

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
    """Improved HGT, Return intermediate embedding For GraphSMOTE"""

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
        if return_embedding:
            return embeddings
        logits = self.classifier(embeddings)
        return logits / self.temperature


class PROptimizedLoss(nn.Module):
    """PR-AUC-optimized loss function"""

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
# Data loading (Support features with industry standards and / without industry standards)
# ==============================================================================

def load_data(data_dir, dataset_name, seed, use_binning=False, use_no_standard=False):
    """
    Load data

    Args:
        use_no_standard: True=Use features without industry standards, False=Use features with industry standards
    """
    # Select graph paths based on the parameter
    if use_no_standard:
        graph_path = '/graph_fusion_easy/'
        print(f"  → Use features without industry standards")
    else:
        graph_path = '/graph_fusion/'
        print(f"  → Use features with industry standards")

    edge_index = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_index.pt', weights_only=False)
    edge_type = torch.load(data_dir + dataset_name + graph_path + 'fused_edge_type.pt', weights_only=False)
    node_to_idx = torch.load(data_dir + dataset_name + graph_path + 'node_to_idx.pt', weights_only=False)

    # 🔑 Key change: Based on use_no_standard Select different feature files
    if use_no_standard:
        # features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features_easy.csv')     # _no_rule
        features_df = pd.read_csv(data_dir + dataset_name + '/data_process/llm_enhanced_features_easy.csv')  # _no_rule
    else:
        # features_df = pd.read_csv(data_dir + dataset_name + '/data_process/fused_features.csv')
        features_df = pd.read_csv(data_dir + dataset_name + '/data_process/llm_enhanced_features_v6_industry.csv')

    # features_df = pd.read_csv(data_dir + dataset_name + '/csv_data/reduced_features.csv')

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
            if len(val.shape) > 1:
                val = val[0]
            raw_features[i] = val
        labels.append(label_map.get(nid, 0))
    raw_features = np.nan_to_num(raw_features)

    # Feature processing
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
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return hetero_data, train_mask, val_mask, test_mask


# ==============================================================================
# Training function
# ==============================================================================

def train_single_model_with_smote(data, train_mask, val_mask, model_seed, config):
    """Train a single model"""
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

    # Stage 1: Pretraining
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

    # Stage 2: Generate GraphSMOTE Synthetic samples
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

    # Stage 3: Retrain
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

    best_val_pr = 0
    patience_counter = 0
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
            best_val_pr = val_pr_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= 50:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate the model"""
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
        if f1 > best_f1:
            best_f1 = f1

    return {'f1': best_f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}


def random_guess_baseline(y_true):
    """
    Random-guessing baseline: Randomly predict 0 or 1

    Args:
        y_true: True labels

    Returns:
        dict: Including 4 metrics in the dictionary
    """
    num_samples = len(y_true)

    # Random guessing 0 or 1
    random_preds = np.random.randint(0, 2, num_samples)

    # Compute F1
    f1 = f1_score(y_true, random_preds, zero_division=0)

    # To compute ROC-AUC and PR-AUC, Probability scores are required
    # Random guessing: Randomly assign each sample [0,1] Score
    random_probs = np.random.rand(num_samples)

    try:
        roc_auc = roc_auc_score(y_true, random_probs)
    except:
        roc_auc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(y_true, random_probs)
        pr_auc = auc(rec, prec)
    except:
        # PR-AUC random-baseline value equals the positive-class prevalence
        pr_auc = (y_true == 1).sum() / len(y_true)

    try:
        ks_stat = ks_2samp(random_probs[y_true == 1], random_probs[y_true == 0]).statistic
    except:
        ks_stat = 0.0

    return {'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks': ks_stat}


def train_and_evaluate(data_dir, dataset, seed, config, use_no_standard=False, permute_labels=False):
    """
    Train and evaluate a single seed

    Args:
        permute_labels: Whether to shuffle training and validation labels
    """
    # Load data
    data, train_mask, val_mask, test_mask = load_data(
        data_dir, dataset, seed,
        use_binning=config['binning'],
        use_no_standard=use_no_standard
    )

    # 🔑 Key fix: If labels need shuffling, Shuffle both training and validation labels
    if permute_labels:
        original_labels = data['enterprise'].y.clone()

        # Shuffle training labels
        train_indices = torch.where(train_mask)[0]
        permuted_train_indices = torch.randperm(len(train_indices))
        permuted_train_labels = original_labels[train_indices][permuted_train_indices]

        # ✅ Added: Shuffle validation labels
        val_indices = torch.where(val_mask)[0]
        permuted_val_indices = torch.randperm(len(val_indices))
        permuted_val_labels = original_labels[val_indices][permuted_val_indices]

        # Create a new label tensor
        permuted_labels = original_labels.clone()
        permuted_labels[train_indices] = permuted_train_labels
        permuted_labels[val_indices] = permuted_val_labels  # ✅ Also shuffle validation labels

        # Train using shuffled labels
        data['enterprise'].y = permuted_labels
        print(f"  ✅ Training labels shuffled ({len(train_indices)} samples)")
        print(f"     Original positive count: {(original_labels[train_indices] == 1).sum().item()}")
        print(f"     Positive count after shuffling: {(permuted_train_labels == 1).sum().item()}")
        print(f"  ✅ Validation labels shuffled ({len(val_indices)} samples)")
        print(f"     Original positive count: {(original_labels[val_indices] == 1).sum().item()}")
        print(f"     Positive count after shuffling: {(permuted_val_labels == 1).sum().item()}")

    # Train the model
    model = train_single_model_with_smote(data, train_mask, val_mask, seed, config)

    # Use original labels for evaluation
    if permute_labels:
        data['enterprise'].y = original_labels

    # Evaluate on the test set
    metrics = evaluate(model, data, test_mask)

    return metrics


# ==============================================================================
# Save results to CSV
# ==============================================================================

def save_full_ablation_results(dataset, seed, exp1, exp2, exp3, exp4, exp5,
                               pos_ratio, output_dir='./permutation_results'):
    """
    Save the complete 5 experiment groups' results

    Args:
        exp1: With industry standards+Original labels
        exp2: With industry standards+Shuffled labels
        exp3: Without industry standards+Original labels
        exp4: Without industry standards+Shuffled labels
        exp5: Random guessing
    """
    # ✅ Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Output directory created/Confirm: {output_dir}")
    except Exception as e:
        print(f"  ❌ Directory creation failed: {e}")
        return None

    # Build DataFrame
    results_df = pd.DataFrame({
        'Experiment': [
            'WithStd_Original',
            'WithStd_Permuted',
            'NoStd_Original',
            'NoStd_Permuted',
            'Random_Guess'
        ],
        'Features': ['WithStd', 'WithStd', 'NoStd', 'NoStd', 'Random'],
        'Labels': ['Original', 'Permuted', 'Permuted', 'Permuted', 'Random'],
        'PR-AUC': [
            exp1['pr_auc'],
            exp2['pr_auc'],
            exp3['pr_auc'],
            exp4['pr_auc'],
            exp5['pr_auc']
        ],
        'F1': [
            exp1['f1'],
            exp2['f1'],
            exp3['f1'],
            exp4['f1'],
            exp5['f1']
        ],
        'ROC-AUC': [
            exp1['roc_auc'],
            exp2['roc_auc'],
            exp3['roc_auc'],
            exp4['roc_auc'],
            exp5['roc_auc']
        ],
        'KS': [
            exp1['ks'],
            exp2['ks'],
            exp3['ks'],
            exp4['ks'],
            exp5['ks']
        ]
    })

    # Add metadata
    results_df['Dataset'] = dataset
    results_df['Seed'] = seed
    results_df['Pos_Ratio'] = pos_ratio
    results_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save files
    filename = f"{dataset}_Label_Test.csv"
    filepath = os.path.join(output_dir, filename)

    try:
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"  💾 Complete experiment results saved: {filepath}")

        # ✅ Verify the file exists
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"  ✓ File existence confirmed, Size: {file_size} bytes")
        else:
            print(f"  ❌ Warning: File missing after saving!")

    except Exception as e:
        print(f"  ❌ Save CSV Failure: {e}")
        import traceback
        traceback.print_exc()
        return None

    return filepath


# ==============================================================================
# Main function for complete ablation experiments
# ==============================================================================

def run_full_ablation_experiment(data_dir, datasets_to_test=None,
                                 output_dir='./permutation_results'):
    """
    Run the complete 5 ablation experiment groups
    """
    print("=" * 120)
    print("🔬 Label-permutation test - Complete ablation experiment (5 comparison groups)")
    print("=" * 120)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Output directory: {output_dir}")
    print("=" * 120)

    if datasets_to_test is None:
        datasets_to_test = list(VALID_SEEDS.keys())

    all_results = {}
    saved_files = []

    for dataset in datasets_to_test:
        if dataset not in VALID_SEEDS:
            print(f"\n❌ {dataset} No validation seeds, Skip")
            continue

        seeds = VALID_SEEDS[dataset][:1]
        config = DATASET_CONFIGS[dataset]

        print(f"\n{'=' * 120}")
        print(f"📊 Dataset: {dataset}")
        print(f"⚙️  Parameters: Hidden={config['hidden']}, Ratio={config['ratio']}, Lambda={config['lambda_rank']}")
        print("=" * 120)

        for seed in seeds:
            # Compute positive-class prevalence
            data, train_mask, val_mask, test_mask = load_data(
                data_dir, dataset, seed, use_binning=config['binning'], use_no_standard=False
            )
            y_test = data['enterprise'].y[test_mask].cpu().numpy()
            pos_ratio = (y_test == 1).sum() / len(y_test)
            print(f"  📊 Test-set positive-class prevalence: {pos_ratio:.4f}")

            # Experiment 1: With industry standards+Original labels
            print(f"\n{'─' * 120}")
            print(f"📈 [1/5] With industry standards + Original labels")
            print(f"{'─' * 120}")
            exp1 = train_and_evaluate(data_dir, dataset, seed, config,
                                      use_no_standard=False, permute_labels=False)
            print(f"  Results: PR={exp1['pr_auc']:.4f}, F1={exp1['f1']:.4f}, "
                  f"ROC={exp1['roc_auc']:.4f}, KS={exp1['ks']:.4f}")

            # Experiment 2: With industry standards+Shuffled labels (Training set+Validation set)
            print(f"\n{'─' * 120}")
            print(f"🔀 [2/5] With industry standards + Shuffled labels (Training set+Validation set)")
            print(f"{'─' * 120}")
            exp2 = train_and_evaluate(data_dir, dataset, seed, config,
                                      use_no_standard=False, permute_labels=True)
            print(f"  Results: PR={exp2['pr_auc']:.4f}, F1={exp2['f1']:.4f}, "
                  f"ROC={exp2['roc_auc']:.4f}, KS={exp2['ks']:.4f}")

            # Experiment 3: Without industry standards+Original labels
            print(f"\n{'─' * 120}")
            print(f"📈 [3/5] Without industry standards + Original labels")
            print(f"{'─' * 120}")
            exp3 = train_and_evaluate(data_dir, dataset, seed, config,
                                      use_no_standard=True, permute_labels=False)
            print(f"  Results: PR={exp3['pr_auc']:.4f}, F1={exp3['f1']:.4f}, "
                  f"ROC={exp3['roc_auc']:.4f}, KS={exp3['ks']:.4f}")

            # Experiment 4: Without industry standards+Shuffled labels (Training set+Validation set)
            print(f"\n{'─' * 120}")
            print(f"🔀 [4/5] Without industry standards + Shuffled labels (Training set+Validation set)")
            print(f"{'─' * 120}")
            exp4 = train_and_evaluate(data_dir, dataset, seed, config,
                                      use_no_standard=True, permute_labels=True)
            print(f"  Results: PR={exp4['pr_auc']:.4f}, F1={exp4['f1']:.4f}, "
                  f"ROC={exp4['roc_auc']:.4f}, KS={exp4['ks']:.4f}")

            # Experiment 5: Random guessing
            print(f"\n{'─' * 120}")
            print(f"🎲 [5/5] Random-guessing baseline")
            print(f"{'─' * 120}")
            exp5 = random_guess_baseline(y_test)
            print(f"  Results: PR={exp5['pr_auc']:.4f}, F1={exp5['f1']:.4f}, "
                  f"ROC={exp5['roc_auc']:.4f}, KS={exp5['ks']:.4f}")

            # Print the comparison table
            print(f"\n{'=' * 120}")
            print(f"📊 Complete experiment comparison (Seed {seed})")
            print(f"{'=' * 120}")
            print(f"{'Experiment':<30} {'PR-AUC':<12} {'F1':<12} {'ROC-AUC':<12} {'KS':<12}")
            print(f"{'─' * 120}")
            print(f"{'1. With industry standards+Original labels':<30} {exp1['pr_auc']:<12.4f} {exp1['f1']:<12.4f} "
                  f"{exp1['roc_auc']:<12.4f} {exp1['ks']:<12.4f}")
            print(f"{'2. With industry standards+Shuffled labels':<30} {exp2['pr_auc']:<12.4f} {exp2['f1']:<12.4f} "
                  f"{exp2['roc_auc']:<12.4f} {exp2['ks']:<12.4f}")
            print(f"{'3. Without industry standards+Original labels':<30} {exp3['pr_auc']:<12.4f} {exp3['f1']:<12.4f} "
                  f"{exp3['roc_auc']:<12.4f} {exp3['ks']:<12.4f}")
            print(f"{'4. Without industry standards+Shuffled labels':<30} {exp4['pr_auc']:<12.4f} {exp4['f1']:<12.4f} "
                  f"{exp4['roc_auc']:<12.4f} {exp4['ks']:<12.4f}")
            print(f"{'5. Random guessing':<30} {exp5['pr_auc']:<12.4f} {exp5['f1']:<12.4f} "
                  f"{exp5['roc_auc']:<12.4f} {exp5['ks']:<12.4f}")
            print(f"{'=' * 120}")

            # Compute performance drops
            with_std_drop = ((exp1['pr_auc'] - exp2['pr_auc']) / exp1['pr_auc']) * 100
            no_std_drop = ((exp3['pr_auc'] - exp4['pr_auc']) / exp3['pr_auc']) * 100

            with_std_f1_drop = ((exp1['f1'] - exp2['f1']) / exp1['f1']) * 100
            no_std_f1_drop = ((exp3['f1'] - exp4['f1']) / exp3['f1']) * 100

            print(f"\n📉 Performance-drop analysis:")
            print(f"  With industry standards:")
            print(f"    - PR-AUC drop: {with_std_drop:.1f}%")
            print(f"    - F1 drop: {with_std_f1_drop:.1f}%")
            print(f"  Without industry standards:")
            print(f"    - PR-AUC drop: {no_std_drop:.1f}%")
            print(f"    - F1 drop: {no_std_f1_drop:.1f}%")

            # Determine
            if with_std_drop > 10 and no_std_drop < 10:
                print(f"\n✅ Conclusion: Industry standards effectively prevent data leakage")
                print(f"   - With industry standards: Significant performance drop after shuffling (PR↓{with_std_drop:.1f}%, F1↓{with_std_f1_drop:.1f}%)")
                print(f"   - Without industry standards: No clear performance drop after shuffling (PR↓{no_std_drop:.1f}%, F1↓{no_std_f1_drop:.1f}%)")
            elif with_std_drop > 10 and no_std_drop > 10:
                print(f"\n⚠️  Conclusion: Neither method exhibits leakage")
                print(f"   - With industry standards: PR↓{with_std_drop:.1f}%, F1↓{with_std_f1_drop:.1f}%")
                print(f"   - Without industry standards: PR↓{no_std_drop:.1f}%, F1↓{no_std_f1_drop:.1f}%")
            else:
                print(f"\n❌ Conclusion: Further checks are needed")

            # Save results
            filepath = save_full_ablation_results(
                dataset, seed, exp1, exp2, exp3, exp4, exp5, pos_ratio, output_dir
            )
            if filepath:
                saved_files.append(filepath)

            all_results[f"{dataset}_seed{seed}"] = {
                'exp1_with_std_original': exp1,
                'exp2_with_std_permuted': exp2,
                'exp3_no_std_original': exp3,
                'exp4_no_std_permuted': exp4,
                'exp5_random_guess': exp5,
                'pos_ratio': pos_ratio,
                'csv_file': filepath
            }

    print("\n" + "=" * 120)
    print(f"📊 All experiment results saved")
    print("=" * 120)
    for f in saved_files:
        print(f"  📄 {f}")
    print("=" * 120)
    print(f"✅ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    return all_results


# ==============================================================================
# Main function
# ==============================================================================

def main():

    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'

    # Run the complete ablation experiment
    datasets_to_test = ['2022_SZ', '2022_SH']  # Select 2 representative datasets
    results = run_full_ablation_experiment(
        data_dir, datasets_to_test,
        output_dir='/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/analyze/Label_Test/'
    )

    return results


if __name__ == '__main__':
    main()