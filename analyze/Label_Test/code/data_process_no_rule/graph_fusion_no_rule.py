# src/graph_fusion.py

import pandas as pd
import numpy as np
import torch
import networkx as nx
from collections import defaultdict

# Keep existing dataset column names compatible with stored CSV files.
_ENTERPRISE_A = '\u4f01\u4e1aA'


graph_select = '/sampled_clustering_no_rule'
DATA_DIR = '/graph_fusion_no_rule/'

def load_graphs(data_dir):
    """Load the pruned graph and feature graph"""

    print("=" * 60)
    print("Load graph data")
    print("=" * 60)

    # 1. Investment edges
    pruned_edges = pd.read_csv(data_dir + '/data_process/enterprise_relations_dib_pruned_GJ_e4_no_rule.csv')
    print(f"\n✓ Investment edges (no_rule): {len(pruned_edges)} entries")


    # 2. Feature-similarity edges
    feature_edges = pd.read_csv(

        data_dir + graph_select + '/sampled_edges.csv'
    )
    print(f"✓ Feature-similarity edges: {len(feature_edges)} entries")

    # 3. Node data
    all_features = pd.read_csv(data_dir + '/data_process/fused_features.csv')
    sampled_nodes = pd.read_csv(
        data_dir + graph_select + '/sampled_nodes.csv'
    )

    print(f"✓ All nodes: {len(all_features)} items")
    print(f"✓ Sampled nodes: {len(sampled_nodes)} items")

    # 4. Label
    labels = pd.read_csv(data_dir + '/csv_data/enterprise_labels.csv')
    print(f"✓ Label count: {len(labels)} items")

    return pruned_edges, feature_edges, all_features, sampled_nodes, labels


# def fuse_graphs(pruned_edges, feature_edges, all_features, sampled_nodes):
#     """Fuse the pruned graph and feature graph"""
#
#     print("\n" + "=" * 60)
#     print("Graph fusion")
#     print("=" * 60)
#
#     # 1. Build node ID to index mapping (Based on all nodes)
#     # ✅ Fix: Normalize to integer strings, Avoid  '0.0' vs '0'  mismatches
#     if 'enterprise_id' in all_features.columns:
#         id_col = 'enterprise_id'
#     elif ' enterprise ID' in all_features.columns:
#         id_col = ' enterprise ID'
#     else:
#         raise ValueError(f"Node not found ID columns!Available columns: {list(all_features.columns)}")
#
#     # ✅ Key fix: First convert to int then to str, Remove decimal points
#     all_node_ids = sorted([str(int(float(nid))) for nid in all_features[id_col].unique()])
#     node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
#
#     print(f"\n Build the node mapping: {len(node_to_idx)} nodes")
#     print(f"  Use column name: {id_col}")
#     print(f"  Node ID Type: {type(list(node_to_idx.keys())[0])}")
#     print(f"  Example ID: {list(node_to_idx.keys())[:5]}")
#
#     # 2. Process pruned investment edges
#     print("\n Process pruned investment edges...")
#     investment_edges = []
#
#     for _, row in pruned_edges.iterrows():
#         # ✅ Normalize to integer strings
#         src_id = str(int(float(row[' enterprise A'])))
#         dst_id = str(int(float(row[' enterprise B'])))
#
#         if src_id in node_to_idx and dst_id in node_to_idx:
#             src_idx = node_to_idx[src_id]
#             dst_idx = node_to_idx[dst_id]
#             investment_edges.append([src_idx, dst_idx])
#
#     # ✅ Handle empty edges
#     if len(investment_edges) > 0:
#         investment_edge_index = torch.tensor(
#             investment_edges, dtype=torch.long
#         ).t().contiguous()
#         print(f"  ✓ Investment edges after conversion: {investment_edge_index.shape[1]} entries")
#     else:
#         investment_edge_index = torch.zeros((2, 0), dtype=torch.long)
#         print(f"  ⚠ Investment edges after conversion: 0 entries (None of the edge nodes are in the full graph)")
#         print(f"  ⚠ Possible cause: Node ID Column names or formats do not match")
#         print(f"     - Investment-edge enterprise ID Example: {pruned_edges[' enterprise A'].iloc[:3].tolist()}")
#         print(f"     - Full graph nodes ID Example: {list(node_to_idx.keys())[:3]}")
#
#     # 3. Process feature-similarity edges (Convert to the full-graph index space)
#     print("\n Process feature-similarity edges...")
#
#     # ✅ Add debugging information
#     print(f"  Example feature-edge data:")
#     if len(feature_edges) > 0:
#         print(f"     enterprise A Type: {type(feature_edges.iloc[0][' enterprise A'])}")
#         print(f"     enterprise A Example: {feature_edges.iloc[0][' enterprise A']}")
#         print(f"  Full graph nodes ID Type: {type(list(node_to_idx.keys())[0])}")
#         print(f"  Full graph nodes ID Example: {list(node_to_idx.keys())[:3]}")
#
#     similarity_edges = []
#     skipped = 0
#
#     for _, row in feature_edges.iterrows():
#         # ✅ Normalize to integer strings
#         try:
#             src_id = str(int(float(row[' enterprise A'])))
#             dst_id = str(int(float(row[' enterprise B'])))
#         except (ValueError, TypeError):
#             # If conversion fails, Try converting directly to strings
#             src_id = str(row[' enterprise A'])
#             dst_id = str(row[' enterprise B'])
#
#         # Convert to full-graph indices
#         if src_id in node_to_idx and dst_id in node_to_idx:
#             src_idx = node_to_idx[src_id]
#             dst_idx = node_to_idx[dst_id]
#             similarity_edges.append([src_idx, dst_idx])
#         else:
#             skipped += 1
#
#     # ✅ Handle empty edges
#     if len(similarity_edges) > 0:
#         similarity_edge_index = torch.tensor(
#             similarity_edges, dtype=torch.long
#         ).t().contiguous()
#         print(f"  ✓ Feature edges after conversion: {similarity_edge_index.shape[1]} entries")
#     else:
#         # Create an empty  [2, 0]  tensor
#         similarity_edge_index = torch.zeros((2, 0), dtype=torch.long)
#         print(f"  ⚠ Feature edges after conversion: 0 entries (None of the edge nodes are in the full graph)")
#
#     if skipped > 0:
#         print(f"  ⚠ Skip: {skipped} entries (Nodes are absent from the full graph)")
#
#     # 4. Fuse edges
#     print("\n Fuse edges...")
#
#     # ✅ Check whether any edges are available
#     has_investment = investment_edge_index.shape[1] > 0
#     has_similarity = similarity_edge_index.shape[1] > 0
#
#     if not has_investment and not has_similarity:
#         raise ValueError("❌ Error: Both investment and feature edges are empty!Check whether node ID values match.")
#
#     # ✅ Handle empty feature edges
#     if has_investment and has_similarity:
#         fused_edge_index = torch.cat(
#             [investment_edge_index, similarity_edge_index],
#             dim=1
#         )
#
#         # Create edge-type indicators
#         edge_type = torch.cat([
#             torch.zeros(investment_edge_index.shape[1], dtype=torch.long),  # 0=Investment edges
#             torch.ones(similarity_edge_index.shape[1], dtype=torch.long)  # 1=Feature edges
#         ])
#
#         print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
#         print(f"    - Investment edges: {investment_edge_index.shape[1]} entries (type=0)")
#         print(f"    - Feature edges: {similarity_edge_index.shape[1]} entries (type=1)")
#
#     elif has_investment:
#         # Only investment edges
#         fused_edge_index = investment_edge_index
#         edge_type = torch.zeros(investment_edge_index.shape[1], dtype=torch.long)
#
#         print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
#         print(f"    - Investment edges: {investment_edge_index.shape[1]} entries (type=0)")
#         print(f"    - Feature edges: 0 entries")
#         print(f"  ⚠ Warning: No feature-similarity edges, The fused graph contains investment edges only")
#
#     else:
#         # Only feature edges
#         fused_edge_index = similarity_edge_index
#         edge_type = torch.ones(similarity_edge_index.shape[1], dtype=torch.long)
#
#         print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
#         print(f"    - Investment edges: 0 entries")
#         print(f"    - Feature edges: {similarity_edge_index.shape[1]} entries (type=1)")
#         print(f"  ⚠ Warning: No investment edges, The fused graph contains feature edges only")
#
#     # 5. Deduplicate (Duplicate edges may exist)
#     print("\n Remove duplicate edges...")
#
#     # Convert to a set for deduplication
#     edge_set = set()
#     unique_edges = []
#     unique_types = []
#
#     for i in range(fused_edge_index.shape[1]):
#         src = fused_edge_index[0, i].item()
#         dst = fused_edge_index[1, i].item()
#         edge_tuple = (min(src, dst), max(src, dst))
#
#         if edge_tuple not in edge_set:
#             edge_set.add(edge_tuple)
#             unique_edges.append([src, dst])
#             unique_types.append(edge_type[i].item())
#
#     fused_edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
#     edge_type = torch.tensor(unique_types, dtype=torch.long)
#
#     print(f"  ✓ Edge count after deduplication: {fused_edge_index.shape[1]} entries")
#
#     return fused_edge_index, edge_type, node_to_idx


def fuse_graphs(pruned_edges, feature_edges, all_features, sampled_nodes):
    """Fuse the pruned graph and feature graph"""

    print("\n" + "=" * 60)
    print("Graph fusion")
    print("=" * 60)

    # 1. Build node ID to index mapping (Based on all nodes)
    # ✅ Fix: Normalize to integer strings, Avoid  '0.0' vs '0'  mismatches
    if 'enterprise_id' in all_features.columns:
        id_col = 'enterprise_id'
    elif '\u4f01\u4e1aID' in all_features.columns:
        id_col = '\u4f01\u4e1aID'
    else:
        raise ValueError(f"Node not found ID columns!Available columns: {list(all_features.columns)}")

    # ✅ Key fix: First convert to int then to str, Remove decimal points
    all_node_ids = sorted([str(int(float(nid))) for nid in all_features[id_col].unique()])
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    print(f"\n Build the node mapping: {len(node_to_idx)} nodes")
    print(f"  Use column name: {id_col}")
    print(f"  Node ID Type: {type(list(node_to_idx.keys())[0])}")
    print(f"  Example ID: {list(node_to_idx.keys())[:5]}")

    # 2. Process pruned investment edges
    print("\n Process pruned investment edges...")
    investment_edges = []

    for _, row in pruned_edges.iterrows():
        # ✅ Normalize to integer strings
        src_id = str(int(float(row['\u4f01\u4e1aA'])))
        dst_id = str(int(float(row['\u4f01\u4e1aB'])))

        if src_id in node_to_idx and dst_id in node_to_idx:
            src_idx = node_to_idx[src_id]
            dst_idx = node_to_idx[dst_id]
            investment_edges.append([src_idx, dst_idx])

    # ✅ Handle empty edges
    if len(investment_edges) > 0:
        investment_edge_index = torch.tensor(
            investment_edges, dtype=torch.long
        ).t().contiguous()
        print(f"  ✓ Investment edges after conversion: {investment_edge_index.shape[1]} entries")
    else:
        investment_edge_index = torch.zeros((2, 0), dtype=torch.long)
        print(f"  ⚠ Investment edges after conversion: 0 entries (None of the edge nodes are in the full graph)")
        print(f"  ⚠ Possible cause: Node ID Column names or formats do not match")
        print(f"     - Investment-edge enterprise ID Example: {pruned_edges[_ENTERPRISE_A].iloc[:3].tolist()}")
        print(f"     - Full graph nodes ID Example: {list(node_to_idx.keys())[:3]}")

    # 3. Process feature-similarity edges (Convert to the full-graph index space)
    print("\n Process feature-similarity edges...")

    # ✅ Add debugging information
    print(f"  Example feature-edge data:")
    if len(feature_edges) > 0:
        print(f"     enterprise A Type: {type(feature_edges.iloc[0][_ENTERPRISE_A])}")
        print(f"     enterprise A Example: {feature_edges.iloc[0][_ENTERPRISE_A]}")
        print(f"  Full graph nodes ID Type: {type(list(node_to_idx.keys())[0])}")
        print(f"  Full graph nodes ID Example: {list(node_to_idx.keys())[:3]}")

    similarity_edges = []
    skipped = 0

    for _, row in feature_edges.iterrows():
        # ✅ Normalize to integer strings
        try:
            src_id = str(int(float(row['\u4f01\u4e1aA'])))
            dst_id = str(int(float(row['\u4f01\u4e1aB'])))
        except (ValueError, TypeError):
            # If conversion fails, Try converting directly to strings
            src_id = str(row['\u4f01\u4e1aA'])
            dst_id = str(row['\u4f01\u4e1aB'])

        # Convert to full-graph indices
        if src_id in node_to_idx and dst_id in node_to_idx:
            src_idx = node_to_idx[src_id]
            dst_idx = node_to_idx[dst_id]
            similarity_edges.append([src_idx, dst_idx])
        else:
            skipped += 1

    # ✅ Handle empty edges
    if len(similarity_edges) > 0:
        similarity_edge_index = torch.tensor(
            similarity_edges, dtype=torch.long
        ).t().contiguous()
        print(f"  ✓ Feature edges after conversion: {similarity_edge_index.shape[1]} entries")
    else:
        # Create an empty  [2, 0]  tensor
        similarity_edge_index = torch.zeros((2, 0), dtype=torch.long)
        print(f"  ⚠ Feature edges after conversion: 0 entries (None of the edge nodes are in the full graph)")

    if skipped > 0:
        print(f"  ⚠ Skip: {skipped} entries (Nodes are absent from the full graph)")

    # 4. Fuse edges
    print("\n Fuse edges...")

    # ✅ Check whether any edges are available
    has_investment = investment_edge_index.shape[1] > 0
    has_similarity = similarity_edge_index.shape[1] > 0

    if not has_investment and not has_similarity:
        raise ValueError("❌ Error: Both investment and feature edges are empty!Check whether node ID values match.")

    # ✅ Handle empty feature edges
    if has_investment and has_similarity:
        fused_edge_index = torch.cat(
            [investment_edge_index, similarity_edge_index],
            dim=1
        )

        # Create edge-type indicators
        edge_type = torch.cat([
            torch.zeros(investment_edge_index.shape[1], dtype=torch.long),  # 0=Investment edges
            torch.ones(similarity_edge_index.shape[1], dtype=torch.long)  # 1=Feature edges
        ])

        print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
        print(f"    - Investment edges: {investment_edge_index.shape[1]} entries (type=0)")
        print(f"    - Feature edges: {similarity_edge_index.shape[1]} entries (type=1)")

    elif has_investment:
        # Only investment edges
        fused_edge_index = investment_edge_index
        edge_type = torch.zeros(investment_edge_index.shape[1], dtype=torch.long)

        print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
        print(f"    - Investment edges: {investment_edge_index.shape[1]} entries (type=0)")
        print(f"    - Feature edges: 0 entries")
        print(f"  ⚠ Warning: No feature-similarity edges, The fused graph contains investment edges only")

    else:
        # Only feature edges
        fused_edge_index = similarity_edge_index
        edge_type = torch.ones(similarity_edge_index.shape[1], dtype=torch.long)

        print(f"  ✓ Total edges after fusion: {fused_edge_index.shape[1]} entries")
        print(f"    - Investment edges: 0 entries")
        print(f"    - Feature edges: {similarity_edge_index.shape[1]} entries (type=1)")
        print(f"  ⚠ Warning: No investment edges, The fused graph contains feature edges only")

    # 5. Deduplicate (Duplicate edges may exist) - ✅ Fix: Keep directed edges
    print("\n Remove duplicate edges...")

    # Convert to a set for deduplication - Preserve direction
    edge_set = set()
    unique_edges = []
    unique_types = []

    for i in range(fused_edge_index.shape[1]):
        src = fused_edge_index[0, i].item()
        dst = fused_edge_index[1, i].item()
        edge_tuple = (src, dst)  # ✅ Preserve direction, No longer use  (min, max)

        if edge_tuple not in edge_set:
            edge_set.add(edge_tuple)
            unique_edges.append([src, dst])
            unique_types.append(edge_type[i].item())

    fused_edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(unique_types, dtype=torch.long)

    print(f"  ✓ Edge count after deduplication: {fused_edge_index.shape[1]} entries")

    return fused_edge_index, edge_type, node_to_idx



def analyze_fused_graph(fused_edge_index, edge_type, all_features, labels):
    """Analyze the fused graph"""

    print("\n" + "=" * 60)
    print("Fused graph analysis")
    print("=" * 60)

    # 1. Basic statistics
    num_nodes = len(all_features)
    num_edges = fused_edge_index.shape[1]
    num_investment = (edge_type == 0).sum().item()
    num_similarity = (edge_type == 1).sum().item()

    print(f"\n Basic statistics:")
    print(f"  Node count: {num_nodes}")
    print(f"  Total edges: {num_edges}")
    print(f"  Investment edges: {num_investment} ({num_investment / num_edges * 100:.1f}%)")
    print(f"  Feature edges: {num_similarity} ({num_similarity / num_edges * 100:.1f}%)")
    print(f"  Average degree: {num_edges * 2 / num_nodes:.1f}")

    # 2. Build NetworkX graph to analyze connectivity
    G = nx.Graph()
    edges = fused_edge_index.t().cpu().numpy()
    G.add_edges_from(edges)

    print(f"\n Connectivity:")
    if nx.is_connected(G):
        print(f"  Fully connected ✓")
    else:
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        print(f"  Not fully connected")
        print(f"  Connected components: {len(components)}")
        print(f"  Largest component: {len(largest)} nodes ({len(largest) / num_nodes * 100:.1f}%)")

    # 3. Degree distribution
    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    print(f"\n Degree distribution:")
    print(f"  Minimum degree: {min(degree_values)}")
    print(f"  Maximum degree: {max(degree_values)}")
    print(f"  Average degree: {np.mean(degree_values):.1f}")
    print(f"  Median degree: {np.median(degree_values):.1f}")

    # 4. Analyze degrees by edge type
    investment_edges = fused_edge_index[:, edge_type == 0]
    similarity_edges = fused_edge_index[:, edge_type == 1]

    G_inv = nx.Graph()
    G_inv.add_edges_from(investment_edges.t().cpu().numpy())

    G_sim = nx.Graph()
    G_sim.add_edges_from(similarity_edges.t().cpu().numpy())

    print(f"\n Investment-edge subgraph:")
    print(f"  Node count: {G_inv.number_of_nodes()}")
    print(f"  Average degree: {sum(dict(G_inv.degree()).values()) / G_inv.number_of_nodes():.1f}")

    print(f"\n Feature-edge subgraph:")
    print(f"  Node count: {G_sim.number_of_nodes()}")
    print(f"  Average degree: {sum(dict(G_sim.degree()).values()) / G_sim.number_of_nodes():.1f}")

    # 5. Label coverage
    label_dict = labels.set_index('\u4f01\u4e1aID')['\u6807\u7b7e'].to_dict()
    nodes_in_graph = set(G.nodes())

    risk_nodes_in_graph = sum(1 for nid in nodes_in_graph if label_dict.get(nid, 0) == 1)
    total_risk = sum(1 for label in label_dict.values() if label == 1)

    print(f"\n Label coverage:")
    print(f"  At-risk nodes in the graph: {risk_nodes_in_graph}")
    print(f"  Total at-risk nodes: {total_risk}")
    print(f"  Coverage: {risk_nodes_in_graph / total_risk * 100:.1f}%")


def save_fused_graph(fused_edge_index, edge_type, node_to_idx, data_dir):
    """Save the fused graph"""

    print("\n" + "=" * 60)
    print("Save the fused graph")
    print("=" * 60)

    # data_dir += ''
    data_dir += DATA_DIR

    # 1. Save edge indices
    edge_index_path = data_dir + 'fused_edge_index.pt'
    torch.save(fused_edge_index, edge_index_path)
    print(f"\n✓ Edge indices saved: {edge_index_path}")

    # 2. Save edge types
    edge_type_path = data_dir + 'fused_edge_type.pt'
    torch.save(edge_type, edge_type_path)
    print(f"✓ Edge types saved: {edge_type_path}")

    # 3. Save the node mapping
    node_mapping_path = data_dir + 'node_to_idx.pt'
    torch.save(node_to_idx, node_mapping_path)
    print(f"✓ Node mapping saved: {node_mapping_path}")

    # 4. Save as CSV (Readable format)
    idx_to_node = {i: nid for nid, i in node_to_idx.items()}

    edges_data = []
    for i in range(fused_edge_index.shape[1]):
        src_idx = fused_edge_index[0, i].item()
        dst_idx = fused_edge_index[1, i].item()
        etype = edge_type[i].item()

        edges_data.append({
            '\u4f01\u4e1aA': idx_to_node[src_idx],
            '\u4f01\u4e1aB': idx_to_node[dst_idx],
            '\u8fb9\u7c7b\u578b': '\u6295\u8d44' if etype == 0 else '\u7279\u5f81'
        })

    edges_csv_path = data_dir + 'fused_edges.csv'
    pd.DataFrame(edges_data).to_csv(edges_csv_path, index=False)
    print(f"✓ Fuse edges CSV Saved: {edges_csv_path}")

    print("\n" + "=" * 60)
    print("Graph fusion completed!")
    print("=" * 60)


def main():
    """Main function"""

    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    data_file = ['2022_SZ', '2022_SH']
    # for data in data_file:
    for data in data_file:
        print("Process data:", data)

        # 1. Load data
        pruned_edges, feature_edges, all_features, sampled_nodes, labels = load_graphs(data_dir + data)

        # 2. Fused graph
        fused_edge_index, edge_type, node_to_idx = fuse_graphs(
            pruned_edges, feature_edges, all_features, sampled_nodes
        )

        # 3. Analyze
        analyze_fused_graph(fused_edge_index, edge_type, all_features, labels)

        # 4. Save
        save_fused_graph(fused_edge_index, edge_type, node_to_idx, data_dir + data)

    return fused_edge_index, edge_type, node_to_idx


if __name__ == "__main__":

    # graph_select = '/sampled_clustering_no_rule'
    # DATA_DIR = '/graph_fusion_no_rule/'
    DAT_DIR_list = ['/graph_fusion_easy/', '/graph_fusion_my_rule/']
    graph_select_list = ['/sampled_clustering_easy/', '/sampled_clustering_my_rule/']

    for DATA_DIR, graph_select in zip(DAT_DIR_list, graph_select_list):

        fused_edge_index, edge_type, node_to_idx = main()