# clustering_with_new_edges_v2.py

"""
Cluster sampling + LLM stratified edge generation

Improvements: 
1. 5 LLM risk edges + 15 financial edges (Stratified generation)
2. Noisy-OR weight fusion (Increase the weights of duplicate edges)
3. Remain fully compatible with edge_pruning.py and graph_fusion.py

Output format: 
- sampled_nodes.csv: Node features (Compatible with the original format)
- sampled_edges.csv:  edges (Compatible with the original format:  enterprise A,  enterprise B, Weight, Reason)
- sampled_labels.csv: Label (Compatible with the original format)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
import json
from tqdm import tqdm
import time
import re

# Keep existing dataset column names compatible with stored CSV files.
_EDGE_WEIGHT = '\u6743\u91cd'
_ENTERPRISE_ID = '\u4f01\u4e1aID'


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def clustering_with_llm_edges(
        fused_path, labels_path,
        n_clusters,
        deepseek_api_key,
        output_dir):
    """Unified cluster sampling + LLM stratified edge generation (improved v2)"""

    print(f"=" * 80)
    print(f"Cluster sampling + LLM stratified edge generation v2")
    print(f"=" * 80)
    print(f"Total clusters: {n_clusters} clusters")
    print(f"Edge generation strategy: 5 LLM risk edges + 15 financial edges\n")

    # 1. Load data
    fused_df = pd.read_csv(fused_path)
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.rename(columns={'\u4f01\u4e1aID': '\u4f01\u4e1aID', '\u6807\u7b7e': 'label'})

    # Merge labels
    fused_df = pd.merge(fused_df, labels_df[['\u4f01\u4e1aID', 'label']], on='\u4f01\u4e1aID', how='left')

    print(f"✓ Node data: {len(fused_df)} enterprises")
    print(f"  At risk: {(fused_df['label'] == 1).sum()} enterprises")
    print(f"  Not at risk: {(fused_df['label'] == 0).sum()} enterprises")

    # 2. Prepare features
    exclude_cols = ['\u4f01\u4e1aID', '\u4f01\u4e1aID', 'label', 'Unnamed: 0', 'risk_type', 'risk_reason']
    all_feature_cols = [col for col in fused_df.columns
                        if col not in exclude_cols
                        and fused_df[col].dtype in ['float64', 'int64']]

    print(f"✓ Use fused features ({len(all_feature_cols)} columns)")

    # Distinguish LLM features from base features
    llm_features = [c for c in all_feature_cols if 'risk' in c.lower()]
    base_features = [c for c in all_feature_cols if c not in llm_features]

    print(f"  - LLM features: {len(llm_features)} items")
    print(f"  - Base features: {len(base_features)} items")

    # Extract feature data
    features_data = fused_df[all_feature_cols].values

    # Handle missing values
    if np.isnan(features_data).any():
        print(f"  ⚠ Missing values found, Fill with medians")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        features_data = imputer.fit_transform(features_data)

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_data)

    print(f"✓ Feature matrix shape: {features_scaled.shape}")

    # 3. Clustering
    print(f"\n--Start clustering...")
    n_clusters_actual = min(n_clusters, len(fused_df))

    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
    fused_df['cluster'] = kmeans.fit_predict(features_scaled)
    cluster_centers = kmeans.cluster_centers_

    print(f"✓ Grouped into: {n_clusters_actual} clusters")

    # 4. Select representative nodes from each cluster
    representatives = []
    repr_indices = []

    for cluster_id in range(n_clusters_actual):
        cluster_mask = fused_df['cluster'] == cluster_id
        cluster_data = fused_df[cluster_mask]
        cluster_features = features_scaled[cluster_mask.values]

        if len(cluster_data) == 0:
            continue

        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        closest_idx = np.argmin(distances)

        representative = cluster_data.iloc[closest_idx]
        representatives.append(representative['\u4f01\u4e1aID'])
        repr_indices.append(cluster_data.index[closest_idx])

    print(f"✓ Selected: {len(representatives)} representative nodes")

    # 5. Extract representative-node data
    sampled_ids = set([str(eid) for eid in representatives])
    sampled_fused = fused_df.iloc[repr_indices].copy()
    sampled_labels = labels_df[labels_df['\u4f01\u4e1aID'].astype(str).isin(sampled_ids)].copy()

    print(f"\n✓ Sampling completed: {len(sampled_ids)} nodes")
    print(f"  - At risk: {(sampled_fused['label'] == 1).sum()} items")
    print(f"  - Not at risk: {(sampled_fused['label'] == 0).sum()} items")
    print(f"  - Sampling ratio: {len(sampled_ids) / len(fused_df) * 100:.1f}%")

    # 6. ✨ Use LLM stratified edge generation (5:15 strategy + Noisy-OR Fusion)
    print(f"\n--Use DeepSeek LLM stratified edge generation...")
    new_edges_df = generate_edges_stratified_with_llm(
        sampled_fused, deepseek_api_key, llm_features, base_features
    )

    # 7. Build the graph and check connectivity
    if len(new_edges_df) > 0:
        G = nx.from_pandas_edgelist(new_edges_df, '\u4f01\u4e1aA', '\u4f01\u4e1aB', edge_attr='\u6743\u91cd')
        n_components = nx.number_connected_components(G)
        print(f"✓ Connected components: {n_components}")

        if n_components > 1:
            components = list(nx.connected_components(G))
            sizes = sorted([len(c) for c in components], reverse=True)
            print(f"  ⚠ Graph has {n_components} connected components")
            print(f"  Component sizes: {sizes[:5]}...")
        else:
            print(f"  Graph is fully connected ✓")

        degree_dict = dict(G.degree())
        degrees = list(degree_dict.values())
        print(f"✓ Average degree: {np.mean(degrees):.1f}")
        print(f"  Degree range: [{np.min(degrees)}, {np.max(degrees)}]")

    # 8. Save results (Maintain original format compatibility)
    print(f"\n--Save results...")
    sampled_fused.to_csv(f'{output_dir}/sampled_nodes.csv', index=False)
    new_edges_df.to_csv(f'{output_dir}/sampled_edges.csv', index=False)
    sampled_labels.to_csv(f'{output_dir}/sampled_labels.csv', index=False)

    print(f"✓ Files saved to: {output_dir}/")
    print(f"  - sampled_nodes.csv ({len(sampled_fused)} rows, {len(all_feature_cols)} features)")
    print(f"  - sampled_edges.csv ({len(new_edges_df)} edges)")
    print(f"  - sampled_labels.csv ({len(sampled_labels)} rows)")

    # 9. Visualization
    visualize_sampling_v2(sampled_fused, new_edges_df, llm_features, base_features, output_dir)
    print('\n')

    return sampled_fused, new_edges_df, sampled_labels


def generate_edges_stratified_with_llm(sampled_fused, api_key, llm_features, base_features):
    """
    Improvements: Stratified edge generation + Noisy-OR Fusion

     strategy: 
    1. For each node, generate 20 edges: 5 entries LLM risk + 15 financial edges
    2. For duplicate edges, use Noisy-OR to fuse weights
    3. Output format matches the original code:  enterprise A,  enterprise B, Weight, Reason
    """

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Edge dictionary: For deduplication and fusion
    edge_dict = {}  # {(src, dst): {'weights': [], 'sources': [], 'reasons': []}}

    # Build the node list
    node_cols = ['\u4f01\u4e1aID'] + llm_features + base_features
    node_list = sampled_fused[node_cols].to_dict('records')

    print(f"Start generating for {len(node_list)} nodes using stratified generation...")
    print(f"   strategy: 5 LLM risk edges + 15 financial edges")

    # Generate edges for each node
    for idx, node in enumerate(tqdm(node_list, desc="LLM Generate edges")):

        # Build the candidate enterprise list
        candidates = [n for n in node_list if n['\u4f01\u4e1aID'] != node['\u4f01\u4e1aID']]

        # ========== Design stratified Prompt ==========
        prompt = design_stratified_prompt(node, candidates, llm_features, base_features)

        try:
            # Call LLM
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            # Process LLM risk edges
            for edge in result.get('risk_pattern_edges', []):
                add_edge_to_dict(
                    edge_dict,
                    str(node['\u4f01\u4e1aID']),
                    str(edge['target_id']),
                    edge['weight'],
                    edge.get('reason', ''),
                    source='llm_risk'
                )

            # Process financial edges
            for edge in result.get('financial_similarity_edges', []):
                add_edge_to_dict(
                    edge_dict,
                    str(node['\u4f01\u4e1aID']),
                    str(edge['target_id']),
                    edge['weight'],
                    edge.get('reason', ''),
                    source='base_financial'
                )

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"\n Node {node[_ENTERPRISE_ID]} Edge generation failed: {e}")
            continue

    # ========== Noisy-OR weight fusion ==========
    print(f"\n Merging duplicate edges (Noisy-OR)...")
    edges_list = []

    for edge_key, edge_info in edge_dict.items():
        weights = edge_info['weights']
        sources = edge_info['sources']
        reasons = edge_info['reasons']

        # Noisy-OR Fusion
        if len(weights) == 1:
            final_weight = weights[0]
            source_tag = sources[0]
        else:
            # Duplicate edges: Noisy-OR
            p_wrong = 1.0
            for w in weights:
                p_wrong *= (1 - w)
            final_weight = 1 - p_wrong
            source_tag = 'both'

        edges_list.append({
            '\u4f01\u4e1aA': edge_info['src'],
            '\u4f01\u4e1aB': edge_info['dst'],
            '\u6743\u91cd': final_weight,
            '\u539f\u56e0': ' | '.join(reasons)
        })

    edges_df = pd.DataFrame(edges_list)

    # Statistics
    n_llm = sum(1 for info in edge_dict.values() if len(info['sources']) == 1 and info['sources'][0] == 'llm_risk')
    n_base = sum(
        1 for info in edge_dict.values() if len(info['sources']) == 1 and info['sources'][0] == 'base_financial')
    n_both = sum(1 for info in edge_dict.values() if len(info['sources']) > 1)

    print(f"\n✓ LLM Generate {len(edges_df)} edges (After deduplication)")
    print(f"  - Only LLM risk: {n_llm} ({n_llm / len(edges_df) * 100:.1f}%)")
    print(f"  - Financial only: {n_base} ({n_base / len(edges_df) * 100:.1f}%)")
    print(f"  - Duplicate edges(Fusion): {n_both} ({n_both / len(edges_df) * 100:.1f}%)")

    if len(edges_df) > 0:
        print(f"  - Weight range: [{edges_df[_EDGE_WEIGHT].min():.3f}, {edges_df[_EDGE_WEIGHT].max():.3f}]")

        # Compare duplicate-edge and single-edge weights
        if n_both > 0:
            both_weights = [edge_dict[k]['weights'] for k in edge_dict.keys() if len(edge_dict[k]['sources']) > 1]
            both_avg = np.mean([1 - np.prod([1 - w for w in ws]) for ws in both_weights])

            single_weights = [edge_dict[k]['weights'][0] for k in edge_dict.keys() if len(edge_dict[k]['sources']) == 1]
            single_avg = np.mean(single_weights)

            print(f"\n  📊 Weight comparison:")
            print(f"    - Mean duplicate-edge weight: {both_avg:.3f}")
            print(f"    - Mean single-edge weight: {single_avg:.3f}")
            print(f"    - Relative increase: +{(both_avg - single_avg):.3f}")

    return edges_df


def add_edge_to_dict(edge_dict, src, dst, weight, reason, source):
    """Add an edge to the dictionary (For deduplication and fusion)"""
    # Undirected edge: Sort to ensure consistency
    edge_key = tuple(sorted([src, dst]))

    if edge_key not in edge_dict:
        edge_dict[edge_key] = {
            'src': src,
            'dst': dst,
            'weights': [],
            'sources': [],
            'reasons': []
        }

    edge_dict[edge_key]['weights'].append(weight)
    edge_dict[edge_key]['sources'].append(source)
    edge_dict[edge_key]['reasons'].append(reason)


def design_stratified_prompt(node, candidates, llm_features, base_features):
    """Design the stratified edge-generation Prompt"""

    # LLM feature description
    llm_desc = "\n".join([
        f"  - {feat}: {node[feat]:.2f}"
        for feat in llm_features
    ])

    # Base feature description (Show the first 10 items)
    base_desc_top10 = "\n".join([
        f"  - {feat}: {node[feat]:.3f}"
        for feat in base_features[:10]
    ])

    # Example candidate enterprises
    candidate_ids = [str(c['\u4f01\u4e1aID']) for c in candidates[:20]]

    prompt = f"""
        You are an expert in enterprise risk relationship analysis. Analyze enterprises similar to the target enterprise from two perspectives.
        
        ## Target enterprise
        ID: {node[_ENTERPRISE_ID]}
        
        ## Perspective 1: LLM risk assessment (5 dimensions)
        {llm_desc}
        
        ## Perspective 2: Original financial indicators (First 10 shown; total: {len(base_features)} items)
        {base_desc_top10}
        ... (another {len(base_features) - 10} financial indicators)
        
        ## Candidate enterprises
        Total: {len(candidates)} items
        Example ID: {', '.join(candidate_ids[:20])}
        ... (another {len(candidates) - 20} items)
        
        ## Task: Generate 20 edges, split into two categories
        
        **Category A (5 edges): Similar risk patterns**
        - Focus: matching LLM risk scores
        - Example: Both have high leverage_risk + high liquidity_risk
        
        **Category B (15 edges): Similar financial indicators**
        - Focus: Match original financial data
        - Example: Similar debt ratios and cash flows
        
        ## Important rules
        1. Category A must contain exactly 5 edges
        2. Category B must contain exactly 15 edges
        3. Avoid duplicates (If an enterprise matches both categories, place it in category B)
        4. Weight range: 0.6-1.0
        
        ## Output format (strict JSON)
        {{
          "risk_pattern_edges": [
            {{"target_id": "enterprise_id", "weight": 0.85, "reason": "similar high leverage + liquidity"}},
            ... // exactly 5 entries
          ],
          "financial_similarity_edges": [
            {{"target_id": "enterprise_id", "weight": 0.82, "reason": "similar debt structure"}},
            ... // exactly 15 entries
          ]
        }}
    """

    return prompt


def visualize_sampling_v2(sampled_fused, edges_df, llm_features, base_features, output_dir):
    """Visualize sampling results (v2 Enhanced version)"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Risk-score scatter plot
    ax = axes[0, 0]
    if len(llm_features) >= 2:
        for label, color, name in [(0, 'blue', 'No Risk'), (1, 'red', 'At Risk')]:
            data = sampled_fused[sampled_fused['label'] == label]
            ax.scatter(data[llm_features[0]], data[llm_features[1]],
                       c=color, label=name, alpha=0.6, s=50)
        ax.set_xlabel(llm_features[0])
        ax.set_ylabel(llm_features[1])
        ax.set_title('Sampled Nodes by Risk Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough risk features',
                ha='center', va='center', transform=ax.transAxes)

    # 2. Edge-weight distribution
    ax = axes[0, 1]
    if len(edges_df) > 0:
        ax.hist(edges_df['\u6743\u91cd'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Edge Weight')
        ax.set_ylabel('Count')
        ax.set_title('Edge Weight Distribution')
        ax.axvline(edges_df['\u6743\u91cd'].median(), color='r', linestyle='--',
                   label=f"Median: {edges_df[_EDGE_WEIGHT].median():.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 3. Label distribution
    ax = axes[0, 2]
    labels = ['At Risk', 'No Risk']
    sampled_counts = [
        (sampled_fused['label'] == 1).sum(),
        (sampled_fused['label'] == 0).sum()
    ]
    colors = ['red', 'blue']
    bars = ax.bar(labels, sampled_counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, sampled_counts):
        height = bar.get_height()
        pct = count / sum(sampled_counts) * 100 if sum(sampled_counts) > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom')

    # 4. Feature-dimension statistics
    ax = axes[1, 0]
    llm_count = len(llm_features)
    base_count = len(base_features)

    categories = ['LLM Features', 'Base Features']
    counts = [llm_count, base_count]
    colors_feat = ['orange', 'green']
    bars = ax.bar(categories, counts, color=colors_feat, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Feature Composition')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count}', ha='center', va='bottom')

    # 5. Degree distribution
    ax = axes[1, 1]
    if len(edges_df) > 0:
        G = nx.from_pandas_edgelist(edges_df, '\u4f01\u4e1aA', '\u4f01\u4e1aB')
        degrees = [d for n, d in G.degree()]
        ax.hist(degrees, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title('Degree Distribution')
        ax.axvline(np.mean(degrees), color='r', linestyle='--',
                   label=f'Mean: {np.mean(degrees):.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 6. Weight vs and degree relationship
    ax = axes[1, 2]
    if len(edges_df) > 0:
        G = nx.from_pandas_edgelist(edges_df, '\u4f01\u4e1aA', '\u4f01\u4e1aB', edge_attr='\u6743\u91cd')

        node_avg_weights = {}
        for node in G.nodes():
            node_edges = [(u, v, d['\u6743\u91cd']) for u, v, d in G.edges(node, data=True)]
            if node_edges:
                avg_weight = np.mean([w for _, _, w in node_edges])
                node_avg_weights[node] = avg_weight

        degrees = [G.degree(n) for n in node_avg_weights.keys()]
        avg_weights = list(node_avg_weights.values())

        ax.scatter(degrees, avg_weights, alpha=0.6, s=30)
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Average Edge Weight')
        ax.set_title('Degree vs Average Weight')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sampling_visualization_v2.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_dir}/sampling_visualization_v2.png")


if __name__ == "__main__":

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = "deepseek-apikey"

    # Path
    data_dir = '/Users/quanhongsong/Documents/Python/pycharm/PythonProject/GNN/data/'
    data_file = ['2023_SH', '2023_SZ', '2022_SH', '2022_SZ', '2022_all', '2023_all', '2022_USA']

    # for data in data_file[4:7]:
    for data in ['2022_BSS']:
        n_total_clusters = 100

        print("=" * 80)
        print(f"Run processing: {data}")
        print("=" * 80)

        nodes, edges, labels = clustering_with_llm_edges(
            fused_path=data_dir + data + '/data_process/fused_features.csv',
            # fused_path=data_dir + data + '/csv_data/reduced_features.csv',
            labels_path=data_dir + data + '/csv_data/enterprise_labels.csv',
            n_clusters=n_total_clusters,
            deepseek_api_key=DEEPSEEK_API_KEY,
            output_dir=data_dir + data + '/sampled_clustering',
            # output_dir=data_dir + data + '/sampled_clustering_nolf'
        )

        print(f"\n{'=' * 80}")
        print(f"{data} Processing completed!")
        print(f"{'=' * 80}\n")