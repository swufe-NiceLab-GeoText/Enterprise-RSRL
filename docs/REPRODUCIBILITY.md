# Reproducibility and implementation notes

Detailed data, preprocessing, evaluation and implementation notes for RSRL. See the [main README](../README.md) for the model overview and training entry point, and [PROMPTS.md](PROMPTS.md) for the LLM templates and API settings.

## Prediction task and datasets

Each node represents an enterprise. The positive label (`1`) indicates a future adverse event, interpreted as non-resilience; `0` indicates no recorded event in the target period.

According to the paper, Chinese labels cover legal disputes, ST/risk warnings, trading suspensions, regulatory penalties, and delisting in the following fiscal year. Chinese observed relations represent institutional co-holdings. The U.S. task predicts subsequent delisting; its construction scripts are under [`base_GNN/USA/`](../base_GNN/USA/).

The following counts are taken from the local `csv_data/enterprise_labels.csv` files. They describe the available data, without implying that every dataset is enabled in the current training configuration.

| Dataset | Enterprises | Positive labels | Negative labels |
| --- | ---: | ---: | ---: |
| 2022-BJ | 183 | 28 | 155 |
| 2022-SH | 2,187 | 744 | 1,443 |
| 2022-SZ | 2,760 | 853 | 1,907 |
| 2022-BSS | 5,130 | 1,625 | 3,505 |
| 2023-BJ | 236 | 33 | 203 |
| 2023-SH | 2,270 | 779 | 1,491 |
| 2023-SZ | 2,846 | 786 | 2,060 |
| 2023-BSS | 5,352 | 1,598 | 3,754 |
| 2022-USA | 4,167 | 390 | 3,777 |

**BSS** combines the Beijing, Shanghai, and Shenzhen exchanges within a year. Directory names use underscores, for example `2022_BSS`. The `*_all` directories have the same node and label counts as BSS but contain different feature/graph artifacts; they should not be substituted without checking the intended experiment. `2022_BSS 2` is an additional local copy and is not a separate benchmark.

## Repository structure

```text
Enterprise-RSRL/
├── README.md
├── docs/
│   ├── PROMPTS.md                   # LLM templates and usage details
│   └── REPRODUCIBILITY.md           # Data and implementation notes
├── DST2027_Enterprise.pdf           # Paper draft
├── src/                            # Main processing and training scripts
├── data/                           # Dataset snapshots and derived artifacts
├── base_GNN/
│   ├── data/                       # Baseline dataset copies
│   ├── HetCAN/                     # Baseline implementations and result logs
│   └── USA/                        # U.S. data construction scripts/materials
├── analyze/                        # Ablations, comparisons, and plotting
└── record.txt                      # Historical experiment console output
```

### Main pipeline

| Script | Purpose | Main output |
| --- | --- | --- |
| [`src/convert_dat_to_csv.py`](../src/convert_dat_to_csv.py) | Convert feature matrices, labels, and relations to CSV | `csv_data/enterprise_*.csv` |
| [`src/select_features.py`](../src/select_features.py) | LLM-based feature selection; nominal target of 50 features | `csv_data/reduced_features.csv` |
| [`src/llm_enhancement.py`](../src/llm_enhancement.py) | Generate five risk scores, risk type, and reason | `data_process/llm_enhanced_features_*.csv` |
| [`src/feature_fusion.py`](../src/feature_fusion.py) | Join financial and semantic features by enterprise ID | `data_process/fused_features.csv` |
| [`src/edge_pruning.py`](../src/edge_pruning.py) | Score and prune observed relations with DIB | `data_process/enterprise_relations_dib_pruned_*.csv` |
| [`src/clustering_with_new_edges.py`](../src/clustering_with_new_edges.py) | Select representatives and generate semantic edges | `sampled_clustering/` |
| [`src/graph_fusion.py`](../src/graph_fusion.py) | Map node IDs and combine relation types | `graph_fusion/` |
| [`src/train.py`](../src/train.py) | Train and report classification/ranking metrics | Console report |

### Dataset layout

A processed dataset typically contains:

```text
data/<dataset>/
├── node.dat / link.dat / label.dat / label.dat.test
├── info.dat / meta.dat / features_norm.npy
├── csv_data/
│   ├── enterprise_features.csv
│   ├── enterprise_labels.csv
│   ├── enterprise_relations.csv
│   └── reduced_features.csv
├── data_process/
│   ├── llm_enhanced_features_*.csv
│   ├── fused_features.csv
│   └── enterprise_relations_*pruned*.csv
├── sampled_clustering/
│   ├── sampled_nodes.csv
│   ├── sampled_labels.csv
│   └── sampled_edges.csv
└── graph_fusion/
    ├── fused_edge_index.pt
    ├── fused_edge_type.pt
    ├── node_to_idx.pt
    └── fused_edges.csv
```

Feature and label tables use `\u4f01\u4e1aID`; LLM outputs use `enterprise_id`. Observed edges use `\u4f01\u4e1aA`, `\u4f01\u4e1aB`, `\u5173\u7cfb\u7c7b\u578b`, and `\u6743\u91cd`. Semantic edge tables use `\u4f01\u4e1aA`, `\u4f01\u4e1aB`, `\u6743\u91cd`, and `\u539f\u56e0`. Fused graph edge types are `0` for observed/investment relations and `1` for semantic/similarity relations.

Suffixes such as `_kimi`, `_qwen`, `_25`, `_200`, `_no_DIB`, and `_pruned_random` identify alternate experiment artifacts. Availability varies by dataset.

## Environment

The main implementation uses Python and the following packages, inferred from source imports:

| Scope | Packages |
| --- | --- |
| Training and DIB | `torch`, `torch-geometric`, `numpy`, `pandas`, `scikit-learn`, `scipy` |
| LLM preprocessing | `openai`, `tqdm` |
| Graph analysis and plotting | `networkx`, `matplotlib` |
| Additional baselines | `dgl`, `xgboost` |

Dependency versions are not pinned, and a verified environment specification is not included. The installation commands in the main README list packages inferred from source imports. Match PyTorch, PyG, DGL, and CUDA versions to the target environment. The paper reports experiments on an NVIDIA RTX 4090; the current main training script does not explicitly move the model and data to CUDA.

## Running experiments

### Train with existing processed artifacts

Run from the repository root, with the packages above installed. Training from cached artifacts does not require an LLM API key.

```bash
python -c "from src.train import validate_and_report; validate_and_report('data/', ['2022_SH'])"
```

The selected dataset must contain the three `.pt` graph files, `data_process/fused_features.csv`, and `csv_data/enterprise_labels.csv`. Parameters and random seeds are defined in `DATASET_CONFIGS` and `VALID_SEEDS` in `src/train.py`.

The current configuration enables `2022_SH`, `2022_SZ`, `2023_SH`, `2023_SZ`, `2022_BSS`, and `2022_USA`. It creates stratified random train/validation/test splits of **60%/20%/20%**, rather than using `label.dat.test`. The default HGT uses two layers, hidden dimension 64, four attention heads, and dropout 0.4.

Reported metrics are **F1, ROC-AUC, PR-AUC, and KS**. PR-AUC is computed by trapezoidal integration of the precision-recall curve. The current F1 evaluator selects the best threshold on the evaluated split; it does not use a threshold fixed on validation data.

The command above documents the current Python entry point and its input files have been checked for `2022_SH`. A complete training run has not been execution-validated as part of this documentation update.

### Rebuild preprocessing artifacts

Follow the main pipeline table in order, with DIB pruning and representative-based augmentation as two branches after feature fusion. Before running a stage:

1. Set its dataset selection and data root. Several `__main__` blocks still contain absolute paths from earlier macOS/Linux environments.
2. Align the producer and consumer filenames. For example, the active LLM enhancement entry point writes an `_industry_qianwen.csv` filename, while feature fusion reads `_industry.csv`; the default client remains `deepseek-chat`.
3. Create the selected output directories where the script does not create them automatically.
4. For LLM stages, supply a credential to the API-key argument. Keep credentials outside tracked files. The current scripts use `deepseek-apikey` placeholders; they do not automatically read environment variables.
5. Select the intended observed-edge input in `graph_fusion.py`. Its current active input is the original relation table, while the pruned inputs are commented out.

The source client defaults to the DeepSeek endpoint with the `deepseek-chat` model identifier. The paper names DeepSeek-V3, but the source does not pin a dated model revision. Full prompts and API settings are in [`docs/PROMPTS.md`](PROMPTS.md).

## Evaluation and analysis

| Entry point | Experiment |
| --- | --- |
| [`analyze/train_ablation.py`](../analyze/train_ablation.py) | Remove semantic features, semantic edges, or DIB; includes configuration screening |
| [`analyze/train_contrast.py`](../analyze/train_contrast.py) | LLM variants, representative counts, and pruning strategies |
| [`analyze/train_GNN.py`](../analyze/train_GNN.py) | GCN, GAT, GraphSAGE, and HGT backbone comparisons |
| [`analyze/Label_Test/code/train_LabelTest.py`](../analyze/Label_Test/code/train_LabelTest.py) | Label-permutation and prompt-rubric comparisons |
| [`analyze/DIB_effect/`](../analyze/DIB_effect/) | Retained-edge budget and label-homophily analysis |
| [`analyze/param/`](../analyze/param/) | Parameter sweeps and plots |
| [`analyze/time/`](../analyze/time/) | Training/inference timing and plots |
| [`base_GNN/HetCAN/`](../base_GNN/HetCAN/) | LR, XGBoost, MLP, GCN, GAT, GraphSAGE, DAGNN, PinSAGE, PRAGA, and UnGSL implementations |

Check each analysis script's paths, active settings, and expected artifact variants before execution. Historical logs and `record.txt` contain outputs from multiple configurations and are not a single verified reproduction record for the paper tables.

## Implementation status

This repository currently preserves a research experiment snapshot. The paper's method and the active scripts differ in several material details:

- **DIB objective and ranking.** The paper specifies a squared code norm with SmoothL1 prediction loss and norm-based pruning. The source uses a scaled L2 norm with binary cross-entropy, then mixes normalized DIB scores and relation weights at 0.7/0.3 before top-k retention.
- **Weighted propagation.** The paper defines an edge-weight bias in attention. The current fusion outputs and HGT calls retain edge indices and relation types, without passing edge weights to attention.
- **GraphSMOTE training.** The source computes interpolated positive embeddings, but does not use them in retraining. Added positive nodes have zero features and receive no new incident edges. Zero initialization is described in Appendix F; the contextual message propagation described there is not established by the current augmentation code.
- **Semantic prompting.** The paper's main text uses scores in `[0, 1]` and excludes identity/class-proportion information. The appendix and active prompts use integer scores from 1 to 10, an internal enterprise index, and low/medium/high score-distribution instructions. These score-distribution instructions are distinct from binary outcome class proportions.
- **Evaluation protocol.** Scaling is fit before splitting, F1 thresholds are selected on the evaluated split, and the main script uses preselected seeds. These choices should be accounted for when interpreting or reproducing reported results.

These differences document the present files; they do not establish which code revision produced each stored artifact or paper result.

## Paper reference

> *When Structure Lies: Risk-Semantic Relational Learning for Enterprise Resilience Assessment*. Anonymous manuscript; see the [local paper draft](../DST2027_Enterprise.pdf).

Publication metadata and an explicit code/data license are not supplied in this snapshot.
