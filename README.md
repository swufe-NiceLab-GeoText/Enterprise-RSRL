# RSRL

Code for *When Structure Lies: Risk-Semantic Relational Learning for Enterprise Resilience Assessment*.

RSRL predicts future adverse events from enterprise financial attributes and relational data, combining LLM-derived risk semantics with graph denoising and augmentation.

## Model

Financial indicators are selected and scored by an LLM using a fixed financial-risk rubric. The resulting representation covers financial health, liquidity, leverage, operational efficiency and market/systemic risk.

A deterministic information bottleneck (DIB) scores observed relations for pruning. K-means representatives guide the generation of risk-pattern and financial-similarity edges, with repeated evidence combined through Noisy-OR. A residual heterogeneous graph transformer (HGT) integrates the relation types and is trained with focal and pairwise ranking losses.

## Reproducibility files

| Content | Files |
|---|---|
| Financial feature selection and risk scoring | [src/select_features.py](src/select_features.py), [src/llm_enhancement.py](src/llm_enhancement.py) |
| Relation denoising | [src/edge_pruning.py](src/edge_pruning.py) |
| Representative sampling and semantic edges | [src/clustering_with_new_edges.py](src/clustering_with_new_edges.py) |
| Feature and graph fusion | [src/feature_fusion.py](src/feature_fusion.py), [src/graph_fusion.py](src/graph_fusion.py) |
| Model, training, hyperparameters and seeds | [src/train.py](src/train.py) |
| Ablations, comparisons and plotting | [analyze/](analyze/) |
| Prompt templates and LLM settings | [docs/PROMPTS.md](docs/PROMPTS.md) |
| Data layout, evaluation and implementation notes | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |

## Data

The experiments cover enterprises listed in Beijing, Shanghai and Shenzhen for 2022 and 2023, and U.S. enterprises for 2022. `BSS` combines the three Chinese exchanges within a year. Positive labels indicate a future adverse event; negative labels indicate no recorded event in the target period.

**Data availability.** The full datasets and generated feature and graph artifacts are not included in this repository because of their size. The auxiliary files retained under `data/` are insufficient for training. Prepare the required inputs separately under `data/<dataset>/` before running an experiment.

Training requires fused features, enterprise labels, edge indices, relation types and a node-index mapping. File names and directory layout are specified in the [data documentation](docs/REPRODUCIBILITY.md#dataset-layout).

## Run

After preparing the data, install the training dependencies and run from the repository root:

```bash
pip install torch torch-geometric numpy pandas scikit-learn scipy
python -c "from src.train import validate_and_report; validate_and_report('data/', ['2022_SH'])"
```

The configured datasets are `2022_SH`, `2022_SZ`, `2023_SH`, `2023_SZ`, `2022_BSS` and `2022_USA`. Change the dataset list in the command as needed. Parameters and fixed seeds are defined in `DATASET_CONFIGS` and `VALID_SEEDS` in `src/train.py`.

The current training script uses stratified random splits of 60%/20%/20% and reports **F1, ROC-AUC, PR-AUC and KS**. Training with prepared artifacts makes no LLM API calls; generating new LLM outputs requires a valid credential in place of `deepseek-apikey`.

Dependency versions are unpinned. Preprocessing requirements, evaluation details and differences between the paper and the active scripts are recorded in the [implementation notes](docs/REPRODUCIBILITY.md).
