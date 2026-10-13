# RSRL

Risk-semantic relational learning for enterprise resilience assessment.

[Paper: *When Structure Lies: Risk-Semantic Relational Learning for Enterprise Resilience Assessment*](DST2027_Enterprise.pdf)

RSRL predicts future adverse events by combining enterprise financial attributes, observed relations and LLM-derived risk semantics.

## Model

The paper's pipeline has four components:

1. **Risk semantics:** select financial indicators and generate five LLM risk scores covering financial health, liquidity, leverage, operational efficiency and market/systemic risk.
2. **Relation denoising:** use a deterministic information bottleneck (DIB) to score and prune observed relations.
3. **Semantic augmentation:** select representative enterprises with K-means, generate semantic edges and combine repeated evidence with Noisy-OR.
4. **Prediction:** combine relation types in a residual heterogeneous graph transformer (HGT), trained with focal and pairwise ranking losses.

## Files

| Content | Location |
|---|---|
| Preprocessing and training | [src/](src/) |
| Model, hyperparameters and seeds | [src/train.py](src/train.py) |
| Ablations and plots | [analyze/](analyze/) |
| Baselines | [base_GNN/HetCAN/](base_GNN/HetCAN/) |

## Data

Available datasets cover enterprises from the Beijing, Shanghai and Shenzhen exchanges for 2022 and 2023, and U.S. enterprises for 2022. `BSS` combines the three Chinese exchanges within a year. Label `1` denotes a future adverse event; `0` denotes no recorded event in the target period.

Processed features, labels and graphs are stored in `data/<dataset>/`. Training on cached artifacts requires no API key. Regenerating LLM outputs requires a credential in place of `deepseek-apikey`; preprocessing paths and filenames must be aligned as described in the documentation.

## Run

Train on cached `2022_SH` artifacts from the repository root:

```bash
pip install torch torch-geometric numpy pandas scikit-learn scipy
python -c "from src.train import validate_and_report; validate_and_report('data/', ['2022_SH'])"
```

Other configured datasets are `2022_SZ`, `2023_SH`, `2023_SZ`, `2022_BSS` and `2022_USA`. Change the dataset list in the command to select one or more datasets.

Hyperparameters and fixed seeds are defined in `DATASET_CONFIGS` and `VALID_SEEDS` in `src/train.py`. The current script uses stratified random splits of 60%/20%/20% and reports **F1, ROC-AUC, PR-AUC and KS**. Dependency versions are unpinned; environment notes and evaluation details are linked below.

## Documentation

- [Prompts and LLM settings](docs/PROMPTS.md)
- [Data, preprocessing, evaluation and implementation differences](docs/REPRODUCIBILITY.md)
