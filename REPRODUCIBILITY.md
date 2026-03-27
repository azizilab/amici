# Reproducibility Guide

This document maps scripts and workflows in this repository to the figures they generate in the AMICI manuscript.

## Benchmarks

The full benchmark pipeline is orchestrated by [`benchmarks/Snakefile`](benchmarks/Snakefile) with configuration in [`benchmarks/benchmark_config.yaml`](benchmarks/benchmark_config.yaml). The pipeline:

1. Generates semisynthetic datasets ([`benchmarks/generate_dataset.py`](benchmarks/generate_dataset.py))
2. Computes ground truth scores ([`benchmarks/gene_task/generate_gt_gene_scores.py`](benchmarks/gene_task/generate_gt_gene_scores.py), [`benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py`](benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py), [`benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py`](benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py))
3. Trains all models ([`benchmarks/train_amici_model.py`](benchmarks/train_amici_model.py), [`benchmarks/train_gitiii_model.py`](benchmarks/train_gitiii_model.py), [`benchmarks/train_ncem_model.py`](benchmarks/train_ncem_model.py), [`benchmarks/train_cgcom_model.py`](benchmarks/train_cgcom_model.py))
4. Scores each model on all tasks and generates PR curves ([`benchmarks/gene_task/generate_amici_scores.py`](benchmarks/gene_task/generate_amici_scores.py), [`benchmarks/gene_task/generate_amici_pr.py`](benchmarks/gene_task/generate_amici_pr.py), and analogous scripts per model/task)
5. Produces the final plots ([`benchmarks/gene_task/plot_boxplots.py`](benchmarks/gene_task/plot_boxplots.py), [`benchmarks/gene_task/plot_pr_curves.py`](benchmarks/gene_task/plot_pr_curves.py), [`benchmarks/neighbor_interaction_task/plot_boxplots.py`](benchmarks/neighbor_interaction_task/plot_boxplots.py), [`benchmarks/neighbor_interaction_task/plot_pr_curves.py`](benchmarks/neighbor_interaction_task/plot_pr_curves.py), [`benchmarks/receiver_subtype_task/plot_boxplots.py`](benchmarks/receiver_subtype_task/plot_boxplots.py), [`benchmarks/receiver_subtype_task/plot_pr_curves.py`](benchmarks/receiver_subtype_task/plot_pr_curves.py), [`benchmarks/length_scale_task/plot_kde.py`](benchmarks/length_scale_task/plot_kde.py))

Sensitivity analysis scripts (S12--S17) are standalone and train their own models internally.

### [`benchmarks/generate_dataset.py`](benchmarks/generate_dataset.py)

Generates the semisynthetic spatial transcriptomics dataset with grid structure, cell types, and subtypes. Orchestrated by the Snakefile rule `generate_dataset`.

- **Figure 2a**: Spatial scatter plot of the semisynthetic dataset colored by cell type

### [`benchmarks/length_scale_task/plot_kde.py`](benchmarks/length_scale_task/plot_kde.py)

KDE density plot of inferred interaction length scales. Orchestrated by Snakefile rule `plot_amici_length_scale_boxplots`.

- **Figure 2b**: Density plot of AMICI's inferred length scales with ground truth overlay

### [`benchmarks/gene_task/plot_boxplots.py`](benchmarks/gene_task/plot_boxplots.py)

AUPRC boxplots for the gene prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_gene_task_boxplots`.

- **Figure 2c**: Gene prediction AUPRC boxplots (AMICI, GITIII, NicheDE, NCEM)

### [`benchmarks/neighbor_interaction_task/plot_boxplots.py`](benchmarks/neighbor_interaction_task/plot_boxplots.py)

AUPRC boxplots for the sender cell prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_neighbor_interaction_task_boxplots`.

- **Figure 2d**: Sender cell prediction AUPRC boxplots (AMICI, GITIII, CGCom)

### [`benchmarks/receiver_subtype_task/plot_boxplots.py`](benchmarks/receiver_subtype_task/plot_boxplots.py)

AUPRC boxplots for the receiver cell prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_receiver_subtype_task_boxplots`.

- **Figure 2e**: Receiver cell prediction AUPRC boxplots (AMICI, GITIII, CGCom)

### [`benchmarks/gene_task/plot_pr_curves.py`](benchmarks/gene_task/plot_pr_curves.py), [`benchmarks/neighbor_interaction_task/plot_pr_curves.py`](benchmarks/neighbor_interaction_task/plot_pr_curves.py), [`benchmarks/receiver_subtype_task/plot_pr_curves.py`](benchmarks/receiver_subtype_task/plot_pr_curves.py)

Precision-recall curves for each benchmark task. Use `plot_pr_curves()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py).

- **Fig S2a**: PR curves for all three tasks on the semisynthetic dataset

### [`benchmarks/train_cgcom_model.py`](benchmarks/train_cgcom_model.py), [`benchmarks/train_gitiii_model.py`](benchmarks/train_gitiii_model.py), [`benchmarks/train_ncem_model.py`](benchmarks/train_ncem_model.py)

Model training scripts that also generate interaction network visualizations and loss curves. Use `plot_interaction_matrix()` and `plot_interaction_graph()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py).

- **Fig S2b**: Predicted interaction networks from the semisynthetic dataset (AMICI vs NCEM)
- **Fig S2c**: CGCom training/validation loss curves

### [`benchmarks/sensitivity_scripts/head_analysis.py`](benchmarks/sensitivity_scripts/head_analysis.py)

Sensitivity analysis varying number of attention heads (h=2, 4, 6, 8) across 10 seeds.

- **Fig S12**: Number of attention heads sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/neighbor_analysis.py`](benchmarks/sensitivity_scripts/neighbor_analysis.py)

Sensitivity analysis varying number of nearest neighbors (k=50, 70, 80, 90) across 10 seeds.

- **Fig S13**: Number of neighbors sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/value_penalty_analysis.py`](benchmarks/sensitivity_scripts/value_penalty_analysis.py)

Sensitivity analysis varying the value vector L1 penalty coefficient.

- **Fig S14**: Value L1 penalty sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/attention_penalty_analysis.py`](benchmarks/sensitivity_scripts/attention_penalty_analysis.py)

Sensitivity analysis varying the attention entropy penalty coefficient.

- **Fig S15**: Attention penalty sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/length_scale_task/sensitivity_test.py`](benchmarks/length_scale_task/sensitivity_test.py)

Length scale sensitivity to attention threshold parameter across sender-receiver pairs.

- **Fig S17**: Length scale sensitivity (boxplot of d_scale vs alpha threshold)

## Cortex Analysis

Prerequisites (run in order):

1. [`cortex/cortex_preprocess.py`](cortex/cortex_preprocess.py) — Loads raw MERFISH data (counts, cell labels, segmentation CSVs from the Brain Image Library), extracts cell centroids from polygon boundaries, normalizes counts, and performs train/test split.
2. [`cortex/cortex_sweep_main.py`](cortex/cortex_sweep_main.py) — Runs a Weights & Biases hyperparameter sweep to train the AMICI model on the preprocessed cortex data. Sweep config: [`cortex/cortex_sweep.yaml`](cortex/cortex_sweep.yaml).
3. [`cortex/cortex_analysis.py`](cortex/cortex_analysis.py) — Loads the trained model and generates all cortex figures (see below).

### [`cortex/cortex_analysis.py`](cortex/cortex_analysis.py)

MERFISH mouse cortex analysis. Generates spatial distributions, directed interaction networks, and gene dot plots using `AMICIAblationModule`.

- **Figure 3a** (left): Spatial scatter plot of MERFISH cortex (`visualize_spatial_distribution()`)
- **Figure 3a** (right): Directed interaction graph (`plot_interaction_directed_graph()`)
- **Figure 3b**: Downstream gene dot plots for Astrocyte and Sst receivers (`plot_featurewise_contributions_dotplot()`)
- **Fig S2d**: Full directed interaction network of all cortex cell types (`plot_interaction_directed_graph()`)

## Xenium Breast Cancer Analysis

Prerequisites (run in order):

1. [`xenium/xenium_preprocess.py`](xenium/xenium_preprocess.py) — Loads raw Xenium spatial transcriptomics data, filters highly variable genes, normalizes counts, corrects DCIS labels via nearest-neighbor voting, and performs train/test split by spatial region.
2. [`xenium/xenium_sweep_main.py`](xenium/xenium_sweep_main.py) — Runs a Weights & Biases hyperparameter sweep to train the AMICI model on the preprocessed Xenium data. Sweep config: [`xenium/xenium_sweep.yaml`](xenium/xenium_sweep.yaml). Alternatively, [`xenium/xenium_train.py`](xenium/xenium_train.py) trains a single model with fixed hyperparameters.
3. Analysis and plotting scripts (each loads the trained model independently): [`xenium/xenium_analysis.py`](xenium/xenium_analysis.py), [`xenium/xenium_spatial_analysis.py`](xenium/xenium_spatial_analysis.py), [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py), [`xenium/xenium_niche_prediction_analysis.py`](xenium/xenium_niche_prediction_analysis.py), [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py), [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py), [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py), [`xenium/xenium_null_z_distribution.py`](xenium/xenium_null_z_distribution.py), [`xenium/runtime_benchmark/plot_benchmark.py`](xenium/runtime_benchmark/plot_benchmark.py).

### [`xenium/xenium_analysis.py`](xenium/xenium_analysis.py)

Primary Xenium breast cancer analysis. Generates spatial distributions, interaction networks, gene dot plots, length scale comparisons, and volcano plots using `AMICIAblationModule` and `AMICICounterfactualAttentionModule`.

- **Figure 4a**: Spatial scatter plots of both Xenium replicates (`visualize_spatial_distribution()`)
- **Figure 4b**: Directed interaction networks — full and immune-tumor subset (`plot_interaction_directed_graph()`)
- **Figure 4c**: Downstream gene dot plots for M1 macrophages, CD8 T cells, invasive tumor (`plot_featurewise_contributions_dotplot()`)
- **Figure 4d**: Length scale distributions and length-scale-dependent gene analysis (`plot_length_scale_distribution()`)
- **Fig S3c**: Explained variance by attention head (`plot_explained_variance_barplot()`)
- **Fig S6**: Volcano plots of neighbor contribution vs Wald statistic per receiver cell type

### [`xenium/xenium_spatial_analysis.py`](xenium/xenium_spatial_analysis.py)

Spatial attention pattern analysis for the Xenium dataset. Generates proximity scores, attention heatmaps, and gene expression spatial plots using `AMICIAblationModule` and `AMICIAttentionModule`.

- **Figure 4e**: Four-panel spatial analysis — proximity scores, attention heatmaps, ESR1 gene expression
- **Fig S7**: AGR3 subpopulation spatial analysis (attention + gene expression)

### [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py)

Communication hub analysis. Clusters cells by AMICI interaction patterns, compares with composition-based niches, and performs grid search over hub parameters.

- **Figure 4f** (top): Spatial scatter plots colored by communication hub assignment
- **Figure 4f** (bottom): Alluvial/Sankey diagrams of hub sender-receiver composition
- **Fig S4a**: Hub vs composition cluster spatial comparison
- **Fig S4b**: ARI/AMI comparison between hubs, composition clusters, and cell-type labels
- **Fig S8a**: Silhouette score vs number of clusters
- **Fig S8b**: Alluvial/Sankey diagrams for all 10 communication hubs
- **Fig S18**: Hub grid search heatmaps — fixed k, varying quantile threshold
- **Fig S19**: Hub grid search heatmaps — fixed quantile, varying k

### [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py)

Gene set enrichment analysis comparing communication hubs to composition clusters. Supports GSEA (MSigDB Hallmark), KEGG Signaling, and Reactome pathway databases.

- **Fig S9**: GSEA barplots comparing hub-unique vs shared vs composition-unique significant pathways
- **Fig S10**: GSEA butterfly charts per cell type (hub vs composition enrichment scores)

### [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py)

Cross-replicate validation of AMICI interaction predictions.

- **Fig S3a**: Interaction strength heatmaps for replicate 1 and replicate 2
- **Fig S3b**: Replicate scatter plot with Spearman correlation

### [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py)

Cell segmentation artifact analysis. Validates that identified genes are not artifacts of segmentation errors.

- **Fig S5a**: Interaction network including stromal cells (showing segmentation artifact dominance)
- **Fig S5b**: Segmentation validation dot plot (Mann-Whitney U test for interaction-mediated genes)

### [`xenium/xenium_null_z_distribution.py`](xenium/xenium_null_z_distribution.py)

Monte Carlo validation of the Wald test normality assumption using 50 randomly initialized models.

- **Fig S16**: Null z-value distribution histogram + Q-Q plot

### [`xenium/runtime_benchmark/plot_benchmark.py`](xenium/runtime_benchmark/plot_benchmark.py)

Runtime scaling visualization. Data generated by [`xenium/runtime_benchmark/run_benchmark.py`](xenium/runtime_benchmark/run_benchmark.py).

- **Fig S11**: Runtime benchmark (CPU wall-clock time + GPU seconds/epoch vs number of cells)

## Data and Model Artifacts

Some of the analyses above depend on pretrained AMICI models and preprocessed datasets. The table below lists each artifact required to reproduce the figures.

### Cortex (MERFISH)

| Dataset              | Local Dataset Path                             | Local Model Path                                                          | Model YAML Config | Figshare Link |
| -------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- | ----------------- | ------------- |
| MERFISH Mouse Cortex | `cortex/data/cortex_processed_2025-04-28.h5ad` | `cortex/saved_models/cortex_18_sweep_plm73bmg_xrtcnlt0_params_2025-05-05` | TODO              | TODO          |

### CosMx

Prerequisites (run in order):

1. [`cosmx/preprocess_cosmx.py`](cosmx/preprocess_cosmx.py) — Loads CosMx liver data, normalizes expression, creates spatial coordinates, filters cell types, and splits into training/test sets.
2. [`cosmx/tune_cosmx.py`](cosmx/tune_cosmx.py) — Runs a Weights & Biases hyperparameter sweep. Alternatively, [`cosmx/train_cosmx.py`](cosmx/train_cosmx.py) trains a single model with fixed hyperparameters.

| Dataset                    | Local Dataset Path                                                | Local Model Path                                          | Model YAML Config | Figshare Link |
| -------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- | ----------------- | ------------- |
| CosMx Liver Cancer (train) | `/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_train.h5ad` | `cosmx/saved_models/cosmx_liver_cancer_sub_khushi_params` | TODO              | TODO          |
| CosMx Liver Cancer (test)  | `/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_test.h5ad`  | —                                                         | —                 | TODO          |

### Xenium Breast Cancer

| Dataset            | Local Dataset Path                                                        | Local Model Path                                                                   | Model YAML Config | Figshare Link |
| ------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------- | ------------- |
| Xenium Full        | `xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad`      | `xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02`      | TODO              | TODO          |
| Xenium Replicate 1 | `xenium/data/xenium_sample1/xenium_sample1_rep1_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep1_proseg_sweep_2025-05-01_model_2025-05-13` | TODO              | TODO          |
| Xenium Replicate 2 | `xenium/data/xenium_sample1/xenium_sample1_rep2_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep2_proseg_sweep_2025-05-01_model_2025-05-14` | TODO              | TODO          |
