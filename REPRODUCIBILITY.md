# Reproducibility Guide

This document maps scripts and workflows in this repository to the figures they generate in the AMICI manuscript.

Figures 1, S1 are schematics/diagrams and were not generated from data scripts.

## Benchmarks

**Base data**: The semisynthetic datasets are generated from the [Fresh 68k PBMCs (Donor A)](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) dataset, downloaded programmatically via `scvi.data.dataset_10x(dataset_name="fresh_68k_pbmc_donor_a")`.

The full benchmark pipeline is orchestrated by [`benchmarks/Snakefile`](benchmarks/Snakefile) with configuration in [`benchmarks/benchmark_config.yaml`](benchmarks/benchmark_config.yaml). The pipeline:

1. Generates semisynthetic datasets ([`benchmarks/generate_dataset.py`](benchmarks/generate_dataset.py))
2. Computes ground truth scores ([`benchmarks/gene_task/generate_gt_gene_scores.py`](benchmarks/gene_task/generate_gt_gene_scores.py), [`benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py`](benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py), [`benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py`](benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py))
3. Trains all models ([`benchmarks/train_amici_model.py`](benchmarks/train_amici_model.py), [`benchmarks/train_gitiii_model.py`](benchmarks/train_gitiii_model.py), [`benchmarks/train_ncem_model.py`](benchmarks/train_ncem_model.py), [`benchmarks/train_cgcom_model.py`](benchmarks/train_cgcom_model.py))
4. Scores each model on all tasks and generates PR curves ([`benchmarks/gene_task/generate_amici_scores.py`](benchmarks/gene_task/generate_amici_scores.py), [`benchmarks/gene_task/generate_amici_pr.py`](benchmarks/gene_task/generate_amici_pr.py), and analogous scripts per model/task)
5. Produces the final plots ([`benchmarks/gene_task/plot_boxplots.py`](benchmarks/gene_task/plot_boxplots.py), [`benchmarks/gene_task/plot_pr_curves.py`](benchmarks/gene_task/plot_pr_curves.py), [`benchmarks/neighbor_interaction_task/plot_boxplots.py`](benchmarks/neighbor_interaction_task/plot_boxplots.py), [`benchmarks/neighbor_interaction_task/plot_pr_curves.py`](benchmarks/neighbor_interaction_task/plot_pr_curves.py), [`benchmarks/receiver_subtype_task/plot_boxplots.py`](benchmarks/receiver_subtype_task/plot_boxplots.py), [`benchmarks/receiver_subtype_task/plot_pr_curves.py`](benchmarks/receiver_subtype_task/plot_pr_curves.py), [`benchmarks/length_scale_task/plot_kde.py`](benchmarks/length_scale_task/plot_kde.py))

Sensitivity analysis scripts (S21--S26) are standalone and train their own models internally.

### [`benchmarks/generate_dataset.py`](benchmarks/generate_dataset.py)

Generates the PBMC semisynthetic spatial transcriptomics dataset with grid structure, cell types, and subtypes. Orchestrated by the Snakefile rule `generate_dataset`.

- **Figure 2a**: Spatial scatter plot of the PBMC semisynthetic dataset colored by cell type

### [`benchmarks/generate_realistic_dataset.py`](benchmarks/generate_realistic_dataset.py)

Generates the realistic breast cancer semisynthetic dataset by combining Flex scRNA-seq expression profiles with Xenium spatial coordinates. Uses real tissue geometry and cell-type composition with programmatic interactions (Macrophages→DCIS, T Cells→Endothelial, Invasive Tumor→Myoepi). Orchestrated by the Snakefile.

- **Figure 2f**: Spatial scatter plot of the realistic semisynthetic dataset colored by cell type

### [`benchmarks/length_scale_task/plot_kde.py`](benchmarks/length_scale_task/plot_kde.py)

KDE density plot of inferred interaction length scales. Orchestrated by Snakefile rule `plot_amici_length_scale_boxplots`.

- **Figure 2b**: Density plot of AMICI's inferred length scales with ground truth overlay (PBMC)
- **Figure 2g**: Density plot of AMICI's inferred length scales with ground truth overlay (realistic)

### [`benchmarks/gene_task/plot_boxplots.py`](benchmarks/gene_task/plot_boxplots.py)

AUPRC boxplots for the gene prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_gene_task_boxplots`.

- **Figure 2c**: Gene prediction AUPRC boxplots — PBMC (AMICI, GITIII, NicheDE, NCEM)
- **Figure 2h**: Gene prediction AUPRC boxplots — realistic (AMICI, GITIII, NicheDE, NCEM)

### [`benchmarks/neighbor_interaction_task/plot_boxplots.py`](benchmarks/neighbor_interaction_task/plot_boxplots.py)

AUPRC boxplots for the sender cell prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_neighbor_interaction_task_boxplots`.

- **Figure 2d**: Sender cell prediction AUPRC boxplots — PBMC (AMICI, GITIII, CGCom)
- **Figure 2i**: Sender cell prediction AUPRC boxplots — realistic (AMICI, GITIII, CGCom)

### [`benchmarks/receiver_subtype_task/plot_boxplots.py`](benchmarks/receiver_subtype_task/plot_boxplots.py)

AUPRC boxplots for the receiver cell prediction task. Uses `plot_boxplots()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py). Orchestrated by Snakefile rule `plot_receiver_subtype_task_boxplots`.

- **Figure 2e**: Receiver cell prediction AUPRC boxplots — PBMC (AMICI, GITIII, CGCom)
- **Figure 2j**: Receiver cell prediction AUPRC boxplots — realistic (AMICI, GITIII, CGCom)

### [`benchmarks/gene_task/plot_pr_curves.py`](benchmarks/gene_task/plot_pr_curves.py), [`benchmarks/neighbor_interaction_task/plot_pr_curves.py`](benchmarks/neighbor_interaction_task/plot_pr_curves.py), [`benchmarks/receiver_subtype_task/plot_pr_curves.py`](benchmarks/receiver_subtype_task/plot_pr_curves.py)

Precision-recall curves for each benchmark task. Use `plot_pr_curves()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py).

- **Fig S2a**: PR curves for all three tasks on the semisynthetic dataset
- **Fig S3**: PR curves for all baselines after hyperparameter sweeps (best model by lowest MSE test loss)

### [`benchmarks/train_amici_model.py`](benchmarks/train_amici_model.py), [`benchmarks/train_cgcom_model.py`](benchmarks/train_cgcom_model.py), [`benchmarks/train_gitiii_model.py`](benchmarks/train_gitiii_model.py), [`benchmarks/train_ncem_model.py`](benchmarks/train_ncem_model.py)

Model training scripts that also generate interaction network visualizations and loss curves. Use `plot_interaction_matrix()` and `plot_interaction_graph()` from [`benchmarks/benchmark_utils.py`](benchmarks/benchmark_utils.py).

- **Fig S2b**: Predicted interaction networks from the semisynthetic dataset (AMICI, NCEM, CGCom, GITIII)
- **Fig S2c**: Training/validation loss curves (CGCom, NCEM, GITIII)

### [`benchmarks/sensitivity_scripts/head_analysis.py`](benchmarks/sensitivity_scripts/head_analysis.py)

Sensitivity analysis varying number of attention heads (h=2, 4, 6, 8) across 10 seeds.

- **Fig S21**: Number of attention heads sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/neighbor_analysis.py`](benchmarks/sensitivity_scripts/neighbor_analysis.py)

Sensitivity analysis varying number of nearest neighbors (k=50, 70, 80, 90) across 10 seeds.

- **Fig S22**: Number of neighbors sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/value_penalty_analysis.py`](benchmarks/sensitivity_scripts/value_penalty_analysis.py)

Sensitivity analysis varying the value vector L1 penalty coefficient.

- **Fig S23**: Value L1 penalty sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/sensitivity_scripts/attention_penalty_analysis.py`](benchmarks/sensitivity_scripts/attention_penalty_analysis.py)

Sensitivity analysis varying the attention entropy penalty coefficient.

- **Fig S24**: Attention penalty sensitivity (AUPRC boxplots + PR curves)

### [`benchmarks/length_scale_task/sensitivity_test.py`](benchmarks/length_scale_task/sensitivity_test.py)

Length scale sensitivity to attention threshold parameter across sender-receiver pairs.

- **Fig S26**: Length scale sensitivity (boxplot of d_scale vs alpha threshold)

## Cortex Analysis

**Base data**: MERFISH mouse cortex data from the [Brain Image Library](https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/processed_data/).

Prerequisites (run in order):

1. [`cortex/cortex_preprocess.py`](cortex/cortex_preprocess.py) — Loads raw MERFISH data (counts, cell labels, segmentation CSVs), extracts cell centroids from polygon boundaries, normalizes counts, and performs train/test split.
2. [`cortex/cortex_sweep_main.py`](cortex/cortex_sweep_main.py) — Runs a Weights & Biases hyperparameter sweep to train the AMICI model on the preprocessed cortex data. Sweep config: [`cortex/cortex_sweep.yaml`](cortex/cortex_sweep.yaml).
3. [`cortex/cortex_analysis.py`](cortex/cortex_analysis.py) — Loads the trained model and generates all cortex figures (see below).

### [`cortex/cortex_analysis.py`](cortex/cortex_analysis.py)

MERFISH mouse cortex analysis. Generates spatial distributions, directed interaction networks, and gene dot plots using `AMICIAblationModule`.

- **Figure 3a**: Spatial scatter plot of MERFISH cortex (`visualize_spatial_distribution()`)
- **Figure 3b**: Directed interaction graph of inferred interacting cell types (`plot_interaction_directed_graph()`)
- **Figure 3c**: Downstream gene dot plots for Astrocyte and Sst receivers (`plot_featurewise_contributions_dotplot()`)
- **Fig S4**: Full directed interaction network of all cortex cell types (`plot_interaction_directed_graph()`)

## Xenium Breast Cancer Analysis

**Base data**: [10x Genomics Xenium FFPE Human Breast Cancer](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast), resegmented with ProSeg and reannotated with resolVI. The preprocessed datasets used in the analyses are available on Figshare (see [Data and Model Artifacts](#data-and-model-artifacts)).

Prerequisites (run in order):

1. [`xenium/xenium_preprocess.py`](xenium/xenium_preprocess.py) — Loads the resegmented/reannotated Xenium data, filters highly variable genes, normalizes counts, corrects DCIS labels via nearest-neighbor voting, and performs train/test split by spatial region.
2. [`xenium/xenium_sweep_main.py`](xenium/xenium_sweep_main.py) — Runs a Weights & Biases hyperparameter sweep to train the AMICI model on the preprocessed Xenium data. Sweep config: [`xenium/xenium_sweep.yaml`](xenium/xenium_sweep.yaml). Alternatively, [`xenium/xenium_train.py`](xenium/xenium_train.py) trains a single model with fixed hyperparameters.
3. Analysis and plotting scripts (each loads the trained model independently): [`xenium/xenium_analysis.py`](xenium/xenium_analysis.py), [`xenium/xenium_spatial_analysis.py`](xenium/xenium_spatial_analysis.py), [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py), [`xenium/xenium_niche_prediction_analysis.py`](xenium/xenium_niche_prediction_analysis.py), [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py), [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py), [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py), [`xenium/segmentation_analysis_gitiii.py`](xenium/segmentation_analysis_gitiii.py), [`xenium/xenium_lr_analysis.py`](xenium/xenium_lr_analysis.py), [`xenium/xenium_null_z_distribution.py`](xenium/xenium_null_z_distribution.py), [`xenium/runtime_benchmark/plot_benchmark.py`](xenium/runtime_benchmark/plot_benchmark.py).

### [`xenium/xenium_analysis.py`](xenium/xenium_analysis.py)

Primary Xenium breast cancer analysis. Generates spatial distributions, interaction networks, gene dot plots, length scale comparisons, and volcano plots using `AMICIAblationModule` and `AMICICounterfactualAttentionModule`.

- **Figure 4a**: Spatial scatter plots of both Xenium replicates (`visualize_spatial_distribution()`)
- **Figure 4b**: Directed interaction networks — full and immune-tumor subset (`plot_interaction_directed_graph()`)
- **Figure 4c**: Downstream gene dot plots for M1 macrophages, CD8 T cells, invasive tumor (`plot_featurewise_contributions_dotplot()`)
- **Figure 4d**: Length scale distributions and length-scale-dependent gene analysis (`plot_length_scale_distribution()`)
- **Fig S5c**: Explained variance by attention head (`plot_explained_variance_barplot()`)
- **Fig S7**: Volcano plots of neighbor contribution vs Wald statistic per receiver cell type

### [`xenium/xenium_spatial_analysis.py`](xenium/xenium_spatial_analysis.py)

Spatial attention pattern analysis for the Xenium dataset. Generates proximity scores, attention heatmaps, and gene expression spatial plots using `AMICIAblationModule` and `AMICIAttentionModule`.

- **Figure 4e**: Four-panel spatial analysis — proximity scores, attention heatmaps, ESR1 gene expression
- **Fig S9**: AGR3 subpopulation spatial analysis (attention + gene expression)

### [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py)

Communication hub analysis. Clusters cells by AMICI interaction patterns, compares with composition-based niches, and performs grid search over hub parameters.

- **Figure 4f** (top): Spatial scatter plots colored by communication hub assignment
- **Figure 4f** (bottom): Alluvial/Sankey diagrams of hub sender-receiver composition
- **Fig S12a**: Silhouette score vs number of clusters
- **Fig S12b**: Alluvial/Sankey diagrams for all 10 communication hubs
- **Fig S13a**: Hub vs composition cluster spatial comparison
- **Fig S13b**: ARI/AMI comparison between hubs, composition clusters, and cell-type labels
- **Fig S27**: Hub grid search heatmaps — fixed k, varying quantile threshold
- **Fig S28**: Hub grid search heatmaps — fixed quantile, varying k

### [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py)

Gene set enrichment analysis comparing communication hubs to composition clusters. Supports GSEA (MSigDB Hallmark), KEGG Signaling, and Reactome pathway databases. Generates per-cell-type butterfly charts and combined figures split by tumor, immune, and stromal groups.

- **Fig S14**: GSEA barplots comparing hub-unique vs shared vs composition-unique significant pathways
- **Fig S15**: GSEA butterfly charts — tumor cell types (MSigDB Hallmark)
- **Fig S16**: GSEA butterfly charts — immune cell types (Reactome)
- **Fig S17**: GSEA butterfly charts — stromal cell types (Reactome)

### [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py)

Cross-replicate validation of AMICI interaction predictions.

- **Fig S5a**: Interaction strength heatmaps for replicate 1 and replicate 2
- **Fig S5b**: Replicate scatter plot with Spearman correlation

### [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py)

Cell segmentation artifact analysis. Validates that identified genes are not artifacts of segmentation errors using a Mann-Whitney U test.

- **Fig S6a**: Interaction network including stromal cells (showing segmentation artifact dominance)
- **Fig S6b**: Segmentation validation dot plot (Mann-Whitney U test for interaction-mediated genes)
- **Fig S10**: ESR1 segmentation overlap between invasive tumor and DCIS 2

### [`xenium/segmentation_analysis_gitiii.py`](xenium/segmentation_analysis_gitiii.py)

Segmentation artifact test applied to genes identified by GITIII for comparison.

- **Fig S11**: GITIII segmentation validation dot plot (Mann-Whitney U test)

### [`xenium/xenium_lr_analysis.py`](xenium/xenium_lr_analysis.py)

Ligand-receptor gene cross-reference analysis. Identifies significant downstream genes that overlap with the OmniPath LR database, filtered through the segmentation artifact test.

- **Fig S8**: LR gene dot plot per cell-type interaction

### [`xenium/xenium_null_z_distribution.py`](xenium/xenium_null_z_distribution.py)

Monte Carlo validation of the Wald test normality assumption using 50 randomly initialized models.

- **Fig S25**: Null z-value distribution histogram + Q-Q plot

### [`xenium/runtime_benchmark/plot_benchmark.py`](xenium/runtime_benchmark/plot_benchmark.py)

Runtime scaling visualization. Data generated by [`xenium/runtime_benchmark/run_benchmark.py`](xenium/runtime_benchmark/run_benchmark.py).

- **Fig S20**: Runtime benchmark (CPU wall-clock time + GPU seconds/epoch vs number of cells)

## Xenium Low-Resolution Analysis

Prerequisites:

1. [`xenium/xenium_preprocess_lowres.py`](xenium/xenium_preprocess_lowres.py) — Preprocesses the Xenium data with low-resolution (merged) cell-type labels.
2. [`xenium/xenium_sweep_main_lowres.py`](xenium/xenium_sweep_main_lowres.py) — Trains the AMICI model on low-resolution labels.
3. [`xenium/xenium_analysis_lowres.py`](xenium/xenium_analysis_lowres.py) — Compares interaction matrices between high-res and low-res models.

### [`xenium/xenium_analysis_lowres.py`](xenium/xenium_analysis_lowres.py)

Compares interaction strength matrices between the high-resolution model (aggregated to low-res labels) and a model trained directly on low-resolution labels. Tests significance via a permutation null.

- **Fig S18**: Low-resolution comparison matrices (high-res, aggregated, and low-res-trained)
- **Fig S19**: Interaction matrix similarity vs shuffled null (cosine similarity and Pearson r histograms)

## Data and Model Artifacts

Some of the analyses above depend on pretrained AMICI models and preprocessed datasets. The table below lists each artifact required to reproduce the figures.

### Cortex (MERFISH)

| Dataset              | Local Dataset Path                             | Local Model Path                                                          | Model YAML Config | Figshare Link |
| -------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- | ----------------- | ------------- |
| MERFISH Mouse Cortex | `cortex/data/cortex_processed_2025-04-28.h5ad` | `cortex/saved_models/cortex_18_sweep_plm73bmg_xrtcnlt0_params_2025-05-05` | [`reproducibility/cortex_config.yaml`](reproducibility/cortex_config.yaml) | TBA |

### Xenium Breast Cancer

| Dataset            | Local Dataset Path                                                        | Local Model Path                                                                   | Model YAML Config | Figshare Link |
| ------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------- | ------------- |
| Xenium Full        | `xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad`      | `xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02`      | [`reproducibility/xenium_full_config.yaml`](reproducibility/xenium_full_config.yaml) | TBA |
| Xenium Replicate 1 | `xenium/data/xenium_sample1/xenium_sample1_rep1_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep1_proseg_sweep_2025-05-01_model_2025-05-13` | [`reproducibility/xenium_rep1_config.yaml`](reproducibility/xenium_rep1_config.yaml) | TBA |
| Xenium Replicate 2 | `xenium/data/xenium_sample1/xenium_sample1_rep2_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep2_proseg_sweep_2025-05-01_model_2025-05-14` | [`reproducibility/xenium_rep2_config.yaml`](reproducibility/xenium_rep2_config.yaml) | TBA |
