# Reproducibility Guide

This document maps scripts and workflows in this repository to the figures they generate in the AMICI manuscript.

Figures 1, S1 are schematics/diagrams and were not generated from data scripts.

## Benchmarks

**Base data**: The semisynthetic datasets are generated from the [Fresh 68k PBMCs (Donor A)](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) dataset, downloaded programmatically via `scvi.data.dataset_10x(dataset_name="fresh_68k_pbmc_donor_a")`.

The full benchmark pipeline is orchestrated by [`benchmarks/Snakefile`](benchmarks/Snakefile) with configuration in [`benchmarks/benchmark_config.yaml`](benchmarks/benchmark_config.yaml) (main benchmark, Figure 2; loaded by default). The spatial cross-validation benchmark (Fig S35a) uses [`benchmarks/benchmark_config_cv.yaml`](benchmarks/benchmark_config_cv.yaml), run with `snakemake --configfile benchmark_config_cv.yaml` from `benchmarks/`. The pipeline:

1. Generates semisynthetic datasets ([`benchmarks/generate_dataset.py`](benchmarks/generate_dataset.py))
2. Computes ground truth scores ([`benchmarks/gene_task/generate_gt_gene_scores.py`](benchmarks/gene_task/generate_gt_gene_scores.py), [`benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py`](benchmarks/neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py), [`benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py`](benchmarks/receiver_subtype_task/generate_gt_receiver_subtype_scores.py))
3. Trains all models ([`benchmarks/train_amici_model.py`](benchmarks/train_amici_model.py), [`benchmarks/train_gitiii_model.py`](benchmarks/train_gitiii_model.py), [`benchmarks/train_ncem_model.py`](benchmarks/train_ncem_model.py), [`benchmarks/train_cgcom_model.py`](benchmarks/train_cgcom_model.py))
4. Scores each model on all tasks and generates PR curves ([`benchmarks/gene_task/generate_amici_scores.py`](benchmarks/gene_task/generate_amici_scores.py), [`benchmarks/gene_task/generate_amici_pr.py`](benchmarks/gene_task/generate_amici_pr.py), and analogous scripts per model/task)
5. Produces the final plots ([`benchmarks/gene_task/plot_boxplots.py`](benchmarks/gene_task/plot_boxplots.py), [`benchmarks/gene_task/plot_pr_curves.py`](benchmarks/gene_task/plot_pr_curves.py), [`benchmarks/neighbor_interaction_task/plot_boxplots.py`](benchmarks/neighbor_interaction_task/plot_boxplots.py), [`benchmarks/neighbor_interaction_task/plot_pr_curves.py`](benchmarks/neighbor_interaction_task/plot_pr_curves.py), [`benchmarks/receiver_subtype_task/plot_boxplots.py`](benchmarks/receiver_subtype_task/plot_boxplots.py), [`benchmarks/receiver_subtype_task/plot_pr_curves.py`](benchmarks/receiver_subtype_task/plot_pr_curves.py), [`benchmarks/length_scale_task/plot_kde.py`](benchmarks/length_scale_task/plot_kde.py))

The sensitivity and robustness scripts under [`benchmarks/sensitivity_scripts/`](benchmarks/sensitivity_scripts) (S21--S24, S26, S29--S33, S35, S37) are standalone: they generate their own datasets and train their own models internally, outside the Snakemake pipeline.

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

## Benchmark Sensitivity and Robustness Analyses

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

### [`benchmarks/sensitivity_scripts/amici_variants/`](benchmarks/sensitivity_scripts/amici_variants)

Comparison of the released AMICI attention against a unimodal-attention variant and an unconstrained positional-encoding variant, across 10 seeds of both semisynthetic datasets. Sweeps: [`current_amici_sweep.py`](benchmarks/sensitivity_scripts/amici_variants/current_amici_sweep.py), [`unimodal_attention_sweep.py`](benchmarks/sensitivity_scripts/amici_variants/unimodal_attention_sweep.py), [`unconstrained_attention_sweep.py`](benchmarks/sensitivity_scripts/amici_variants/unconstrained_attention_sweep.py). Plots: [`plot_amici_variant_comparison.py`](benchmarks/sensitivity_scripts/amici_variants/plot_amici_variant_comparison.py).

- **Fig S29a–b**: Learned unimodal peak parameter and attention-drop distance vs ground truth
- **Fig S29c**: Length scales from the unconstrained positional-encoding variant
- **Fig S29d**: AUPRC comparison of the three variants on all three tasks

### Length scale uncertainty (Fig S30)

- **Fig S30a**: [`benchmarks/sensitivity_scripts/length_scale_uncertainty.py`](benchmarks/sensitivity_scripts/length_scale_uncertainty.py) — bootstrap over resampled sender cells within one trained model
- **Fig S30b**: [`benchmarks/sensitivity_scripts/length_scale_dataset_bootstrap_ci.py`](benchmarks/sensitivity_scripts/length_scale_dataset_bootstrap_ci.py) — 50 bootstrapped replicates of the PBMC semisynthetic dataset
- **Fig S30c–d**: [`benchmarks/sensitivity_scripts/realistic_length_scale_dataset_bootstrap_ci.py`](benchmarks/sensitivity_scripts/realistic_length_scale_dataset_bootstrap_ci.py) — the same over the realistic semisynthetic dataset (c), plus that dataset's cell-type composition (d)
- **Fig S30e**: [`benchmarks/sensitivity_scripts/length_scale_coordinate_shuffle_sensitivity.py`](benchmarks/sensitivity_scripts/length_scale_coordinate_shuffle_sensitivity.py) — within-cell-type coordinate shuffling at 30%, 50% and 100%
- **Fig S30f**: [`xenium/xenium_length_scale_seed_sweep.py`](xenium/xenium_length_scale_seed_sweep.py) trains 20 Xenium models across seeds; [`xenium/xenium_length_scale_seed_sweep_reanalysis.py`](xenium/xenium_length_scale_seed_sweep_reanalysis.py) sweeps every head of those cached models and produces the plotted panel
- **Fig S30g**: [`human_tonsil/human_tonsil_length_scale_seed_sweep.py`](human_tonsil/human_tonsil_length_scale_seed_sweep.py) and [`human_tonsil/human_tonsil_length_scale_seed_sweep_reanalysis.py`](human_tonsil/human_tonsil_length_scale_seed_sweep_reanalysis.py) — the same for the tonsil dataset

### Multi-scale and unimodal interaction recovery (Fig S31)

- **Fig S31a–c**: [`benchmarks/sensitivity_scripts/2_phase/`](benchmarks/sensitivity_scripts/2_phase) — [`two_phase_amici_sweep.py`](benchmarks/sensitivity_scripts/2_phase/two_phase_amici_sweep.py) (dataset, training, AUPRCs), [`two_phase_head_attention_summary.py`](benchmarks/sensitivity_scripts/2_phase/two_phase_head_attention_summary.py) and [`two_phase_selected_head_length_scales.py`](benchmarks/sensitivity_scripts/2_phase/two_phase_selected_head_length_scales.py) (per-head length scales)
- **Fig S31d–f**: [`benchmarks/sensitivity_scripts/unimodal_two_phase/`](benchmarks/sensitivity_scripts/unimodal_two_phase) — [`unimodal_two_phase_amici_sweep.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_amici_sweep.py), [`unimodal_two_phase_baseline_sweep.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_baseline_sweep.py) (GITIII/CGCom via [`snakemake_shim.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/snakemake_shim.py)), [`unimodal_two_phase_task_error_analysis.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_task_error_analysis.py) (e), [`unimodal_two_phase_head_attention_summary.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_head_attention_summary.py) and [`unimodal_two_phase_selected_head_length_scales.py`](benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_selected_head_length_scales.py) (f)
- **Fig S31g–i**: Model-free distance-binned expression profiles — [`cortex/cortex_empirical_distance_expression.py`](cortex/cortex_empirical_distance_expression.py) (g), [`atera_breast/atera_breast_empirical_distance_expression.py`](atera_breast/atera_breast_empirical_distance_expression.py) (h), [`xenium/xenium_empirical_distance_expression.py`](xenium/xenium_empirical_distance_expression.py) (i)

### [`benchmarks/sensitivity_scripts/neighbor_occlusion_analysis.py`](benchmarks/sensitivity_scripts/neighbor_occlusion_analysis.py), [`benchmarks/sensitivity_scripts/neutral_sampling_negative_control.py`](benchmarks/sensitivity_scripts/neutral_sampling_negative_control.py)

Neighborhood occlusion and density-matched negative controls.

- **Fig S32a**: AUPRC and PR curves under increasing neighbor occlusion (both semisynthetic datasets)
- **Fig S32b–d**: AUPRC curves for the grid-based (b), realistic three-cell-type (c) and realistic breast cancer (d) density-matched negative controls

### [`benchmarks/sensitivity_scripts/matched_neighbor_occlusion_analysis.py`](benchmarks/sensitivity_scripts/matched_neighbor_occlusion_analysis.py)

Occlusion of high-attention neighbors against distance- and cell-type-matched low-attention neighbors.

- **Fig S33**: Matched occlusion (reconstruction error by arm, paired differences, occluded-neighbor distances)

### [`benchmarks/sensitivity_scripts/test_region_composition_diagnostic.py`](benchmarks/sensitivity_scripts/test_region_composition_diagnostic.py)

Cell-type composition of the held-out spatial test region against the full tissue, for the Xenium, Atera breast and cortex datasets.

- **Fig S35b**: Jensen-Shannon divergence per dataset
- **Fig S35c**: Largest per-cell-type composition shifts

Fig S35a comes from the Snakemake pipeline run with [`benchmarks/benchmark_config_cv.yaml`](benchmarks/benchmark_config_cv.yaml) (spatial cross-validation on the realistic breast cancer dataset).

### [`benchmarks/sensitivity_scripts/coordinate_noise_sensitivity.py`](benchmarks/sensitivity_scripts/coordinate_noise_sensitivity.py)

AMICI performance under an x-axis gradient of 2D Gaussian coordinate noise, for both semisynthetic datasets.

- **Fig S37**: AUPRC for all three tasks vs coordinate noise level

### [`benchmarks/sensitivity_scripts/family_wide_bh_interaction_networks.py`](benchmarks/sensitivity_scripts/family_wide_bh_interaction_networks.py), [`benchmarks/sensitivity_scripts/spatial_block_jackknife_gene_significance.py`](benchmarks/sensitivity_scripts/spatial_block_jackknife_gene_significance.py)

Robustness of downstream-gene significance to the multiple-testing family and to spatial dependence, for the Xenium, Atera breast and CosMx tonsil datasets.

- **Fig S39a, c, e**: Per-pair vs family-wide Benjamini-Hochberg interaction matrices (`family_wide_bh_interaction_networks.py`)
- **Fig S39b, d, f**: Fraction of significant genes retained under the spatial block-jackknife effective sample size (`spatial_block_jackknife_gene_significance.py`)

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
3. Analysis and plotting scripts (each loads the trained model independently): [`xenium/xenium_analysis.py`](xenium/xenium_analysis.py), [`xenium/xenium_spatial_analysis.py`](xenium/xenium_spatial_analysis.py), [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py), [`xenium/xenium_niche_prediction_analysis.py`](xenium/xenium_niche_prediction_analysis.py), [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py), [`xenium/xenium_hub_analysis.py`](xenium/xenium_hub_analysis.py), [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py), [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py), [`xenium/segmentation_analysis_gitiii.py`](xenium/segmentation_analysis_gitiii.py), [`xenium/xenium_stromal_bleeding_analysis.py`](xenium/xenium_stromal_bleeding_analysis.py), [`xenium/xenium_lr_analysis.py`](xenium/xenium_lr_analysis.py), [`xenium/xenium_null_z_distribution.py`](xenium/xenium_null_z_distribution.py), [`benchmarks/sensitivity_scripts/xenium_adjusted_cd8_tumor_effect.py`](benchmarks/sensitivity_scripts/xenium_adjusted_cd8_tumor_effect.py), [`xenium/runtime_benchmark/plot_benchmark.py`](xenium/runtime_benchmark/plot_benchmark.py).

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

### [`benchmarks/sensitivity_scripts/xenium_adjusted_cd8_tumor_effect.py`](benchmarks/sensitivity_scripts/xenium_adjusted_cd8_tumor_effect.py)

Covariate-adjusted ESR1 and AGR3 expression in invasive tumor cells of replicate 1, removing fitted effects of local tumor purity, proliferation and spatial region while retaining the CD8 attention term.

- **Fig S10**: Raw, adjusted and difference spatial maps for ESR1 (top) and AGR3 (bottom)

### [`xenium/xenium_niche_analysis.py`](xenium/xenium_niche_analysis.py)

Communication hub analysis. Clusters cells by AMICI interaction patterns and performs grid search over hub parameters.

- **Figure 4f** (top): Spatial scatter plots colored by communication hub assignment
- **Figure 4f** (bottom): Alluvial/Sankey diagrams of hub sender-receiver composition
- **Fig S13a**: Silhouette score vs number of clusters
- **Fig S13b**: Alluvial/Sankey diagrams for all 10 communication hubs
- **Fig S27**: Hub grid search heatmaps — fixed k, varying quantile threshold
- **Fig S28**: Hub grid search heatmaps — fixed quantile, varying k

### [`xenium/xenium_niche_prediction_analysis.py`](xenium/xenium_niche_prediction_analysis.py)

Comparison of communication hubs against composition-based niches.

- **Fig S14a**: Hub vs composition cluster spatial comparison
- **Fig S14b**: ARI/AMI comparison between hubs, composition clusters, and cell-type labels

### [`xenium/xenium_niche_validation_gsea.py`](xenium/xenium_niche_validation_gsea.py), [`xenium/xenium_hub_analysis.py`](xenium/xenium_hub_analysis.py)

Gene set enrichment analysis comparing communication hubs to composition clusters and to graph-based baselines. `xenium_niche_validation_gsea.py` supports GSEA (MSigDB Hallmark), KEGG Signaling and Reactome pathway databases and generates the per-cell-type butterfly charts; `xenium_hub_analysis.py` (with [`xenium/hub_graph_baseline_utils.py`](xenium/hub_graph_baseline_utils.py)) adds the fixed-radius and kNN graph baselines.

- **Fig S15a**: GSEA barplots comparing hub-unique vs shared vs composition-unique significant pathways
- **Fig S15b–c**: ARI/AMI against graph baselines, and pathway overlap across methods
- **Fig S16**: GSEA butterfly charts — tumor cell types (MSigDB Hallmark)
- **Fig S17**: GSEA butterfly charts — immune cell types (Reactome)
- **Fig S18**: GSEA butterfly charts — stromal cell types (Reactome)

### [`xenium/xenium_replicate_validation.py`](xenium/xenium_replicate_validation.py)

Cross-replicate validation of AMICI interaction predictions.

- **Fig S5a**: Interaction strength heatmaps for replicate 1 and replicate 2
- **Fig S5b**: Replicate scatter plot with Spearman correlation

### [`xenium/segmentation_analysis.py`](xenium/segmentation_analysis.py)

Cell segmentation artifact analysis. Validates that identified genes are not artifacts of segmentation errors using a Mann-Whitney U test.

- **Fig S6a**: Interaction network including stromal cells (showing segmentation artifact dominance)
- **Fig S6b**: Segmentation validation dot plot (Mann-Whitney U test for interaction-mediated genes)
- **Fig S11**: ESR1 segmentation overlap between invasive tumor and DCIS 2

### [`xenium/segmentation_analysis_gitiii.py`](xenium/segmentation_analysis_gitiii.py)

Segmentation artifact test applied to genes identified by GITIII for comparison.

- **Fig S12**: GITIII segmentation validation dot plot (Mann-Whitney U test)

### [`xenium/xenium_stromal_bleeding_analysis.py`](xenium/xenium_stromal_bleeding_analysis.py)

Quantifies how stromal interactions influence the inferred network, comparing models trained with and without stromal cells and testing stromal attention against a distance-matched null.

- **Fig S34a**: Interaction score heatmaps with and without stromal cells
- **Fig S34b**: Score differences and pairwise correlation before vs after stromal exclusion
- **Fig S34c**: Distance-matched null comparison of stromal attention

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
4. [`xenium/xenium_within_type_autocorrelation.py`](xenium/xenium_within_type_autocorrelation.py) — Computes the within-label spatial autocorrelation diagnostic.
5. [`xenium/make_supp_lowres_figure.py`](xenium/make_supp_lowres_figure.py) — Assembles the full supplementary figure from the cached results of steps 3 and 4.

The low-resolution model is a refit at the hyperparameters of wandb sweep `xsjrwnof` run `5tfi78lp` (see [`reproducibility/xenium_lowres_config.yaml`](reproducibility/xenium_lowres_config.yaml)); the original run's weights were lost, so the model was retrained on 2026-09-19 with the same config.

### [`xenium/xenium_analysis_lowres.py`](xenium/xenium_analysis_lowres.py)

Compares interaction strength matrices between the high-resolution model (aggregated to low-res labels) and a model trained directly on low-resolution labels. Tests significance via a permutation null.

- **Fig S19a–c**: High-resolution, aggregated, and low-resolution-trained interaction matrices
- **Fig S19d**: Interaction matrix similarity vs shuffled null (cosine similarity and Pearson r histograms)

### [`xenium/xenium_within_type_autocorrelation.py`](xenium/xenium_within_type_autocorrelation.py)

Within-label spatial autocorrelation (Moran's I of expression PCs within each coarse label) against a high-resolution subtype oracle.

- **Fig S19e**: Deployable vs oracle autocorrelation score per coarse cell type

## Atera Breast Cancer Analysis

**Base data**: Atera whole-transcriptome spatial profiling of breast cancer. The preprocessed dataset and trained model are available on Figshare (see [Data and Model Artifacts](#data-and-model-artifacts)).

Prerequisites (run in order):

1. [`atera_breast/atera_breast_preprocess.py`](atera_breast/atera_breast_preprocess.py) — Loads and filters the Atera data, normalizes counts, and performs a spatial train/test split.
2. [`atera_breast/atera_breast_train.py`](atera_breast/atera_breast_train.py) — Trains the AMICI model. Sweep config: [`atera_breast/atera_breast_sweep.yaml`](atera_breast/atera_breast_sweep.yaml), launched via [`atera_breast/run_atera_breast_sweep.sh`](atera_breast/run_atera_breast_sweep.sh).
3. [`atera_breast/atera_breast_analysis.py`](atera_breast/atera_breast_analysis.py) — Loads the trained model and generates the Atera figures.

### [`atera_breast/atera_breast_analysis.py`](atera_breast/atera_breast_analysis.py)

Primary Atera breast cancer analysis. Generates the spatial overview, interaction heatmap, downstream gene dot plots, and the CAF subpopulation analysis.

- **Figure 5a**: Spatial scatter plot colored by cell type with the held-out test region boxed (`visualize_spatial_distribution()`)
- **Figure 5b**: Interaction strength heatmap for all cell-type pairs
- **Figure 5c**: Downstream gene dot plots for stromal, immune and tumor interactions
- **Fig S38a–b**: CAF subclustering (UMAP, EMILIN1/C3 expression, spatial distribution)
- **Fig S38c**: Log fold change for the top DEGs between high-attention senders of each CAF population

### [`atera_breast/atera_breast_segmentation_analysis.py`](atera_breast/atera_breast_segmentation_analysis.py)

Segmentation artifact test (one-sided Mann-Whitney U) for the Atera dataset. Determines which genes are bolded in Figure 5c.

### [`atera_breast/atera_breast_attention_consistency.py`](atera_breast/atera_breast_attention_consistency.py)

Stability of attention scores across models trained over varying hyperparameters and seeds.

- **Fig S36a**: Test reconstruction loss vs mean Spearman correlation across runs, and the pairwise correlation matrix

## Human Tonsil (CosMx) Analysis

**Base data**: CosMx 1000-gene panel human tonsil. The preprocessed dataset and trained model are available on Figshare (see [Data and Model Artifacts](#data-and-model-artifacts)).

Prerequisites (run in order):

1. [`human_tonsil/human_tonsil_preprocess.py`](human_tonsil/human_tonsil_preprocess.py) — Filters unlabeled annotations, normalizes counts, and performs a spatial train/test split.
2. [`human_tonsil/human_tonsil_train.py`](human_tonsil/human_tonsil_train.py) — Trains the AMICI model. Sweep config: [`human_tonsil/human_tonsil_sweep.yaml`](human_tonsil/human_tonsil_sweep.yaml), launched via [`human_tonsil/run_human_tonsil_sweep.sh`](human_tonsil/run_human_tonsil_sweep.sh).
3. [`human_tonsil/human_tonsil_analysis.py`](human_tonsil/human_tonsil_analysis.py) — Loads the trained model and generates the tonsil figures.

### [`human_tonsil/human_tonsil_analysis.py`](human_tonsil/human_tonsil_analysis.py)

Primary human tonsil analysis. Generates the spatial overview, interaction network and heatmaps, downstream gene dot plots, and length scale distributions (converted from CosMx pixels to micrometers).

- **Figure 6a**: Spatial scatter plot colored by cell type with the held-out test region boxed (`visualize_spatial_distribution()`)
- **Figure 6b**: Interaction network for cell types of interest, and the germinal center interaction heatmap
- **Figure 6c**: Downstream gene dot plots for stromal and immune interactions
- **Figure 6d**: Length scale distributions for GC B cell receivers per attention head

### [`human_tonsil/human_tonsil_attention_consistency.py`](human_tonsil/human_tonsil_attention_consistency.py)

Stability of attention scores across models trained over varying hyperparameters and seeds.

- **Fig S36b**: Test reconstruction loss vs mean Spearman correlation across runs, and the pairwise correlation matrix

## Data and Model Artifacts

Some of the analyses above depend on pretrained AMICI models and preprocessed datasets. The table below lists each artifact required to reproduce the figures. All datasets and models are available on [Figshare](https://doi.org/10.6084/m9.figshare.31927956).

### Cortex (MERFISH)

| Dataset              | Local Dataset Path                             | Local Model Path                                                          | Model YAML Config |
| -------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- | ----------------- |
| MERFISH Mouse Cortex | `cortex/data/cortex_processed_2025-04-28.h5ad` | `cortex/saved_models/cortex_sweep_2025-04-28_model_2025-05-05/cortex_18_sweep_plm73bmg_xrtcnlt0_params_2025-05-05` | [`reproducibility/cortex_config.yaml`](reproducibility/cortex_config.yaml) |

### Xenium Breast Cancer

| Dataset            | Local Dataset Path                                                        | Local Model Path                                                                   | Model YAML Config |
| ------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------- |
| Xenium Full        | `xenium/data/xenium_sample1_filtered_2025-05-01.h5ad`      | `xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02/xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02` | [`reproducibility/xenium_full_config.yaml`](reproducibility/xenium_full_config.yaml) |
| Xenium Replicate 1 | `xenium/data/xenium_sample1_rep1_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep1_proseg_sweep_2025-05-01_model_2025-05-13/xenium_42_sweep_4jrcb6jd_6xyu2ted_params_2025-05-13` | [`reproducibility/xenium_rep1_config.yaml`](reproducibility/xenium_rep1_config.yaml) |
| Xenium Replicate 2 | `xenium/data/xenium_sample1_rep2_filtered_2025-05-01.h5ad` | `xenium/saved_models/xenium_sample1_rep2_proseg_sweep_2025-05-01_model_2025-05-14/xenium_22_sweep_pwyd8qid_8h73cxui_params_2025-05-14` | [`reproducibility/xenium_rep2_config.yaml`](reproducibility/xenium_rep2_config.yaml) |
| Xenium Low-Res     | `xenium/data/xenium_sample1_filtered_lowres_2026-03-27.h5ad` | `xenium/saved_models/xenium_sample1_lowres_sweep_2026-03-27_model_2026-09-19/xenium_42_retrain_5tfi78lp_params_2026-09-19` | [`reproducibility/xenium_lowres_config.yaml`](reproducibility/xenium_lowres_config.yaml) |

### Human Tonsil (CosMx)

| Dataset      | Local Dataset Path                                              | Local Model Path                                                                    | Model YAML Config |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------- |
| Human Tonsil | `human_tonsil/data/human_tonsil_filtered_2026-07-25.h5ad` (train/test splits: `human_tonsil_filtered_train_2026-07-25.h5ad` / `human_tonsil_filtered_test_2026-07-25.h5ad`) | `human_tonsil/saved_models/human_tonsil_sweep_2026-07-25_model_2026-07-26/human_tonsil_40_sweep_0f9rk0na_4303v0zu_params_2026-07-26` | [`reproducibility/human_tonsil_config.yaml`](reproducibility/human_tonsil_config.yaml) |

### Atera Breast Cancer

| Dataset      | Local Dataset Path                                               | Local Model Path                                                                    | Model YAML Config |
| ------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------- |
| Atera Breast | `atera_breast/data/atera_breast_filtered_2026-07-22.h5ad` (train/test splits: `atera_breast_filtered_train_2026-07-22.h5ad` / `atera_breast_filtered_test_2026-07-22.h5ad`) | `atera_breast/saved_models/atera_breast_sweep_2026-07-22_model_2026-07-28/atera_breast_33_sweep_25xpxkuk_zhtraubt_params_2026-07-28` | [`reproducibility/atera_breast_config.yaml`](reproducibility/atera_breast_config.yaml) |
