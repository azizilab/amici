# Source Data for the AMICI manuscript

This folder rebuilds the Source Data workbooks for the main figures from the exact datasets and AMICI
checkpoints that produced them. Each exporter verifies the dataset and `model.pt` by md5, recomputes every
plotted quantity, checks it against the published panel, and writes per-panel CSVs. `build_workbooks.py`
then assembles one workbook per figure.

| Output | Contents |
| --- | --- |
| `output/Source_Data_Fig3.xlsx` | Fig. 3a-c (MERFISH cortex) |
| `output/Source_Data_Fig4.xlsx` + `Source_Data_Fig4_large_tables.zip` | Fig. 4a-f (Xenium breast cancer); the per-cell tables of 4a, 4d (length scales) and 4f (hubs) are CSVs in the zip |
| `output/Source_Data_Fig5.xlsx` + `Source_Data_Fig5_large_tables.zip` | Fig. 5a-c (Atera breast cancer); the 5a cell map is a CSV in the zip |
| `output/Source_Data_Fig6.xlsx` + `Source_Data_Fig6_large_tables.zip` | Fig. 6a-d (CosMx tonsil); the 6a cell map is a CSV in the zip |
| `output/Source_Data_Fig2.xlsx` | Fig. 2a-j (semisynthetic benchmarks), see below |
| `output/figures/Fig2g_rerun.{pdf,svg,png}` | Redrawn Fig. 2g, made from the Fig2g sheet by `plot_fig2g.py` |

Tables longer than 150,000 rows go into the companion zip so each workbook stays a few MB.
Every workbook opens with a `README` sheet giving the dataset and checkpoint provenance (paths, md5, W&B
sweep/run, training config, Figshare and bucket locations) and a description of each panel sheet with units.
Figure 1 is a schematic and has no Source Data.

## Reproducing

```bash
conda activate amici            # /opt/conda/envs/amici on the AMICI VM
python source_data/fig3_cortex.py
python source_data/fig4_xenium.py
python source_data/fig5_atera.py
python source_data/fig6_tonsil.py
python -I source_data/fig2_benchmarks.py --results benchmarks/results_fig2_original --example-seed 72  # see Figure 2 below
python source_data/build_workbooks.py
```

Datasets and checkpoints are listed with md5 sums and remote copies in [`checkpoints.yaml`](checkpoints.yaml).
The exporters reuse the cached all-cell-type ablation scores when present (they recompute them otherwise),
and independently recompute every dot-plot receiver from the checkpoint; in all four datasets the fresh values
equal the cached ones exactly. Run time on one A100 is roughly 5 min (Fig. 3, first run computes the cortex
cache), 10 min (Figs. 5 and 6) and 20 min (Fig. 4).

## Checkpoint per figure

| Figure | Dataset | Checkpoint (W&B sweep / run) | model.pt md5 | Remote copy |
| --- | --- | --- | --- | --- |
| 2 | semisynthetic PBMC (10 seeds) and realistic breast cancer | per-seed `amici_*_best_model` from the original Snakemake run | in `fig2_manifest.txt` | gs://justin-amici/semisyn/fig2_amici_results/ (PBMC only) |
| 3 | `cortex_processed_2025-04-28.h5ad` | `plm73bmg` / `xrtcnlt0` | `76eadfaa...` | Figshare 63409695; bucket |
| 4 | `xenium_sample1_filtered_2025-05-01.h5ad` | `g3mucw4s` / `te7pkv3z` | `c118e838...` | Figshare 63409710; bucket |
| 5 | `atera_breast_filtered_2026-07-22.h5ad` | `25xpxkuk` / `zhtraubt` | `f53e6ab5...` | local only |
| 6 | `human_tonsil_filtered_2026-07-25.h5ad` | `0f9rk0na` / `4303v0zu` | `e12cc392...` | local only |

## How each panel was checked against the manuscript

| Panel | Check | Result |
| --- | --- | --- |
| 3b | edges of the 5-node subgraph at the 0.86 quantile (threshold 25 genes) | 11 of 12 match the drawn graph |
| 3c | gene selection for Astrocytes (n_top 8, min z 15) and Sst (n_top 10, min z 5) | exact |
| 4c | published genes vs the ranked top genes per sender | published genes are the top-ranked genes in order, truncated; TIGIT (rank 6, CD4 -> CD8) is omitted |
| 4c | bold genes vs the segmentation test | 18/19 reproduced; C15orf48 differs |
| 4d | violin rows and their ascending-median order | exact (13 rows) |
| 4e | head with the largest attention variance | head 2, as published |
| 4f | hub labels vs `hub_analysis_cache/hub_cluster_assignments.csv` | identical (ARI 1.0) |
| 5b | Ward-clustered row and column order | exact |
| 5c | gene selection (n_top 8, min z -10) | exact for all three receivers |
| 5c | bold genes vs the segmentation test | 12/30 reproduced |
| 6b | Ward-clustered GC heatmap order | exact |
| 6c | gene selection (n_top 10, min z -10) | exact for both receivers |
| 6d | head 3 (n_top 10) and head 6 (n_top 5) gene selection; length scales vs the cached CSV | exact; length scales equal to 2e-13 um |

## Figure 2

The original benchmark run was found in `/home/khushidesai/amici_results` on khushi-gpu-migrated (the Spatial
HCM connection). Its Fig. 2 files (per-seed AUPRC CSVs for every method, the ten per-seed AMICI models and
best-seed files, the length-scale plots and the boxplots; 288 files, 342 MiB) were copied to
gs://justin-amici/semisyn/fig2_amici_results/ with an md5 manifest (`fig2_manifest.txt`) and downloaded to
`benchmarks/results_fig2_original/` (gitignored); every file matches the manifest.

- **Fig. 2c-e** come straight from those AUPRC files. All ten PBMC seeds are complete, and the
  Bonferroni-corrected two-sided Mann-Whitney tests reproduce every p-value in the caption exactly (0.0013,
  0.0013, 0.0023 and 3 x 0.0013 on the gene task; 0.028 and 0.0003 for senders; 0.028 and 0.0004 for receivers).
  The `Fig2c_stats`, `Fig2d_stats` and `Fig2e_stats` sheets also carry, for each of the eleven AMICI-vs-baseline
  comparisons, the Mann-Whitney `U` statistic of the AMICI sample (`U_max_n1n2` = n1 * n2 = 100) and the two
  effect sizes derived from it, rank-biserial r = 2U/(n1*n2) - 1 and the common-language P(AMICI > baseline)
  = U/(n1*n2); the non-significant GITIII gene-task comparison (U = 54) is listed alongside the rest. These are
  the numbers tabulated in [`legend_numbers.md`](legend_numbers.md).
- **Fig. 2b** is seed 72, the only seed whose length-scale plot has the short-range A -> B shoulder of the
  published panel. Its sheet holds the KDE curves read exactly from that run's `length_scale_distribution.svg`
  (modes 10.9 um for A -> B and 20.6 um for C -> A), because the per-cell values cannot be recomputed (next point).
- **Fig. 2a** is rebuilt from two files the run itself kept for seed 72: GITIII's processed input
  (`saved_models/data/processed/slide1.csv`, every cell's coordinates and type) and
  `ground_truth_receiver_subtypes.csv` (interacting receivers), both in dataset order; all 2,896 flagged cells are
  receiver types A or B and none of the 5,084 type C cells is, which confirms the alignment. The run's `.h5ad`
  inputs were not kept, and `benchmarks/data/3ct_dataset_2way_*.h5ad` on the AMICI VM are later regenerations (the
  original seed-72 model was trained on 18,033 cells and a gene list sharing only 495 of 500 genes with the local
  file), so they are not used.
- **Fig. 2f-j.** The outputs of the initial realistic breast cancer run were not kept, so a second AMICI sweep was
  run to get a best model for the benchmark, with `reproducibility/benchmarks/fig2_realistic_benchmark_config.yaml`
  on a byte-identical copy of `data/breast_cancer_42.h5ad` (md5 1e185c0bddb7795d30e8729d289ad8d3). That config is
  the committed April config (commit 3a3cc16) with the extra attention heads and seeds that an April-2026 message
  records, and the best model is the one with the lowest test loss (14 heads, seed 42). Re-run outputs:
  AMICI's 30 runs in `benchmarks/results_realistic_rerun_april_family/` (the lowest-test-loss model, 14 heads and
  seed 42, is scored in `scores/run_329/`; that model, its scores and the sweep logs are also in
  `gs://justin-amici/semisyn/fig2_realistic_april_sweep/`); GITIII at 3a3cc16 in
  `../amici-3a3cc16/benchmarks/results/`; NicheDE in `benchmarks/results_realistic_rerun/`; NCEM in
  `benchmarks/results_cv/`. NicheDE and NCEM are deterministic and
  reproduce the published bars on this dataset (NicheDE sigma = 20: 0.2408 vs 0.2402; NCEM 0.2505 / 0.2441 / 0.2486
  vs 0.2498 / 0.2434 / 0.2479), which identifies it as the published dataset. CGCom was not re-run.
  - **2f** (dataset map) is exported from that dataset.
  - **2h-j are kept as published.** Their Source Data are the published bar heights, which were read off the bars
    of a vector copy of the figure on 2026-07-28 (`benchmarks/results_cv/breast_cancer/regular_vs_cv_barplots/
    regular_recovered_auprc_values.csv`; the vector copy itself is gone) and sit about 0.0007 below the values the
    deterministic baselines reproduce. Each sheet also gives the re-run AUPRC, the difference and the re-run's
    output folder. The re-run gives AMICI 0.975 / 0.513 / 0.944 (gene / sender / receiver) against 0.986 / 0.522 /
    0.928 published, and GITIII 0.933 / 0.410 / 0.949 against 0.929 / 0.391 / 0.950.
  - **2g is redrawn** from the re-run's selected AMICI model (`output/figures/Fig2g_rerun.*`), because no data for
    the published curves survive. The redrawn curves have the published shapes and
    order; their peaks sit at 41.6 / 27.1 / 17.1 um (Macrophages -> DCIS, T cells -> Endothelial, Invasive tumor ->
    Myoepi), against about 44 / 30 / 19 um read off the published panel and 40 / 25 / 15 um ground truth.

Export with
`python -I source_data/fig2_benchmarks.py --results benchmarks/results_fig2_original --realistic-results benchmarks/results_realistic_rerun_april_family/scores/run_329 --example-seed 72`
plus one `--rerun-source METHOD=PATH` per method (the folders above; `CGCom=` marks it as not re-run), then
`python source_data/plot_fig2g.py` and `python source_data/build_workbooks.py Fig2`.

## For the submission

Inventory of Supporting Information (one line per file):

- Source Data Fig. 2 | Source_Data_Fig2.xlsx | Numerical data underlying Fig. 2a-j
- Source Data Fig. 3 | Source_Data_Fig3.xlsx | Numerical data underlying Fig. 3a-c
- Source Data Fig. 4 | Source_Data_Fig4.xlsx, Source_Data_Fig4_large_tables.zip | Numerical data underlying Fig. 4a-f
- Source Data Fig. 5 | Source_Data_Fig5.xlsx, Source_Data_Fig5_large_tables.zip | Numerical data underlying Fig. 5a-c
- Source Data Fig. 6 | Source_Data_Fig6.xlsx, Source_Data_Fig6_large_tables.zip | Numerical data underlying Fig. 6a-d
- Source Data ED Fig. 1 | Source_Data_ED1.xlsx | Numerical data underlying Extended Data Fig. 1c, f-i
- Source Data ED Fig. 2 | Source_Data_ED2.xlsx | Numerical data underlying Extended Data Fig. 2a, g, h
- Source Data ED Fig. 3 | Source_Data_ED3.xlsx, Source_Data_ED3_large_tables.zip | Numerical data underlying Extended Data Fig. 3b-j
- Source Data ED Fig. 4 | Source_Data_ED4.xlsx | Numerical data underlying Extended Data Fig. 4b-h
- Source Data ED Fig. 5 | Source_Data_ED5.xlsx, Source_Data_ED5_large_tables.zip | Numerical data underlying Extended Data Fig. 5c, d, h-j
- Source Data ED Fig. 6 | Source_Data_ED6.xlsx | Numerical data underlying Extended Data Fig. 6a-d
- Source Data ED Fig. 7 | Source_Data_ED7.xlsx | Numerical data underlying Extended Data Fig. 7b, c, e-g, i-k
- Source Data ED Fig. 8 | Source_Data_ED8.xlsx | Numerical data underlying Extended Data Fig. 8a-e, g
- Source Data ED Fig. 9 | Source_Data_ED9.xlsx | Numerical data underlying Extended Data Fig. 9a-i

Extended Data panels with no numerical graph, and so no Source Data: 1a, 1b, 2e, 2f, 4a, 5a, 5b, 7d, 7h
(architecture schematics and spatial maps).

Extended Data panels whose numbers were never written to disk, and which therefore need their analysis
rerun before they can be exported: 1d, 1e, 2b-d, 3a, 5e-g, 6e-i, 7a, 8f, 8h, 8i. Of these, 6e-h (the
hyperparameter sweeps) have no saved models on the VM or in gs://justin-amici, so regenerating them means
retraining, and the retrained models would not be the ones behind the published panels.

Run `python source_data/extended_data.py --survey` for the current coverage and
`--export` to rewrite the Extended Data panel CSVs.

Data Availability: "Source data are provided with this paper."
