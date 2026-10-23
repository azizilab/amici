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
| 3b | edges of the 5-node subgraph at the 0.86 quantile (threshold 25 genes) | 11 of 12 match the drawn graph (see issues) |
| 3c | gene selection for Astrocytes (n_top 8, min z 15) and Sst (n_top 10, min z 5) | exact |
| 4c | published genes vs the ranked top genes per sender | published genes are the top-ranked genes in order, truncated; TIGIT (rank 6, CD4 -> CD8) is omitted |
| 4c | bold genes vs the segmentation test | 18/19 reproduced; C15orf48 differs (see issues) |
| 4d | violin rows and their ascending-median order | exact (13 rows) |
| 4e | head with the largest attention variance | head 2, as published |
| 4f | hub labels vs `hub_analysis_cache/hub_cluster_assignments.csv` | identical (ARI 1.0) |
| 5b | Ward-clustered row and column order | exact |
| 5c | gene selection (n_top 8, min z -10) | exact for all three receivers |
| 5c | bold genes vs the segmentation test | 12/30 reproduced (see issues) |
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
  vs 0.2498 / 0.2434 / 0.2479), which identifies it as the published dataset. CGCom was not re-run (issue 2).
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

## Issues found while rebuilding the panels

These do not change the Source Data values, but should be resolved or acknowledged before submission.

1. **Fig. 2h NicheDE sigma = 200 and 500 do not reproduce.** NicheDE is deterministic, the installed NicheDE
   (GitHub commit 87e0e89, installed 2026-03-09) predates the realistic dataset, and both the re-run and the June
   CV run give 0.803 / 0.752 on this dataset, against 0.732 / 0.673 published. The settings that produced the
   published bars are not in the repository. `generate_nichede_scores.R` also installs NicheDE and sceasy from
   GitHub at run time without a pinned commit.
2. **CGCom attention overflows on the realistic dataset.** Its attention scores there range from about 4,600 to
   infinity (they should lie in [0, 1]; on the PBMC datasets they do), in both the re-run and the June CV run, so
   the CGCom bars in Fig. 2i-j rest on overflowed scores. A plotting step added to `train_cgcom_model.py` in June
   2026 crashed on these values; `benchmark_utils.plot_interaction_graph` now skips non-finite edges (figure only).
3. **Fig. 2c-e use 9 distinct datasets, not 10.** In the original run, seeds 58 and 123 give identical AUPRCs
   for every deterministic baseline (all NicheDE and NCEM settings), and the regenerated seed-58 and seed-123
   files on the AMICI VM are byte-identical, so the generator produced the same dataset for both seeds. NCEM
   with radius 15 and radius 20 also gives identical AUPRCs on every seed.
4. **Fig. 3b differs from the recomputed graph by two edges.** The published network lacks Oligodendrocytes ->
   L4/5 IT (25 genes, exactly at the 0.86-quantile threshold) and shows a faint Astrocytes -> L2/3 IT (21 genes,
   below it). The library code from before the 2025-09-26 z-value rewrite (commit e768bc5) gives an identical
   matrix, so the difference is not from that change. Sheet `Fig3b` lists every pair with its weight; consider
   redrawing Fig. 3b from it.
5. **Fig. 5c bolding is not reproducible.** The segmentation test described in the Methods (one-sided
   Mann-Whitney, both p < 0.05) reproduces the bold/non-bold pattern for 12 of 30 genes; e.g. CCND1, TFF1 and
   GATA3 pass with p < 1e-20 but are not bold, while C3, CXCL12 and EMILIN1 fail but are bold.
   `atera_breast_segmentation_analysis.py` currently tests unrelated High_Grade_DCIS pairs, so the analysis
   that set the published bolding is not in the repository. The recomputed p-values are in sheet `Fig5c`.
6. **Fig. 4c, C15orf48** is bold but fails the far-receiver comparison added in the revision (near vs far
   receiver p = 0.98; near vs far sender p = 9e-9). The bolding predates the revised two-comparison test.
7. **Segmentation threshold.** The Methods use p < 0.05; the analysis scripts use p < 0.1
   (`significant_both`). Sheets report both calls; they agree for every Fig. 4c gene.
8. **Benchmark statistics.** The Methods say the benchmark p-values are BH-adjusted, but
   `benchmark_utils.plot_boxplots` (and the caption values) use Bonferroni.
9. **Figshare v1 is incomplete.** It has no Atera or tonsil data or models (REPRODUCIBILITY.md says it does),
   and its low-resolution Xenium model is run `t4t8pzvi`, not the 2026-09-19 retrain `5tfi78lp` the guide cites.
   The five Figshare `model.pt` files are unlabelled; `checkpoints.yaml` maps each file id to its run.
10. **Analysis scripts no longer match some published panels.** `cortex_analysis.py` draws L6b instead of Sst in
   Fig. 3b and only the Astrocyte dot plot; `xenium_analysis.py` lacks the multi-receiver Fig. 4c and the
   head-resolved Fig. 4d dot plot; `xenium_niche_analysis.py` sets `hub_cluster = 9`, which is legend hub 10,
   while the published Sankey "hub 9" is cluster index 8; `xenium_spatial_analysis.py` plots AGR3, not ESR1;
   `human_tonsil_analysis.py` ends on the CD4 T cell receiver rather than the GC B cell panels. No script in the
   repository draws the Fig. 2h-j bar plots; the pipeline's `plot_boxplots` draws box plots across dataset seeds.
   The exporters here record the settings that do reproduce each panel.
11. **Paths.** `cortex_analysis.py` loads `./saved_models/cortex_18_sweep_...` without the
   `cortex_sweep_2025-04-28_model_2025-05-05/` folder that REPRODUCIBILITY.md documents (the checkpoint now sits
   at the documented path). The Xenium 2025-05-01 train/test splits that the Xenium scripts load were missing
   and were restored from gs://justin-amici.
12. **Fig. 5b axis label.** The receiver axis reads "Plasma Myeloid Mix" while the sender axis reads "Plasma
   Mast Mix"; both are the `Plasma_Mast_Mixture` label.
13. **Tonsil segmentation flags use pixels.** `human_tonsil_analysis.py` calls the dot plot with
   `flag_segmentation_artifacts=True` on CosMx pixel coordinates, so its 20/100 thresholds are 2.4/12 um rather
   than 20/100 um. Fig. 6c draws every gene bold, so the figure is unaffected, but the Methods say the test was
   applied to the tonsil data. The library version also seeds its subsampling with Python's `hash()`, which
   changes between processes unless `PYTHONHASHSEED` is set.

## For the submission

Inventory of Supporting Information (one line per file):

- Source Data Fig. 2 | Source_Data_Fig2.xlsx | Numerical data underlying Fig. 2a-j
- Source Data Fig. 3 | Source_Data_Fig3.xlsx | Numerical data underlying Fig. 3a-c
- Source Data Fig. 4 | Source_Data_Fig4.xlsx, Source_Data_Fig4_large_tables.zip | Numerical data underlying Fig. 4a-f
- Source Data Fig. 5 | Source_Data_Fig5.xlsx, Source_Data_Fig5_large_tables.zip | Numerical data underlying Fig. 5a-c
- Source Data Fig. 6 | Source_Data_Fig6.xlsx, Source_Data_Fig6_large_tables.zip | Numerical data underlying Fig. 6a-d

Data Availability: "Source data are provided with this paper."
