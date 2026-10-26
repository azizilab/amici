# Numbers for the Fig. 4d, 6d and 2c-e legends

Everything below was recomputed on 2026-10-22 from the checkpoints in
[`checkpoints.yaml`](checkpoints.yaml) and from the original benchmark run in
`benchmarks/results_fig2_original/`. No figure changes; these are legend values only.

## What *n* means in Fig. 4d and 6d

Each violin (4d) or box (6d) is one **sender cell per observation**: the length scale is evaluated once
per candidate sender cell, from that cell's own counterfactual attention profile
(`AMICICounterfactualAttentionModule._calculate_length_scales`). The candidate set for a
sender → receiver pair is every cell of the sender type that appears in at least one 50-nearest-neighbor
neighborhood of a receiver cell of that type, which is the set
`AMICICounterfactualAttentionModule.compute` iterates over.

Two consequences worth knowing before wording the legend:

- **n is a property of the sender → receiver pair, not of the head.** The same sender cells are evaluated
  on every head, so the two heads of a pair (e.g. CD4 T → CD8 T on heads 4 and 5) carry the same n.
- **n is slightly smaller than the sender type's population**, because it excludes sender cells that are
  not in any receiver neighborhood. The gap is largest for CD8 T → Invasive Tumor (7,116 of 18,268 CD8 T
  cells), where the receiver type is large and the sender type is concentrated elsewhere in the tissue.

## Fig. 4d (top) — Xenium, 13 violins, top to bottom

Ascending median, i.e. the published row order. Median is in µm, attention threshold 0.1.

| Row | Interaction | **n sender cells** | Median (µm) | Receiver cells | Sender type population |
| --- | --- | --- | --- | --- | --- |
| 1 | CD4 T cells → CD8 T cells (head 5) | **19,490** | 4.43 | 18,268 | 19,657 |
| 2 | CD4 T cells → CD8 T cells (head 4) | **19,490** | 5.30 | 18,268 | 19,657 |
| 3 | CD4 T cells → B cells (head 4) | **18,266** | 5.55 | 11,724 | 19,657 |
| 4 | CD8 T cells → M1 macrophages (head 4) | **18,089** | 6.70 | 20,179 | 18,268 |
| 5 | B cells → CD4 T cells (head 5) | **11,272** | 6.77 | 19,657 | 11,724 |
| 6 | CD8 T cells → invasive tumor (head 5) | **7,116** | 7.64 | 56,599 | 18,268 |
| 7 | CD8 T cells → M1 macrophages (head 5) | **18,089** | 7.80 | 20,179 | 18,268 |
| 8 | M2 macrophages → CD4 T cells (head 5) | **4,679** | 7.82 | 19,657 | 5,231 |
| 9 | CD4 T cells → B cells (head 5) | **18,266** | 8.91 | 11,724 | 19,657 |
| 10 | M2 macrophages → CD8 T cells (head 5) | **4,876** | 9.80 | 18,268 | 5,231 |
| 11 | DCIS 1 → M1 macrophages (head 5) | **16,619** | 10.12 | 20,179 | 19,587 |
| 12 | Invasive tumor → M1 macrophages (head 5) | **49,386** | 11.07 | 20,179 | 56,599 |
| 13 | M2 macrophages → M1 macrophages (head 5) | **5,073** | 11.65 | 20,179 | 5,231 |

Full per-cell values: `panels/Fig4/Fig4d_length_scales.csv` (210,711 rows = the 13 row counts above);
per-violin statistics: `panels/Fig4/Fig4d_length_scale_summary.csv`.

## Fig. 4d (bottom) — no violins

The bottom panel is the head-resolved CD4 T → CD8 T dot plot (4 genes × heads 4 and 5), so it has no
per-cell distribution. If the legend needs counts for it, they are the ones behind the ablation:
**18,268 CD8 T cell receivers** and **19,490 CD4 T cell senders** in their neighborhoods.

## Fig. 6d — CosMx tonsil, 4 boxes

Receiver is GC B cell (resting), 60,481 cells. Median in µm, attention threshold 0.1.

| Head | Sender | **n sender cells** | Median (µm) | Sender type population |
| --- | --- | --- | --- | --- |
| 3 | Tfh cell | **14,190** | 41.02 | 15,636 |
| 3 | Follicular dendritic cell | **5,719** | 44.80 | 5,945 |
| 6 | Tfh cell | **14,190** | 12.80 | 15,636 |
| 6 | Follicular dendritic cell | **5,719** | 12.43 | 5,945 |

Full per-cell values: `panels/Fig6/Fig6d_length_scales.csv` (39,818 rows); per-box statistics:
`panels/Fig6/Fig6d_length_scale_summary.csv`.

## Suggested legend wording

> Each observation is one sender cell; n is the number of sender cells of that type lying in at least one
> 50-nearest-neighbor neighborhood of a receiver cell of that type (4d: n = 4,679-49,386 per violin, see
> Source Data; 6d: n = 14,190 Tfh cells and n = 5,719 follicular dendritic cells per box). The two heads of
> a given sender → receiver pair are evaluated on the same sender cells.

The per-violin n is in `Fig4d_length_scale_summary.csv`.

## How the counts were verified

Two independent routes agree exactly on all 12 pair counts:

1. Row counts in the exported per-cell tables, which the exporters built by running the published
   checkpoints (md5-verified) and which reproduce the published violin order and length scales.
2. A rebuild of the 50-nearest-neighbor graph from `obsm["spatial"]` and the cell-type labels alone
   (same construction as `AMICI._compute_nn`: euclidean, same-type neighbors excluded), counting distinct
   sender-type cells across all receiver neighborhoods. No model involved.

Each `n_sender_cells` is also the number of **distinct** cells, confirmed by
`nunique(sender_cell_id) == len(sender_cell_id)` in both per-cell tables.

## Fig. 2c-e — U statistics, effect sizes and medians

The per-seed AUPRCs are on this machine: `benchmarks/results_fig2_original/3ct_dataset_2way_<seed>/*_pr.csv`
for seeds 40, 123, 6, 23, 25, 88, 72, 58, 22, 31. All ten Bonferroni-corrected p-values in the caption
reproduce exactly from them, which confirms that folder is the published run.

n = 10 datasets per group, so U runs to a maximum of 100: U = 100 wherever the groups separate completely,
98 for NicheDE σ = 500, and 83 for GITIII in 2d and 2e.

| Panel | Comparison | **U** | **Rank-biserial r** | **P(AMICI > baseline)** | AMICI median | Baseline median | Median diff. | p (Bonferroni) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2c gene | vs GITIII | 54 | 0.08 | 0.54 | 0.9544 | 0.9488 | 0.0057 | 1.0 (n.s.) |
| 2c gene | vs NicheDE σ=20 | **100** | 1.00 | 1.00 | 0.9544 | 0.4592 | 0.4953 | 0.0013 |
| 2c gene | vs NicheDE σ=200 | **100** | 1.00 | 1.00 | 0.9544 | 0.7500 | 0.2044 | 0.0013 |
| 2c gene | vs NicheDE σ=500 | **98** | 0.96 | 0.98 | 0.9544 | 0.7559 | 0.1985 | 0.0023 |
| 2c gene | vs NCEM rad=10 | **100** | 1.00 | 1.00 | 0.9544 | 0.0706 | 0.8838 | 0.0013 |
| 2c gene | vs NCEM rad=15 | **100** | 1.00 | 1.00 | 0.9544 | 0.0720 | 0.8824 | 0.0013 |
| 2c gene | vs NCEM rad=20 | **100** | 1.00 | 1.00 | 0.9544 | 0.0720 | 0.8824 | 0.0013 |
| 2d sender | vs GITIII | **83** | 0.66 | 0.83 | 0.6248 | 0.5035 | 0.1212 | 0.028 |
| 2d sender | vs CGCom | **100** | 1.00 | 1.00 | 0.6248 | 0.0012 | 0.6236 | 0.0003 |
| 2e receiver | vs GITIII | **83** | 0.66 | 0.83 | 0.9777 | 0.9355 | 0.0421 | 0.028 |
| 2e receiver | vs CGCom | **100** | 1.00 | 1.00 | 0.9777 | 0.1758 | 0.8019 | 0.0004 |

Effect sizes: rank-biserial r = 2U/(n₁n₂) − 1, the standard effect size for a Mann-Whitney U test;
P(AMICI > baseline) = U/(n₁n₂) is the same quantity as a common-language effect size. Per-method medians,
quartiles and ranges, and the per-seed values, are in `Source_Data_Fig2.xlsx` (sheets Fig2c-e).
