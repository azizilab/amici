# Numbers for the Fig. 4d, 6d and 2c-e legends

Everything below was recomputed on 2026-10-22 from the checkpoints in
[`checkpoints.yaml`](checkpoints.yaml) and from the original benchmark run in
`benchmarks/results_fig2_original/`. No figure changes; these are legend values only.

The editor's AIP checklist (NBT-BC69009C) asks for three things in these legends, and
[What the glyphs show](#what-the-glyphs-show) below answers all three:

- *n* as a precise value, with the unit of study defined, in 2d, 2e, 4d and 6d;
- box plots defined in terms of minima, maxima, centre, bounds of box and whiskers, and percentile, in
  2c-e, 4d and 6d;
- the statistical test named in 2c-e, with its sidedness and its multiple-comparison adjustment.

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

## What the glyphs show

Every box and violin in Figs. 2c-e, 4d and 6d is drawn with library defaults, which in all three cases
is the Tukey convention: the centre line (or, in 4d, the white dot) is the **median**, the box spans
the **25th to 75th percentile**, and the whiskers reach the most extreme observation still within
**1.5 x IQR** of the nearer box edge. Observations outside the whiskers are *outliers*, not the minimum
and maximum of the data. The three panels differ in what they draw outside the box:

| Panel | Call | Outside the box |
| --- | --- | --- |
| 2c-e | `matplotlib.axes.Axes.boxplot(..., patch_artist=True)` in `benchmarks/benchmark_utils.plot_boxplots` | outliers as fliers, plus **all 10 per-dataset values overlaid as points and joined across methods**, so every observation is visible |
| 4d (top) | `seaborn.violinplot` (seaborn 0.13.2 defaults: `inner="box"`, `bw_method="scott"`, `bw_adjust=1`, `cut=2`, `gridsize=100`, `density_norm="area"`) in `xenium/xenium_analysis.py` | a Gaussian kernel density estimate, each violin scaled to equal area, extended 2 bandwidths past the data and clipped to the 0-50 µm axis; **no outliers are drawn** |
| 6d | `seaborn.boxplot(..., linewidth=0.9, fliersize=0.8, width=0.65)` in `human_tonsil/human_tonsil_analysis.py` | outliers as points (`fliersize=0.8`) |

The 4d inner box is `matplotlib.cbook.boxplot_stats`, i.e. the same Tukey box as 2c-e and 6d, so the
whisker ends in 4d are 1.5 x IQR limits and not the extremes of the violin, whose tails run past them.

### Minima, maxima, centre, bounds and whiskers, per glyph

Recomputed on 2026-10-28 from the exported per-cell and per-dataset tables with
`matplotlib.cbook.boxplot_stats`, i.e. from the same numbers the panels were drawn from. Figures and
Source Data are unchanged; these are the values a legend or a reviewer would ask for.

**Fig. 4d (top)** — length scale in µm, 13 violins in published (ascending-median) order. `min` and `max`
are the data extremes, which the violin tails show and the inner box does not.

| Row | Interaction | n | Min | Whisker low | Q1 | Median | Q3 | Whisker high | Max | Outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CD4 T → CD8 T (head 5) | 19,490 | 0.15 | 0.15 | 3.15 | 4.43 | 6.12 | 10.56 | 41.08 | 560 |
| 2 | CD4 T → CD8 T (head 4) | 19,490 | 0.93 | 0.93 | 4.09 | 5.30 | 6.67 | 10.52 | 33.99 | 467 |
| 3 | CD4 T → B cells (head 4) | 18,266 | 0.91 | 0.91 | 4.38 | 5.55 | 6.97 | 10.85 | 33.29 | 550 |
| 4 | CD8 T → M1 macrophages (head 4) | 18,089 | 1.53 | 1.53 | 5.34 | 6.70 | 8.37 | 12.91 | 43.42 | 536 |
| 5 | B cells → CD4 T (head 5) | 11,272 | 0.41 | 0.41 | 5.10 | 6.77 | 8.86 | 14.49 | 29.79 | 208 |
| 6 | CD8 T → invasive tumor (head 5) | 7,116 | 1.55 | 2.41 | 6.36 | 7.64 | 9.04 | 13.06 | 77.35 | 237 |
| 7 | CD8 T → M1 macrophages (head 5) | 18,089 | 0.00 | 0.46 | 5.98 | 7.80 | 9.73 | 15.34 | 35.12 | 356 |
| 8 | M2 macrophages → CD4 T (head 5) | 4,679 | 0.00 | 0.82 | 6.03 | 7.82 | 9.99 | 15.91 | 26.68 | 85 |
| 9 | CD4 T → B cells (head 5) | 18,266 | 1.18 | 1.18 | 6.53 | 8.91 | 12.18 | 20.67 | 161.48 | 839 |
| 10 | M2 macrophages → CD8 T (head 5) | 4,876 | 1.47 | 1.47 | 7.40 | 9.80 | 12.87 | 21.05 | 35.15 | 96 |
| 11 | DCIS 1 → M1 macrophages (head 5) | 16,619 | 3.22 | 3.60 | 8.53 | 10.12 | 11.97 | 17.14 | 31.06 | 296 |
| 12 | Invasive tumor → M1 macrophages (head 5) | 49,386 | 1.50 | 2.31 | 8.98 | 11.07 | 13.64 | 20.62 | 60.17 | 1,165 |
| 13 | M2 macrophages → M1 macrophages (head 5) | 5,073 | 0.94 | 0.94 | 8.79 | 11.65 | 14.81 | 23.82 | 37.70 | 74 |

The 0-50 µm `plt.xlim` of the published panel clips the longest tails (rows 6, 9 and 12 run to 77, 161
and 60 µm), so the drawn violins are truncated, not the distributions.

**Fig. 6d** — length scale in µm, receiver GC B cell (resting), 4 boxes.

| Head | Sender | n | Min | Whisker low | Q1 | Median | Q3 | Whisker high | Max | Outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Tfh cell | 14,190 | 6.54 | 6.54 | 26.76 | 41.02 | 67.06 | 127.51 | 1,623.68 | 966 |
| 3 | Follicular dendritic cell | 5,719 | 7.39 | 7.39 | 26.66 | 44.80 | 74.68 | 146.63 | 887.15 | 362 |
| 6 | Tfh cell | 14,190 | 1.99 | 3.61 | 10.51 | 12.80 | 15.23 | 22.29 | 141.48 | 309 |
| 6 | Follicular dendritic cell | 5,719 | 4.15 | 4.15 | 10.05 | 12.43 | 14.97 | 22.28 | 52.33 | 136 |

The published panel's `plt.xlim` also clips here, so the drawn whiskers and outliers stop at the axis
limit while the tabulated maxima do not.

**Figs. 2c-e** — AUPRC, n = 10 datasets per box. Every observation is also drawn as a point, so min and
max are visible in the panel; `min`/`max` and the quartiles are in the `Fig2c/d/e_stats` sheets.

| Panel | Method | Min | Whisker low | Q1 | Median | Q3 | Whisker high | Max | Outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2c | AMICI | 0.8673 | 0.8673 | 0.9116 | 0.9544 | 0.9600 | 0.9827 | 0.9827 | 0 |
| 2c | GITIII | 0.8579 | 0.8985 | 0.9252 | 0.9488 | 0.9573 | 0.9726 | 0.9726 | 1 |
| 2c | NicheDE σ=20 | 0.2910 | 0.2910 | 0.3808 | 0.4592 | 0.4864 | 0.5687 | 0.5687 | 0 |
| 2c | NicheDE σ=200 | 0.5713 | 0.5713 | 0.6496 | 0.7500 | 0.8319 | 0.8522 | 0.8522 | 0 |
| 2c | NicheDE σ=500 | 0.5974 | 0.5974 | 0.6702 | 0.7559 | 0.8459 | 0.8736 | 0.8736 | 0 |
| 2c | NCEM rad=10 | 0.0581 | 0.0581 | 0.0630 | 0.0706 | 0.0777 | 0.0914 | 0.1228 | 1 |
| 2c | NCEM rad=15 | 0.0580 | 0.0580 | 0.0638 | 0.0720 | 0.0800 | 0.0910 | 0.1120 | 1 |
| 2c | NCEM rad=20 | 0.0580 | 0.0580 | 0.0638 | 0.0720 | 0.0800 | 0.0910 | 0.1120 | 1 |
| 2d | AMICI | 0.4516 | 0.4516 | 0.5438 | 0.6248 | 0.7039 | 0.7769 | 0.7769 | 0 |
| 2d | GITIII | 0.4412 | 0.4412 | 0.4633 | 0.5035 | 0.5214 | 0.5696 | 0.5696 | 0 |
| 2d | CGCom | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 0.0098 | 0.0108 | 0.0604 | 2 |
| 2e | AMICI | 0.9492 | 0.9492 | 0.9556 | 0.9777 | 0.9843 | 0.9955 | 0.9955 | 0 |
| 2e | GITIII | 0.8139 | 0.8892 | 0.9027 | 0.9355 | 0.9591 | 0.9837 | 0.9837 | 1 |
| 2e | CGCom | 0.1419 | 0.1419 | 0.1692 | 0.1758 | 0.1951 | 0.1988 | 0.1988 | 0 |

Regenerate either table with
`python -I source_data/box_stats.py` (see [`box_stats.py`](box_stats.py)); it reads only the exported
panel CSVs, so it needs no model and no GPU.

## Suggested legend wording

**Fig. 2c-e.** The unit of study is a simulated dataset, so these are not biological or technical
replicates of a measurement; the current draft calls them "technical semi-synthetic dataset replicates",
which invites exactly the query the checklist warns about. They are independent simulations differing
only in their random seed (40, 123, 6, 23, 25, 88, 72, 58, 22, 31), and all methods are scored on the
same ten.

> n = 10 independently simulated datasets per method (ten random seeds; every method scored on the same
> ten datasets). Box plots show the median (centre line), the 25th-75th percentile (box bounds) and the
> most extreme value within 1.5 x the interquartile range of the box (whiskers); points outside the
> whiskers are outliers, and all ten per-dataset values are overlaid as points and joined across methods.
> P values are from two-sided Mann-Whitney U tests of each baseline against AMICI, Bonferroni-corrected
> for the [7 in 2c / 2 in 2d / 2 in 2e] baseline comparisons in the panel; exact corrected P values,
> U statistics and effect sizes for every comparison are in the Source Data.

**Figs. 4d and 6d.** One tissue section each (Xenium breast cancer replicate 1; CosMx human tonsil), so
the observations are cells within a single section and not independent biological replicates.

> Each observation is one sender cell; n is the number of sender cells of that type lying in at least one
> 50-nearest-neighbor neighborhood of a receiver cell of that type (4d: n = 4,679-49,386 per violin, see
> Source Data; 6d: n = 14,190 Tfh cells and n = 5,719 follicular dendritic cells per box). The two heads of
> a given sender → receiver pair are evaluated on the same sender cells. Data are from one tissue section.
> Violins (4d) are Gaussian kernel density estimates scaled to equal area; the inner box and the boxes in
> 6d show the median (white dot in 4d, centre line in 6d), the 25th-75th percentile (box bounds) and the
> most extreme value within 1.5 x the interquartile range of the box (whiskers). Outliers are drawn in 6d
> and omitted in 4d. Per-glyph minima, maxima, quartiles and whisker limits are in the Source Data.

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

The test is a **two-sided** Mann-Whitney U test of each baseline against AMICI, Bonferroni-corrected by
the number of baseline comparisons in the panel (7 in 2c, 2 in 2d, 2 in 2e), exactly as
`benchmarks/benchmark_utils.plot_boxplots` draws it. Only the GITIII gene-task comparison is not
significant; its raw p = 0.79 times 7 is 5.54, so the caption should report it as p > 0.99 (or n.s.)
rather than the uncapped product that the `Fig2c_stats` sheet carries in `p_bonferroni`.

Effect sizes: rank-biserial r = 2U/(n₁n₂) − 1, the standard effect size for a Mann-Whitney U test;
P(AMICI > baseline) = U/(n₁n₂) is the same quantity as a common-language effect size. Per-method medians,
quartiles and ranges, and the per-seed values, are in `Source_Data_Fig2.xlsx` (sheets Fig2c-e).
