# Source Data

The workbooks the submission attaches, one per figure, and the per-panel tables they are built
from. Both are build artifacts and are not in the branch; `source_data/.gitignore` keeps them out.

| Path | Contents |
| --- | --- |
| `source_data/output/Source_Data_<figure>.xlsx` | one workbook per figure |
| `source_data/output/Source_Data_<figure>_large_tables.zip` | sheets over 150,000 rows, as CSV |
| `source_data/panels/<figure>/` | the per-panel CSVs and a `manifest.json` mapping each to its sheet |

Each workbook opens with a README sheet giving the dataset and checkpoint it came from — path,
md5, W&B sweep and run, training config, deposited object name — and one line per sheet with its
row count and units. Figure 1 is a schematic and has no Source Data.

## Sheets per figure

Sheets marked \* are in the companion zip rather than the workbook. For Fig. 2c-e each panel also
has a `_stats` sheet with the box statistics and the Mann-Whitney results.

- **Fig. 2** — semi-synthetic benchmarks. **2a** `Fig2a`; **2b** `Fig2b`; **2c-e** `Fig2c`, `Fig2d`,
  `Fig2e`; **2f** `Fig2f`; **2g** `Fig2g`; **2h-j** `Fig2h`, `Fig2i`, `Fig2j`
- **Fig. 3** — MERFISH mouse cortex. **3a** `Fig3a`; **3b** `Fig3b`, `Fig3b_full_matrix`; **3c**
  `Fig3c`
- **Fig. 4** — Xenium breast cancer. **4a** `Fig4a`\*; **4b** `Fig4b`, `Fig4b_full_matrix`; **4c**
  `Fig4c`; **4d** `Fig4d`, `Fig4d_attending_CD8`; **4e** `Fig4e_length_scales`\*,
  `Fig4e_length_scale_summary`, `Fig4e_head_dotplot`; **4f** `Fig4f_hubs`\*, `Fig4f_sankey_flows`,
  `Fig4f_hub_summary`
- **Fig. 5** — Atera breast cancer. **5a** `Fig5a`\*; **5b** `Fig5b`; **5c** `Fig5c`
- **Fig. 6** — CosMx human tonsil. **6a** `Fig6a`\*; **6b** `Fig6b_network`, `Fig6b_GC_heatmap`,
  `Fig6b_full_heatmap`; **6c** `Fig6c`; **6d** `Fig6d_head_dotplots`, `Fig6d_length_scales`,
  `Fig6d_length_scale_summary`
- **ED Fig. 1** — architecture, benchmark diagnostics, Xenium replicates. **1c** `ED1c_pr_curves`;
  **1f** `ED1f_sweep_pr_curves`; **1g** `ED1g_cortex_interaction_matrix`; **1h**
  `ED1h_replicate1_interaction_mat`, `ED1h_replicate2_interaction_mat`; **1i**
  `ED1i_replicate_correlation_poin`, `ED1i_replicate_correlation_stat`
- **ED Fig. 2** — further Xenium analyses. **2a** `ED2a_interaction_matrix_with_st`; **2c**
  `ED2c_labeled_genes`, `ED2c_volcano_all_genes`; **2g** `ED2g_adjusted_coefficients`,
  `ED2g_adjusted_spatial_values`, `ED2g_adjustment_summary`; **2h** `ED2h_segmentation_test`
- **ED Fig. 3** — segmentation, length-scale uncertainty, hub selection. **3b**
  `ED3b_xenium_length_scales_by_se`; **3c** `ED3c_tonsil_length_scales_by_se`; **3d**
  `ED3d_hub_cluster_assignments`; **3e** `ED3e_hub_sender_receiver_compos`; **3f**
  `ED3f_length_scale_samples`, `ED3f_bootstrap_ci`; **3g** `ED3g_pbmc_replicate_samples`\*,
  `ED3g_pbmc_bootstrap_ci`; **3h** `ED3h_realistic_replicate_sample`, `ED3h_realistic_bootstrap_ci`;
  **3i** `ED3i_realistic_cell_type_compos`; **3j** `ED3j_coordinate_shuffle_estimat`
- **ED Fig. 4** — hubs against niche composition and graph baselines. **4b** `ED4b_gsea_results`;
  **4c** `ED4c_hub_agreement_ari_ami`; **4d** `ED4d_pathway_overlap_summary`; **4h**
  `ED4h_graph_baseline_ari_ami`
- **ED Fig. 5** — CAF subpopulations, label resolution, runtime, spatial CV. **5c**
  `ED5c_caf_sender_de`, `ED5c_caf_sender_attention_score`; **5d** `ED5d_highres_interaction_matrix`;
  **5h** `ED5h_within_label_autocorrelati`; **5i** `ED5i_runtime`; **5j**
  `ED5j_spatial_cv_task_scores`\*
- **ED Fig. 6** — no-ribosomal tonsil retrain, T zone fibroblasts. **6a**
  `ED6a_interaction_strength_by_pa`, `ED6a_gene_neighbour_contributio`; **6b**
  `ED6b_fibroblast_de_high_vs_low`; **6c** `ED6c_fibroblast_attention_score`
- **ED Fig. 7** — model selection, attention stability, hyperparameters. **7a**
  `ED7a_atera_attention_consistenc`, `ED7a_atera_attenti_correlations`; **7b**
  `ED7b_tonsil_attention_consisten`, `ED7b_tonsil_attent_correlations`; **7c**
  `ED7c_test_region_jsd`; **7d** `ED7d_test_region_composition`; **7e** `ED7e_value_penalty_auprc`,
  `ED7e_value_penalty_pr_curves`; **7f** `ED7f_attention_penalty_auprc`,
  `ED7f_attention_penalty_pr_curve`; **7g** `ED7g_head_auprc`, `ED7g_head_pr_curves`; **7h**
  `ED7h_neighbor_auprc`, `ED7h_neighbor_pr_curves`
- **ED Fig. 8** — thresholds, hub stability, multi-scale interactions. **8b**
  `ED8b_hub_cluster_assignments`; **8e** `ED8e_two_phase_head_length_scal`,
  `ED8e_two_phase_head_summary`; **8f** `ED8f_two_phase_task_scores`; **8g**
  `ED8g_unimodal_baseline_auprc`; **8i** `ED8i_unimodal_length_scales`; **8j**
  `ED8j_tonsil_distance_profiles`; **8k** `ED8k_monotonicity_by_pair`
- **ED Fig. 9** — attention variants, occlusion, negative controls. **9a**
  `ED9a_unimodal_peak_distances`; **9b** `ED9b_unimodal_attention_length_`; **9c**
  `ED9c_unconstrained_length_scale`; **9d** `ED9d_variant_task_scores`; **9e**
  `ED9e_occlusion_auprc`; **9g** `ED9g_matched_occlusion_summary`,
  `ED9g_matched_occlusion_per_cell`, `ED9g_matched_occlusion_distance`
- **ED Fig. 10** — noise, multiple testing, spatial dependence, stroma. **10a**
  `ED10a_noise_sensitivity_runs`; **10b** `ED10b_xenium_family_wide_bh`; **10c**
  `ED10c_xenium_jackknife_500um`; **10d** `ED10d_atera_family_wide_bh`; **10e**
  `ED10e_atera_jackknife_500um`; **10f** `ED10f_tonsil_family_wide_bh`; **10g**
  `ED10g_tonsil_jackknife_500um`; **10h** `ED10h_with_without_stroma_edges`,
  `ED10h_with_stroma_matrix`, `ED10h_without_stroma_matrix`; **10i** `ED10i_distance_matched_null`

Box plots and violins carry their full statistics in the `_summary` and `_stats` sheets: n, mean,
s.d., minimum, quartiles, maximum, the whisker ends, the number of points beyond them, and the
axis limit the panel is drawn to.

Two gene selections do not reproduce as exact lists, by design, and the exporters report it:

- **Fig. 4c** draws a truncation of the ranked top genes per sender, so a rerun computes a longer
  list with the published one as its prefix. *TIGIT* (rank 6, CD4 T to CD8 T) is the one gene
  dropped out of order.
- **Fig. 6c and 6d** select the top genes only after dropping those flagged as segmentation
  artifacts, including the comparison against the senders a receiver borders. Without that step the
  CD4 T cell panel returns the fibroblast chemokines *CCL21* and *CCL19* instead of receiver genes.

## Panels with no Source Data

Schematics and spatial maps, which carry no numerical graph: Figure 1, and Extended Data 1a, 1b,
2e, 2f, 4a, 5a, 5b, 8d, 8h.

Panels the analysis never wrote to disk, which need their script rerun before they can be
exported: Extended Data 1d, 1e, 2b, 3a, 5e-g, 7i, 8a, 9f, 9h, 9i. `extended_data.py --survey`
names the script for each.

## Checkpoint per figure

| Figure | Dataset | Checkpoint (W&B sweep / run) | `model.pt` md5 |
| --- | --- | --- | --- |
| 2 | semi-synthetic PBMC (10 seeds) and realistic breast cancer | per-seed `amici_*_best_model` | in `fig2_manifest.txt` |
| 3 | `cortex_processed_2025-04-28.h5ad` | `plm73bmg` / `xrtcnlt0` | `76eadfaa…` |
| 4 | `xenium_sample1_filtered_2025-05-01.h5ad` | `g3mucw4s` / `te7pkv3z` | `c118e838…` |
| 5 | `atera_breast_filtered_2026-07-22.h5ad` | `25xpxkuk` / `zhtraubt` | `f53e6ab5…` |
| 6 | `human_tonsil_filtered_no_ribo_2026-07-25.h5ad` | seed-40 retrain at the `0f9rk0na` hyperparameters | `b06237db…` |

Extended Data panels reuse these checkpoints. Full paths, md5 sums and remote copies are in
[`source_data/checkpoints.yaml`](source_data/checkpoints.yaml); the per-figure asset lists, with a
`fetch.sh` for each, are in
[`reproducibility/figure_assets.yaml`](reproducibility/figure_assets.yaml).

## Rebuilding

From the repository root, with the `amici` environment active:

```bash
python source_data/fig3_cortex.py
python source_data/fig4_xenium.py
python source_data/fig5_atera.py
python source_data/fig6_tonsil.py
python source_data/fig2_benchmarks.py --results benchmarks/results_fig2_original \
    --realistic-results benchmarks/results_realistic_rerun_april_family/scores/run_329 \
    --example-seed 72 --rerun-source METHOD=PATH   # one per method, see Figure 2 below
python source_data/extended_data.py --export
python source_data/build_workbooks.py
python source_data/plot_fig2g.py                   # redrawn Fig. 2g
```

Each exporter verifies its dataset and `model.pt` by md5 before use and recomputes every plotted
quantity from the checkpoint. On one A100: about 5 min for Fig. 3, 10 min for Fig. 5, 20 min for
Fig. 4 and 25 min for Fig. 6.

## Figure 2

Two panels cannot be rebuilt from the original benchmark run, and say so in the workbook:

- **2g is redrawn** from the re-run's selected AMICI model, because no data for the published
  curves survive. The redrawn curves keep the published shapes and order, with peaks at
  41.6 / 27.1 / 17.1 µm against about 44 / 30 / 19 µm measured off the published panel.
- **2h-j keep the published bar heights.** Each sheet also gives the re-run AUPRC, the difference
  and the re-run's output folder. The deterministic baselines reproduce to within 0.0007, which is
  what identifies the dataset.

CGCom was not re-run. 2a-f come from the original run and its inputs.

## Submission inventory

| Item | Files |
| --- | --- |
| Source Data Fig. 2-6 | `Source_Data_Fig<n>.xlsx`, plus `_large_tables.zip` for 4, 5 and 6 |
| Source Data ED Fig. 1-10 | `Source_Data_ED<n>.xlsx`, plus `_large_tables.zip` for ED 3 and ED 5 |

Data Availability ends with:

> The human tonsil CosMx data are available from NCBI GEO under accession GSE351148. Source data
> are provided with this paper.
