"""Source Data for the Extended Data figures.

The editor asks for Source Data behind every graph in the Figures *and* the Extended Data items.
`fig2_benchmarks.py` ... `fig6_tonsil.py` cover the main figures; this script covers ED 1-9.

Each ED panel is declared below with where its numbers live. Panels fall into three kinds:

``CACHE``
    The numbers were written to disk by the analysis that drew the panel, so the exporter copies or
    reshapes them. No model and no GPU.
``RECOMPUTE``
    The analysis plots straight from memory and keeps nothing, so the panel needs its script rerun
    before it can be exported. Listed with the script to run.
``IMAGE``
    A schematic, a spatial map or a micrograph, which carries no numerical graph. Recorded so the
    inventory is complete and so that "no Source Data" is a statement rather than an omission.

Run ``python source_data/extended_data.py --survey`` to print the coverage report: which panels are
exportable right now, which need a rerun, and which cached file is missing if any.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE = "cache"
RECOMPUTE = "recompute"
IMAGE = "image"


@dataclass
class Panel:
    """One Extended Data panel and the provenance of its numbers."""

    panel: str
    description: str
    kind: str
    sources: list[str] = field(default_factory=list)
    # Script to rerun for RECOMPUTE panels, or the script that wrote the cache for CACHE panels.
    script: str = ""
    note: str = ""

    def missing_sources(self) -> list[str]:
        return [s for s in self.sources if not os.path.exists(os.path.join(REPO_ROOT, s))]


# ED 1: architecture, benchmark diagnostics, reproducibility across Xenium sections
ED1 = [
    Panel("a", "Multi-headed attention architecture", IMAGE),
    Panel("b", "Framework and downstream tasks", IMAGE),
    Panel(
        "c",
        "PR curves for the three tasks, PBMC semi-synthetic",
        CACHE,
        ["benchmarks/results_fig2_original"],
        "benchmarks/gene_task/plot_pr_curves.py",
        "per-seed *_pr.csv under each 3ct_dataset_2way_<seed>/ folder",
    ),
    Panel(
        "d",
        "Predicted interaction networks vs ground truth",
        CACHE,
        ["benchmarks/results_fig2_original"],
        "benchmarks/benchmark_utils.py",
        "edge lists are derivable from the per-seed score CSVs",
    ),
    Panel("e", "Training and validation loss curves for the baselines", RECOMPUTE, [], "benchmarks/train_*_model.py"),
    Panel(
        "f",
        "PR curves after hyperparameter sweeps",
        CACHE,
        ["benchmarks/results_fig2_original"],
        "benchmarks/*/plot_pr_curves.py",
    ),
    Panel(
        "g",
        "Full MERFISH cortex interaction network",
        CACHE,
        ["source_data/panels/Fig3/Fig3b_full_matrix.csv"],
        "cortex/cortex_analysis.py",
        "the full matrix already exported for Fig. 3b is the same quantity",
    ),
    Panel(
        "h",
        "Interaction strength matrices for the two Xenium sections",
        CACHE,
        [
            "xenium/figures/replicate_validation/rep1_interaction_weight_matrix.csv",
            "xenium/figures/replicate_validation/rep2_interaction_weight_matrix.csv",
        ],
        "xenium/xenium_replicate_validation.py",
    ),
    Panel(
        "i",
        "Correlation of interaction strengths between sections",
        CACHE,
        [
            "xenium/figures/replicate_validation/Fig_S5b_redraw_points.csv",
            "xenium/figures/replicate_validation/Fig_S5b_redraw_stats.csv",
        ],
        "xenium/redraw_fig_s5b.py",
    ),
]

# ED 2: additional Xenium analyses
ED2 = [
    Panel("a", "Interaction network including stromal cells", CACHE, ["xenium/figures/stromal_bleeding_analysis/cache/interaction_weight_matrix.csv"], "xenium/xenium_stromal_bleeding_analysis.py"),
    Panel("b", "Maximum explained variance per head and cell type", RECOMPUTE, [], "xenium/xenium_analysis.py"),
    Panel("c", "Volcano plots per receiver cell type", RECOMPUTE, [], "xenium/xenium_analysis.py", "needs the all-cell-type ablation cache, which is on the VM but not exported"),
    Panel("d", "Ligand-receptor overlap dot plot", RECOMPUTE, [], "xenium/xenium_lr_analysis.py"),
    Panel("e", "AGR3 subpopulation attention and expression", IMAGE),
    Panel("f", "ESR1 expression and attention maps", IMAGE),
    Panel("g", "Covariate-adjusted ESR1 and AGR3 expression", CACHE, ["benchmarks/sensitivity_scripts/figures/xenium_adjusted_cd8_tumor_effect"], "benchmarks/sensitivity_scripts/xenium_adjusted_cd8_tumor_effect.py"),
    Panel(
        "h",
        "Segmentation test for the Xenium dot-plot genes",
        CACHE,
        ["source_data/panels/Fig4/Fig4c.csv"],
        "xenium/segmentation_analysis.py",
        "per-gene p-values and the n per group are already exported in the Fig. 4c sheet",
    ),
]

# ED 3: segmentation testing, length-scale uncertainty, hub selection
ED3 = [
    Panel("a", "GITIII segmentation test", RECOMPUTE, [], "xenium/segmentation_analysis_gitiii.py"),
    Panel("b", "Xenium length scales across 20 refits", CACHE, ["xenium/results/xenium_length_scale_seed_sweep_reanalysis_v4"], "xenium/xenium_length_scale_seed_sweep_reanalysis.py"),
    Panel("c", "Tonsil length scales across refits", CACHE, ["human_tonsil/results"], "human_tonsil/human_tonsil_length_scale_seed_sweep_reanalysis.py"),
    Panel("d", "Silhouette score vs number of clusters", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache"], "xenium/xenium_niche_analysis.py"),
    Panel("e", "Sender and receiver composition of all hubs", CACHE, ["source_data/panels/Fig4/Fig4f_sankey_flows.csv"], "xenium/xenium_niche_analysis.py"),
    Panel("f", "Length-scale bootstrap over sender cells", CACHE, ["benchmarks/sensitivity_scripts/figures/length_scale_uncertainty"], "benchmarks/sensitivity_scripts/length_scale_uncertainty.py"),
    Panel("g", "Length scales over 50 PBMC dataset replicates", CACHE, ["benchmarks/sensitivity_scripts/figures/length_scale_dataset_bootstrap_ci"], "benchmarks/sensitivity_scripts/length_scale_dataset_bootstrap_ci.py"),
    Panel("h", "Length scales over 10 realistic dataset replicates", CACHE, ["benchmarks/sensitivity_scripts/figures/realistic_length_scale_dataset_bootstrap_ci"], "benchmarks/sensitivity_scripts/realistic_length_scale_dataset_bootstrap_ci.py"),
    Panel("i", "Cell-type population counts, realistic dataset", CACHE, ["benchmarks/sensitivity_scripts/figures/realistic_length_scale_dataset_bootstrap_ci"], "benchmarks/sensitivity_scripts/realistic_length_scale_dataset_bootstrap_ci.py"),
    Panel("j", "Length scales under coordinate shuffling", CACHE, [
            "benchmarks/sensitivity_scripts/figures/length_scale_coordinate_0p3_shuffle_sensitivity",
            "benchmarks/sensitivity_scripts/figures/length_scale_coordinate_0p5_shuffle_sensitivity",
            "benchmarks/sensitivity_scripts/figures/length_scale_coordinate_1p0_shuffle_sensitivity",
        ], "benchmarks/sensitivity_scripts/length_scale_coordinate_shuffle_sensitivity.py"),
]

# ED 4: hubs vs niche composition and graph baselines
ED4 = [
    Panel("a", "Niche composition clusters in space", IMAGE),
    Panel("b", "Uniquely significant GSEA gene sets per method", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_gsea_results.csv"], "xenium/xenium_niche_validation_gsea.py"),
    Panel("c", "Agreement of hubs with alternative label sets", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_ari_ami.csv"], "xenium/xenium_hub_analysis.py"),
    Panel("d", "GSEA pathway overlap across methods", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_pathway_overlap_summary.csv"], "xenium/xenium_niche_validation_gsea.py"),
    Panel("e", "GSEA butterfly plot, stromal cell types", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_gsea_results.csv"], "xenium/xenium_niche_validation_gsea.py"),
    Panel("f", "GSEA butterfly plot, tumor cell types", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_gsea_results.csv"], "xenium/xenium_niche_validation_gsea.py"),
    Panel("g", "GSEA butterfly plot, immune cell types", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_gsea_results.csv"], "xenium/xenium_niche_validation_gsea.py"),
    Panel("h", "ARI and AMI against graph baselines", CACHE, ["xenium/data/xenium_sample1/hub_graph_baseline_cache"], "xenium/hub_graph_baseline_utils.py"),
]

# ED 5: CAF subpopulations, cell-type resolution, runtime, spatial CV
ED5 = [
    Panel("a", "CAF UMAP and marker expression", IMAGE),
    Panel("b", "CAF populations in space and on the UMAP", IMAGE),
    Panel("c", "Log fold change of top DEGs between high-attention T-cell senders", CACHE, ["atera_breast/figures/atera_breast_analysis/atera_breast_t_cell_caf_group_de.csv", "atera_breast/figures/atera_breast_analysis/atera_breast_t_cell_caf_group_attention_scores.csv"], "atera_breast/atera_breast_analysis.py"),
    Panel("d", "High-resolution interaction matrix", CACHE, ["source_data/panels/Fig4/Fig4b_full_matrix.csv"], "xenium/xenium_analysis.py"),
    Panel("e", "High-resolution matrix aggregated to coarse labels", RECOMPUTE, [], "xenium/xenium_analysis_lowres.py"),
    Panel("f", "Low-resolution interaction matrix", RECOMPUTE, [], "xenium/xenium_analysis_lowres.py"),
    Panel("g", "Permutation null for matrix similarity", RECOMPUTE, [], "xenium/xenium_analysis_lowres.py", "1,000 seeded permutations; the script keeps neither the null draws nor the observed values"),
    Panel("h", "Within-label spatial autocorrelation", CACHE, ["xenium/data/xenium_sample1/within_type_autocorrelation_cache/within_type_autocorrelation_validation_scores.csv"], "xenium/xenium_within_type_autocorrelation.py"),
    Panel(
        "i",
        "Runtime on CPU and GPU",
        CACHE,
        [
            "xenium/runtime_benchmark/results/results_atera_gene_scaling_rerun_2026-07-28/benchmark_results_cpu.json",
            "xenium/runtime_benchmark/results/results_atera_gene_scaling_rerun_2026-07-28/benchmark_results_gpu.json",
        ],
        "xenium/runtime_benchmark/plot_benchmark.py",
    ),
    Panel("j", "Spatial cross-validation AUPRCs", CACHE, ["benchmarks/results_cv"], "benchmarks snakemake pipeline"),
]

# ED 6: model selection, attention stability, hyperparameter sensitivity
ED6 = [
    Panel("a", "Atera model selection and attention stability", CACHE, ["atera_breast/results/attention_consistency"], "atera_breast/atera_breast_attention_consistency.py"),
    Panel("b", "Tonsil model selection and attention stability", CACHE, ["human_tonsil/results"], "human_tonsil/human_tonsil_attention_consistency.py"),
    Panel("c", "Jensen-Shannon divergence of held-out composition", CACHE, ["benchmarks/figures/test_region_composition_diagnostic/test_region_composition_jsd_summary.csv"], "benchmarks/sensitivity_scripts/test_region_composition_diagnostic.py"),
    Panel("d", "Largest per-cell-type composition shifts", CACHE, ["benchmarks/figures/test_region_composition_diagnostic/test_region_cell_type_composition.csv"], "benchmarks/sensitivity_scripts/test_region_composition_diagnostic.py"),
    Panel("e", "Sensitivity to the attention value penalty", CACHE, ["benchmarks/sensitivity_scripts/results/value_penalty_analysis"], "benchmarks/sensitivity_scripts/value_penalty_analysis.py"),
    Panel("f", "Sensitivity to the attention penalty", CACHE, ["benchmarks/sensitivity_scripts/results/attention_penalty_analysis"], "benchmarks/sensitivity_scripts/attention_penalty_analysis.py"),
    Panel("g", "Sensitivity to the number of heads", CACHE, ["benchmarks/sensitivity_scripts/results/head_analysis"], "benchmarks/sensitivity_scripts/head_analysis.py"),
    Panel("h", "Sensitivity to the number of neighbors", CACHE, ["benchmarks/sensitivity_scripts/results/neighbor_analysis"], "benchmarks/sensitivity_scripts/neighbor_analysis.py"),
    Panel("i", "Empirical null distribution of Wald statistics", RECOMPUTE, [], "xenium/xenium_null_z_distribution.py"),
]

# ED 7: threshold sensitivity, hub stability, multi-scale interactions, monotonicity
ED7 = [
    Panel("a", "Length scale vs attention threshold", RECOMPUTE, [], "benchmarks/length_scale_task/sensitivity_test.py", "plots per-sender-cell length scales from memory; nothing is written to disk"),
    Panel("b", "Hub grid search over cluster number", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache"], "xenium/xenium_niche_analysis.py"),
    Panel("c", "Hub grid search over quantile threshold", CACHE, ["xenium/data/xenium_sample1/hub_analysis_cache"], "xenium/xenium_niche_analysis.py"),
    Panel("d", "Two-phase dataset in space", IMAGE),
    Panel("e", "Per-head length scales, two-phase dataset", CACHE, ["benchmarks/sensitivity_scripts/2_phase"], "benchmarks/sensitivity_scripts/2_phase/two_phase_selected_head_length_scales.py"),
    Panel("f", "AUPRC on the two-phase dataset", CACHE, ["benchmarks/sensitivity_scripts/2_phase"], "benchmarks/sensitivity_scripts/2_phase/two_phase_amici_sweep.py"),
    Panel("g", "AUPRC on the unimodal dataset vs baselines", CACHE, ["benchmarks/sensitivity_scripts/unimodal_two_phase"], "benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_baseline_sweep.py"),
    Panel("h", "Unimodal dataset recovery in space", IMAGE),
    Panel("i", "Length scales on the unimodal dataset", CACHE, ["benchmarks/sensitivity_scripts/unimodal_two_phase"], "benchmarks/sensitivity_scripts/unimodal_two_phase/unimodal_two_phase_selected_head_length_scales.py"),
    Panel("j", "Distance profile for an example pair", CACHE, ["human_tonsil/figures/empirical_distance_expression"], "human_tonsil/human_tonsil_empirical_distance_expression.py"),
    Panel("k", "Monotonic decay calls across all pairs", CACHE, ["figures/monotonicity/monotonicity_by_pair.csv"], "benchmarks/make_monotonicity_figure.py"),
]

# ED 8: attention variants, neighbor occlusion, negative controls
ED8 = [
    Panel("a", "Learned unimodal peak parameter", CACHE, ["benchmarks/sensitivity_scripts/results/unimodal_attention_analysis/unimodal_attention_length_scale_samples.csv"], "benchmarks/sensitivity_scripts/amici_variants"),
    Panel("b", "Attention-drop distance", CACHE, ["benchmarks/sensitivity_scripts/results/unimodal_attention_analysis/unimodal_attention_length_scale_samples.csv"], "benchmarks/sensitivity_scripts/amici_variants"),
    Panel("c", "Length scales from the unconstrained PE variant", CACHE, ["benchmarks/sensitivity_scripts/results/unconstrained_attention_analysis/unconstrained_attention_length_scale_samples.csv"], "benchmarks/sensitivity_scripts/amici_variants"),
    Panel("d", "AUPRC of the three variants", CACHE, ["benchmarks/sensitivity_scripts/results/amici_variants/amici_variant_task_scores.csv"], "benchmarks/sensitivity_scripts/amici_variants"),
    Panel("e", "AUPRC vs occlusion fraction", CACHE, ["benchmarks/sensitivity_scripts/figures/neighbor_occlusion_analysis/neighbor_occlusion_summary.csv"], "benchmarks/sensitivity_scripts/neighbor_occlusion_analysis.py"),
    Panel("f", "PR curves per occlusion level", RECOMPUTE, [], "benchmarks/sensitivity_scripts/neighbor_occlusion_analysis.py", "only the AUPRC summary is written; the curves are drawn from memory"),
    Panel(
        "g",
        "Matched high- vs low-attention occlusion",
        CACHE,
        [
            "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_summary.csv",
            "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_per_cell.csv",
            "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_diagnostics.csv",
        ],
        "benchmarks/sensitivity_scripts/matched_neighbor_occlusion_analysis.py",
    ),
    Panel("h", "PR curves, grid-based negative control", RECOMPUTE, [], "benchmarks/sensitivity_scripts/neutral_sampling_negative_control.py", "the script contains no to_csv call"),
    Panel("i", "PR curves, realistic negative control", RECOMPUTE, [], "benchmarks/sensitivity_scripts/neutral_sampling_negative_control.py", "the script contains no to_csv call"),
]

# ED 9: noise, multiple-testing family, spatial dependence, stromal exclusion
ED9 = [
    Panel("a", "AUPRC under coordinate and expression noise", CACHE, ["benchmarks/sensitivity_scripts/figures/coordinate_noise_gradient_sensitivity_sweep/coordinate_noise_sensitivity_all_runs.csv"], "benchmarks/sensitivity_scripts/coordinate_noise_sensitivity.py"),
    Panel("b", "Per-pair vs family-wide BH, Xenium", CACHE, ["benchmarks/sensitivity_scripts/results/family_wide_bh"], "benchmarks/sensitivity_scripts/family_wide_bh_interaction_networks.py"),
    Panel("c", "Block-jackknife retention, Xenium", CACHE, ["benchmarks/sensitivity_scripts/results/spatial_block_jackknife"], "benchmarks/sensitivity_scripts/spatial_block_jackknife_gene_significance.py"),
    Panel("d", "Per-pair vs family-wide BH, Atera", CACHE, ["benchmarks/sensitivity_scripts/results/family_wide_bh"], "benchmarks/sensitivity_scripts/family_wide_bh_interaction_networks.py"),
    Panel("e", "Block-jackknife retention, Atera", CACHE, ["benchmarks/sensitivity_scripts/results/spatial_block_jackknife"], "benchmarks/sensitivity_scripts/spatial_block_jackknife_gene_significance.py"),
    Panel("f", "Per-pair vs family-wide BH, tonsil", CACHE, ["benchmarks/sensitivity_scripts/results/family_wide_bh"], "benchmarks/sensitivity_scripts/family_wide_bh_interaction_networks.py"),
    Panel("g", "Block-jackknife retention, tonsil", CACHE, ["benchmarks/sensitivity_scripts/results/spatial_block_jackknife"], "benchmarks/sensitivity_scripts/spatial_block_jackknife_gene_significance.py"),
    Panel(
        "h",
        "Interaction scores with and without stromal cells",
        CACHE,
        [
            "xenium/figures/stromal_bleeding_analysis/cache/with_without_stroma_edge_scores.csv",
            "xenium/figures/stromal_bleeding_analysis/cache/with_stroma_nonstromal_interaction_matrix.csv",
            "xenium/figures/stromal_bleeding_analysis/cache/without_stroma_interaction_matrix.csv",
        ],
        "xenium/xenium_stromal_bleeding_analysis.py",
    ),
    Panel(
        "i",
        "Distance-matched null for stromal attention",
        CACHE,
        ["xenium/figures/stromal_bleeding_analysis/cache/distance_matched_stromal_attention_null_metrics.csv"],
        "xenium/xenium_stromal_bleeding_analysis.py",
    ),
]

EXTENDED_DATA = {1: ED1, 2: ED2, 3: ED3, 4: ED4, 5: ED5, 6: ED6, 7: ED7, 8: ED8, 9: ED9}


def survey() -> None:
    """Print which ED panels can be exported now, which need a rerun, and which caches are missing."""
    ready: list[str] = []
    missing: list[tuple[str, list[str]]] = []
    reruns: list[tuple[str, str]] = []
    images: list[str] = []

    for number, panels in EXTENDED_DATA.items():
        for panel in panels:
            name = f"ED {number}{panel.panel}"
            if panel.kind == IMAGE:
                images.append(name)
            elif panel.kind == RECOMPUTE:
                reruns.append((name, panel.script))
            else:
                gaps = panel.missing_sources()
                if gaps:
                    missing.append((name, gaps))
                else:
                    ready.append(name)

    total = sum(len(p) for p in EXTENDED_DATA.values())
    graphs = total - len(images)
    print(f"{total} Extended Data panels: {graphs} graphs, {len(images)} images or schematics\n")
    print(f"Exportable from cache now ({len(ready)}/{graphs} graphs):")
    print("  " + ", ".join(ready) + "\n")
    print(f"Cache declared but not on this machine ({len(missing)}):")
    for name, gaps in missing:
        print(f"  {name}: {', '.join(gaps)}")
    print(f"\nNeeds the analysis rerun ({len(reruns)}):")
    for name, script in reruns:
        print(f"  {name}: {script}")
    print(f"\nNo numerical graph, no Source Data needed ({len(images)}):")
    print("  " + ", ".join(images))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", action="store_true", help="print the coverage report and exit")
    parser.add_argument("--export", action="store_true", help="write the exportable panels to source_data/panels/ED*/")
    args = parser.parse_args()
    if args.survey:
        survey()
    if args.export:
        export()
    if not (args.survey or args.export):
        parser.error("pass --survey or --export")



# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------
# Each entry maps a panel to the tables that hold its numbers. ``copy`` writes one cached table out
# as it stands; ``concat`` gathers a set of per-seed or per-dataset tables into one tidy table, adding
# the column that distinguishes them (taken from the file name). Panel kinds that need a rerun have no
# entry here, so the exporter reports them as outstanding rather than inventing anything.

EXPORT_ROOT = os.path.join(REPO_ROOT, "source_data", "panels")

EXPORTS: dict[str, list[dict]] = {
    # ED 1
    "1c": [{"glob": "benchmarks/results_fig2_original/3ct_dataset_2way_*/*_task_pr.csv", "dest": "ED1c_pr_curves.csv",
            "key": "source", "mode": "pr_concat"}],
    # ED 1d: the published run kept only PR curves, not the per-method interaction scores the
    # networks are drawn from, so this panel needs the benchmark rerun rather than a cached table.

    "1f": [{"glob": "benchmarks/results/3ct_dataset_2way_*/*_task_pr.csv", "dest": "ED1f_sweep_pr_curves.csv",
            "key": "source", "mode": "pr_concat"}],
    "1g": [{"src": "source_data/panels/Fig3/Fig3b_full_matrix.csv", "dest": "ED1g_cortex_interaction_matrix.csv"}],
    "1h": [{"src": "xenium/figures/replicate_validation/rep1_interaction_weight_matrix.csv", "dest": "ED1h_replicate1_interaction_matrix.csv"},
           {"src": "xenium/figures/replicate_validation/rep2_interaction_weight_matrix.csv", "dest": "ED1h_replicate2_interaction_matrix.csv"}],
    "1i": [{"src": "xenium/figures/replicate_validation/Fig_S5b_redraw_points.csv", "dest": "ED1i_replicate_correlation_points.csv"},
           {"src": "xenium/figures/replicate_validation/Fig_S5b_redraw_stats.csv", "dest": "ED1i_replicate_correlation_stats.csv"}],
    # ED 2
    "2a": [{"src": "xenium/figures/stromal_bleeding_analysis/cache/interaction_weight_matrix.csv", "dest": "ED2a_interaction_matrix_with_stroma.csv"}],
    "2g": [{"src": "benchmarks/sensitivity_scripts/figures/xenium_adjusted_cd8_tumor_effect/cd8_tumor_adjusted_coefficients.csv", "dest": "ED2g_adjusted_coefficients.csv"},
           {"src": "benchmarks/sensitivity_scripts/figures/xenium_adjusted_cd8_tumor_effect/cd8_tumor_adjusted_spatial_values.csv", "dest": "ED2g_adjusted_spatial_values.csv"},
           {"src": "benchmarks/sensitivity_scripts/figures/xenium_adjusted_cd8_tumor_effect/cd8_tumor_adjustment_summary.csv", "dest": "ED2g_adjustment_summary.csv"}],
    "2h": [{"src": "source_data/panels/Fig4/Fig4c.csv", "dest": "ED2h_segmentation_test.csv"}],
    # ED 3
    "3b": [{"src": "xenium/results/xenium_length_scale_seed_sweep_reanalysis_v4/xenium_length_scale_seed_reanalysis_by_rank.csv", "dest": "ED3b_xenium_length_scales_by_seed.csv"}],
    "3c": [{"src": "human_tonsil/results/human_tonsil_length_scale_seed_sweep_reanalysis/human_tonsil_length_scale_seed_reanalysis_long.csv", "dest": "ED3c_tonsil_length_scales_by_seed.csv"}],
    "3d": [{"src": "xenium/data/xenium_sample1/hub_analysis_cache/hub_cluster_assignments.csv", "dest": "ED3d_hub_cluster_assignments.csv"}],
    "3e": [{"src": "source_data/panels/Fig4/Fig4f_sankey_flows.csv", "dest": "ED3e_hub_sender_receiver_composition.csv"}],
    "3f": [{"src": "benchmarks/sensitivity_scripts/figures/length_scale_uncertainty/length_scale_samples.csv", "dest": "ED3f_length_scale_samples.csv"},
           {"src": "benchmarks/sensitivity_scripts/figures/length_scale_uncertainty/length_scale_bootstrap_ci.csv", "dest": "ED3f_bootstrap_ci.csv"}],
    "3g": [{"src": "benchmarks/sensitivity_scripts/figures/length_scale_dataset_bootstrap_ci/length_scale_samples.csv", "dest": "ED3g_pbmc_replicate_samples.csv"},
           {"src": "benchmarks/sensitivity_scripts/figures/length_scale_dataset_bootstrap_ci/length_scale_dataset_bootstrap_ci.csv", "dest": "ED3g_pbmc_bootstrap_ci.csv"}],
    "3h": [{"src": "benchmarks/sensitivity_scripts/figures/realistic_length_scale_dataset_bootstrap_ci/length_scale_samples.csv", "dest": "ED3h_realistic_replicate_samples.csv"},
           {"src": "benchmarks/sensitivity_scripts/figures/realistic_length_scale_dataset_bootstrap_ci/realistic_length_scale_dataset_bootstrap_ci.csv", "dest": "ED3h_realistic_bootstrap_ci.csv"}],
    "3i": [{"src": "benchmarks/sensitivity_scripts/figures/realistic_length_scale_dataset_bootstrap_ci/length_scale_model_estimates.csv", "dest": "ED3i_realistic_model_estimates.csv"}],
    "3j": [{"glob": "benchmarks/sensitivity_scripts/figures/length_scale_coordinate_*_shuffle_sensitivity/*_estimates.csv",
            "dest": "ED3j_coordinate_shuffle_estimates.csv", "key": "source", "mode": "concat"}],
    # ED 4
    # ED 4b and the ED 4e-g butterfly plots all read the same GSEA table, subset by cell-type group,
    # so it is exported once.
    "4b": [{"src": "xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_gsea_results.csv", "dest": "ED4b_gsea_results.csv"}],
    "4c": [{"src": "xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_ari_ami.csv", "dest": "ED4c_hub_agreement_ari_ami.csv"}],
    "4d": [{"src": "xenium/data/xenium_sample1/hub_analysis_cache/hub_analysis_pathway_overlap_summary.csv", "dest": "ED4d_pathway_overlap_summary.csv"}],
    "4h": [{"src": "xenium/data/xenium_sample1/hub_graph_baseline_cache/combined_ari_ami_metrics.csv", "dest": "ED4h_graph_baseline_ari_ami.csv"}],
    # ED 5
    "5c": [{"src": "atera_breast/figures/atera_breast_analysis/atera_breast_t_cell_caf_group_de.csv", "dest": "ED5c_caf_sender_de.csv"},
           {"src": "atera_breast/figures/atera_breast_analysis/atera_breast_t_cell_caf_group_attention_scores.csv", "dest": "ED5c_caf_sender_attention_scores.csv"}],
    "5d": [{"src": "source_data/panels/Fig4/Fig4b_full_matrix.csv", "dest": "ED5d_highres_interaction_matrix.csv"}],
    "5h": [{"src": "xenium/data/xenium_sample1/within_type_autocorrelation_cache/within_type_autocorrelation_validation_scores.csv", "dest": "ED5h_within_label_autocorrelation.csv"}],
    "5i": [{"srcs": ["xenium/runtime_benchmark/results/results_atera_gene_scaling_rerun_2026-07-28/benchmark_results_cpu.json",
                     "xenium/runtime_benchmark/results/results_atera_gene_scaling_rerun_2026-07-28/benchmark_results_gpu.json"],
            "dest": "ED5i_runtime.csv", "mode": "runtime_json"}],
    "5j": [{"glob": "benchmarks/results_cv/breast_cancer_42/*_task_scores.csv", "dest": "ED5j_spatial_cv_task_scores.csv", "key": "source", "mode": "concat"}],
    # ED 6
    "6a": [{"src": "atera_breast/results/attention_consistency/atera_breast_attention_consistency_runs.csv", "dest": "ED6a_atera_attention_consistency_runs.csv"},
           {"src": "atera_breast/results/attention_consistency/atera_breast_attention_consistency_correlations.csv", "dest": "ED6a_atera_attention_consistency_correlations.csv"}],
    "6b": [{"src": "human_tonsil/results/attention_consistency/human_tonsil_attention_consistency_runs.csv", "dest": "ED6b_tonsil_attention_consistency_runs.csv"},
           {"src": "human_tonsil/results/attention_consistency/human_tonsil_attention_consistency_correlations.csv", "dest": "ED6b_tonsil_attention_consistency_correlations.csv"}],
    "6c": [{"src": "benchmarks/figures/test_region_composition_diagnostic/test_region_composition_jsd_summary.csv", "dest": "ED6c_test_region_jsd.csv"}],
    "6d": [{"src": "benchmarks/figures/test_region_composition_diagnostic/test_region_cell_type_composition.csv", "dest": "ED6d_test_region_composition.csv"}],
    "6e": [{"src": "benchmarks/sensitivity_scripts/results/value_penalty_analysis/value_penalty_analysis_auprc.csv", "dest": "ED6e_value_penalty_auprc.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/value_penalty_analysis/value_penalty_analysis_pr_curves.csv", "dest": "ED6e_value_penalty_pr_curves.csv"}],
    "6f": [{"src": "benchmarks/sensitivity_scripts/results/attention_penalty_analysis/attention_penalty_analysis_auprc.csv", "dest": "ED6f_attention_penalty_auprc.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/attention_penalty_analysis/attention_penalty_analysis_pr_curves.csv", "dest": "ED6f_attention_penalty_pr_curves.csv"}],
    "6g": [{"src": "benchmarks/sensitivity_scripts/results/head_analysis/head_analysis_auprc.csv", "dest": "ED6g_head_auprc.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/head_analysis/head_analysis_pr_curves.csv", "dest": "ED6g_head_pr_curves.csv"}],
    "6h": [{"src": "benchmarks/sensitivity_scripts/results/neighbor_analysis/neighbor_analysis_auprc.csv", "dest": "ED6h_neighbor_auprc.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/neighbor_analysis/neighbor_analysis_pr_curves.csv", "dest": "ED6h_neighbor_pr_curves.csv"}],
    # ED 7
    "7b": [{"src": "xenium/data/xenium_sample1/hub_analysis_cache/hub_cluster_assignments.csv", "dest": "ED7b_hub_cluster_assignments.csv"}],
    "7e": [{"src": "benchmarks/sensitivity_scripts/2_phase/results/two_phase_amici_sweep/two_phase_selected_head_length_scale_samples.csv", "dest": "ED7e_two_phase_head_length_scales.csv"},
           {"src": "benchmarks/sensitivity_scripts/2_phase/results/two_phase_amici_sweep/two_phase_head_length_scale_summary.csv", "dest": "ED7e_two_phase_head_summary.csv"}],
    "7f": [{"src": "benchmarks/sensitivity_scripts/2_phase/results/two_phase_amici_sweep/two_phase_task_scores.csv", "dest": "ED7f_two_phase_task_scores.csv"}],
    "7g": [{"src": "benchmarks/sensitivity_scripts/unimodal_two_phase/results/unimodal_two_phase_baseline_sweep/unimodal_baseline_auprc_comparison.csv", "dest": "ED7g_unimodal_baseline_auprc.csv"}],
    "7i": [{"src": "benchmarks/sensitivity_scripts/unimodal_two_phase/results/unimodal_two_phase_amici_sweep/unimodal_two_phase_length_scale_estimates.csv", "dest": "ED7i_unimodal_length_scales.csv"}],
    "7j": [{"src": "human_tonsil/figures/empirical_distance_expression/human_tonsil_distance_binned_expression.csv", "dest": "ED7j_tonsil_distance_profiles.csv"}],
    "7k": [{"src": "figures/monotonicity/monotonicity_by_pair.csv", "dest": "ED7k_monotonicity_by_pair.csv"}],
    # ED 8
    "8a": [{"src": "benchmarks/sensitivity_scripts/results/amici_variants/unimodal_peak_interaction_distances.csv", "dest": "ED8a_unimodal_peak_distances.csv"}],
    "8b": [{"src": "benchmarks/sensitivity_scripts/results/unimodal_attention_analysis/unimodal_attention_length_scale_samples.csv", "dest": "ED8b_unimodal_attention_length_scales.csv"}],
    "8c": [{"src": "benchmarks/sensitivity_scripts/results/unconstrained_attention_analysis/unconstrained_attention_length_scale_samples.csv", "dest": "ED8c_unconstrained_length_scales.csv"}],
    "8d": [{"src": "benchmarks/sensitivity_scripts/results/amici_variants/amici_variant_task_scores.csv", "dest": "ED8d_variant_task_scores.csv"}],
    "8e": [{"src": "benchmarks/sensitivity_scripts/figures/neighbor_occlusion_analysis/neighbor_occlusion_summary.csv", "dest": "ED8e_occlusion_auprc.csv"}],
    "8g": [{"src": "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_summary.csv", "dest": "ED8g_matched_occlusion_summary.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_per_cell.csv", "dest": "ED8g_matched_occlusion_per_cell.csv"},
           {"src": "benchmarks/sensitivity_scripts/results/matched_neighbor_occlusion_analysis/matched_occlusion_diagnostics.csv", "dest": "ED8g_matched_occlusion_distance_summary.csv", "mode": "occlusion_diagnostics"}],
    # ED 9
    "9a": [{"src": "benchmarks/sensitivity_scripts/figures/coordinate_noise_gradient_sensitivity_sweep/coordinate_noise_sensitivity_all_runs.csv", "dest": "ED9a_noise_sensitivity_runs.csv"}],
    "9b": [{"src": "benchmarks/sensitivity_scripts/results/family_wide_bh/xenium_per_pair_counts.csv", "dest": "ED9b_xenium_family_wide_bh.csv"}],
    "9c": [{"src": "benchmarks/sensitivity_scripts/results/spatial_block_jackknife/xenium_jackknife_500um_per_pair.csv", "dest": "ED9c_xenium_jackknife_500um.csv"}],
    "9d": [{"src": "benchmarks/sensitivity_scripts/results/family_wide_bh/atera_breast_per_pair_counts.csv", "dest": "ED9d_atera_family_wide_bh.csv"}],
    "9e": [{"src": "benchmarks/sensitivity_scripts/results/spatial_block_jackknife/atera_breast_jackknife_500um_per_pair.csv", "dest": "ED9e_atera_jackknife_500um.csv"}],
    "9f": [{"src": "benchmarks/sensitivity_scripts/results/family_wide_bh/human_tonsil_per_pair_counts.csv", "dest": "ED9f_tonsil_family_wide_bh.csv"}],
    "9g": [{"src": "benchmarks/sensitivity_scripts/results/spatial_block_jackknife/human_tonsil_jackknife_500um_per_pair.csv", "dest": "ED9g_tonsil_jackknife_500um.csv"}],
    "9h": [{"src": "xenium/figures/stromal_bleeding_analysis/cache/with_without_stroma_edge_scores.csv", "dest": "ED9h_with_without_stroma_edges.csv"},
           {"src": "xenium/figures/stromal_bleeding_analysis/cache/with_stroma_nonstromal_interaction_matrix.csv", "dest": "ED9h_with_stroma_matrix.csv"},
           {"src": "xenium/figures/stromal_bleeding_analysis/cache/without_stroma_interaction_matrix.csv", "dest": "ED9h_without_stroma_matrix.csv"}],
    "9i": [{"src": "xenium/figures/stromal_bleeding_analysis/cache/distance_matched_stromal_attention_null_metrics.csv", "dest": "ED9i_distance_matched_null.csv"}],
}


PR_CURVE_POINTS = 501


def _downsample_pr_curve(frame, points: int = PR_CURVE_POINTS):
    """Thin a precision-recall curve onto a fixed recall grid.

    The raw curves carry one row per decision threshold, which runs to millions of rows for the
    cell-level tasks and would put a single Source Data file far over the journal's size guidance.
    Interpolating onto `points` evenly spaced recall values preserves the curve at print resolution
    and keeps avg_precision_score, which is the number quoted in the text, exactly as computed.
    """
    import numpy as np
    import pandas as pd

    if len(frame) <= points:
        return frame
    ordered = frame.sort_values("recall")
    grid = np.linspace(ordered["recall"].min(), ordered["recall"].max(), points)
    thinned = pd.DataFrame(
        {
            "recall": grid,
            "precision": np.interp(grid, ordered["recall"].to_numpy(), ordered["precision"].to_numpy()),
        }
    )
    for column in frame.columns:
        if column not in ("recall", "precision"):
            thinned[column] = frame[column].iloc[0]
    thinned.attrs["downsampled_from"] = len(frame)
    return thinned


def _summarize_occlusion_diagnostics(source: str, destination: str) -> int:
    """Summarise the per-edge matching diagnostics that ED 8g's right sub-panel shows.

    The raw table is one row per candidate neighbor edge (4.2 million rows, 250 MB). The panel plots
    the distance distribution per arm, so the distances are summarised per dataset and arm instead.
    """
    import pandas as pd

    raw = pd.read_csv(source)
    summary = (
        raw.groupby(["dataset", "arm"])["distance"]
        .agg(n="size", mean="mean", std="std", minimum="min", q1=lambda v: v.quantile(0.25),
             median="median", q3=lambda v: v.quantile(0.75), maximum="max")
        .reset_index()
    )
    summary.to_csv(destination, index=False)
    return len(summary)


def _runtime_json_to_table(sources: list[str], destination: str) -> int:
    """Flatten the runtime benchmark JSONs into the table ED 5i plots.

    Each JSON holds a metadata block and one record per (cells, genes, accelerator) run; the panel
    plots wall-clock time and seconds per epoch against the number of cells, one line per gene panel.
    """
    import json as jsonmod

    import pandas as pd

    rows = []
    for source in sources:
        with open(source) as handle:
            payload = jsonmod.load(handle)
        epochs = payload.get("metadata", {}).get("train_params", {}).get("max_epochs")
        for record in payload["results"]:
            entry = dict(record)
            if epochs:
                entry["n_epochs"] = epochs
                entry["runtime_s_per_epoch"] = entry["runtime_train_s"] / epochs
            rows.append(entry)
    table = pd.DataFrame(rows)
    table.to_csv(destination, index=False)
    return len(table)


def _export_one(spec: dict, out_dir: str) -> tuple[str, int]:
    """Write one exported table. Returns the output path and its row count."""
    import glob as globmod

    import pandas as pd

    destination = os.path.join(out_dir, spec["dest"])
    if spec.get("mode") in ("concat", "pr_concat"):
        paths = sorted(globmod.glob(os.path.join(REPO_ROOT, spec["glob"])))
        if not paths:
            raise FileNotFoundError(spec["glob"])
        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            if spec.get("mode") == "pr_concat":
                frame = _downsample_pr_curve(frame)
            # The distinguishing information is in the path: dataset folder and file stem.
            frame.insert(0, spec.get("key", "source"), os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)))
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(destination, index=False)
        return destination, len(table)

    if spec.get("mode") == "runtime_json":
        sources = [os.path.join(REPO_ROOT, src) for src in spec["srcs"]]
        return destination, _runtime_json_to_table(sources, destination)

    source = os.path.join(REPO_ROOT, spec["src"])
    if spec.get("mode") == "occlusion_diagnostics":
        return destination, _summarize_occlusion_diagnostics(source, destination)
    if spec["dest"].endswith(".json"):
        import shutil

        shutil.copyfile(source, destination)
        return destination, 0
    table = pd.read_csv(source)
    table.to_csv(destination, index=False)
    return destination, len(table)


def export() -> None:
    """Write every exportable Extended Data panel into source_data/panels/ED<n>/."""
    import json

    written: list[dict] = []
    failures: list[tuple[str, str]] = []

    for number, panels in EXTENDED_DATA.items():
        out_dir = os.path.join(EXPORT_ROOT, f"ED{number}")
        manifest: list[dict] = []
        for panel in panels:
            key = f"{number}{panel.panel}"
            specs = EXPORTS.get(key)
            if not specs:
                continue
            os.makedirs(out_dir, exist_ok=True)
            for spec in specs:
                try:
                    path, rows = _export_one(spec, out_dir)
                except Exception as exc:  # a cache that moved or a malformed table
                    failures.append((key, f"{spec.get('src') or spec.get('glob') or spec.get('srcs')}: {exc}"))
                    continue
                entry = {
                    "panel": f"ED {number}{panel.panel}",
                    "description": panel.description,
                    "file": os.path.basename(path),
                    "rows": rows,
                    "source": spec.get("src") or spec.get("glob") or ", ".join(spec.get("srcs", [])),
                    "produced_by": panel.script,
                }
                manifest.append(entry)
                written.append(entry)
        if manifest:
            titles = {
                1: "Extended Data Fig. 1: architecture, benchmark diagnostics and reproducibility across sections",
                2: "Extended Data Fig. 2: additional analyses of the Xenium breast cancer dataset",
                3: "Extended Data Fig. 3: segmentation testing, length-scale uncertainty and hub selection",
                4: "Extended Data Fig. 4: communication hubs against niche composition and graph baselines",
                5: "Extended Data Fig. 5: CAF subpopulations, cell-type resolution, runtime and spatial cross-validation",
                6: "Extended Data Fig. 6: model selection, attention stability and hyperparameter sensitivity",
                7: "Extended Data Fig. 7: threshold sensitivity, hub stability, multi-scale interactions and monotonicity",
                8: "Extended Data Fig. 8: attention variants, neighbor occlusion and negative controls",
                9: "Extended Data Fig. 9: noise, multiple-testing family, spatial dependence and stromal exclusion",
            }
            shared = {
                "ED 4e": "ED 4e (stromal cell types) is the same GSEA table as ED 4b, subset by cell-type group: sheet ED4b_gsea_results",
                "ED 4f": "ED 4f (tumor cell types): sheet ED4b_gsea_results",
                "ED 4g": "ED 4g (immune cell types): sheet ED4b_gsea_results",
                "ED 7c": "ED 7c (quantile threshold grid) shares the hub assignments in sheet ED7b_hub_cluster_assignments",
            }
            sheets = []
            readme = [["Figure", titles[number]]]
            for entry in manifest:
                if entry["file"].endswith(".json"):
                    readme.append([entry["panel"], f"{entry['description']} - {entry['file']} (JSON, in the companion zip)"])
                    continue
                sheet_name = entry["file"].rsplit(".", 1)[0][:31]
                sheets.append(
                    {
                        "sheet": sheet_name,
                        "csv": entry["file"],
                        "description": f"{entry['description']}. Source: {entry['source']}"
                        + (f"; produced by {entry['produced_by']}" if entry["produced_by"] else ""),
                        "index": False,
                        "rows": entry["rows"],
                    }
                )
                readme.append([entry["panel"], f"Sheet {sheet_name}"])
            for panel_name, text in shared.items():
                if panel_name.startswith(f"ED {number}"):
                    readme.append([panel_name, text])
            with open(os.path.join(out_dir, "manifest.json"), "w") as handle:
                json.dump({"figure": f"ED{number}", "provenance": {}, "readme": readme, "sheets": sheets}, handle, indent=2)

    print(f"wrote {len(written)} tables across {len({w['panel'][:5] for w in written})} Extended Data figures")
    for entry in written:
        print(f"  {entry['panel']:7s} {entry['file']:52s} {entry['rows']:>8,} rows")
    if failures:
        print(f"\n{len(failures)} failed:")
        for key, message in failures:
            print(f"  ED {key}: {message}")

if __name__ == "__main__":
    main()
