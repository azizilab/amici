"""Source Data for Figure 2 (semisynthetic benchmarks).

Panels
  2a  Spatial cell-type map of the example PBMC grid dataset.
  2b  AMICI length scales on that dataset, with the ground-truth length scales.
  2c-e  AUPRC per PBMC dataset seed for the gene, sender and receiver tasks, with the two-sided Mann-Whitney
        U tests against AMICI (Bonferroni-corrected, as in benchmarks/benchmark_utils.plot_boxplots).
  2f  Spatial cell-type map of the realistic breast cancer dataset and its interacting subpopulations.
  2g  AMICI length scales on the realistic dataset, with the ground-truth length scales.
  2h-j  AUPRC per method on the realistic dataset.

Reads the outputs of the Snakemake benchmark (benchmarks/Snakefile, benchmarks/benchmark_config.yaml).
Run from the repository root:
    python source_data/fig2_benchmarks.py --results benchmarks/results --example-seed <seed>
Panels whose inputs are missing are reported and skipped; nothing is written for an incomplete task unless
--allow-incomplete is given.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import _common as C

FIGURE = "Fig2"
CONFIG = C.REPO / "benchmarks" / "benchmark_config.yaml"

PBMC = "3ct_dataset_2way"
REALISTIC = "breast_cancer"
REALISTIC_SEED = 42
PBMC_LABELS = {"0": "Cell type A", "2": "Cell type B", "3": "Cell type C"}
PBMC_INTERACTING = {"0_sub1": "Cell type A (interacting)", "2_sub1": "Cell type B (interacting)"}
REALISTIC_LABELS = {
    "T_Cells": "T Cells",
    "B_Cells": "B Cells",
    "Invasive_Tumor": "Invasive Tumor",
    "Myoepi": "Myoepithelial",
}

TASKS = {
    "gene": {
        "panels": ("2c", "2h"),
        "title": "Task 1: predicting genes downstream of interaction",
        "methods": [
            ("AMICI", "amici_gene_task_pr.csv"),
            ("GITIII", "gitiii_gene_task_pr.csv"),
            ("NicheDE (sigma = 20)", "nichede_20_gene_task_pr.csv"),
            ("NicheDE (sigma = 200)", "nichede_200_gene_task_pr.csv"),
            ("NicheDE (sigma = 500)", "nichede_500_gene_task_pr.csv"),
            ("NCEM (rad = 10)", "ncem_10_gene_task_pr.csv"),
            ("NCEM (rad = 15)", "ncem_15_gene_task_pr.csv"),
            ("NCEM (rad = 20)", "ncem_20_gene_task_pr.csv"),
        ],
    },
    "sender": {
        "panels": ("2d", "2i"),
        "title": "Task 2: sender cell prediction",
        "methods": [
            ("AMICI", "amici_neighbor_interaction_task_pr.csv"),
            ("GITIII", "gitiii_neighbor_interaction_task_pr.csv"),
            ("CGCom", "cgcom_neighbor_interaction_task_pr.csv"),
        ],
    },
    "receiver": {
        "panels": ("2e", "2j"),
        "title": "Task 3: receiver cell prediction",
        "methods": [
            ("AMICI", "amici_receiver_subtype_task_pr.csv"),
            ("GITIII", "gitiii_receiver_subtype_task_pr.csv"),
            ("CGCom", "cgcom_receiver_subtype_task_pr.csv"),
        ],
    },
}


# Fig. 2h-j bar heights as published. The realistic run behind them was not kept; these values were read off the
# bars of a vector copy of the figure (2026-07-28) and sit about 0.0007 below the AUPRCs that deterministic methods
# (NCEM, NicheDE sigma = 20) reproduce on the same dataset.
PUBLISHED_REALISTIC_VALUES = (
    C.REPO / "benchmarks" / "results_cv" / "breast_cancer" / "regular_vs_cv_barplots" / "regular_recovered_auprc_values.csv"
)
PUBLISHED_TASK_NAMES = {"gene": "Gene", "sender": "Sender", "receiver": "Receiver"}
PUBLISHED_METHOD_NAMES = {
    "AMICI": "AMICI",
    "GITIII": "GITIII",
    "CGCom": "CGCom",
    "NicheDE (sigma = 20)": "NicheDE_20",
    "NicheDE (sigma = 200)": "NicheDE_200",
    "NicheDE (sigma = 500)": "NicheDE_500",
    "NCEM (rad = 10)": "NCEM_10",
    "NCEM (rad = 15)": "NCEM_15",
    "NCEM (rad = 20)": "NCEM_20",
}


def read_auprc(path):
    if not path.exists():
        return np.nan
    # "+ 0.0" turns the -0.0 stored for degenerate CGCom runs into 0.0.
    return float(pd.read_csv(path)["avg_precision_score"].values[0]) + 0.0


# Bonferroni-corrected p-values printed in the Fig. 2 caption, used to confirm the results folder is the published run.
CAPTION_P = {
    "gene": {
        "NicheDE (sigma = 20)": 0.0013,
        "NicheDE (sigma = 200)": 0.0013,
        "NicheDE (sigma = 500)": 0.0023,
        "NCEM (rad = 10)": 0.0013,
        "NCEM (rad = 15)": 0.0013,
        "NCEM (rad = 20)": 0.0013,
    },
    "sender": {"GITIII": 0.028, "CGCom": 0.0003},
    "receiver": {"GITIII": 0.028, "CGCom": 0.0004},
}


def stars(p):
    if p < 0.001:
        return "****"
    if p < 0.005:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def pbmc_task_panels(results, seeds, task, allow_incomplete):
    spec = TASKS[task]
    rows = []
    for seed in seeds:
        folder = results / f"{PBMC}_{seed}"
        row = {"dataset_seed": seed}
        for name, fname in spec["methods"]:
            row[name] = read_auprc(folder / fname)
        rows.append(row)
    wide = pd.DataFrame(rows)
    methods = [m for m, _ in spec["methods"]]
    complete = wide[methods].notna().all(axis=1)
    missing = wide.loc[~complete, "dataset_seed"].tolist()
    if missing:
        print(f"  {spec['panels'][0]}: seeds with missing files: {missing}")
        if not allow_incomplete:
            return None
    # plot_boxplots.py drops a seed when any method's file is missing for that task.
    wide = wide[complete].reset_index(drop=True)

    amici = wide["AMICI"].to_numpy()
    n_comparisons = len(methods) - 1
    summary = []
    for name in methods:
        vals = wide[name].to_numpy()
        entry = {
            "method": name,
            "n_datasets": len(vals),
            "median": np.median(vals),
            "q1": np.percentile(vals, 25),
            "q3": np.percentile(vals, 75),
            "min": vals.min(),
            "max": vals.max(),
        }
        if name != "AMICI":
            # scipy returns the U of the first sample, i.e. U of AMICI over this baseline; it runs from 0 to
            # n1 * n2 (= 100 here, 10 datasets per group).
            u, p = stats.mannwhitneyu(amici, vals, alternative="two-sided")
            u_max = len(amici) * len(vals)
            entry.update(
                {
                    "mannwhitney_U": u,
                    "U_max_n1n2": u_max,
                    "mannwhitney_p_vs_AMICI": p,
                    "p_bonferroni": p * n_comparisons,
                    "significance_stars": stars(p * n_comparisons),
                    "rank_biserial_r": 2.0 * u / u_max - 1.0,
                    "cles_prob_amici_higher": u / u_max,
                }
            )
        summary.append(entry)
    return wide, pd.DataFrame(summary)


def pbmc_cells_from_run(results, seed):
    """Per-cell coordinates and labels of a PBMC dataset, rebuilt from files the benchmark run itself kept.

    GITIII's processed input (saved_models/data/processed/slide1.csv) holds every cell's coordinates and cell type
    in dataset order, and ground_truth_receiver_subtypes.csv flags the interacting receiver subpopulations in the
    same order. Returns None when either file is missing.
    """
    folder = results / f"{PBMC}_{seed}"
    slide = folder / "saved_models" / "data" / "processed" / "slide1.csv"
    gt = folder / "ground_truth_receiver_subtypes.csv"
    if not (slide.exists() and gt.exists()):
        return None
    cells = pd.read_csv(slide, usecols=["centerx", "centery", "subclass"])
    flags = pd.read_csv(gt)
    assert len(cells) == len(flags), "slide1.csv and ground_truth_receiver_subtypes.csv differ in length"
    leiden = cells["subclass"].str.replace("ct_", "", regex=False)
    interacting = flags["class"].to_numpy() == 1
    # Only receiver types (A, B) carry interacting subpopulations; a row offset would put flags on type C cells.
    assert not (interacting & (leiden == "3").to_numpy()).any(), "files are not in the same cell order"
    cell_type = leiden.map(PBMC_LABELS)
    legend = cell_type.where(~interacting, cell_type + " (interacting)")
    return pd.DataFrame(
        {
            "cell_id": flags["cell_idx"].values,
            "x_um": cells["centerx"].values,
            "y_um": cells["centery"].values,
            "cell_type": cell_type.values,
            "interacting_subpopulation": interacting,
            "legend_label": legend.values,
        }
    )


def data_path(results, dataset, seed):
    """Dataset a run used: its own data/ copy when the run set dir_path there, else benchmarks/data/."""
    local = results / "data" / f"{dataset}_{seed}.h5ad"
    return local if local.exists() else C.REPO / "benchmarks" / "data" / f"{dataset}_{seed}.h5ad"


def dataset_matches_run(dataset, seed, results):
    """True when benchmarks/data/{dataset}_{seed}.h5ad has the genes the run's AMICI model was trained on."""
    import anndata as ad
    import torch

    model_file = results / f"{dataset}_{seed}" / "saved_models" / f"amici_{dataset}_{seed}_best_model" / "model.pt"
    data_file = data_path(results, dataset, seed)
    if not (model_file.exists() and data_file.exists()):
        return False
    trained = np.asarray(torch.load(model_file, map_location="cpu", weights_only=False)["var_names"]).astype(str)
    return np.array_equal(trained, np.asarray(ad.read_h5ad(data_file, backed="r").var_names).astype(str))


def selected_amici_model(results, dataset, seed):
    """Head count and training seed of the AMICI model a sweep selected, e.g. '14 heads, seed 42'."""
    import torch

    models = results / f"{dataset}_{seed}" / "saved_models"
    state = torch.load(models / f"amici_{dataset}_{seed}_best_model" / "model.pt", map_location="cpu", weights_only=False)
    n_heads = state["attr_dict"]["init_params_"]["kwargs"]["model_kwargs"]["n_heads"]
    training_seed = (models / f"amici_{dataset}_{seed}_best_seed.txt").read_text().strip()
    return f"{n_heads} heads, seed {training_seed}"


def length_scale_panel(dataset, seed, results, dataset_config):
    """Per-sender-cell length scales on the max-explained-variance head (benchmarks/length_scale_task/plot_kde.py)."""
    import scanpy as sc

    model_dir = results / f"{dataset}_{seed}" / "saved_models" / f"amici_{dataset}_{seed}_best_model"
    if not (model_dir / "model.pt").exists():
        print(f"  no AMICI model at {model_dir}; skipping length scales")
        return None
    import torch

    from amici import AMICI

    adata = sc.read_h5ad(data_path(results, dataset, seed))
    trained_genes = np.asarray(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)["var_names"])
    if not np.array_equal(trained_genes.astype(str), np.asarray(adata.var_names).astype(str)):
        print(f"  {dataset}_{seed}.h5ad is not the dataset this model was trained on (gene lists differ); skipping")
        return None
    model = AMICI.load(str(model_dir), adata=adata)
    AMICI.setup_anndata(adata, labels_key=dataset_config["labels_key"], coord_obsm_key="spatial", n_neighbors=50)
    # The head is chosen by explained variance alone, so the permutation test (p-values only) is skipped.
    expl_variance = model.get_expl_variance_scores(adata=adata, run_permutation_test=False)
    frames = []
    for interaction in dataset_config["gt_interactions"].values():
        receiver, sender = interaction["receiver"], interaction["sender"]
        cf = model.get_counterfactual_attention_patterns(adata=adata, cell_type=receiver)
        ls = cf._calculate_length_scales(
            head_idxs=np.arange(model.module.n_heads), sender_types=[sender], attention_threshold=0.1
        )
        head = expl_variance.compute_max_explained_variance_head(cell_type=receiver)
        ls = ls[ls["head_idx"] == head].copy()
        ls["sender"] = sender
        ls["receiver"] = receiver
        ls["ground_truth_length_scale_um"] = interaction["length_scale"]
        frames.append(ls)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"neighbor_idx": "sender_cell_id", "length_scale": "length_scale_um", "head_idx": "head"})
    return out[["sender", "receiver", "head", "sender_cell_id", "length_scale_um", "ground_truth_length_scale_um"]]


def kde_curves_from_svg(svg_path, labels):
    """Recover the KDE curves drawn in a matplotlib length_scale_distribution.svg, in data units.

    Tick positions and their labels (written by matplotlib as comments) give the exact affine map from SVG
    coordinates to data coordinates. The curves are the long line2d paths; seaborn draws the hue levels in
    reverse, so each curve is matched to its label by stroke colour in the default palette order.
    """
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    import re

    text = Path(svg_path).read_text()

    def ticks(axis):
        out = []
        for m in re.finditer(rf'<g id="{axis}tick_\d+">(.*?)<!-- (.*?) -->', text, re.S):
            pos = re.search(r'<use [^>]*{}="([-0-9.]+)"'.format("x" if axis == "x" else "y"), m.group(1))
            out.append((float(pos.group(1)), float(m.group(2))))
        return np.array(out)

    def affine(pairs):
        slope, intercept = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
        resid = np.abs(pairs[:, 0] * slope + intercept - pairs[:, 1]).max()
        assert resid < 1e-6 * max(1.0, np.abs(pairs[:, 1]).max()), "ticks are not evenly mapped"
        return slope, intercept

    sx, ix = affine(ticks("x"))
    sy, iy = affine(ticks("y"))
    paths = re.findall(r'<g id="line2d_\d+">\s*<path d="([^"]+)"[^>]*style="[^"]*stroke: (#[0-9a-f]{6})', text)
    curves = [(d, color) for d, color in paths if d.count("L") > 50]
    assert len(curves) == len(labels), f"expected {len(labels)} curves, found {len(curves)}"
    frames = []
    for d, color in curves:
        label = labels[palette.index(color)]
        pts = np.array([[float(v) for v in xy.split()] for xy in re.findall(r"[ML] ([-0-9.]+ [-0-9.]+)", d)])
        frames.append(
            pd.DataFrame(
                {"interaction": label, "length_scale_um": pts[:, 0] * sx + ix, "density": pts[:, 1] * sy + iy, "svg_stroke": color}
            )
        )
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(C.REPO / "benchmarks" / "results"))
    parser.add_argument("--example-seed", type=int, default=None, help="PBMC seed shown in Fig. 2a-b")
    parser.add_argument(
        "--realistic-results", default=None, help="results folder of the realistic run, if not the same as --results"
    )
    parser.add_argument(
        "--rerun-source",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="results folder to read a method family's re-run AUPRCs from (AMICI, GITIII, CGCom, NicheDE, NCEM); "
        "defaults to --realistic-results",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    results = Path(args.results)
    config = yaml.safe_load(open(CONFIG))
    pbmc_cfg = config["datasets"][PBMC]
    real_cfg = config["datasets"][REALISTIC]
    C.ensure_dirs()
    panels = []

    # ---- 2a / 2b -------------------------------------------------------------------------------------
    example_matches = args.example_seed is not None and dataset_matches_run(PBMC, args.example_seed, results)
    if args.example_seed is not None and not example_matches:
        cells = pbmc_cells_from_run(results, args.example_seed)
        if cells is None:
            print(f"  2a: {PBMC}_{args.example_seed}.h5ad is not the run's dataset and the run kept no cell table; not written")
        else:
            print(f"  2a: rebuilt from the run's GITIII input and ground-truth subtypes ({len(cells)} cells)")
            panels.append(
                {
                    "sheet": "Fig2a",
                    "description": f"PBMC grid semisynthetic dataset (seed {args.example_seed}) as used by the benchmark "
                    "run: one row per cell with coordinates (um), cell type and interacting subpopulation, from the "
                    "run's GITIII input (slide1.csv) and ground_truth_receiver_subtypes.csv (same cell order).",
                    "df": cells,
                }
            )
    if example_matches:
        import anndata as ad

        a = ad.read_h5ad(data_path(results, PBMC, args.example_seed))
        coords = a.obsm["spatial"]
        leiden = a.obs["leiden"].astype(str)
        subtype = a.obs["subtype"].astype(str)
        panels.append(
            {
                "sheet": "Fig2a",
                "description": f"PBMC grid semisynthetic dataset (seed {args.example_seed}): one row per cell with "
                "coordinates (um), cell type and interacting subpopulation.",
                "df": pd.DataFrame(
                    {
                        "cell_id": a.obs_names,
                        "x_um": np.asarray(coords["X"]),
                        "y_um": np.asarray(coords["Y"]),
                        "cell_type": leiden.map(PBMC_LABELS).values,
                        "subtype": subtype.values,
                        "legend_label": subtype.map(PBMC_INTERACTING).fillna(leiden.map(PBMC_LABELS)).values,
                        "train_test_split": a.obs["train_test_split"].astype(str).values,
                    }
                ),
            }
        )
    if args.example_seed is not None:
        ls = length_scale_panel(PBMC, args.example_seed, results, pbmc_cfg)
        svg = results / f"{PBMC}_{args.example_seed}" / "length_scale_distribution.svg"
        if ls is None and svg.exists():
            gt = {}
            labels = []
            for inter in pbmc_cfg["gt_interactions"].values():
                label = f"{PBMC_LABELS[inter['sender']]} -> {PBMC_LABELS[inter['receiver']]}"
                labels.append(label)
                gt[label] = inter["length_scale"]
            curves = kde_curves_from_svg(svg, labels)
            curves["ground_truth_length_scale_um"] = curves["interaction"].map(gt)
            for label, g in curves.groupby("interaction"):
                print(f"  2b {label}: KDE mode {g.loc[g['density'].idxmax(), 'length_scale_um']:.2f} um (ground truth {gt[label]})")
            panels.append(
                {
                    "sheet": "Fig2b",
                    "description": f"KDE curves of AMICI length scales (um) on the PBMC dataset (seed {args.example_seed}), "
                    "read from the benchmark run's length_scale_distribution.svg (head with the largest explained "
                    "variance, attention threshold 0.1), with the ground-truth length scales drawn as dashed lines. "
                    "Per-cell values cannot be recomputed because the run's input dataset was not kept.",
                    "df": curves,
                }
            )
        if ls is not None:
            ls["interaction"] = ls["sender"].map(PBMC_LABELS) + " -> " + ls["receiver"].map(PBMC_LABELS)
            panels.append(
                {
                    "sheet": "Fig2b",
                    "description": "AMICI length scale (um) per sender cell on the head with the largest explained "
                    "variance for the receiver (attention threshold 0.1); the KDE of these values is drawn with the "
                    "ground-truth length scale as a dashed line.",
                    "df": ls,
                }
            )

    # ---- 2c-e ----------------------------------------------------------------------------------------
    for task, spec in TASKS.items():
        out = pbmc_task_panels(results, pbmc_cfg["seeds"], task, args.allow_incomplete)
        if out is None:
            continue
        wide, summary = out
        panel = spec["panels"][0]
        for method, published in CAPTION_P[task].items():
            got = float(summary.loc[summary["method"] == method, "p_bonferroni"].iloc[0])
            status = "MATCH" if round(got, 4) == published else "MISMATCH"
            print(f"  [{status}] {panel} {method}: p = {got:.4f} (caption {published})")
        panels.append(
            {
                "sheet": f"Fig{panel}",
                "description": f"{spec['title']}: AUPRC of each method on each PBMC dataset seed (one row per seed; "
                "rows are the connected points of the box plot).",
                "df": wide,
            }
        )
        panels.append(
            {
                "sheet": f"Fig{panel}_stats",
                "description": "Box-plot statistics and two-sided Mann-Whitney U tests of each method against AMICI, "
                "Bonferroni-corrected over the compared methods (stars: **** < 0.001, *** < 0.005, ** < 0.01, "
                "* < 0.05). mannwhitney_U is the U statistic of the AMICI sample against that baseline "
                "(scipy.stats.mannwhitneyu with AMICI as the first sample); it runs from 0 to "
                "U_max_n1n2 = n1 * n2 = 100, the product of the two group sizes (10 datasets each). "
                "rank_biserial_r = 2U/(n1*n2) - 1 is the rank-biserial correlation, and "
                "cles_prob_amici_higher = U/(n1*n2) is the common-language effect size, the probability that "
                "AMICI scores higher than the baseline on a randomly drawn pair of datasets; both are +1 when "
                "AMICI exceeds the baseline on every dataset. The p-values use the normal approximation "
                "(scipy's method=\"auto\" is asymptotic at n = 10, with a continuity correction and the tie "
                "correction; it switches to the exact test only below n = 8), as in the published caption. "
                "Every baseline is listed, including comparisons that are not significant.",
                "df": summary,
            }
        )

    # ---- 2f / 2g -------------------------------------------------------------------------------------
    import anndata as ad

    real_results = Path(args.realistic_results) if args.realistic_results else results
    if not (real_results / f"{REALISTIC}_{REALISTIC_SEED}").exists():
        print(f"  2f-j: no {REALISTIC}_{REALISTIC_SEED} folder in {real_results}; realistic panels not written")
        readme = README_ROWS
        C.save_panels(FIGURE, panels, readme, provenance_block(results, pbmc_cfg, args))
        return
    real_path = data_path(real_results, REALISTIC, REALISTIC_SEED)
    a = ad.read_h5ad(real_path)
    coords = np.asarray(a.obsm["spatial"])
    cell_type = a.obs["cell_type"].astype(str)
    subtype = a.obs["subtype"].astype(str)
    interacting = subtype.str.endswith("_sub1")
    panels.append(
        {
            "sheet": "Fig2f",
            "description": "Realistic breast cancer semisynthetic dataset (seed 42): one row per cell with Xenium "
            "coordinates (um), cell type, subtype, whether the cell is an interacting subpopulation (highlighted in the "
            "right map) and the train/test split.",
            "df": pd.DataFrame(
                {
                    "cell_id": a.obs_names,
                    "x_um": coords[:, 0],
                    "y_um": coords[:, 1],
                    "cell_type": cell_type.map(lambda s: REALISTIC_LABELS.get(s, s)).values,
                    "subtype": subtype.values,
                    "interacting_subpopulation": interacting.values,
                    "train_test_split": a.obs["train_test_split"].astype(str).values,
                }
            ),
        }
    )
    ls = length_scale_panel(REALISTIC, REALISTIC_SEED, real_results, real_cfg)
    if ls is not None:
        ls["interaction"] = (
            ls["sender"].map(lambda s: REALISTIC_LABELS.get(s, s)) + " -> " + ls["receiver"].map(lambda s: REALISTIC_LABELS.get(s, s))
        )
        model_label = selected_amici_model(real_results, REALISTIC, REALISTIC_SEED)
        panels.append(
            {
                "sheet": "Fig2g",
                "description": "AMICI length scale (um) per sender cell on the max-explained-variance head for each "
                "ground-truth interaction of the realistic dataset (attention threshold 0.1), with the ground-truth "
                f"length scales. From the lowest-test-loss AMICI model ({model_label}) of the re-run April sweep "
                "(reproducibility/benchmarks/fig2_realistic_benchmark_config.yaml); Fig. 2g is redrawn from this sheet "
                "by source_data/plot_fig2g.py (KDE per interaction, ground truth as dashed lines).",
                "df": ls,
            }
        )

    # ---- 2h-j ----------------------------------------------------------------------------------------
    # The published bars are kept; their values are the Source Data, with the re-run alongside as a reproduction.
    published = pd.read_csv(PUBLISHED_REALISTIC_VALUES).set_index(["task", "method"])["auprc"]
    sources = {}
    for item in args.rerun_source:
        family, path = item.split("=", 1)
        sources[family] = Path(path) if path else None
    amici_root = sources.get("AMICI", real_results)
    amici_label = selected_amici_model(amici_root, REALISTIC, REALISTIC_SEED) if amici_root is not None else "not re-run"
    for task, spec in TASKS.items():
        rows = []
        for name, fname in spec["methods"]:
            family = name.split(" (")[0]
            root = sources.get(family, real_results)
            rerun = read_auprc(root / f"{REALISTIC}_{REALISTIC_SEED}" / fname) if root is not None else np.nan
            bar = float(published[(PUBLISHED_TASK_NAMES[task], PUBLISHED_METHOD_NAMES[name])])
            rows.append(
                {
                    "method": name,
                    "auprc_published_bar": bar,
                    "auprc_rerun": rerun,
                    "rerun_minus_published": rerun - bar,
                    "rerun_source": "not re-run" if root is None else os.path.relpath(root, C.REPO),
                }
            )
        vals = pd.DataFrame(rows)
        missing = vals.loc[vals["auprc_rerun"].isna(), "method"].tolist()
        if missing:
            print(f"  {spec['panels'][1]}: re-run AUPRC not available yet for {missing}")
        print(vals.round(4).to_string(index=False))
        notes = []
        if any(name.startswith("NicheDE") for name, _ in spec["methods"]):
            notes.append("NicheDE with sigma = 200 and 500 does not reproduce (re-run about 0.07-0.08 higher).")
        if "CGCom" in missing:
            notes.append(
                "CGCom was not re-run: its attention scores overflow on this dataset (source_data/README.md, issue 2)."
            )
        panels.append(
            {
                "sheet": f"Fig{spec['panels'][1]}",
                "description": f"{spec['title']} on the realistic breast cancer dataset. auprc_published_bar is the "
                "bar height as published, read off the bars of a vector copy of the figure (AMICI's and GITIII's are "
                "also recorded, to two decimals, in an April-2026 message); it sits about 0.0007 below the AUPRCs that "
                "deterministic methods reproduce. auprc_rerun is an independent re-run, from the folder in "
                f"rerun_source: AMICI is the lowest-test-loss model ({amici_label}) of the re-run April sweep "
                "(reproducibility/benchmarks/fig2_realistic_benchmark_config.yaml), GITIII is re-run at commit "
                "3a3cc16. AMICI does not reproduce exactly, because the sweep's near-tied models give widely "
                "different AUPRCs (reproducibility/benchmarks/README.md). " + " ".join(notes),
                "df": vals,
            }
        )

    C.save_panels(FIGURE, panels, README_ROWS, provenance_block(results, pbmc_cfg, args))


README_ROWS = [
    ("Figure", "Figure 2: benchmarking on semisynthetic data"),
    ("Fig. 2a", "Sheet Fig2a (left half of the panel is a schematic)"),
    ("Fig. 2b", "Sheet Fig2b"),
    ("Fig. 2c-e", "Sheets Fig2c, Fig2d, Fig2e and their _stats sheets"),
    ("Fig. 2f", "Sheet Fig2f"),
    ("Fig. 2g", "Sheet Fig2g"),
    ("Fig. 2h-j", "Sheets Fig2h, Fig2i, Fig2j"),
]


def provenance_block(results, pbmc_cfg, args):
    return {
        "dataset": "semisynthetic benchmarks",
        "results_folder": str(results),
        "realistic_results_folder": str(args.realistic_results or results),
        "pipeline": "benchmarks/Snakefile, benchmarks/benchmark_config.yaml",
        "pbmc_seeds": pbmc_cfg["seeds"],
        "example_pbmc_seed": args.example_seed,
    }


if __name__ == "__main__":
    main()
