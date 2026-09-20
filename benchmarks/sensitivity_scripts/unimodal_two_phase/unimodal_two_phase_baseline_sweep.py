"""Run the GITIII and CGCom baselines on the unimodal synthetic dataset.

Scores and PR/AUPRC are produced by the Snakemake pipeline's own scripts, driven through
``snakemake_shim`` rather than reimplemented, so the numbers are comparable to the benchmark
results elsewhere in the paper. Each step runs in the conda env the corresponding Snakemake
rule declares: GITIII and CGCom in their own envs, ground truth and PR in the AMICI env
(which supersedes the pipeline's lightweight ``basic`` env).

CGCom is only wired for the neighbor-interaction and receiver-subtype tasks in the pipeline;
there is no CGCom gene-task scorer, so the gene task compares AMICI against GITIII only.

Steps are skipped when their outputs already exist, so the script can be re-run cheaply.
"""

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
SHIM = os.path.join(SCRIPT_DIR, "snakemake_shim.py")

DATASET = "3ct_dataset_1int_unimodal_10_30"
SEED = 42
RUN_KEY = f"{DATASET}_{SEED}"

RESULTS_PATH = "results"  # relative to BENCHMARKS_DIR; depth matters (see snakemake_shim)
RUN_DIR = os.path.join(RESULTS_PATH, RUN_KEY)
ADATA_PATH = os.path.join("data", f"{RUN_KEY}.h5ad")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "unimodal_two_phase_baseline_sweep")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures", "unimodal_two_phase_baseline_sweep")
AMICI_SCORES = os.path.join(SCRIPT_DIR, "results/unimodal_two_phase_amici_sweep/unimodal_two_phase_task_scores.csv")

PYTHON = {
    "amici": "/opt/conda/envs/amici/bin/python",
    "gitiii": "/opt/conda/envs/gitiii/bin/python",
    "cgcom": "/opt/conda/envs/cgcom/bin/python",
}

CONFIG = {
    "results_path": RESULTS_PATH + "/",
    "dir_path": "data/",
    "datasets": {
        DATASET: {
            "labels_key": "leiden",
            "subtype_labels_key": "subtype",
            "use_cross_validation": False,
            # GITIII has no sweep grid in the pipeline; this drives CGCom's
            # 2 learning rates x 3 neighbor-threshold ratios.
            "sweep_baselines": True,
            "cgcom_batch_size": 32,
            "cgcom_scoring_batch_size": 4,
            "gt_interactions": {
                "interaction_unimodal": {
                    "sender": "3",
                    "receiver": "0",
                    "interaction_subtype": "0_sub1",
                    "neutral_subtype": "0_sub0",
                    "length_scale": 20,
                }
            },
        }
    },
}


def path_in_run(name):
    """Resolve a path inside this sweep's run directory."""
    return os.path.join(RUN_DIR, name)


GT_GENES = path_in_run("ground_truth_genes.csv")
GT_NEIGHBORS = path_in_run("ground_truth_neighbor_interactions.csv")
GT_RECEIVERS = path_in_run("ground_truth_receiver_subtypes.csv")
GITIII_MODEL = path_in_run("saved_models/GRIT_best.pth")
CGCOM_MODEL = path_in_run(f"saved_models/cgcom_{RUN_KEY}_model.pt")

STEPS = [
    # (name, env, script, input, output, primary output to test for skipping)
    ("gt_genes", "amici", "gene_task/generate_gt_gene_scores.py", {"adata_path": ADATA_PATH}, {"out": GT_GENES}),
    (
        "gt_neighbors",
        "amici",
        "neighbor_interaction_task/generate_gt_neighbor_interaction_scores.py",
        {"adata_path": ADATA_PATH},
        {"out": GT_NEIGHBORS},
    ),
    (
        "gt_receivers",
        "amici",
        "receiver_subtype_task/generate_gt_receiver_subtype_scores.py",
        {"adata_path": ADATA_PATH},
        {"out": GT_RECEIVERS},
    ),
    ("gitiii_train", "gitiii", "train_gitiii_model.py", {"adata_path": ADATA_PATH}, {"out": GITIII_MODEL}),
    (
        "gitiii_gene_scores",
        "gitiii",
        "gene_task/generate_gitiii_scores.py",
        {"adata_path": ADATA_PATH, "model_path": GITIII_MODEL},
        {"out": path_in_run("gitiii_gene_task_scores.csv")},
    ),
    (
        "gitiii_neighbor_scores",
        "gitiii",
        "neighbor_interaction_task/generate_gitiii_scores.py",
        {"adata_path": ADATA_PATH, "model_path": GITIII_MODEL},
        {"out": path_in_run("gitiii_neighbor_interaction_scores.csv")},
    ),
    (
        "gitiii_receiver_scores",
        "gitiii",
        "receiver_subtype_task/generate_gitiii_scores.py",
        {"adata_path": ADATA_PATH, "model_path": GITIII_MODEL},
        {"out": path_in_run("gitiii_receiver_subtype_scores.csv")},
    ),
    ("cgcom_train", "cgcom", "train_cgcom_model.py", {"adata_path": ADATA_PATH}, {"out": CGCOM_MODEL}),
    (
        "cgcom_neighbor_scores",
        "cgcom",
        "neighbor_interaction_task/generate_cgcom_scores.py",
        {"adata_path": ADATA_PATH, "model_path": CGCOM_MODEL},
        {"out": path_in_run("cgcom_neighbor_interaction_scores.csv")},
    ),
    (
        "cgcom_receiver_scores",
        "cgcom",
        "receiver_subtype_task/generate_cgcom_scores.py",
        {"adata_path": ADATA_PATH, "model_path": CGCOM_MODEL},
        {"out": path_in_run("cgcom_receiver_subtype_scores.csv")},
    ),
    # PR / AUPRC, using the pipeline's own scoring logic
    (
        "gitiii_gene_pr",
        "amici",
        "gene_task/generate_gitiii_pr.py",
        {"gt_gene_scores_path": GT_GENES, "gitiii_scores_path": path_in_run("gitiii_gene_task_scores.csv")},
        {"out": path_in_run("gitiii_gene_task_pr.csv")},
    ),
    (
        "gitiii_neighbor_pr",
        "amici",
        "neighbor_interaction_task/generate_gitiii_pr.py",
        {
            "gt_neighbor_interactions_path": GT_NEIGHBORS,
            "gitiii_scores_path": path_in_run("gitiii_neighbor_interaction_scores.csv"),
        },
        {"out": path_in_run("gitiii_neighbor_interaction_task_pr.csv")},
    ),
    (
        "gitiii_receiver_pr",
        "amici",
        "receiver_subtype_task/generate_gitiii_pr.py",
        {
            "gt_receiver_subtypes_path": GT_RECEIVERS,
            "gitiii_scores_path": path_in_run("gitiii_receiver_subtype_scores.csv"),
        },
        {"out": path_in_run("gitiii_receiver_subtype_task_pr.csv")},
    ),
    (
        "cgcom_neighbor_pr",
        "amici",
        "neighbor_interaction_task/generate_cgcom_pr.py",
        {
            "gt_neighbor_interactions_path": GT_NEIGHBORS,
            "cgcom_scores_path": path_in_run("cgcom_neighbor_interaction_scores.csv"),
        },
        {"out": path_in_run("cgcom_neighbor_interaction_task_pr.csv")},
    ),
    (
        "cgcom_receiver_pr",
        "amici",
        "receiver_subtype_task/generate_cgcom_pr.py",
        {
            "gt_receiver_subtypes_path": GT_RECEIVERS,
            "cgcom_scores_path": path_in_run("cgcom_receiver_subtype_scores.csv"),
        },
        {"out": path_in_run("cgcom_receiver_subtype_task_pr.csv")},
    ),
]

PR_FILES = {
    ("GITIII", "Gene"): path_in_run("gitiii_gene_task_pr.csv"),
    ("GITIII", "Neighbor interaction"): path_in_run("gitiii_neighbor_interaction_task_pr.csv"),
    ("GITIII", "Receiver subtype"): path_in_run("gitiii_receiver_subtype_task_pr.csv"),
    ("CGCom", "Neighbor interaction"): path_in_run("cgcom_neighbor_interaction_task_pr.csv"),
    ("CGCom", "Receiver subtype"): path_in_run("cgcom_receiver_subtype_task_pr.csv"),
}


def run_step(name, env, script, inputs, outputs):
    """Execute one pipeline script through the shim in the given conda env."""
    primary = os.path.join(BENCHMARKS_DIR, outputs["out"])
    if os.path.exists(primary):
        print(f"[skip] {name} -> {outputs['out']} already exists", flush=True)
        return True

    spec_path = os.path.join(OUTPUT_DIR, f"spec_{name}.json")
    with open(spec_path, "w") as handle:
        json.dump(
            {
                "config": CONFIG,
                "wildcards": {"dataset": DATASET, "seed": str(SEED)},
                "input": inputs,
                "output": outputs,
            },
            handle,
            indent=2,
        )

    os.makedirs(os.path.dirname(primary), exist_ok=True)
    print(
        f"\n{'=' * 78}\n[run ] {name}  (env={env})\n  script={script}\n  out={outputs['out']}\n{'=' * 78}", flush=True
    )
    result = subprocess.run(
        [PYTHON[env], SHIM, "--script", script, "--spec", spec_path, "--benchmarks-dir", BENCHMARKS_DIR],
        cwd=BENCHMARKS_DIR,
    )
    if result.returncode != 0:
        print(f"[FAIL] {name} exited {result.returncode}", flush=True)
        return False
    if not os.path.exists(primary):
        print(f"[FAIL] {name} produced no output at {outputs['out']}", flush=True)
        return False
    print(f"[ ok ] {name}", flush=True)
    return True


def collect_auprc(step_status):
    """Assemble the AUPRC comparison table from the PR csvs plus the AMICI sweep result."""
    records = []
    amici = pd.read_csv(AMICI_SCORES)
    for _, row in amici.iterrows():
        records.append({"model": "AMICI", "task": row["task"], "auprc": row["auprc"], "status": row["status"]})

    for (model, task), rel_path in PR_FILES.items():
        path = os.path.join(BENCHMARKS_DIR, rel_path)
        if not os.path.exists(path):
            records.append({"model": model, "task": task, "auprc": np.nan, "status": "missing"})
            continue
        pr = pd.read_csv(path)
        records.append(
            {"model": model, "task": task, "auprc": float(pr["avg_precision_score"].iloc[0]), "status": "success"}
        )

    # CGCom has no gene-task scorer in the pipeline.
    records.append({"model": "CGCom", "task": "Gene", "auprc": np.nan, "status": "not_implemented"})
    return pd.DataFrame(records)


def plot_comparison(scores):
    """Grouped bar chart of AUPRC per task and model."""
    tasks = ["Gene", "Neighbor interaction", "Receiver subtype"]
    models = ["AMICI", "GITIII", "CGCom"]
    colors = {"AMICI": "#D55E00", "GITIII": "#0072B2", "CGCom": "#009E73"}

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.26
    x = np.arange(len(tasks))
    for offset, model in enumerate(models):
        values = []
        for task in tasks:
            row = scores[(scores["model"] == model) & (scores["task"] == task)]
            values.append(float(row["auprc"].iloc[0]) if len(row) and np.isfinite(row["auprc"].iloc[0]) else np.nan)
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, label=model, color=colors[model])
        for pos, value in zip(positions, values, strict=False):
            if np.isfinite(value):
                ax.text(pos, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
            else:
                ax.text(pos, 0.02, "n/a", ha="center", fontsize=8, rotation=90, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("AUPRC (average precision)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Unimodal synthetic dataset: benchmark AUPRC by task")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"unimodal_baseline_auprc_comparison.{ext}"), dpi=300)
    plt.close(fig)


def main():
    """Run each baseline scoring step, then collect and plot the AUPRC comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BENCHMARKS_DIR, RUN_DIR, "saved_models"), exist_ok=True)

    step_status = {}
    for name, env, script, inputs, outputs in STEPS:
        step_status[name] = run_step(name, env, script, inputs, outputs)

    scores = collect_auprc(step_status)
    scores.to_csv(os.path.join(OUTPUT_DIR, "unimodal_baseline_auprc_comparison.csv"), index=False)
    pd.DataFrame([{"step": k, "succeeded": v} for k, v in step_status.items()]).to_csv(
        os.path.join(OUTPUT_DIR, "unimodal_baseline_step_status.csv"), index=False
    )
    plot_comparison(scores)

    print("\n=== Step status ===", flush=True)
    for name, ok in step_status.items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}", flush=True)
    print("\n=== AUPRC comparison ===", flush=True)
    print(scores.pivot(index="task", columns="model", values="auprc").to_string(), flush=True)
    print(f"\nWrote results to {OUTPUT_DIR}\nWrote figure to {FIGURE_DIR}", flush=True)

    if not all(step_status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
