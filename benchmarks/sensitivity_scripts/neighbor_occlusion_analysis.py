# %% Import libraries
import json
import os
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scvi
import torch
from scipy.stats import spearmanr
from sklearn.metrics import PrecisionRecallDisplay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from amici_benchmark_utils import (  # noqa: E402
    get_amici_gene_task_scores,
    get_amici_neighbor_interaction_scores,
    get_amici_receiver_subtype_scores,
)
from benchmark_utils import (  # noqa: E402
    get_interaction_gt_neighbor_classes,
    get_model_precision_recall_auc,
    get_receiver_gt_ranked_genes,
)
from generate_dataset import _create_interaction_df, generate_synthetic_dataset  # noqa: E402
from generate_realistic_dataset import generate_realistic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
RUN_NAME = "neighbor_occlusion_analysis"
DATASET_SEED = 42
OCCLUSION_FRACTIONS = [0.0, 0.25, 0.5, 0.75]
OCCLUSION_SEEDS = [0, 1, 2]
BASELINE_OCCLUSION_SEED = 0

EXP_DEFAULTS = {
    "epochs": 400,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
}

SYNTHETIC_INTERACTIONS = {
    "interaction_1": {
        "sender": "0",
        "receiver": "2",
        "interaction_subtype": "2_sub1",
        "neutral_subtype": "2_sub0",
        "length_scale": 10,
    },
    "interaction_2": {
        "sender": "3",
        "receiver": "0",
        "interaction_subtype": "0_sub1",
        "neutral_subtype": "0_sub0",
        "length_scale": 20,
    },
}
REALISTIC_INTERACTIONS = {
    "interaction_1": {
        "sender": "Macrophages",
        "receiver": "DCIS",
        "interaction_subtype": "DCIS_sub1",
        "neutral_subtype": "DCIS_sub0",
        "length_scale": 40,
    },
    "interaction_2": {
        "sender": "T_Cells",
        "receiver": "Endothelial",
        "interaction_subtype": "Endothelial_sub1",
        "neutral_subtype": "Endothelial_sub0",
        "length_scale": 25,
    },
    "interaction_3": {
        "sender": "Invasive_Tumor",
        "receiver": "Myoepi",
        "interaction_subtype": "Myoepi_sub1",
        "neutral_subtype": "Myoepi_sub0",
        "length_scale": 15,
    },
}

DATASET_CONFIGS = {
    "3ct_dataset_2way": {
        "kind": "synthetic",
        "labels_key": "leiden",
        "subtype_key": "subtype",
        "gt_interactions": SYNTHETIC_INTERACTIONS,
        "run": {
            "end_val": 1e-3,
            "value_l1": 1e-5,
            "train_seed": 22,
            "epoch_start": 10,
            "epoch_end": 30,
            "batch_size": 128,
            "n_heads": 8,
            "lr": 1e-3,
            "n_neighbors": 50,
            "flavor": "linear",
        },
    },
    "breast_cancer": {
        "kind": "realistic",
        "labels_key": "cell_type",
        "subtype_key": "subtype",
        "gt_interactions": REALISTIC_INTERACTIONS,
        "run": {
            "end_val": 1e-5,
            "value_l1": 1e-5,
            "train_seed": 22,
            "epoch_start": 15,
            "epoch_end": 30,
            "batch_size": 256,
            "n_heads": 10,
            "lr": 1e-3,
            "n_neighbors": 50,
            "flavor": "linear",
        },
        "flex_h5_path": "data/GSM7782698_count_raw_feature_bc_matrix.h5",
        "annot_path": "data/41467_2023_43458_MOESM4_ESM.xlsx",
        "xenium_path": "data/xenium_rep1_io.h5ad",
        "scvi_model_dir": "data/scvi_model",
        "n_cv_folds": 3,
    },
}

# %% Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
saved_models_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figure_dir = os.path.join(base_dir, "figures", RUN_NAME)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)


# %% Helper functions
def benchmark_path(path):
    """Resolve benchmark config paths relative to the benchmarks directory."""
    return path if os.path.isabs(path) else os.path.join(benchmark_dir, path)


def dataset_path(dataset_name):
    """Return the cached h5ad path for this analysis."""
    return os.path.join(data_dir, f"{dataset_name}_{DATASET_SEED}.h5ad")


def ensure_dataset(dataset_name, dataset_config):
    """Generate the real-interaction semisynthetic dataset if it is not already cached."""
    adata_path = dataset_path(dataset_name)
    if os.path.exists(adata_path):
        return

    if dataset_config["kind"] == "synthetic":
        interaction_df = _create_interaction_df(dataset_config["gt_interactions"])
        random.seed(DATASET_SEED)
        np.random.seed(DATASET_SEED)
        torch.random.manual_seed(DATASET_SEED)
        scvi.settings.seed = DATASET_SEED
        generate_synthetic_dataset(interaction_df, adata_path)
        return

    if dataset_config["kind"] == "realistic":
        np.random.seed(DATASET_SEED)
        torch.manual_seed(DATASET_SEED)
        scvi.settings.seed = DATASET_SEED
        generate_realistic_dataset(
            benchmark_path(dataset_config["flex_h5_path"]),
            benchmark_path(dataset_config["annot_path"]),
            benchmark_path(dataset_config["xenium_path"]),
            adata_path,
            scvi_model_dir=benchmark_path(dataset_config["scvi_model_dir"]),
            n_cv_folds=dataset_config.get("n_cv_folds", 3),
        )
        return

    raise ValueError(f"Unknown dataset kind: {dataset_config['kind']}")


def train_or_load_model(dataset_name, adata, dataset_config):
    """Train one AMICI model for the base dataset, or load the cached model."""
    run = dataset_config["run"]
    model_path = os.path.join(saved_models_dir, dataset_name, "model")
    result_path = os.path.join(saved_models_dir, dataset_name, "result.json")

    if os.path.exists(os.path.join(model_path, "model.pt")) and os.path.exists(result_path):
        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run["n_neighbors"]),
        )
        return AMICI.load(model_path, adata=adata)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    pl.seed_everything(run["train_seed"])
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    AMICI.setup_anndata(
        adata_train,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model = AMICI(
        adata_train,
        n_heads=int(run["n_heads"]),
        value_l1_penalty_coef=float(run["value_l1"]),
    )
    model.train(
        max_epochs=EXP_DEFAULTS["epochs"],
        batch_size=int(run["batch_size"]),
        plan_kwargs={"lr": float(run["lr"])},
        early_stopping=EXP_DEFAULTS["early_stopping"],
        early_stopping_monitor=EXP_DEFAULTS["early_stopping_monitor"],
        early_stopping_patience=EXP_DEFAULTS["early_stopping_patience"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                start_val=1e-6,
                end_val=float(run["end_val"]),
                epoch_start=int(run["epoch_start"]),
                epoch_end=int(run["epoch_end"]),
                flavor=run["flavor"],
            ),
        ],
    )

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model.save(model_path, overwrite=True)
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]
    test_loss = (
        model.get_reconstruction_error(adata, indices=test_indices, batch_size=int(run["batch_size"]))[
            "reconstruction_loss"
        ]
        .detach()
        .cpu()
        .numpy()
        .item()
    )
    with open(result_path, "w") as f:
        json.dump({**run, "test_loss": test_loss, "model_path": model_path}, f, indent=2)
    return model


def setup_inference_anndata(adata, dataset_config):
    """Register AMICI nearest-neighbor fields before optional occlusion."""
    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(dataset_config["run"]["n_neighbors"]),
    )


def make_neighbor_attention_mask(adata, occlusion_fraction, occlusion_seed):
    """Create the same kind of binary keep/drop attention mask used by AMICI neighbor dropout."""
    attention_mask = np.ones_like(np.asarray(adata.obsm["_nn_idx"]), dtype=int)
    if occlusion_fraction > 0:
        rng = np.random.default_rng(occlusion_seed)
        attention_mask[rng.random(attention_mask.shape) < occlusion_fraction] = 0
    return attention_mask


def get_gt_gene_scores(adata, dataset_config):
    """Create ground-truth gene labels for all configured interactions."""
    all_gt_gene_scores = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        gt_gene_scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            dataset_config["subtype_key"],
        )
        gt_gene_scores["interaction"] = interaction_name
        all_gt_gene_scores.append(gt_gene_scores)
    return pd.concat(all_gt_gene_scores, ignore_index=True)


def get_gt_receiver_classes(adata, dataset_config):
    """Create ground-truth receiver-subtype labels."""
    receiver_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in dataset_config["gt_interactions"].values():
        receiver_mask |= adata.obs[dataset_config["subtype_key"]] == interaction_config["interaction_subtype"]
    return pd.DataFrame({"cell_idx": adata.obs_names, "class": receiver_mask.astype(float)})


def get_attention_score_vector(model, adata, attention_mask=None):
    """Return max-over-head attention scores keyed by cell-neighbor pair."""
    attention_scores = get_amici_neighbor_interaction_scores(model, adata, attention_mask=attention_mask)
    return attention_scores.rename(columns={"amici_scores": "attention_score"})


def compute_pr_curves(
    model, adata, dataset_config, gt_gene_scores, gt_neighbor_classes, gt_receiver_classes, attention_mask=None
):
    """Compute PR curves for attention weights and downstream AMICI predictions."""
    all_gene_scores = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        gene_scores = get_amici_gene_task_scores(
            model,
            adata,
            interaction_config["sender"],
            interaction_config["receiver"],
            attention_mask=attention_mask,
        )
        gene_scores["interaction"] = interaction_name
        all_gene_scores.append(gene_scores)
    gene_scores_df = pd.concat(all_gene_scores, ignore_index=True)
    gene_precision, gene_recall, gene_auprc = get_model_precision_recall_auc(
        gene_scores_df,
        gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata, attention_mask=attention_mask)
    neighbor_precision, neighbor_recall, neighbor_auprc = get_model_precision_recall_auc(
        neighbor_scores_df,
        gt_neighbor_classes,
        merge_cols=["cell_idx", "neighbor_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    receiver_scores_df = get_amici_receiver_subtype_scores(model, adata, attention_mask=attention_mask)
    receiver_precision, receiver_recall, receiver_auprc = get_model_precision_recall_auc(
        receiver_scores_df,
        gt_receiver_classes,
        merge_cols=["cell_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    return {
        "Gene Task": {"precision": gene_precision, "recall": gene_recall, "auprc": gene_auprc},
        "Neighbor Interaction Task": {
            "precision": neighbor_precision,
            "recall": neighbor_recall,
            "auprc": neighbor_auprc,
        },
        "Receiver Subtype Task": {
            "precision": receiver_precision,
            "recall": receiver_recall,
            "auprc": receiver_auprc,
        },
    }


def compute_attention_stability(baseline_attention_scores, occluded_attention_scores):
    """Compare baseline and occluded attention scores for matched cell-neighbor pairs."""
    merged = pd.merge(
        baseline_attention_scores,
        occluded_attention_scores,
        on=["cell_idx", "neighbor_idx"],
        how="inner",
        suffixes=("_baseline", "_occluded"),
    )
    if len(merged) < 2:
        return np.nan
    return spearmanr(merged["attention_score_baseline"], merged["attention_score_occluded"]).correlation


def plot_pr_curves(pr_curve_records):
    """Plot PR curves for each dataset and task, colored by occlusion fraction."""
    task_names = ["Neighbor Interaction Task", "Gene Task", "Receiver Subtype Task"]
    dataset_names = list(pr_curve_records)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(OCCLUSION_FRACTIONS)))

    fig, axes = plt.subplots(len(dataset_names), len(task_names), figsize=(18, 9), squeeze=False)
    for row_idx, dataset_name in enumerate(dataset_names):
        for col_idx, task_name in enumerate(task_names):
            ax = axes[row_idx][col_idx]
            for color, occlusion_fraction in zip(colors, OCCLUSION_FRACTIONS, strict=False):
                pr = pr_curve_records[dataset_name][task_name][occlusion_fraction]
                PrecisionRecallDisplay(recall=pr["recall"], precision=pr["precision"]).plot(
                    ax=ax,
                    name=f"{occlusion_fraction:.0%} occluded (AUPRC={pr['auprc']:.2f})",
                    color=color,
                )
            ax.set_title(f"{dataset_name}\n{task_name}", fontsize=10)
            ax.legend(fontsize=7)

    fig.suptitle("AMICI robustness to random neighbor occlusion", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "neighbor_occlusion_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_auprc_and_attention_stability(summary_df):
    """Plot average AUPRC and attention-rank stability across occlusion seeds."""
    task_names = ["Neighbor Interaction Task", "Gene Task", "Receiver Subtype Task"]
    dataset_names = summary_df["dataset"].unique()

    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(13, 8), squeeze=False)
    for row_idx, dataset_name in enumerate(dataset_names):
        dataset_df = summary_df[summary_df["dataset"] == dataset_name]

        ax = axes[row_idx][0]
        for task_name in task_names:
            task_df = (
                dataset_df[dataset_df["task"] == task_name]
                .groupby("occlusion_fraction")["auprc"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                task_df["occlusion_fraction"], task_df["mean"], yerr=task_df["std"], marker="o", label=task_name
            )
        ax.set_title(f"{dataset_name}: PR performance")
        ax.set_xlabel("Occlusion fraction")
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[row_idx][1]
        stability_df = (
            dataset_df[dataset_df["task"] == "Neighbor Interaction Task"]
            .groupby("occlusion_fraction")["attention_spearman"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            stability_df["occlusion_fraction"],
            stability_df["mean"],
            yerr=stability_df["std"],
            marker="o",
            color="steelblue",
        )
        ax.set_title(f"{dataset_name}: attention-rank stability")
        ax.set_xlabel("Occlusion fraction")
        ax.set_ylabel("Spearman rho vs. baseline")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Neighbor occlusion robustness summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "neighbor_occlusion_summary.png"), dpi=300, bbox_inches="tight")
    plt.show()


# %% Train models, occlude neighbors, and compute PR curves
select_gpu()
random.seed(DATASET_SEED)
np.random.seed(DATASET_SEED)
torch.random.manual_seed(DATASET_SEED)
torch.manual_seed(DATASET_SEED)
scvi.settings.seed = DATASET_SEED

summary_records = []
pr_curve_records = {}

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    ensure_dataset(dataset_name, dataset_config)
    adata = sc.read_h5ad(dataset_path(dataset_name))
    adata.obs_names_make_unique()

    setup_inference_anndata(adata, dataset_config)
    model = train_or_load_model(dataset_name, adata, dataset_config)
    gt_gene_scores = get_gt_gene_scores(adata, dataset_config)
    gt_neighbor_classes = get_interaction_gt_neighbor_classes(
        adata, dataset_config["gt_interactions"], dataset_config["labels_key"]
    )
    gt_receiver_classes = get_gt_receiver_classes(adata, dataset_config)

    baseline_attention_scores = None
    pr_curve_records[dataset_name] = {
        "Neighbor Interaction Task": {},
        "Gene Task": {},
        "Receiver Subtype Task": {},
    }

    for occlusion_fraction in OCCLUSION_FRACTIONS:
        seed_pr_curves = []
        seed_attention_stability = []

        for occlusion_seed in OCCLUSION_SEEDS:
            if occlusion_fraction == 0.0 and occlusion_seed != BASELINE_OCCLUSION_SEED:
                continue

            attention_mask = make_neighbor_attention_mask(
                adata,
                occlusion_fraction,
                occlusion_seed,
            )
            pr_curves = compute_pr_curves(
                model,
                adata,
                dataset_config,
                gt_gene_scores,
                gt_neighbor_classes,
                gt_receiver_classes,
                attention_mask=attention_mask,
            )
            attention_scores = get_attention_score_vector(model, adata, attention_mask=attention_mask)
            if occlusion_fraction == 0.0:
                baseline_attention_scores = attention_scores
                attention_spearman = 1.0
            else:
                attention_spearman = compute_attention_stability(baseline_attention_scores, attention_scores)

            for task_name, pr in pr_curves.items():
                summary_records.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "occlusion_fraction": occlusion_fraction,
                        "occlusion_seed": occlusion_seed,
                        "auprc": pr["auprc"],
                        "attention_spearman": attention_spearman,
                    }
                )
            seed_pr_curves.append(pr_curves)
            seed_attention_stability.append(attention_spearman)

        representative_curves = seed_pr_curves[0]
        for task_name, pr in representative_curves.items():
            pr_curve_records[dataset_name][task_name][occlusion_fraction] = pr

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(figure_dir, "neighbor_occlusion_summary.csv"), index=False)

# %% Plot robustness curves
plot_pr_curves(pr_curve_records)
plot_auprc_and_attention_stability(summary_df)
