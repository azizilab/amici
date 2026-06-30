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
from sklearn.metrics import PrecisionRecallDisplay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import generate_realistic_dataset as realistic_dataset  # noqa: E402
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
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
RUN_NAME = "coordinate_noise_sensitivity"
DATASET_SEEDS = list(range(10))
NOISE_MAX_SIGMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
NOISE_RANDOM_SEED_OFFSET = 1000

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
    "semisynthetic": {
        "kind": "synthetic",
        "cache_template": "semisyn_{seed}.h5ad",
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
    "realistic_semisynthetic": {
        "kind": "realistic",
        "cache_template": os.path.join("realistic_length_scale_dataset_bootstrap_ci", "breast_cancer_{seed}.h5ad"),
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
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
perturbed_data_dir = os.path.join(data_dir, RUN_NAME)
model_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figure_dir = os.path.join(base_dir, "figures", RUN_NAME)

os.makedirs(perturbed_data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)


# %% Helper functions
def benchmark_path(path):
    """Resolve paths relative to the benchmarks directory."""
    return path if os.path.isabs(path) else os.path.join(benchmark_dir, path)


def set_all_seeds(seed):
    """Set all RNG seeds used by dataset generation and AMICI training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def sigma_slug(max_sigma):
    """Create a filename-safe representation of a noise sigma."""
    return str(max_sigma).replace(".", "p")


def source_dataset_path(dataset_config, dataset_seed):
    """Return the preferred cached source h5ad path for a dataset seed."""
    return os.path.join(data_dir, dataset_config["cache_template"].format(seed=dataset_seed))


def perturbed_dataset_path(dataset_name, dataset_seed, max_sigma):
    """Return the cached perturbed h5ad path."""
    return os.path.join(
        perturbed_data_dir,
        f"{dataset_name}_seed_{dataset_seed}_xgraded_sigma_{sigma_slug(max_sigma)}.h5ad",
    )


def ensure_source_dataset(dataset_config, dataset_seed):
    """Load existing generated datasets when possible; otherwise create them."""
    adata_path = source_dataset_path(dataset_config, dataset_seed)
    if os.path.exists(adata_path):
        return adata_path

    os.makedirs(os.path.dirname(adata_path), exist_ok=True)
    set_all_seeds(dataset_seed)
    if dataset_config["kind"] == "synthetic":
        interaction_df = _create_interaction_df(dataset_config["gt_interactions"])
        generate_synthetic_dataset(interaction_df, adata_path)
        return adata_path

    if dataset_config["kind"] == "realistic":
        realistic_dataset.SEED = dataset_seed
        realistic_dataset.generate_realistic_dataset(
            benchmark_path(dataset_config["flex_h5_path"]),
            benchmark_path(dataset_config["annot_path"]),
            benchmark_path(dataset_config["xenium_path"]),
            adata_path,
            scvi_model_dir=benchmark_path(dataset_config["scvi_model_dir"]),
            n_cv_folds=dataset_config.get("n_cv_folds", 3),
        )
        return adata_path

    raise ValueError(f"Unknown dataset kind: {dataset_config['kind']}")


def add_left_to_right_coordinate_noise(adata, max_sigma, random_seed):
    """Add 2D Gaussian coordinate noise with sigma increasing along the x-axis."""
    perturbed = adata.copy()
    coords = np.asarray(perturbed.obsm["spatial"], dtype=float).copy()
    perturbed.uns["original_spatial"] = coords.copy()

    x = coords[:, 0]
    x_span = x.max() - x.min()
    if x_span <= 0:
        raise ValueError("Cannot apply x-graded coordinate noise with zero x span.")

    x_weight = (x - x.min()) / x_span
    cell_sigmas = max_sigma * x_weight
    rng = np.random.default_rng(random_seed)
    coords += rng.normal(loc=0.0, scale=cell_sigmas[:, None], size=coords.shape)

    perturbed.obsm["spatial"] = coords
    perturbed.obs["coordinate_noise_max_sigma"] = float(max_sigma)
    perturbed.obs["coordinate_noise_x_weight"] = x_weight
    perturbed.uns["coordinate_noise"] = {
        "mode": "left_to_right_gaussian_2d",
        "max_sigma": float(max_sigma),
        "random_seed": int(random_seed),
    }
    return perturbed


def ensure_perturbed_dataset(dataset_name, dataset_config, dataset_seed, max_sigma):
    """Load or create a perturbed dataset for one seed and noise level."""
    output_path = perturbed_dataset_path(dataset_name, dataset_seed, max_sigma)
    if os.path.exists(output_path):
        adata = sc.read_h5ad(output_path)
        adata.obs_names_make_unique()
        return adata

    source_path = ensure_source_dataset(dataset_config, dataset_seed)
    adata = sc.read_h5ad(source_path)
    adata.obs_names_make_unique()
    noise_seed = NOISE_RANDOM_SEED_OFFSET + dataset_seed
    perturbed = add_left_to_right_coordinate_noise(adata, max_sigma, noise_seed)
    perturbed.write_h5ad(output_path)
    return perturbed


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


def train_or_load_model(adata, dataset_name, dataset_seed, max_sigma, dataset_config):
    """Train one AMICI model for a perturbed dataset, or load its cached model."""
    run = dataset_config["run"]
    sigma_name = sigma_slug(max_sigma)
    run_dir = os.path.join(model_dir, dataset_name, f"dataset_seed_{dataset_seed}", f"sigma_{sigma_name}")
    model_path = os.path.join(run_dir, "model")
    result_path = os.path.join(run_dir, "result.json")
    os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(os.path.join(model_path, "model.pt")) and os.path.exists(result_path):
        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run["n_neighbors"]),
        )
        with open(result_path) as f:
            return AMICI.load(model_path, adata=adata), json.load(f)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

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
    result = {
        **run,
        "dataset": dataset_name,
        "dataset_seed": dataset_seed,
        "noise_max_sigma": max_sigma,
        "test_loss": test_loss,
        "model_path": model_path,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return model, result


def compute_pr_metrics(model, adata, dataset_config):
    """Compute AMICI PR metrics for all benchmark tasks."""
    gt_gene_scores = get_gt_gene_scores(adata, dataset_config)
    gt_neighbor_classes = get_interaction_gt_neighbor_classes(
        adata,
        dataset_config["gt_interactions"],
        dataset_config["labels_key"],
    )
    gt_receiver_classes = get_gt_receiver_classes(adata, dataset_config)

    gene_scores = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        scores = get_amici_gene_task_scores(
            model,
            adata,
            interaction_config["sender"],
            interaction_config["receiver"],
        )
        scores["interaction"] = interaction_name
        gene_scores.append(scores)
    gene_scores_df = pd.concat(gene_scores, ignore_index=True)
    gene_precision, gene_recall, gene_auprc = get_model_precision_recall_auc(
        gene_scores_df,
        gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata)
    neighbor_precision, neighbor_recall, neighbor_auprc = get_model_precision_recall_auc(
        neighbor_scores_df,
        gt_neighbor_classes,
        merge_cols=["cell_idx", "neighbor_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    receiver_scores_df = get_amici_receiver_subtype_scores(model, adata)
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


def plot_summary(summary_df):
    """Plot mean AUPRC across dataset seeds for each coordinate-noise level."""
    task_names = ["Neighbor Interaction Task", "Gene Task", "Receiver Subtype Task"]
    dataset_names = list(DATASET_CONFIGS)
    fig, axes = plt.subplots(len(dataset_names), len(task_names), figsize=(16, 8), squeeze=False, sharex=True)

    for row_idx, dataset_name in enumerate(dataset_names):
        dataset_df = summary_df[summary_df["dataset"] == dataset_name]
        for col_idx, task_name in enumerate(task_names):
            ax = axes[row_idx][col_idx]
            task_df = dataset_df[dataset_df["task"] == task_name]
            stats = task_df.groupby("noise_max_sigma")["auprc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats["noise_max_sigma"], stats["mean"], yerr=stats["std"], marker="o")
            ax.set_title(f"{dataset_name}\n{task_name}", fontsize=10)
            ax.set_xlabel("Maximum Gaussian noise sigma")
            ax.set_ylabel("AUPRC")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AMICI sensitivity to left-to-right coordinate noise", fontsize=14)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(figure_dir, f"{RUN_NAME}_summary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_representative_pr_curves(pr_curve_records):
    """Plot representative seed-0 PR curves for each noise level."""
    task_names = ["Neighbor Interaction Task", "Gene Task", "Receiver Subtype Task"]
    dataset_names = list(DATASET_CONFIGS)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(NOISE_MAX_SIGMAS)))
    fig, axes = plt.subplots(len(dataset_names), len(task_names), figsize=(18, 9), squeeze=False)

    for row_idx, dataset_name in enumerate(dataset_names):
        for col_idx, task_name in enumerate(task_names):
            ax = axes[row_idx][col_idx]
            for color, max_sigma in zip(colors, NOISE_MAX_SIGMAS, strict=False):
                pr = pr_curve_records[dataset_name][task_name].get(max_sigma)
                if pr is None:
                    continue
                PrecisionRecallDisplay(recall=pr["recall"], precision=pr["precision"]).plot(
                    ax=ax,
                    name=f"sigma {max_sigma:g} (AUPRC={pr['auprc']:.2f})",
                    color=color,
                )
            ax.set_title(f"{dataset_name}\n{task_name}", fontsize=10)
            ax.legend(fontsize=7)

    fig.suptitle("Representative PR curves under coordinate noise", fontsize=14)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(figure_dir, f"{RUN_NAME}_pr_curves.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% Train AMICI across datasets and coordinate-noise levels
summary_path = os.path.join(figure_dir, f"{RUN_NAME}_summary.csv")
summary_records = []
pr_curve_records = {
    dataset_name: {"Neighbor Interaction Task": {}, "Gene Task": {}, "Receiver Subtype Task": {}}
    for dataset_name in DATASET_CONFIGS
}

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for dataset_seed in DATASET_SEEDS:
        for max_sigma in NOISE_MAX_SIGMAS:
            print(f"Processing {dataset_name}, dataset seed {dataset_seed}, max sigma {max_sigma}")
            adata = ensure_perturbed_dataset(dataset_name, dataset_config, dataset_seed, max_sigma)
            AMICI.setup_anndata(
                adata,
                labels_key=dataset_config["labels_key"],
                coord_obsm_key="spatial",
                n_neighbors=int(dataset_config["run"]["n_neighbors"]),
            )
            model, result = train_or_load_model(adata, dataset_name, dataset_seed, max_sigma, dataset_config)
            pr_metrics = compute_pr_metrics(model, adata, dataset_config)

            for task_name, pr in pr_metrics.items():
                summary_records.append(
                    {
                        "dataset": dataset_name,
                        "dataset_seed": dataset_seed,
                        "noise_max_sigma": max_sigma,
                        "task": task_name,
                        "auprc": pr["auprc"],
                        "test_loss": result["test_loss"],
                        "model_path": result["model_path"],
                    }
                )
                if dataset_seed == DATASET_SEEDS[0]:
                    pr_curve_records[dataset_name][task_name][max_sigma] = pr

            pd.DataFrame(summary_records).to_csv(summary_path, index=False)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(summary_path, index=False)
plot_summary(summary_df)
plot_representative_pr_curves(pr_curve_records)
