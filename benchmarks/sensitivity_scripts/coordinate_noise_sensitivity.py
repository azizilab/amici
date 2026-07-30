# %% Import libraries
import itertools
import json
import os
import random
import shutil
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scvi
import torch

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
RUN_NAME = "coordinate_noise_gradient_sensitivity_sweep"
DEFAULT_RANDOM_SEED = 42
NOISE_SIGMAS = [0.0, 0.01, 0.05, 0.1]
DATASETS_TO_RUN = ["3ct_dataset_2way", "breast_cancer"]

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
        "dataset_seed": 58,
        "labels_key": "leiden",
        "subtype_key": "subtype",
        "gt_interactions": SYNTHETIC_INTERACTIONS,
        "source_h5ad_path": "sensitivity_scripts/data/semisyn_58.h5ad",
        "sweep_params": {
            "end_attention_penalty": [3e-3, 3e-4, 3e-5],
            "attention_penalty_schedule": [[10, 40]],
            "seed": [21, 33, 88, 99],
            "value_l1_penalty_coef": [3e-6, 3e-5],
            "batch_size": [128],
            "lr": [1e-3],
            "n_neighbors": [50],
            "penalty_flavor_params": ["linear"],
            "n_heads": [8],
        },
    },
    "breast_cancer": {
        "kind": "realistic",
        "dataset_seed": 2,
        "labels_key": "cell_type",
        "subtype_key": "subtype",
        "gt_interactions": REALISTIC_INTERACTIONS,
        "base_h5ad_path": "sensitivity_scripts/data/realistic_length_scale_dataset_bootstrap_ci/breast_cancer_2.h5ad",
        "flex_h5_path": "data/GSM7782698_count_raw_feature_bc_matrix.h5",
        "annot_path": "data/41467_2023_43458_MOESM4_ESM.xlsx",
        "xenium_path": "data/xenium_rep1_io.h5ad",
        "scvi_model_dir": "data/scvi_model",
        "n_cv_folds": 3,
        "sweep_params": {
            "end_attention_penalty": [1e-5],
            "attention_penalty_schedule": [[15, 30]],
            "seed": [21, 22, 33, 88, 99],
            "value_l1_penalty_coef": [1e-5],
            "batch_size": [256],
            "lr": [1e-3],
            "n_neighbors": [50],
            "penalty_flavor_params": ["linear"],
            "n_heads": [10],
        },
    },
}


# %% Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data", RUN_NAME)
saved_models_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figure_dir = os.path.join(base_dir, "figures", RUN_NAME)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)


# %% Helpers
def benchmark_path(path):
    """Resolve benchmark-relative paths."""
    return path if os.path.isabs(path) else os.path.join(benchmark_dir, path)


def sigma_tag(sigma):
    """Return a filesystem-safe sigma tag."""
    return f"sigma_{str(sigma).replace('.', 'p')}"


def base_dataset_path(dataset_name, dataset_config):
    """Return the cached unperturbed dataset path."""
    return os.path.join(data_dir, f"{dataset_name}_{dataset_config['dataset_seed']}_base.h5ad")


def noisy_dataset_path(dataset_name, dataset_config, sigma):
    """Return the cached perturbed dataset path."""
    return os.path.join(data_dir, f"{dataset_name}_{dataset_config['dataset_seed']}_{sigma_tag(sigma)}.h5ad")


def build_sweep_runs(dataset_config):
    """Build the AMICI sweep used to select the model for each noise level."""
    sweep = dataset_config["sweep_params"]

    run_configs = []
    for run_idx, (end_val, schedule, train_seed, value_l1, batch_size, lr, n_neighbors, flavor, n_heads) in enumerate(
        itertools.product(
            sweep["end_attention_penalty"],
            sweep["attention_penalty_schedule"],
            sweep["seed"],
            sweep["value_l1_penalty_coef"],
            sweep["batch_size"],
            sweep["lr"],
            sweep["n_neighbors"],
            sweep["penalty_flavor_params"],
            sweep["n_heads"],
        )
    ):
        run_configs.append(
            {
                "end_val": end_val,
                "flavor": flavor,
                "value_l1": value_l1,
                "train_seed": train_seed,
                "epoch_start": schedule[0],
                "epoch_end": schedule[1],
                "batch_size": batch_size,
                "n_heads": n_heads,
                "lr": lr,
                "n_neighbors": n_neighbors,
                "run_idx": run_idx,
            }
        )
        if len(run_configs) == 6:
            break
    return run_configs


def ensure_base_dataset(dataset_name, dataset_config):
    """Create or copy the unperturbed semi-synthetic dataset."""
    adata_path = base_dataset_path(dataset_name, dataset_config)
    if os.path.exists(adata_path):
        return adata_path

    source_h5ad_path = dataset_config.get("source_h5ad_path")
    if source_h5ad_path and os.path.exists(benchmark_path(source_h5ad_path)):
        shutil.copyfile(benchmark_path(source_h5ad_path), adata_path)
        return adata_path

    if dataset_config["kind"] == "synthetic":
        dataset_seed = int(dataset_config["dataset_seed"])
        random.seed(dataset_seed)
        np.random.seed(dataset_seed)
        torch.random.manual_seed(dataset_seed)
        scvi.settings.seed = dataset_seed
        interaction_df = _create_interaction_df(dataset_config["gt_interactions"])
        generate_synthetic_dataset(interaction_df, adata_path)
        return adata_path

    base_h5ad_path = benchmark_path(dataset_config["base_h5ad_path"])
    if os.path.exists(base_h5ad_path):
        shutil.copyfile(base_h5ad_path, adata_path)
        return adata_path

    generate_realistic_dataset(
        benchmark_path(dataset_config["flex_h5_path"]),
        benchmark_path(dataset_config["annot_path"]),
        benchmark_path(dataset_config["xenium_path"]),
        adata_path,
        scvi_model_dir=benchmark_path(dataset_config["scvi_model_dir"]),
        n_cv_folds=dataset_config.get("n_cv_folds", 3),
    )
    return adata_path


def add_coordinate_noise(adata, sigma, seed):
    """Add x-gradient Gaussian noise in min-max normalized coordinate space."""
    adata = adata.copy()
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    mins = coords.min(axis=0, keepdims=True)
    spans = coords.max(axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0

    if sigma > 0:
        rng = np.random.default_rng(seed)
        normalized_coords = (coords - mins) / spans
        cell_sigmas = sigma * normalized_coords[:, [0]]
        normalized_coords = normalized_coords + rng.normal(0, cell_sigmas, size=normalized_coords.shape)
        adata.obsm["spatial"] = normalized_coords * spans + mins
    return adata


def ensure_noisy_dataset(dataset_name, dataset_config, sigma):
    """Create or load a coordinate-perturbed dataset."""
    adata_path = noisy_dataset_path(dataset_name, dataset_config, sigma)
    if os.path.exists(adata_path):
        return sc.read_h5ad(adata_path)

    base_path = ensure_base_dataset(dataset_name, dataset_config)
    adata = sc.read_h5ad(base_path)
    adata.obs_names_make_unique()
    adata = add_coordinate_noise(
        adata,
        sigma=sigma,
        seed=DEFAULT_RANDOM_SEED + int(dataset_config["dataset_seed"]) * 1000 + int(sigma * 1_000_000),
    )
    adata.write_h5ad(adata_path)
    return adata


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


def train_or_load_run(dataset_name, adata, dataset_config, sigma, run):
    """Train or load one AMICI run for a coordinate-noise condition."""
    run_idx = int(run["run_idx"])
    run_dir = os.path.join(saved_models_dir, dataset_name, sigma_tag(sigma), f"run_{run_idx}")
    model_path = os.path.join(run_dir, "model")
    result_path = os.path.join(run_dir, "result.json")
    if os.path.exists(os.path.join(model_path, "model.pt")) and os.path.exists(result_path):
        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run["n_neighbors"]),
        )
        return AMICI.load(model_path, adata=adata), json.load(open(result_path))

    os.makedirs(run_dir, exist_ok=True)
    pl.seed_everything(run["train_seed"])
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    AMICI.setup_anndata(
        adata_train,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model = AMICI(adata_train, n_heads=int(run["n_heads"]), value_l1_penalty_coef=float(run["value_l1"]))
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
            )
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
    result = {**run, "test_loss": test_loss, "model_path": model_path, "noise_sigma": sigma}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return model, result


def evaluate_sweep_condition(dataset_name, adata, dataset_config, sigma):
    """Run a compact sweep and return all scores plus the best run summary."""
    records = []
    for run in build_sweep_runs(dataset_config):
        run_idx = int(run["run_idx"])
        run_dir = os.path.join(saved_models_dir, dataset_name, sigma_tag(sigma), f"run_{run_idx}")
        scores_path = os.path.join(run_dir, "task_auprc_scores.csv")
        print(f"  run_{run_idx}: {run}", flush=True)
        try:
            if os.path.exists(scores_path):
                run_df = pd.read_csv(scores_path)
                result = json.load(open(os.path.join(run_dir, "result.json")))
            else:
                model, result = train_or_load_run(dataset_name, adata, dataset_config, sigma, run)
                scores = evaluate_model(model, adata, dataset_config)
                run_df = pd.DataFrame(
                    [
                        {
                            "task": task,
                            "auprc": auprc,
                            "status": "success",
                            "error": None,
                        }
                        for task, auprc in scores.items()
                    ]
                )
                run_df.to_csv(scores_path, index=False)

            for _, row in run_df.iterrows():
                records.append(
                    {
                        "dataset": dataset_name,
                        "dataset_seed": dataset_config["dataset_seed"],
                        "noise_sigma": sigma,
                        "noise_variance": sigma**2,
                        "run_idx": run_idx,
                        "task": row["task"],
                        "auprc": row["auprc"],
                        "status": row.get("status", "success"),
                        "error": row.get("error", None),
                        "test_loss": result["test_loss"],
                        "model_path": result["model_path"],
                        **{key: result.get(key, run.get(key)) for key in run},
                    }
                )
        except Exception as exc:  # noqa: BLE001
            for task in ["Gene Task", "Neighbor Interaction Task", "Receiver Subtype Task"]:
                records.append(
                    {
                        "dataset": dataset_name,
                        "dataset_seed": dataset_config["dataset_seed"],
                        "noise_sigma": sigma,
                        "noise_variance": sigma**2,
                        "run_idx": run_idx,
                        "task": task,
                        "auprc": np.nan,
                        "status": "failed",
                        "test_loss": np.nan,
                        "model_path": None,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        **run,
                    }
                )

    condition_df = pd.DataFrame(records)
    mean_scores = (
        condition_df[condition_df["status"] == "success"]
        .groupby("run_idx", as_index=False)["auprc"]
        .mean()
        .rename(columns={"auprc": "mean_auprc"})
    )
    if mean_scores.empty:
        return condition_df, condition_df

    best_run_idx = int(mean_scores.sort_values("mean_auprc", ascending=False).iloc[0]["run_idx"])
    best_df = condition_df[condition_df["run_idx"] == best_run_idx].copy()
    best_df["mean_auprc"] = float(mean_scores.loc[mean_scores["run_idx"] == best_run_idx, "mean_auprc"].iloc[0])
    return condition_df, best_df


def evaluate_model(model, adata, dataset_config):
    """Evaluate AMICI on all three semi-synthetic benchmark tasks."""
    gt_gene_scores = get_gt_gene_scores(adata, dataset_config)
    gt_neighbor_classes = get_interaction_gt_neighbor_classes(
        adata, dataset_config["gt_interactions"], dataset_config["labels_key"]
    )
    gt_receiver_classes = get_gt_receiver_classes(adata, dataset_config)

    all_gene_scores = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        gene_scores = get_amici_gene_task_scores(
            model,
            adata,
            interaction_config["sender"],
            interaction_config["receiver"],
        )
        gene_scores["interaction"] = interaction_name
        all_gene_scores.append(gene_scores)
    _, _, gene_auprc = get_model_precision_recall_auc(
        pd.concat(all_gene_scores, ignore_index=True),
        gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    neighbor_scores = get_amici_neighbor_interaction_scores(model, adata)
    _, _, neighbor_auprc = get_model_precision_recall_auc(
        neighbor_scores,
        gt_neighbor_classes,
        merge_cols=["cell_idx", "neighbor_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    receiver_scores = get_amici_receiver_subtype_scores(model, adata)
    _, _, receiver_auprc = get_model_precision_recall_auc(
        receiver_scores,
        gt_receiver_classes,
        merge_cols=["cell_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )
    return {
        "Gene Task": gene_auprc,
        "Neighbor Interaction Task": neighbor_auprc,
        "Receiver Subtype Task": receiver_auprc,
    }


def plot_summary(summary_df):
    """Plot task AUPRC as coordinate noise increases."""
    task_order = ["Neighbor Interaction Task", "Gene Task", "Receiver Subtype Task"]
    dataset_order = [dataset for dataset in DATASETS_TO_RUN if dataset in set(summary_df["dataset"])]
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(13, 4.2), sharey=True)
    if len(dataset_order) == 1:
        axes = [axes]
    for ax, dataset_name in zip(axes, dataset_order, strict=False):
        dataset_df = summary_df[summary_df["dataset"] == dataset_name]
        for task in task_order:
            task_df = dataset_df[dataset_df["task"] == task].sort_values("noise_sigma")
            ax.plot(task_df["noise_sigma"], task_df["auprc"], marker="o", linewidth=2, label=task)
        ax.set_title(dataset_name)
        ax.set_xlabel("Maximum Gaussian noise sigma\n(normalized coordinate units)")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("AMICI robustness to mild coordinate perturbation", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figure_dir, f"coordinate_noise_sensitivity_summary.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


# %% Run
select_gpu()
random.seed(DEFAULT_RANDOM_SEED)
np.random.seed(DEFAULT_RANDOM_SEED)
torch.random.manual_seed(DEFAULT_RANDOM_SEED)
torch.manual_seed(DEFAULT_RANDOM_SEED)
scvi.settings.seed = DEFAULT_RANDOM_SEED

summary_records = []
all_run_records = []
for dataset_name in DATASETS_TO_RUN:
    dataset_config = DATASET_CONFIGS[dataset_name]
    for sigma in NOISE_SIGMAS:
        print(f"Running {dataset_name} with max normalized coordinate sigma={sigma}", flush=True)
        try:
            adata = ensure_noisy_dataset(dataset_name, dataset_config, sigma)
            adata.obs_names_make_unique()
            condition_df, best_df = evaluate_sweep_condition(dataset_name, adata, dataset_config, sigma)
            all_run_records.append(condition_df)
            summary_records.append(best_df)
        except Exception as exc:  # noqa: BLE001
            error_df = pd.DataFrame(
                [
                    {
                        "dataset": dataset_name,
                        "dataset_seed": dataset_config["dataset_seed"],
                        "noise_sigma": sigma,
                        "noise_variance": sigma**2,
                        "run_idx": np.nan,
                        "task": task,
                        "auprc": np.nan,
                        "mean_auprc": np.nan,
                        "test_loss": np.nan,
                        "model_path": None,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    for task in ["Gene Task", "Neighbor Interaction Task", "Receiver Subtype Task"]
                ]
            )
            all_run_records.append(error_df)
            summary_records.append(error_df)

summary_df = pd.concat(summary_records, ignore_index=True)
all_runs_df = pd.concat(all_run_records, ignore_index=True)
all_runs_df.to_csv(os.path.join(figure_dir, "coordinate_noise_sensitivity_all_runs.csv"), index=False)
summary_df.to_csv(os.path.join(figure_dir, "coordinate_noise_sensitivity_summary.csv"), index=False)
plot_summary(summary_df)
print(f"Saved outputs to {figure_dir}", flush=True)
