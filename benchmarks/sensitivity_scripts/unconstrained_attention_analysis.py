# %% Import libraries
import itertools
import json
import os
import random
import shutil
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scvi
import torch

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
from amici._alt_module import AMICIUnconstrainedAttentionModule  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402


class UnconstrainedAttentionAMICI(AMICI):
    """AMICI wrapper that swaps in the old unconstrained positional-embedding module."""

    _module_cls = AMICIUnconstrainedAttentionModule


# %% Config
RUN_NAME = "unconstrained_attention_analysis"
SYNTHETIC_DATASET_SEEDS = [40, 123, 6, 23, 25, 88, 72, 58, 22, 31]
REALISTIC_DATASET_SEEDS = list(range(10))
ATTENTION_THRESHOLD = 0.1

SYNTHETIC_CONFIG = {
    "dataset_type": "synthetic",
    "dataset_name": "3ct_dataset_2way",
    "labels_key": "leiden",
    "subtype_labels_key": "subtype",
    "seeds": SYNTHETIC_DATASET_SEEDS,
    "gt_interactions": {
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
    },
    "sweep_params": {
        "end_attention_penalty": [1e-2, 1e-3, 1e-4],
        "attention_penalty_schedule": [[10, 40]],
        "seed": [22, 38, 42],
        "value_l1_penalty_coef": [1e-6, 1e-5, 1e-4],
        "batch_size": [128],
        "lr": [1e-3],
        "n_neighbors": [50],
        "penalty_flavor_params": ["linear"],
        "n_heads": [8],
    },
}

REALISTIC_CONFIG = {
    "dataset_type": "realistic_semisynthetic",
    "dataset_name": "breast_cancer",
    "labels_key": "cell_type",
    "subtype_labels_key": "subtype",
    "seeds": REALISTIC_DATASET_SEEDS,
    "n_cv_folds": 3,
    "flex_h5_path": "data/GSM7782698_count_raw_feature_bc_matrix.h5",
    "annot_path": "data/41467_2023_43458_MOESM4_ESM.xlsx",
    "xenium_path": "data/xenium_rep1_io.h5ad",
    "scvi_model_dir": "data/scvi_model",
    "gt_interactions": {
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
    },
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
}

EXP_DEFAULTS = {
    "epochs": 400,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
}


def benchmark_path(benchmark_dir, path):
    """Resolve benchmark config paths relative to the benchmarks directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(benchmark_dir, path)


def set_all_seeds(seed):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def build_run_configs(dataset_config):
    """Build AMICI sweep configs from the dataset benchmark config."""
    sweep = dataset_config["sweep_params"]
    run_configs = []
    for end_val, schedule, train_seed, value_l1, batch_size, lr, n_neighbors, flavor, n_heads in itertools.product(
        sweep["end_attention_penalty"],
        sweep["attention_penalty_schedule"],
        sweep["seed"],
        sweep["value_l1_penalty_coef"],
        sweep["batch_size"],
        sweep["lr"],
        sweep["n_neighbors"],
        sweep["penalty_flavor_params"],
        sweep["n_heads"],
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
            }
        )
    return run_configs


def cache_matches_run(result, run):
    """Return whether an on-disk run result was produced with the requested config."""
    keys = [
        "dataset_type",
        "dataset_seed",
        "run_idx",
        "end_val",
        "flavor",
        "value_l1",
        "train_seed",
        "epoch_start",
        "epoch_end",
        "batch_size",
        "n_heads",
        "lr",
        "n_neighbors",
    ]
    return all(result.get(key) == run.get(key) for key in keys)


def generate_or_load_synthetic_dataset(dataset_config, data_dir, dataset_seed):
    """Load a cached synthetic dataset or generate it if missing."""
    adata_path = os.path.join(data_dir, f"semisyn_{dataset_seed}.h5ad")
    if not os.path.exists(adata_path):
        set_all_seeds(dataset_seed)
        interaction_df = _create_interaction_df(dataset_config["gt_interactions"])
        generate_synthetic_dataset(interaction_df, adata_path)

    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()
    return adata, adata_path


def generate_or_load_realistic_dataset(benchmark_dir, dataset_config, data_dir, dataset_seed):
    """Load a cached realistic semi-synthetic dataset or generate it if missing."""
    dataset_name = dataset_config["dataset_name"]
    existing_paths = [
        os.path.join(data_dir, "realistic_length_scale_dataset_bootstrap_ci", f"{dataset_name}_{dataset_seed}.h5ad"),
        os.path.join(data_dir, f"{dataset_name}_{dataset_seed}.h5ad"),
        os.path.join(benchmark_dir, "data", f"{dataset_name}_{dataset_seed}.h5ad"),
    ]
    for adata_path in existing_paths:
        if os.path.exists(adata_path):
            adata = sc.read_h5ad(adata_path)
            adata.obs_names_make_unique()
            return adata, adata_path

    adata_path = os.path.join(data_dir, RUN_NAME, f"{dataset_name}_{dataset_seed}.h5ad")
    os.makedirs(os.path.dirname(adata_path), exist_ok=True)
    set_all_seeds(dataset_seed)
    realistic_dataset.SEED = dataset_seed
    realistic_dataset.generate_realistic_dataset(
        benchmark_path(benchmark_dir, dataset_config["flex_h5_path"]),
        benchmark_path(benchmark_dir, dataset_config["annot_path"]),
        benchmark_path(benchmark_dir, dataset_config["xenium_path"]),
        adata_path,
        scvi_model_dir=benchmark_path(benchmark_dir, dataset_config["scvi_model_dir"])
        if dataset_config.get("scvi_model_dir")
        else None,
        n_cv_folds=dataset_config.get("n_cv_folds", 3),
    )

    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()
    return adata, adata_path


def train_or_load_run(adata, dataset_config, run, run_path, result_path, eval_indices):
    """Train one old AMICI run, load its cached result, or record a failed run."""
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        if cache_matches_run(result, run):
            return result

    try:
        pl.seed_everything(run["train_seed"])
        adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
        UnconstrainedAttentionAMICI.setup_anndata(
            adata_train,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run["n_neighbors"]),
        )
        model = UnconstrainedAttentionAMICI(
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
        UnconstrainedAttentionAMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run["n_neighbors"]),
        )
        model.save(run_path, overwrite=True)
        test_loss = (
            model.get_reconstruction_error(adata, indices=eval_indices, batch_size=int(run["batch_size"]))[
                "reconstruction_loss"
            ]
            .detach()
            .cpu()
            .numpy()
            .item()
        )
        if not np.isfinite(test_loss):
            raise ValueError(f"Non-finite test loss: {test_loss}")

        result = {**run, "status": "success", "test_loss": test_loss, "model_path": run_path, "error": None}
    except Exception as exc:  # noqa: BLE001
        result = {
            **run,
            "status": "failed",
            "test_loss": np.nan,
            "model_path": run_path,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def build_task_ground_truth(adata, dataset_config):
    """Compute ground-truth labels for the three benchmark tasks."""
    subtype_key = dataset_config["subtype_labels_key"]
    labels_key = dataset_config["labels_key"]

    all_gt_gene_scores = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        gt_gene_scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            subtype_key,
        )
        gt_gene_scores["interaction"] = interaction_name
        all_gt_gene_scores.append(gt_gene_scores)
    gt_gene_scores_df = pd.concat(all_gt_gene_scores, ignore_index=True)

    combined_interaction_subtype_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in dataset_config["gt_interactions"].values():
        combined_interaction_subtype_mask |= adata.obs[subtype_key] == interaction_config["interaction_subtype"]
    gt_receiver_classes_df = pd.DataFrame(
        {
            "cell_idx": adata.obs_names,
            "class": combined_interaction_subtype_mask.astype(float),
        }
    )

    gt_neighbor_classes_df = get_interaction_gt_neighbor_classes(
        adata,
        dataset_config["gt_interactions"],
        labels_key,
    )
    return gt_gene_scores_df, gt_neighbor_classes_df, gt_receiver_classes_df


def evaluate_tasks(model, adata, dataset_config):
    """Evaluate gene, neighbor-interaction, and receiver-subtype AUPRC tasks."""
    gt_gene_scores_df, gt_neighbor_classes_df, gt_receiver_classes_df = build_task_ground_truth(adata, dataset_config)

    task_records = []

    try:
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
        gene_scores_df = pd.concat(all_gene_scores, ignore_index=True)
        _, _, auprc = get_model_precision_recall_auc(
            gene_scores_df,
            gt_gene_scores_df,
            merge_cols=["gene", "interaction"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        task_records.append({"task": "Gene", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        task_records.append({"task": "Gene", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    try:
        neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata)
        _, _, auprc = get_model_precision_recall_auc(
            neighbor_scores_df,
            gt_neighbor_classes_df,
            merge_cols=["cell_idx", "neighbor_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        task_records.append({"task": "Neighbor interaction", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        task_records.append({"task": "Neighbor interaction", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    try:
        receiver_scores_df = get_amici_receiver_subtype_scores(model, adata)
        _, _, auprc = get_model_precision_recall_auc(
            receiver_scores_df,
            gt_receiver_classes_df,
            merge_cols=["cell_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        task_records.append({"task": "Receiver subtype", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        task_records.append({"task": "Receiver subtype", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    return pd.DataFrame(task_records)


def process_dataset_type(dataset_config, benchmark_dir, data_dir, model_dir, results_dir):
    """Train/evaluate old AMICI across one dataset type."""
    dataset_type = dataset_config["dataset_type"]
    run_configs = build_run_configs(dataset_config)
    dataset_results = []
    run_result_records = []

    print(f"Processing {dataset_type}: {len(dataset_config['seeds'])} dataset seeds, {len(run_configs)} runs per seed")
    for dataset_seed in dataset_config["seeds"]:
        print(f"Processing {dataset_type} seed {dataset_seed}")
        if dataset_type == "synthetic":
            adata, adata_path = generate_or_load_synthetic_dataset(dataset_config, data_dir, dataset_seed)
        else:
            adata, adata_path = generate_or_load_realistic_dataset(
                benchmark_dir, dataset_config, data_dir, dataset_seed
            )

        dataset_model_dir = os.path.join(model_dir, dataset_type, f"dataset_seed_{dataset_seed}")
        os.makedirs(dataset_model_dir, exist_ok=True)
        task_scores_path = os.path.join(dataset_model_dir, "task_scores.csv")
        best_params_path = os.path.join(dataset_model_dir, "best_model_params.json")
        best_model_path = os.path.join(dataset_model_dir, "best_model")

        if os.path.exists(task_scores_path) and os.path.exists(best_params_path):
            task_scores_df = pd.read_csv(task_scores_path)
            dataset_results.append(task_scores_df)
            continue

        set_all_seeds(dataset_seed)
        UnconstrainedAttentionAMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(run_configs[0]["n_neighbors"]),
        )
        eval_indices = np.where(adata.obs["train_test_split"] == "test")[0]

        run_results = []
        for run_idx, run_config in enumerate(run_configs):
            run = {
                **run_config,
                "dataset_type": dataset_type,
                "dataset_seed": dataset_seed,
                "run_idx": run_idx,
            }
            run_path = os.path.join(dataset_model_dir, f"run_{run_idx}_seed_{run['train_seed']}")
            result_path = os.path.join(dataset_model_dir, f"run_{run_idx}_seed_{run['train_seed']}_results.json")
            result = train_or_load_run(adata, dataset_config, run, run_path, result_path, eval_indices)
            run_results.append(result)
            run_result_records.append(result)

        successful_runs = [result for result in run_results if result.get("status") == "success"]
        if not successful_runs:
            task_scores_df = pd.DataFrame(
                {
                    "dataset_type": dataset_type,
                    "dataset_seed": dataset_seed,
                    "dataset_path": adata_path,
                    "task": ["Gene", "Neighbor interaction", "Receiver subtype"],
                    "auprc": [np.nan, np.nan, np.nan],
                    "status": ["failed", "failed", "failed"],
                    "error": [
                        "No successful training runs",
                        "No successful training runs",
                        "No successful training runs",
                    ],
                }
            )
            task_scores_df.to_csv(task_scores_path, index=False)
            dataset_results.append(task_scores_df)
            continue

        best_result = min(successful_runs, key=lambda x: x["test_loss"])
        best_result = {
            **best_result,
            "selection_metric": "test_loss",
            "n_candidate_runs": len(run_results),
            "n_successful_runs": len(successful_runs),
            "candidate_test_losses": [
                {
                    "run_idx": result["run_idx"],
                    "train_seed": result["train_seed"],
                    "status": result["status"],
                    "test_loss": result["test_loss"],
                    "error": result.get("error"),
                }
                for result in run_results
            ],
        }
        if os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)
        shutil.copytree(best_result["model_path"], best_model_path)

        with open(best_params_path, "w") as f:
            json.dump(best_result, f, indent=2)

        UnconstrainedAttentionAMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(best_result["n_neighbors"]),
        )
        model = UnconstrainedAttentionAMICI.load(best_model_path, adata=adata)
        task_scores_df = evaluate_tasks(model, adata, dataset_config)
        task_scores_df["dataset_type"] = dataset_type
        task_scores_df["dataset_seed"] = dataset_seed
        task_scores_df["dataset_path"] = adata_path
        task_scores_df["best_test_loss"] = best_result["test_loss"]
        task_scores_df["best_train_seed"] = best_result["train_seed"]
        task_scores_df["best_end_val"] = best_result["end_val"]
        task_scores_df["best_value_l1"] = best_result["value_l1"]
        task_scores_df["n_successful_runs"] = len(successful_runs)
        task_scores_df.to_csv(task_scores_path, index=False)
        dataset_results.append(task_scores_df)

    if run_result_records:
        pd.DataFrame(run_result_records).to_csv(
            os.path.join(results_dir, f"{dataset_type}_run_results.csv"),
            index=False,
        )

    return pd.concat(dataset_results, ignore_index=True)


def plot_task_boxplots(results_df, figures_dir):
    """Save boxplots of AUPRC across dataset seeds for each task and dataset type."""
    task_order = ["Gene", "Neighbor interaction", "Receiver subtype"]
    dataset_types = ["synthetic", "realistic_semisynthetic"]
    dataset_titles = {
        "synthetic": "Synthetic",
        "realistic_semisynthetic": "Realistic semi-synthetic",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, dataset_type in zip(axes, dataset_types, strict=False):
        subset = results_df[results_df["dataset_type"] == dataset_type]
        box_data = [subset.loc[subset["task"] == task, "auprc"].dropna().to_numpy() for task in task_order]
        bp = ax.boxplot(box_data, labels=task_order, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.55)

        for x_pos, values in enumerate(box_data, start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(np.full(len(values), x_pos) + jitter, values, color="black", s=18, alpha=0.65, zorder=3)

        ax.set_title(dataset_titles[dataset_type])
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUPRC")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Unconstrained positional-embedding AMICI task performance across dataset seeds", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figures_dir, f"{RUN_NAME}_task_auprc_boxplots.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def compute_length_scale_samples_for_dataset(dataset_config, benchmark_dir, data_dir, model_dir, results_dir):
    """Compute length-scale samples for cached best models of one dataset type."""
    dataset_type = dataset_config["dataset_type"]
    length_scale_path = os.path.join(results_dir, f"{dataset_type}_length_scale_samples.csv")
    if os.path.exists(length_scale_path):
        return pd.read_csv(length_scale_path)

    sample_records = []
    for dataset_seed in dataset_config["seeds"]:
        print(f"Computing length scales for {dataset_type} seed {dataset_seed}")
        if dataset_type == "synthetic":
            adata, _ = generate_or_load_synthetic_dataset(dataset_config, data_dir, dataset_seed)
        else:
            adata, _ = generate_or_load_realistic_dataset(benchmark_dir, dataset_config, data_dir, dataset_seed)

        dataset_model_dir = os.path.join(model_dir, dataset_type, f"dataset_seed_{dataset_seed}")
        best_model_path = os.path.join(dataset_model_dir, "best_model")
        best_params_path = os.path.join(dataset_model_dir, "best_model_params.json")
        if not (os.path.exists(os.path.join(best_model_path, "model.pt")) and os.path.exists(best_params_path)):
            print(f"Skipping {dataset_type} seed {dataset_seed}; cached best model is missing.")
            continue

        with open(best_params_path) as f:
            best_result = json.load(f)

        UnconstrainedAttentionAMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(best_result["n_neighbors"]),
        )
        model = UnconstrainedAttentionAMICI.load(best_model_path, adata=adata)
        explained_variance = model.get_expl_variance_scores(adata=adata)
        attention_patterns = model.get_attention_patterns(adata=adata)
        attention_df = attention_patterns._attention_patterns_df.copy()
        nn_idxs_df = attention_patterns._nn_idxs_df.copy()
        nn_idxs_df["cell_idx"] = adata.obs_names
        labels = adata.obs[dataset_config["labels_key"]]
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        obs_position = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)

        for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
            receiver_type = interaction_config["receiver"]
            sender_type = interaction_config["sender"]
            best_head = explained_variance.compute_max_explained_variance_head(cell_type=receiver_type)
            receiver_attention_df = attention_df.loc[
                (attention_df["label"] == receiver_type) & (attention_df["head"] == best_head)
            ]
            attention_long_df = receiver_attention_df.melt(
                id_vars=["cell_idx"],
                value_vars=[f"neighbor_{idx}" for idx in range(model.n_neighbors)],
                var_name="neighbor_col",
                value_name="attention_score",
            )
            nn_long_df = nn_idxs_df.melt(
                id_vars=["cell_idx"],
                value_vars=[f"neighbor_{idx}" for idx in range(model.n_neighbors)],
                var_name="neighbor_col",
                value_name="neighbor_idx",
            )
            length_scale_df = pd.merge(attention_long_df, nn_long_df, on=["cell_idx", "neighbor_col"], how="inner")
            length_scale_df["neighbor_label"] = labels.reindex(length_scale_df["neighbor_idx"]).to_numpy()
            length_scale_df = length_scale_df.loc[
                (length_scale_df["neighbor_label"] == sender_type)
                & (length_scale_df["attention_score"] >= ATTENTION_THRESHOLD)
            ].copy()
            if length_scale_df.empty:
                length_scale_df["length_scale"] = []
            else:
                cell_positions = obs_position.reindex(length_scale_df["cell_idx"]).to_numpy(dtype=int)
                neighbor_positions = obs_position.reindex(length_scale_df["neighbor_idx"]).to_numpy(dtype=int)
                length_scale_df["length_scale"] = np.linalg.norm(
                    coords[cell_positions] - coords[neighbor_positions],
                    axis=1,
                )
            length_scale_df["dataset_type"] = dataset_type
            length_scale_df["dataset_seed"] = dataset_seed
            length_scale_df["interaction"] = interaction_name
            length_scale_df["interaction_label"] = f"{sender_type} -> {receiver_type}"
            length_scale_df["receiver_type"] = receiver_type
            length_scale_df["sender_type"] = sender_type
            length_scale_df["head_idx"] = best_head
            length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
            length_scale_df["best_test_loss"] = best_result["test_loss"]
            length_scale_df["best_train_seed"] = best_result["train_seed"]
            sample_records.append(length_scale_df)

    if sample_records:
        samples_df = pd.concat(sample_records, ignore_index=True)
    else:
        samples_df = pd.DataFrame()
    samples_df.to_csv(length_scale_path, index=False)
    return samples_df


def plot_length_scale_distributions(length_scale_df, figures_dir):
    """Plot predicted length-scale distributions with ground truth marked by x."""
    if length_scale_df.empty:
        print("No length-scale samples available; skipping length-scale distribution plot.")
        return

    dataset_types = ["synthetic", "realistic_semisynthetic"]
    dataset_titles = {
        "synthetic": "Synthetic",
        "realistic_semisynthetic": "Realistic semi-synthetic",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)

    for ax, dataset_type in zip(axes[0], dataset_types, strict=False):
        plot_df = length_scale_df[length_scale_df["dataset_type"] == dataset_type].dropna(subset=["length_scale"])
        interactions = (
            plot_df[["interaction", "interaction_label", "gt_length_scale"]]
            .drop_duplicates()
            .sort_values("gt_length_scale")
        )
        positions = np.arange(1, len(interactions) + 1)
        violin_data = [
            plot_df.loc[plot_df["interaction"] == interaction, "length_scale"].to_numpy()
            for interaction in interactions["interaction"]
        ]
        if not violin_data:
            ax.set_title(dataset_titles[dataset_type])
            ax.text(0.5, 0.5, "No length-scale samples", ha="center", va="center", transform=ax.transAxes)
            continue

        violins = ax.violinplot(
            violin_data,
            positions=positions,
            vert=False,
            widths=0.8,
            showmeans=True,
            showextrema=True,
        )
        for body in violins["bodies"]:
            body.set_facecolor("steelblue")
            body.set_edgecolor("steelblue")
            body.set_alpha(0.45)
        for part in ("cmeans", "cmins", "cmaxes", "cbars"):
            violins[part].set_color("steelblue")
            violins[part].set_linewidth(1.4)

        gt_positions = interactions["gt_length_scale"].astype(float).to_numpy()
        ax.scatter(gt_positions, positions, marker="x", color="black", s=130, linewidths=2.5, zorder=4)
        ax.set_yticks(positions)
        ax.set_yticklabels(interactions["interaction_label"])
        ax.set_xlabel("Predicted length scale")
        ax.set_title(dataset_titles[dataset_type])
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Unconstrained positional-embedding AMICI length-scale distributions", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figures_dir, f"{RUN_NAME}_length_scale_distributions.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# %% Run analysis
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
results_dir = os.path.join(base_dir, "results", RUN_NAME)
figures_dir = os.path.join(base_dir, "figures", RUN_NAME)
for path in (data_dir, model_dir, results_dir, figures_dir):
    os.makedirs(path, exist_ok=True)

all_results = []
for dataset_config in (SYNTHETIC_CONFIG, REALISTIC_CONFIG):
    all_results.append(process_dataset_type(dataset_config, benchmark_dir, data_dir, model_dir, results_dir))

results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv(os.path.join(results_dir, "unconstrained_attention_task_scores.csv"), index=False)
plot_task_boxplots(results_df, figures_dir)

length_scale_results = []
for dataset_config in (SYNTHETIC_CONFIG, REALISTIC_CONFIG):
    length_scale_results.append(
        compute_length_scale_samples_for_dataset(dataset_config, benchmark_dir, data_dir, model_dir, results_dir)
    )
length_scale_df = pd.concat(length_scale_results, ignore_index=True)
length_scale_df.to_csv(os.path.join(results_dir, "unconstrained_attention_length_scale_samples.csv"), index=False)
plot_length_scale_distributions(length_scale_df, figures_dir)
