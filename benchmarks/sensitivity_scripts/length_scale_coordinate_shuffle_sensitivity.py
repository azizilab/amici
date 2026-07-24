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
from generate_realistic_dataset import generate_realistic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATASET = "breast_cancer"
BASE_DATASET_SEED = 42
SHUFFLE_SEEDS = list(range(10))
ATTENTION_THRESHOLD = 0.1
SHUFFLE_FRACTION = 0.3
SHUFFLE_LABEL = f"{SHUFFLE_FRACTION:g}".replace(".", "p")
TRAIN_SEED = 22
RUN_NAME = f"length_scale_coordinate_{SHUFFLE_LABEL}_shuffle_sensitivity"
OUTPUT_PREFIX = f"length_scale_coordinate_{SHUFFLE_LABEL}_shuffle"
AUPRC_RUN_NAME = "length_scale_coordinate_shuffle_auprc_comparison"
AUPRC_OUTPUT_PREFIX = "length_scale_coordinate_shuffle_auprc"

COORDINATE_SHUFFLE_CONDITIONS = [
    {
        "shuffle_fraction": 0.3,
        "shuffle_label": "0.3",
        "data_template": f"{DATASET}_{BASE_DATASET_SEED}_coord_0p3_shuffle_{{seed}}.h5ad",
        "model_run_name": "length_scale_coordinate_0p3_shuffle_sensitivity",
    },
    {
        "shuffle_fraction": 0.5,
        "shuffle_label": "0.5",
        "data_template": f"{DATASET}_{BASE_DATASET_SEED}_coord_shuffle_{{seed}}.h5ad",
        "model_run_name": "length_scale_coordinate_shuffle_sensitivity",
    },
    {
        "shuffle_fraction": 1.0,
        "shuffle_label": "1.0",
        "data_template": f"{DATASET}_{BASE_DATASET_SEED}_coord_full_shuffle_{{seed}}.h5ad",
        "model_run_name": "length_scale_coordinate_full_shuffle_sensitivity",
    },
]

CONFIG = {
    "dir_path": "data/",
    "datasets_realistic": ["breast_cancer"],
    "datasets": {
        "3ct_dataset_2way": {
            "seeds": [40, 123, 6, 23, 25, 88, 72, 58, 22, 31],
            "labels_key": "leiden",
            "subtype_labels_key": "subtype",
            "ncem_niche_sizes": [10, 15, 20],
            "nichede_niche_sizes": [20, 200, 500],
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
        },
        "breast_cancer": {
            "generation_type": "realistic",
            "seeds": [42],
            "labels_key": "cell_type",
            "subtype_labels_key": "subtype",
            "use_cross_validation": False,
            "n_cv_folds": 3,
            "sweep_baselines": True,
            "ncem_niche_sizes": [10, 15, 20],
            "nichede_niche_sizes": [20, 200, 500],
            "sweep_params": {
                "end_attention_penalty": [1.0e-5],
                "attention_penalty_schedule": [[15, 30]],
                "seed": [21, 22, 33, 88, 99],
                "value_l1_penalty_coef": [1.0e-5],
                "batch_size": [256],
                "lr": [1.0e-3],
                "n_neighbors": [50],
                "penalty_flavor_params": ["linear"],
                "n_heads": [10],
            },
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
        },
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


def build_run_config(dataset_config):
    """Build one fixed AMICI config from the benchmark parameter set."""
    sweep = dataset_config["sweep_params"]
    schedule = sweep["attention_penalty_schedule"][0]
    return {
        "end_val": sweep["end_attention_penalty"][0],
        "flavor": sweep["penalty_flavor_params"][0],
        "value_l1": sweep["value_l1_penalty_coef"][0],
        "train_seed": TRAIN_SEED,
        "epoch_start": schedule[0],
        "epoch_end": schedule[1],
        "batch_size": sweep["batch_size"][0],
        "n_heads": sweep["n_heads"][0],
        "lr": sweep["lr"][0],
        "n_neighbors": sweep["n_neighbors"][0],
    }


def ensure_base_realistic_dataset(benchmark_dir, dataset_config, base_adata_path):
    """Generate the base realistic dataset if it does not already exist."""
    if os.path.exists(base_adata_path):
        return

    np.random.seed(BASE_DATASET_SEED)
    torch.manual_seed(BASE_DATASET_SEED)
    scvi.settings.seed = BASE_DATASET_SEED
    generate_realistic_dataset(
        benchmark_path(benchmark_dir, dataset_config["flex_h5_path"]),
        benchmark_path(benchmark_dir, dataset_config["annot_path"]),
        benchmark_path(benchmark_dir, dataset_config["xenium_path"]),
        base_adata_path,
        scvi_model_dir=benchmark_path(benchmark_dir, dataset_config["scvi_model_dir"])
        if dataset_config.get("scvi_model_dir")
        else None,
        n_cv_folds=dataset_config.get("n_cv_folds", 3),
    )


def shuffle_coordinates_within_cell_type(adata, labels_key, shuffle_seed):
    """Permute coordinates for the configured fraction of cells within each cell type."""
    rng = np.random.default_rng(shuffle_seed)
    adata = adata.copy()
    spatial = np.asarray(adata.obsm["spatial"]).copy()
    shuffled = np.zeros(adata.n_obs, dtype=bool)

    for _, cell_indices in adata.obs.groupby(labels_key, observed=True).indices.items():
        cell_indices = np.asarray(cell_indices)
        n_shuffle = int(np.floor(SHUFFLE_FRACTION * len(cell_indices)))
        if n_shuffle < 2:
            continue

        selected = rng.choice(cell_indices, size=n_shuffle, replace=False)
        permuted = selected.copy()
        rng.shuffle(permuted)
        spatial[selected] = spatial[permuted]
        shuffled[selected] = True

    adata.obsm["spatial"] = spatial
    adata.obs["coordinate_shuffle_seed"] = shuffle_seed
    adata.obs["coordinate_shuffle_fraction"] = SHUFFLE_FRACTION
    adata.obs["coordinates_shuffled"] = shuffled
    return adata


def cache_matches_run(result, run):
    """Return whether an on-disk run result was produced with the requested config."""
    keys = [
        "shuffle_seed",
        "shuffle_fraction",
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


def train_or_load_run(adata, dataset_config, run, run_path, result_path, eval_indices):
    """Train one AMICI run or load its cached result."""
    if os.path.exists(result_path) and os.path.exists(os.path.join(run_path, "model.pt")):
        with open(result_path) as f:
            result = json.load(f)
        if cache_matches_run(result, run):
            return result

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
    result = {**run, "test_loss": test_loss, "model_path": run_path}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def compute_length_scale_estimates(adata, model, dataset_config, shuffle_seed, best_result):
    """Compute mean best-head length scales for all configured interactions."""
    explained_variance = model.get_expl_variance_scores(adata=adata)
    estimate_records = []
    sample_records = []

    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        receiver_type = interaction_config["receiver"]
        sender_type = interaction_config["sender"]
        interaction_label = f"{sender_type} -> {receiver_type}"

        counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
            adata=adata,
            cell_type=receiver_type,
        )
        best_head = explained_variance.compute_max_explained_variance_head(cell_type=receiver_type)
        length_scale_df = counterfactual_attention_patterns._calculate_length_scales(
            head_idxs=[best_head],
            sender_types=[sender_type],
            attention_threshold=ATTENTION_THRESHOLD,
        )
        mean_length_scale = length_scale_df["length_scale"].mean()

        estimate_records.append(
            {
                "shuffle_seed": shuffle_seed,
                "interaction": interaction_name,
                "interaction_label": interaction_label,
                "receiver_type": receiver_type,
                "sender_type": sender_type,
                "head_idx": best_head,
                "mean_length_scale": mean_length_scale,
                "n_length_scale_samples": len(length_scale_df),
                "gt_length_scale": interaction_config["length_scale"],
                "best_end_val": best_result["end_val"],
                "best_value_l1": best_result["value_l1"],
                "best_train_seed": best_result["train_seed"],
                "best_test_loss": best_result["test_loss"],
                "best_model_path": best_result["model_path"],
            }
        )

        length_scale_df["shuffle_seed"] = shuffle_seed
        length_scale_df["interaction"] = interaction_name
        length_scale_df["interaction_label"] = interaction_label
        length_scale_df["receiver_type"] = receiver_type
        length_scale_df["sender_type"] = sender_type
        length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
        sample_records.append(length_scale_df)

    return pd.DataFrame(estimate_records), pd.concat(sample_records, ignore_index=True)


def plot_length_scale_boxplots(estimates_df, figures_dir):
    """Plot mean inferred length scales across shuffled datasets."""
    interactions = list(estimates_df["interaction"].drop_duplicates())
    labels = [
        estimates_df.loc[estimates_df["interaction"] == interaction, "interaction_label"].iloc[0]
        for interaction in interactions
    ]
    box_data = [
        estimates_df.loc[estimates_df["interaction"] == interaction, "mean_length_scale"].dropna().to_numpy()
        for interaction in interactions
    ]
    gt_values = [
        estimates_df.loc[estimates_df["interaction"] == interaction, "gt_length_scale"].iloc[0]
        for interaction in interactions
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(box_data, labels=labels, patch_artist=True, boxprops={"facecolor": "steelblue", "alpha": 0.5})

    rng = np.random.default_rng(0)
    for x_pos, values in enumerate(box_data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), x_pos) + jitter, values, color="black", s=18, alpha=0.7, zorder=3)

    for x_pos, gt in enumerate(gt_values, start=1):
        ax.hlines(gt, x_pos - 0.35, x_pos + 0.35, color="red", linestyle="--", linewidth=1.5)

    ax.set_ylabel("Mean inferred length scale")
    ax.set_xlabel("Ground-truth interaction")
    ax.set_title("Length scale sensitivity to within-cell-type coordinate shuffling")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figures_dir, f"{OUTPUT_PREFIX}_sensitivity.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_task_ground_truth(adata, dataset_config):
    """Compute ground-truth labels for the three benchmark tasks."""
    subtype_key = dataset_config["subtype_labels_key"]
    labels_key = dataset_config["labels_key"]

    gt_gene_records = []
    for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
        gt_gene_scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            subtype_key,
        )
        gt_gene_scores["interaction"] = interaction_name
        gt_gene_records.append(gt_gene_scores)
    gt_gene_scores_df = pd.concat(gt_gene_records, ignore_index=True)

    receiver_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in dataset_config["gt_interactions"].values():
        receiver_mask |= adata.obs[subtype_key] == interaction_config["interaction_subtype"]
    gt_receiver_classes_df = pd.DataFrame({"cell_idx": adata.obs_names, "class": receiver_mask.astype(float)})

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
        gene_score_records = []
        for interaction_name, interaction_config in dataset_config["gt_interactions"].items():
            gene_scores = get_amici_gene_task_scores(
                model,
                adata,
                interaction_config["sender"],
                interaction_config["receiver"],
            )
            gene_scores["interaction"] = interaction_name
            gene_score_records.append(gene_scores)
        gene_scores_df = pd.concat(gene_score_records, ignore_index=True)
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


def evaluate_cached_shuffle_auprcs(condition, data_dir, model_root_dir, figures_dir, dataset_config):
    """Load cached shuffled datasets/models and compute AUPRCs."""
    records = []
    for shuffle_seed in SHUFFLE_SEEDS:
        condition_dir = os.path.join(model_root_dir, condition["model_run_name"], f"shuffle_{shuffle_seed}")
        result_path = os.path.join(condition_dir, "task_auprc_scores.csv")
        model_path = os.path.join(condition_dir, "best_model")
        adata_path = os.path.join(data_dir, condition["data_template"].format(seed=shuffle_seed))

        if os.path.exists(result_path):
            task_df = pd.read_csv(result_path)
        else:
            if not os.path.exists(os.path.join(model_path, "model.pt")):
                raise FileNotFoundError(f"Missing cached AMICI model: {model_path}")
            if not os.path.exists(adata_path):
                raise FileNotFoundError(f"Missing cached shuffled AnnData: {adata_path}")

            adata = sc.read_h5ad(adata_path)
            adata.obs_names_make_unique()
            AMICI.setup_anndata(
                adata,
                labels_key=dataset_config["labels_key"],
                coord_obsm_key="spatial",
                n_neighbors=int(dataset_config["sweep_params"].get("n_neighbors", [50])[0]),
            )
            model = AMICI.load(model_path, adata=adata)
            task_df = evaluate_tasks(model, adata, dataset_config)
            task_df.to_csv(result_path, index=False)

        task_df["shuffle_seed"] = shuffle_seed
        task_df["shuffle_fraction"] = condition["shuffle_fraction"]
        task_df["shuffle_label"] = condition["shuffle_label"]
        task_df["model_path"] = model_path
        records.append(task_df)

    condition_df = pd.concat(records, ignore_index=True)
    condition_df.to_csv(
        os.path.join(figures_dir, f"{AUPRC_OUTPUT_PREFIX}_{condition['shuffle_label'].replace('.', 'p')}.csv"),
        index=False,
    )
    return condition_df


def plot_cached_shuffle_auprcs(auprc_df, figures_dir):
    """Plot task AUPRCs across coordinate shuffling fractions."""
    task_order = ["Gene", "Neighbor interaction", "Receiver subtype"]
    fraction_order = ["0.3", "0.5", "1.0"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    rng = np.random.default_rng(0)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, task in zip(axes, task_order, strict=False):
        subset = auprc_df[(auprc_df["task"] == task) & (auprc_df["status"] == "success")]
        box_data = [
            subset.loc[subset["shuffle_label"] == label, "auprc"].dropna().to_numpy() for label in fraction_order
        ]
        ax.boxplot(
            box_data,
            labels=fraction_order,
            patch_artist=True,
            boxprops={"facecolor": "#d9d9d9", "alpha": 0.7},
            medianprops={"color": "black"},
        )
        for x_pos, values in enumerate(box_data, start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                color=colors[x_pos - 1],
                s=18,
                alpha=0.75,
                zorder=3,
            )
        ax.set_title(task)
        ax.set_xlabel("Coordinate shuffle fraction")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    fig.suptitle("AMICI task recovery after coordinate shuffling", y=1.03)
    plt.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figures_dir, f"{AUPRC_OUTPUT_PREFIX}_boxplots.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# %% Setup paths
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
config = CONFIG
dataset_config = config["datasets"][DATASET]

data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figures_dir = os.path.join(base_dir, "figures", RUN_NAME)
auprc_figures_dir = os.path.join(base_dir, "figures", AUPRC_RUN_NAME)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(auprc_figures_dir, exist_ok=True)

output_paths = [
    os.path.join(figures_dir, f"{OUTPUT_PREFIX}_estimates.csv"),
    os.path.join(figures_dir, f"{OUTPUT_PREFIX}_sensitivity.png"),
    os.path.join(figures_dir, f"{OUTPUT_PREFIX}_sensitivity.svg"),
]
if all(os.path.exists(path) for path in output_paths):
    print("Length scale coordinate shuffle sensitivity has already been plotted and saved. Continuing to AUPRCs.")

base_adata_path = benchmark_path(benchmark_dir, os.path.join(config["dir_path"], f"{DATASET}_{BASE_DATASET_SEED}.h5ad"))
ensure_base_realistic_dataset(benchmark_dir, dataset_config, base_adata_path)
base_adata = sc.read_h5ad(base_adata_path)
base_adata.obs_names_make_unique()

run_config = build_run_config(dataset_config)
estimate_records = []
sample_records = []

# %% Generate shuffled datasets, train AMICI, and collect length scales
for shuffle_seed in SHUFFLE_SEEDS:
    random.seed(shuffle_seed)
    np.random.seed(shuffle_seed)
    torch.manual_seed(shuffle_seed)
    scvi.settings.seed = shuffle_seed

    shuffled_adata_path = os.path.join(
        data_dir,
        f"{DATASET}_{BASE_DATASET_SEED}_coord_{SHUFFLE_LABEL}_shuffle_{shuffle_seed}.h5ad",
    )
    dataset_model_dir = os.path.join(model_dir, f"shuffle_{shuffle_seed}")
    best_model_path = os.path.join(dataset_model_dir, "best_model")
    best_params_path = os.path.join(dataset_model_dir, "best_model_params.json")
    estimates_path = os.path.join(dataset_model_dir, "length_scale_estimates.csv")
    samples_path = os.path.join(dataset_model_dir, "length_scale_samples.csv")
    os.makedirs(dataset_model_dir, exist_ok=True)

    if os.path.exists(estimates_path) and os.path.exists(samples_path):
        estimate_records.append(pd.read_csv(estimates_path))
        sample_records.append(pd.read_csv(samples_path))
        continue

    if os.path.exists(shuffled_adata_path):
        adata = sc.read_h5ad(shuffled_adata_path)
    else:
        adata = shuffle_coordinates_within_cell_type(
            base_adata,
            dataset_config["labels_key"],
            shuffle_seed,
        )
        adata.write_h5ad(shuffled_adata_path)

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(dataset_config["sweep_params"].get("n_neighbors", [50])[0]),
    )
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]

    run = {**run_config, "shuffle_seed": shuffle_seed, "shuffle_fraction": SHUFFLE_FRACTION, "run_idx": 0}
    run_path = os.path.join(dataset_model_dir, "run_0")
    result_path = os.path.join(dataset_model_dir, "run_0_results.json")
    best_result = train_or_load_run(adata, dataset_config, run, run_path, result_path, test_indices)
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(best_result["model_path"], best_model_path)

    with open(best_params_path, "w") as f:
        json.dump(best_result, f, indent=2)

    model = AMICI.load(best_model_path, adata=adata)
    estimate_df, sample_df = compute_length_scale_estimates(adata, model, dataset_config, shuffle_seed, best_result)
    estimate_df.to_csv(estimates_path, index=False)
    sample_df.to_csv(samples_path, index=False)

    estimate_records.append(estimate_df)
    sample_records.append(sample_df)

estimates_df = pd.concat(estimate_records, ignore_index=True)
samples_df = pd.concat(sample_records, ignore_index=True)
estimates_df.to_csv(os.path.join(figures_dir, f"{OUTPUT_PREFIX}_estimates.csv"), index=False)
samples_df.to_csv(os.path.join(figures_dir, f"{OUTPUT_PREFIX}_samples.csv"), index=False)

# %% Plot mean inferred length scales
plot_length_scale_boxplots(estimates_df, figures_dir)

# %% Evaluate cached AMICI models for three coordinate shuffling fractions
auprc_records = []
model_root_dir = os.path.join(base_dir, "saved_models")
for condition in COORDINATE_SHUFFLE_CONDITIONS:
    auprc_records.append(
        evaluate_cached_shuffle_auprcs(
            condition,
            data_dir,
            model_root_dir,
            auprc_figures_dir,
            dataset_config,
        )
    )

auprc_df = pd.concat(auprc_records, ignore_index=True)
auprc_df.to_csv(os.path.join(auprc_figures_dir, f"{AUPRC_OUTPUT_PREFIX}_summary.csv"), index=False)
plot_cached_shuffle_auprcs(auprc_df, auprc_figures_dir)
