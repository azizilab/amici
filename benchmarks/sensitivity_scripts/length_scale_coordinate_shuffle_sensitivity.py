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

from generate_realistic_dataset import generate_realistic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATASET = "breast_cancer"
BASE_DATASET_SEED = 42
SHUFFLE_SEEDS = list(range(10))
ATTENTION_THRESHOLD = 0.1
SHUFFLE_FRACTION = 1.0
TRAIN_SEED = 22
RUN_NAME = "length_scale_coordinate_full_shuffle_sensitivity"

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


def shuffle_half_coordinates_within_cell_type(adata, labels_key, shuffle_seed):
    """Permute coordinates for half of cells within each cell type."""
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
    adata.obs["coordinates_shuffled"] = shuffled
    return adata


def train_or_load_run(adata, dataset_config, run, run_path, result_path, eval_indices):
    """Train one AMICI run or load its cached result."""
    if os.path.exists(result_path) and os.path.exists(os.path.join(run_path, "model.pt")):
        with open(result_path) as f:
            return json.load(f)

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
            os.path.join(figures_dir, f"length_scale_coordinate_full_shuffle_sensitivity.{ext}"),
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
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

output_paths = [
    os.path.join(figures_dir, "length_scale_coordinate_full_shuffle_estimates.csv"),
    os.path.join(figures_dir, "length_scale_coordinate_full_shuffle_sensitivity.png"),
    os.path.join(figures_dir, "length_scale_coordinate_full_shuffle_sensitivity.svg"),
]
if all(os.path.exists(path) for path in output_paths):
    print("Length scale coordinate shuffle sensitivity has already been plotted and saved. Skipping analysis.")
    sys.exit(0)

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
        f"{DATASET}_{BASE_DATASET_SEED}_coord_full_shuffle_{shuffle_seed}.h5ad",
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
        adata = shuffle_half_coordinates_within_cell_type(
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

    run = {**run_config, "shuffle_seed": shuffle_seed, "run_idx": 0}
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
estimates_df.to_csv(os.path.join(figures_dir, "length_scale_coordinate_full_shuffle_estimates.csv"), index=False)
samples_df.to_csv(os.path.join(figures_dir, "length_scale_coordinate_full_shuffle_samples.csv"), index=False)

# %% Plot mean inferred length scales
plot_length_scale_boxplots(estimates_df, figures_dir)
