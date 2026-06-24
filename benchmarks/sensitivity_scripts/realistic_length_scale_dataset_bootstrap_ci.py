# %% Import libraries
import itertools
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

import generate_realistic_dataset as realistic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATASET = "breast_cancer"
DATASET_SEEDS = list(range(10))
ATTENTION_THRESHOLD = 0.1
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 10000
RUN_NAME = "realistic_length_scale_dataset_bootstrap_ci"

CONFIG = {
    "dir_path": "data/",
    "datasets": {
        "breast_cancer": {
            "generation_type": "realistic",
            "seeds": [42],
            "labels_key": "cell_type",
            "subtype_labels_key": "subtype",
            "use_cross_validation": False,
            "n_cv_folds": 3,
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
        }
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


def build_run_configs(dataset_config):
    """Build all AMICI configs from the benchmark sweep parameter set."""
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


def dataset_outputs_current(best_params_path, estimates_path, samples_path, run_configs):
    """Return whether cached length-scale outputs were computed with the current AMICI sweep."""
    if not (os.path.exists(best_params_path) and os.path.exists(estimates_path) and os.path.exists(samples_path)):
        return False
    with open(best_params_path) as f:
        best_params = json.load(f)

    expected_train_seeds = [run["train_seed"] for run in run_configs]
    return (
        best_params.get("n_candidate_runs") == len(run_configs)
        and best_params.get("candidate_train_seeds") == expected_train_seeds
        and best_params.get("selection_metric") == "test_loss"
    )


def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def generate_or_load_realistic_dataset(benchmark_dir, dataset_config, adata_path, dataset_seed):
    """Generate one realistic semi-synthetic dataset with a dataset-specific seed."""
    if os.path.exists(adata_path):
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()
        return adata

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
    return adata


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


def compute_bootstrap_ci(values, confidence_level, n_bootstrap, random_state, statistic=np.mean):
    """Compute a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}

    rng = np.random.default_rng(random_state)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence_level
    return {
        "point_estimate": statistic(values),
        "ci_lower": np.percentile(bootstrap_stats, 100 * alpha / 2),
        "ci_upper": np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)),
        "n": len(values),
    }


def compute_length_scale_estimates(adata, model, dataset_config, dataset_seed, best_result):
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
                "dataset_seed": dataset_seed,
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

        length_scale_df["dataset_seed"] = dataset_seed
        length_scale_df["interaction"] = interaction_name
        length_scale_df["interaction_label"] = interaction_label
        length_scale_df["receiver_type"] = receiver_type
        length_scale_df["sender_type"] = sender_type
        length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
        sample_records.append(length_scale_df)

    return pd.DataFrame(estimate_records), pd.concat(sample_records, ignore_index=True)


def plot_dataset_bootstrap_ci(model_estimates_df, figures_dir):
    """Plot mean length scale distributions across realistic semi-synthetic datasets."""
    plot_df = model_estimates_df.dropna(subset=["mean_length_scale"]).copy()
    plot_interactions = (
        plot_df[["interaction", "interaction_label", "gt_length_scale"]]
        .drop_duplicates()
        .sort_values("gt_length_scale")
    )
    positions = np.arange(1, len(plot_interactions) + 1)
    violin_data = [
        plot_df.loc[plot_df["interaction"] == interaction, "mean_length_scale"].to_numpy()
        for interaction in plot_interactions["interaction"]
    ]
    gt_positions = plot_interactions["gt_length_scale"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
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
        violins[part].set_linewidth(1.6)

    ax.scatter(gt_positions, positions, marker="x", color="black", s=180, linewidths=3.0, zorder=4)
    axis_values = np.concatenate([plot_df["mean_length_scale"].to_numpy(), gt_positions])
    axis_min = np.nanmin(axis_values)
    axis_max = np.nanmax(axis_values)
    axis_padding = 0.05 * (axis_max - axis_min) if axis_max > axis_min else 1.0
    ax.set_xlim(axis_min - axis_padding, axis_max + axis_padding)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_interactions["interaction_label"])
    ax.set_xlabel("Ground truth and inferred length scale")
    ax.set_ylabel("Interaction")
    ax.set_title("Realistic dataset bootstrap length scale distributions for AMICI")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(figures_dir, f"{RUN_NAME}.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(figures_dir, f"{RUN_NAME}_violin.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# %% Setup paths
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
dataset_config = CONFIG["datasets"][DATASET]

data_dir = os.path.join(base_dir, "data", RUN_NAME)
model_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figures_dir = os.path.join(base_dir, "figures", RUN_NAME)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

output_paths = [
    os.path.join(figures_dir, "length_scale_model_estimates.csv"),
    os.path.join(figures_dir, "length_scale_samples.csv"),
    os.path.join(figures_dir, f"{RUN_NAME}.csv"),
    os.path.join(figures_dir, f"{RUN_NAME}_violin.png"),
    os.path.join(figures_dir, f"{RUN_NAME}_violin.svg"),
]
missing_outputs = [path for path in output_paths if not os.path.exists(path)]
if missing_outputs:
    print("Missing cached output files; recomputing realistic length scale dataset bootstrap outputs.")

run_configs = build_run_configs(dataset_config)
print(f"Sweeping {len(run_configs)} AMICI models per realistic dataset.")
estimate_records = []
sample_records = []

# %% Generate realistic datasets, train AMICI, and collect length scales
for dataset_seed in DATASET_SEEDS:
    print(f"Processing realistic dataset seed {dataset_seed}")
    set_all_seeds(dataset_seed)
    adata_path = os.path.join(data_dir, f"{DATASET}_{dataset_seed}.h5ad")
    dataset_model_dir = os.path.join(model_dir, f"dataset_seed_{dataset_seed}")
    best_model_path = os.path.join(dataset_model_dir, "best_model")
    best_params_path = os.path.join(dataset_model_dir, "best_model_params.json")
    estimates_path = os.path.join(dataset_model_dir, "length_scale_estimates.csv")
    samples_path = os.path.join(dataset_model_dir, "length_scale_samples.csv")
    os.makedirs(dataset_model_dir, exist_ok=True)

    if dataset_outputs_current(best_params_path, estimates_path, samples_path, run_configs):
        estimate_records.append(pd.read_csv(estimates_path))
        sample_records.append(pd.read_csv(samples_path))
        continue

    adata = generate_or_load_realistic_dataset(benchmark_dir, dataset_config, adata_path, dataset_seed)
    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run_configs[0]["n_neighbors"]),
    )
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]

    run_results = []
    for run_idx, run_config in enumerate(run_configs):
        run = {**run_config, "dataset_seed": dataset_seed, "run_idx": run_idx}
        run_path = os.path.join(dataset_model_dir, f"run_{run_idx}_seed_{run['train_seed']}")
        result_path = os.path.join(dataset_model_dir, f"run_{run_idx}_seed_{run['train_seed']}_results.json")
        run_results.append(train_or_load_run(adata, dataset_config, run, run_path, result_path, test_indices))

    best_result = min(run_results, key=lambda x: x["test_loss"])
    best_result = {
        **best_result,
        "selection_metric": "test_loss",
        "n_candidate_runs": len(run_results),
        "candidate_train_seeds": [run["train_seed"] for run in run_configs],
        "candidate_test_losses": [
            {
                "run_idx": result["run_idx"],
                "train_seed": result["train_seed"],
                "test_loss": result["test_loss"],
            }
            for result in run_results
        ],
    }
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(best_result["model_path"], best_model_path)

    with open(best_params_path, "w") as f:
        json.dump(best_result, f, indent=2)

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(best_result["n_neighbors"]),
    )
    model = AMICI.load(best_model_path, adata=adata)
    estimate_df, sample_df = compute_length_scale_estimates(adata, model, dataset_config, dataset_seed, best_result)
    estimate_df.to_csv(estimates_path, index=False)
    sample_df.to_csv(samples_path, index=False)

    estimate_records.append(estimate_df)
    sample_records.append(sample_df)

model_estimates_df = pd.concat(estimate_records, ignore_index=True)
length_scale_samples_df = pd.concat(sample_records, ignore_index=True)
model_estimates_df.to_csv(os.path.join(figures_dir, "length_scale_model_estimates.csv"), index=False)
length_scale_samples_df.to_csv(os.path.join(figures_dir, "length_scale_samples.csv"), index=False)

# %% Compute bootstrap confidence intervals across realistic datasets
ci_records = []
for interaction_name, group in model_estimates_df.groupby("interaction", sort=False):
    interaction_config = dataset_config["gt_interactions"][interaction_name]
    ci_record = compute_bootstrap_ci(
        group["mean_length_scale"],
        confidence_level=CONFIDENCE_LEVEL,
        n_bootstrap=N_BOOTSTRAP,
        random_state=interaction_config["length_scale"],
        statistic=np.mean,
    )
    ci_record.update(
        {
            "interaction": interaction_name,
            "interaction_label": group["interaction_label"].iloc[0],
            "receiver_type": group["receiver_type"].iloc[0],
            "sender_type": group["sender_type"].iloc[0],
            "head_idxs": ",".join(map(str, sorted(group["head_idx"].dropna().unique()))),
            "gt_length_scale": interaction_config["length_scale"],
        }
    )
    ci_records.append(ci_record)

ci_df = pd.DataFrame(ci_records)
ci_df.to_csv(os.path.join(figures_dir, f"{RUN_NAME}.csv"), index=False)

# %% Plot length scale distributions
plot_dataset_bootstrap_ci(model_estimates_df, figures_dir)
