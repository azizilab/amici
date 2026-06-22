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

from generate_dataset import _create_interaction_df, generate_synthetic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATASET_SEEDS = list(range(50))
LABELS_KEY = "leiden"
SUBTYPE_KEY = "subtype"
ATTENTION_THRESHOLD = 0.1
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
LENGTH_SCALE_STATISTIC = "mean"
GT_INTERACTIONS = {
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

END_PENALTY_VALUES = [1e-2, 1e-3, 1e-4]
VALUE_L1_VALUES = [1e-6, 1e-5, 1e-4]
TRAIN_SEEDS = [22, 38, 42]
SCHEDULES = [[10, 40]]
EXP_PARAMS = {
    "lr": 1e-3,
    "epochs": 400,
    "batch_size": 128,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
    "n_neighbors": 50,
    "n_heads": 8,
}


def compute_bootstrap_ci(values, confidence_level=0.95, n_bootstrap=10000, random_state=None, statistic="mean"):
    """Return a percentile bootstrap CI for the requested statistic."""
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got '{statistic}'")

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("Cannot compute a bootstrap CI with no finite values.")

    stat_func = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(random_state)
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, len(values), size=len(values))
        bootstrap_stats[i] = stat_func(values[resample_idx])

    alpha = 1 - confidence_level
    return {
        "statistic": statistic,
        "point_estimate": stat_func(values),
        "ci_lower": np.percentile(bootstrap_stats, 100 * alpha / 2),
        "ci_upper": np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)),
        "n_samples": len(values),
    }


def train_or_load_run(adata, run, run_path, result_path, test_indices):
    """Train one AMICI sweep run or load its cached result."""
    if os.path.exists(result_path) and os.path.exists(os.path.join(run_path, "model.pt")):
        with open(result_path) as f:
            return json.load(f)

    pl.seed_everything(run["train_seed"])
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    AMICI.setup_anndata(
        adata_train,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=EXP_PARAMS["n_neighbors"],
    )
    model = AMICI(
        adata_train,
        n_heads=EXP_PARAMS["n_heads"],
        value_l1_penalty_coef=run["value_l1"],
    )
    model.train(
        max_epochs=EXP_PARAMS["epochs"],
        batch_size=EXP_PARAMS["batch_size"],
        plan_kwargs={"lr": EXP_PARAMS["lr"]},
        early_stopping=EXP_PARAMS["early_stopping"],
        early_stopping_monitor=EXP_PARAMS["early_stopping_monitor"],
        early_stopping_patience=EXP_PARAMS["early_stopping_patience"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                start_val=1e-6,
                end_val=run["end_val"],
                epoch_start=run["epoch_start"],
                epoch_end=run["epoch_end"],
                flavor="linear",
            ),
        ],
    )
    AMICI.setup_anndata(
        adata,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=EXP_PARAMS["n_neighbors"],
    )
    model.save(run_path, overwrite=True)
    test_loss = (
        model.get_reconstruction_error(adata, indices=test_indices, batch_size=EXP_PARAMS["batch_size"])[
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


def compute_best_model_length_scale_estimates(adata, model, dataset_seed, best_result):
    """Compute best-head length scale estimates for each ground-truth interaction."""
    explained_variance = model.get_expl_variance_scores(adata=adata)
    estimate_records = []
    sample_records = []

    for interaction_name, interaction_config in GT_INTERACTIONS.items():
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
        length_scale_values = length_scale_df["length_scale"].to_numpy()

        estimate_records.append(
            {
                "dataset_seed": dataset_seed,
                "interaction": interaction_name,
                "interaction_label": interaction_label,
                "receiver_type": receiver_type,
                "sender_type": sender_type,
                "head_idx": best_head,
                "length_scale": np.mean(length_scale_values)
                if LENGTH_SCALE_STATISTIC == "mean"
                else np.median(length_scale_values),
                "n_neighbor_samples": len(length_scale_values),
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
        length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
        sample_records.append(length_scale_df)

    return pd.DataFrame(estimate_records), pd.concat(sample_records, ignore_index=True)


# %% Setup paths
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "saved_models", "length_scale_dataset_bootstrap_ci")
figures_dir = os.path.join(base_dir, "figures", "length_scale_dataset_bootstrap_ci")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

ci_output_paths = [
    os.path.join(figures_dir, "length_scale_dataset_bootstrap_ci.csv"),
    os.path.join(figures_dir, "length_scale_dataset_bootstrap_ci_violin.png"),
    os.path.join(figures_dir, "length_scale_dataset_bootstrap_ci_violin.svg"),
]
if all(os.path.exists(path) for path in ci_output_paths):
    print("Dataset bootstrap length scale CI has already been plotted and saved. Skipping analysis.")
    sys.exit(0)

interaction_df = _create_interaction_df(GT_INTERACTIONS)

all_runs = []
for end_val, value_l1, train_seed, schedule in itertools.product(
    END_PENALTY_VALUES,
    VALUE_L1_VALUES,
    TRAIN_SEEDS,
    SCHEDULES,
):
    all_runs.append(
        {
            "end_val": end_val,
            "value_l1": value_l1,
            "train_seed": train_seed,
            "epoch_start": schedule[0],
            "epoch_end": schedule[1],
        }
    )

# %% Generate datasets, sweep AMICI models, and collect best-model length scales
estimate_records = []
sample_records = []

for dataset_seed in DATASET_SEEDS:
    adata_path = os.path.join(data_dir, f"semisyn_{dataset_seed}.h5ad")
    dataset_model_dir = os.path.join(model_dir, f"dataset_{dataset_seed}")
    best_model_path = os.path.join(dataset_model_dir, "best_model")
    best_params_path = os.path.join(dataset_model_dir, "best_model_params.json")
    estimates_path = os.path.join(dataset_model_dir, "length_scale_model_estimates.csv")
    samples_path = os.path.join(dataset_model_dir, "length_scale_samples.csv")
    os.makedirs(dataset_model_dir, exist_ok=True)

    if os.path.exists(estimates_path) and os.path.exists(samples_path):
        estimate_records.append(pd.read_csv(estimates_path))
        sample_records.append(pd.read_csv(samples_path))
        continue

    random.seed(dataset_seed)
    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    scvi.settings.seed = dataset_seed

    generate_synthetic_dataset(interaction_df, adata_path)
    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()

    AMICI.setup_anndata(
        adata,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=EXP_PARAMS["n_neighbors"],
    )
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]

    run_results = []
    for run_idx, run in enumerate(all_runs):
        run = {**run, "dataset_seed": dataset_seed, "run_idx": run_idx}
        run_path = os.path.join(dataset_model_dir, f"run_{run_idx}")
        result_path = os.path.join(dataset_model_dir, f"run_{run_idx}_results.json")
        run_results.append(train_or_load_run(adata, run, run_path, result_path, test_indices))

    best_result = min(run_results, key=lambda x: x["test_loss"])
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(best_result["model_path"], best_model_path)

    with open(best_params_path, "w") as f:
        json.dump(best_result, f, indent=2)

    model = AMICI.load(best_model_path, adata=adata)
    estimate_df, sample_df = compute_best_model_length_scale_estimates(adata, model, dataset_seed, best_result)
    estimate_df.to_csv(estimates_path, index=False)
    sample_df.to_csv(samples_path, index=False)

    estimate_records.append(estimate_df)
    sample_records.append(sample_df)

model_estimates_df = pd.concat(estimate_records, ignore_index=True)
length_scale_samples_df = pd.concat(sample_records, ignore_index=True)
model_estimates_df.to_csv(os.path.join(figures_dir, "length_scale_model_estimates.csv"), index=False)
length_scale_samples_df.to_csv(os.path.join(figures_dir, "length_scale_samples.csv"), index=False)

# %% Compute bootstrap confidence intervals across best models
ci_records = []
for (interaction_name, sender_type), group in model_estimates_df.groupby(["interaction", "sender_type"]):
    interaction_config = GT_INTERACTIONS[interaction_name]
    ci_record = compute_bootstrap_ci(
        group["length_scale"],
        confidence_level=CONFIDENCE_LEVEL,
        n_bootstrap=N_BOOTSTRAP,
        random_state=interaction_config["length_scale"],
        statistic=LENGTH_SCALE_STATISTIC,
    )
    ci_record.update(
        {
            "interaction": interaction_name,
            "interaction_label": group["interaction_label"].iloc[0],
            "receiver_type": group["receiver_type"].iloc[0],
            "sender_type": sender_type,
            "head_idxs": ",".join(map(str, sorted(group["head_idx"].unique()))),
            "gt_length_scale": interaction_config["length_scale"],
        }
    )
    ci_records.append(ci_record)

ci_df = pd.DataFrame(ci_records)
ci_df.to_csv(os.path.join(figures_dir, "length_scale_dataset_bootstrap_ci.csv"), index=False)

# %% Plot length scale distributions
plot_df = model_estimates_df.dropna(subset=["length_scale"]).copy()
plot_interactions = (
    plot_df[["interaction", "interaction_label", "gt_length_scale"]].drop_duplicates().sort_values("gt_length_scale")
)
positions = plot_interactions["gt_length_scale"].astype(float).to_numpy()
violin_data = [
    plot_df.loc[plot_df["interaction"] == interaction, "length_scale"].to_numpy()
    for interaction in plot_interactions["interaction"]
]
min_spacing = np.min(np.diff(np.sort(positions))) if len(positions) > 1 else 1.0
violin_width = max(0.8, min_spacing * 0.45)

fig, ax = plt.subplots(figsize=(7, 4))
violins = ax.violinplot(
    violin_data,
    positions=positions,
    widths=violin_width,
    showmeans=True,
    showextrema=False,
)
for body in violins["bodies"]:
    body.set_facecolor("steelblue")
    body.set_edgecolor("black")
    body.set_alpha(0.45)
violins["cmeans"].set_color("black")
violins["cmeans"].set_linewidth(1.2)

rng = np.random.default_rng(0)
for x_pos, values in zip(positions, violin_data, strict=False):
    jitter = rng.uniform(-0.08 * violin_width, 0.08 * violin_width, size=len(values))
    ax.scatter(np.full(len(values), x_pos) + jitter, values, color="black", s=12, alpha=0.45, zorder=3)

axis_min = min(np.nanmin(plot_df["length_scale"]), np.nanmin(positions))
axis_max = max(np.nanmax(plot_df["length_scale"]), np.nanmax(positions))
ax.plot([axis_min, axis_max], [axis_min, axis_max], color="red", linestyle="--", linewidth=1.2)
ax.set_xticks(positions)
ax.set_xticklabels([f"{gt:g}" for gt in positions])
ax.set_xlabel("Ground-truth length scale")
ax.set_ylabel("Mean inferred length scale")
ax.set_title("Dataset bootstrap length scale distributions for AMICI")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()

for ext in ("png", "svg"):
    fig.savefig(os.path.join(figures_dir, f"length_scale_dataset_bootstrap_ci.{ext}"), dpi=300, bbox_inches="tight")
    fig.savefig(
        os.path.join(figures_dir, f"length_scale_dataset_bootstrap_ci_violin.{ext}"),
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)
