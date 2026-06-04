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
DATASET_SEED = 6
LABELS_KEY = "leiden"
SUBTYPE_KEY = "subtype"
ATTENTION_THRESHOLD = 0.1
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
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
TRAIN_SEEDS = [22, 38, 17, 11, 42, 33, 18]
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

# %% Setup paths
select_gpu()
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "saved_models", "length_scale_uncertainty")
figures_dir = os.path.join(base_dir, "figures", "length_scale_uncertainty")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

ci_output_paths = [
    os.path.join(figures_dir, "length_scale_bootstrap_ci.csv"),
    os.path.join(figures_dir, "length_scale_bootstrap_ci.png"),
    os.path.join(figures_dir, "length_scale_bootstrap_ci.svg"),
]
if all(os.path.exists(path) for path in ci_output_paths):
    print("Length scale CI has already been plotted and saved. Skipping length scale uncertainty analysis.")
    sys.exit(0)

adata_path = os.path.join(data_dir, f"semisyn_{DATASET_SEED}.h5ad")
best_model_path = os.path.join(model_dir, "best_model")

# %% Generate dataset
random.seed(DATASET_SEED)
np.random.seed(DATASET_SEED)
torch.manual_seed(DATASET_SEED)
scvi.settings.seed = DATASET_SEED

interaction_df = _create_interaction_df(GT_INTERACTIONS)
generate_synthetic_dataset(interaction_df, adata_path)
adata = sc.read_h5ad(adata_path)
adata.obs_names_make_unique()

# %% Train models
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

test_indices = np.where(adata.obs["train_test_split"] == "test")[0]
run_results = []

for run_idx, run in enumerate(all_runs):
    run_path = os.path.join(model_dir, f"run_{run_idx}")
    result_path = os.path.join(model_dir, f"run_{run_idx}_results.json")

    if os.path.exists(result_path) and os.path.exists(os.path.join(run_path, "model.pt")):
        with open(result_path) as f:
            run_results.append(json.load(f))
        continue

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
        model.get_reconstruction_error(adata, indices=test_indices, batch_size=128)["reconstruction_loss"]
        .detach()
        .cpu()
        .numpy()
        .item()
    )
    result = {**run, "run_idx": run_idx, "test_loss": test_loss, "model_path": run_path}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    run_results.append(result)

best_result = min(run_results, key=lambda x: x["test_loss"])
if os.path.exists(best_model_path):
    shutil.rmtree(best_model_path)
shutil.copytree(best_result["model_path"], best_model_path)

with open(os.path.join(model_dir, "best_model_params.json"), "w") as f:
    json.dump(best_result, f, indent=2)

# %% Compute length scale bootstrap confidence intervals
AMICI.setup_anndata(
    adata,
    labels_key=LABELS_KEY,
    coord_obsm_key="spatial",
    n_neighbors=EXP_PARAMS["n_neighbors"],
)
model = AMICI.load(best_model_path, adata=adata)
explained_variance = model.get_expl_variance_scores(adata=adata)

ci_records = []
length_scale_records = []
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
    ci_df = counterfactual_attention_patterns._compute_length_scale_bootstrap_ci(
        length_scale_df,
        confidence_level=CONFIDENCE_LEVEL,
        n_bootstrap=N_BOOTSTRAP,
        random_state=DATASET_SEED,
        statistic="mean",
    )

    length_scale_df["interaction"] = interaction_name
    length_scale_df["interaction_label"] = interaction_label
    length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
    ci_df["interaction"] = interaction_name
    ci_df["interaction_label"] = interaction_label
    ci_df["receiver_type"] = receiver_type
    ci_df["gt_length_scale"] = interaction_config["length_scale"]

    length_scale_records.append(length_scale_df)
    ci_records.append(ci_df)

length_scale_df = pd.concat(length_scale_records, ignore_index=True)
ci_df = pd.concat(ci_records, ignore_index=True)
length_scale_df.to_csv(os.path.join(figures_dir, "length_scale_samples.csv"), index=False)
ci_df.to_csv(os.path.join(figures_dir, "length_scale_bootstrap_ci.csv"), index=False)

# %% Plot confidence intervals
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = np.arange(len(ci_df))
xerr = np.vstack(
    [
        ci_df["point_estimate"] - ci_df["ci_lower"],
        ci_df["ci_upper"] - ci_df["point_estimate"],
    ]
)
ax.errorbar(
    ci_df["point_estimate"],
    y_pos,
    xerr=xerr,
    fmt="o",
    capsize=4,
    color="steelblue",
)
for y, gt in zip(y_pos, ci_df["gt_length_scale"], strict=False):
    ax.scatter(gt, y, marker="x", color="black", zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(ci_df["interaction_label"])
ax.set_xlabel("Length scale")
ax.set_title("Bootstrap confidence intervals for AMICI length scales")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()

for ext in ("png", "svg"):
    fig.savefig(os.path.join(figures_dir, f"length_scale_bootstrap_ci.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
