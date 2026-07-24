# %% Imports
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
SENSITIVITY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BENCHMARK_DIR)
sys.path.insert(0, SENSITIVITY_DIR)

from amici import AMICI  # noqa: E402

# %% Settings
LABELS_KEY = "leiden"
RECEIVER_TYPE = "0"
SENDER_TYPE = "3"
INNER_RADIUS = 10
OUTER_RADIUS = 25
ATTENTION_THRESHOLD = 0.1
MAX_DISTANCE = 60
N_DISTANCE_POINTS = 300

DATA_PATH = "data/3ct_dataset_1int_2phase.h5ad"
BEST_PARAMS_PATH = "results/two_phase_amici_sweep/best_model_params.json"
MODEL_PATH = "saved_models/two_phase_amici_sweep/best_model"
FIGURE_DIR = "figures/two_phase_amici_sweep"
RESULTS_DIR = "results/two_phase_amici_sweep"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# %% Helpers
def sigmoid(x):
    """Compute a numerically stable sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def compute_head_summary(counterfactual_df):
    """Compute per-head attention curves and threshold-derived length scales."""
    df = counterfactual_df[counterfactual_df["neighbor_label"].astype(str) == SENDER_TYPE].copy()
    distances = np.linspace(0, MAX_DISTANCE, N_DISTANCE_POINTS)
    curve_rows = []
    length_rows = []

    for head_idx, head_df in df.groupby("head_idx"):
        base = head_df["base_attention_score"].to_numpy()
        dummy = head_df["dummy_attention_score"].to_numpy()
        pos_coef = head_df["position_coefficient"].to_numpy()
        scale = head_df["distance_kernel_unit_scale"].to_numpy()

        logits = base[:, None] - dummy[:, None] - (pos_coef[:, None] / scale[:, None]) * distances[None, :]
        attention = sigmoid(logits)
        mean_attention = attention.mean(axis=0)
        q25, q75 = np.quantile(attention, [0.25, 0.75], axis=0)

        for distance, mean_value, lo, hi in zip(distances, mean_attention, q25, q75, strict=False):
            curve_rows.append(
                {
                    "head_idx": int(head_idx),
                    "distance": distance,
                    "mean_attention": mean_value,
                    "q25_attention": lo,
                    "q75_attention": hi,
                }
            )

        length_scales = (scale / pos_coef) * (np.log((1 - ATTENTION_THRESHOLD) / ATTENTION_THRESHOLD) + base - dummy)
        length_scales = np.clip(length_scales, a_min=0, a_max=None)
        for neighbor_idx, length_scale in zip(head_df["neighbor_idx"], length_scales, strict=False):
            length_rows.append(
                {
                    "head_idx": int(head_idx),
                    "sender_type": SENDER_TYPE,
                    "receiver_type": RECEIVER_TYPE,
                    "neighbor_idx": neighbor_idx,
                    "length_scale": length_scale,
                    "attention_threshold": ATTENTION_THRESHOLD,
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(length_rows)


def plot_attention_curves(curve_df):
    """Plot mean counterfactual attention against distance by head."""
    heads = sorted(curve_df["head_idx"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(heads) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, head_idx in zip(axes, heads, strict=False):
        sub = curve_df[curve_df["head_idx"] == head_idx]
        ax.plot(sub["distance"], sub["mean_attention"], color="#4C78A8", linewidth=2)
        ax.fill_between(
            sub["distance"].to_numpy(),
            sub["q25_attention"].to_numpy(),
            sub["q75_attention"].to_numpy(),
            color="#4C78A8",
            alpha=0.2,
        )
        ax.axvline(INNER_RADIUS, color="black", linestyle="--", linewidth=1.3)
        ax.axvline(OUTER_RADIUS, color="black", linestyle=":", linewidth=1.6)
        ax.axhline(ATTENTION_THRESHOLD, color="#D62728", linestyle="-.", linewidth=1.1)
        ax.set_title(f"Head {head_idx}")
        ax.grid(alpha=0.2)

    for ax in axes[len(heads) :]:
        ax.axis("off")
    fig.supxlabel("Distance from receiver")
    fig.supylabel("Mean implied attention to sender type 3")
    fig.suptitle("Two-phase sender-to-receiver attention by head", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"two_phase_head_attention_summary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_length_scale_by_head(length_df):
    """Plot threshold-derived length-scale distributions by head."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(data=length_df, x="head_idx", y="length_scale", color="#9ecae1", ax=ax, showfliers=False)
    sns.stripplot(
        data=length_df.sample(min(len(length_df), 3000), random_state=18),
        x="head_idx",
        y="length_scale",
        color="0.25",
        size=1.2,
        alpha=0.25,
        ax=ax,
    )
    ax.axhline(INNER_RADIUS, color="black", linestyle="--", linewidth=1.4, label="GT inner radius 10")
    ax.axhline(OUTER_RADIUS, color="black", linestyle=":", linewidth=1.8, label="GT outer radius 25")
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Length scale at attention threshold 0.1")
    ax.set_title("Two-phase threshold-derived length scales by head")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGURE_DIR, f"two_phase_head_length_scale_summary.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


# %% Run
with open(BEST_PARAMS_PATH) as f:
    best = json.load(f)

adata = sc.read_h5ad(DATA_PATH)
adata.obs_names_make_unique()
AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=int(best["n_neighbors"]))
model = AMICI.load(MODEL_PATH, adata=adata)
counterfactual = model.get_counterfactual_attention_patterns(adata=adata, cell_type=RECEIVER_TYPE)
curve_df, length_df = compute_head_summary(counterfactual._counterfactual_attention_df)

curve_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_head_attention_curves.csv"), index=False)
length_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_head_length_scale_summary.csv"), index=False)
plot_attention_curves(curve_df)
plot_length_scale_by_head(length_df)

print(f"Saved head diagnostics to {FIGURE_DIR}", flush=True)
