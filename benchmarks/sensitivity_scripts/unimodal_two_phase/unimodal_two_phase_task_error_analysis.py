# %% Imports
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from sklearn.metrics import precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
SENSITIVITY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BENCHMARK_DIR)
sys.path.insert(0, SENSITIVITY_DIR)

from amici_benchmark_utils import (  # noqa: E402
    get_amici_neighbor_interaction_scores,
    get_amici_receiver_subtype_scores,
)
from benchmark_utils import get_interaction_gt_neighbor_classes  # noqa: E402

from amici import AMICI  # noqa: E402

# %% Settings
LABELS_KEY = "leiden"
SUBTYPE_KEY = "subtype"
RECEIVER_TYPE = "0"
SENDER_TYPE = "3"
INTERACTION_SUBTYPE = "0_sub1"
INTERACTION_START = 10
INTERACTION_PEAK = 20
INTERACTION_END = 30
DISTANCE_BIN_WIDTH = 2
MAX_PLOT_DISTANCE = 60

GT_INTERACTIONS = {
    "interaction_unimodal": {
        "sender": SENDER_TYPE,
        "receiver": RECEIVER_TYPE,
        "interaction_subtype": INTERACTION_SUBTYPE,
        "neutral_subtype": "0_sub0",
        "length_scale": INTERACTION_PEAK,
        "interaction_start": INTERACTION_START,
        "interaction_end": INTERACTION_END,
    },
}

DATA_PATH = "data/3ct_dataset_1int_unimodal_10_30.h5ad"
BEST_PARAMS_PATH = "results/unimodal_two_phase_amici_sweep/best_model_params.json"
MODEL_PATH = "saved_models/unimodal_two_phase_amici_sweep/best_model"
FIGURE_DIR = "figures/unimodal_two_phase_amici_sweep"
RESULTS_DIR = "results/unimodal_two_phase_amici_sweep"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOME_CLASS_COLORS = {
    "correct": "#0072B2",
    "incorrect": "#D55E00",
}


# %% Helpers
def best_f1_threshold(labels, scores):
    """Return the score threshold maximizing F1, used to turn rankings into calls."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[min(np.argmax(f1), len(thresholds) - 1)])


def label_outcomes(labels, calls):
    """Map (ground truth, call) pairs onto confusion-matrix outcome names."""
    outcomes = np.full(len(labels), "true negative", dtype=object)
    outcomes[(labels == 1) & (calls == 1)] = "true positive"
    outcomes[(labels == 1) & (calls == 0)] = "false negative"
    outcomes[(labels == 0) & (calls == 1)] = "false positive"
    return outcomes


def collapse_outcomes(labels, calls):
    """Collapse the confusion matrix into correct and incorrect calls."""
    return np.where(labels == calls, "correct", "incorrect")


def nearest_sender_distance(adata):
    """Distance from every cell to its closest sender-type cell.

    Matches how the generative process decided which receivers could interact, which used all
    sender cells rather than only the nearest neighbors AMICI sees.
    """
    coords = np.asarray(adata.obsm["spatial"])
    sender_coords = coords[(adata.obs[LABELS_KEY].astype(str) == SENDER_TYPE).to_numpy()]
    distances, _ = cKDTree(sender_coords).query(coords, k=1)
    return distances


def build_receiver_task_table(model, adata):
    """Score, call, and annotate every receiver cell for the receiver-subtype task."""
    scores = get_amici_receiver_subtype_scores(model, adata)
    truth = pd.DataFrame(
        {
            "cell_idx": adata.obs_names,
            "label": (adata.obs[SUBTYPE_KEY].astype(str) == INTERACTION_SUBTYPE).astype(int).to_numpy(),
            "cell_type": adata.obs[LABELS_KEY].astype(str).to_numpy(),
            "nearest_sender_distance": nearest_sender_distance(adata),
            "x": np.asarray(adata.obsm["spatial"])[:, 0],
            "y": np.asarray(adata.obsm["spatial"])[:, 1],
        }
    )
    table = truth.merge(scores[["cell_idx", "amici_scores"]], on="cell_idx", how="inner")
    table = table[table["cell_type"] == RECEIVER_TYPE].reset_index(drop=True)

    threshold = best_f1_threshold(table["label"].to_numpy(), table["amici_scores"].to_numpy())
    table["call"] = (table["amici_scores"] >= threshold).astype(int)
    table["outcome"] = label_outcomes(table["label"].to_numpy(), table["call"].to_numpy())
    table["outcome_class"] = collapse_outcomes(table["label"].to_numpy(), table["call"].to_numpy())
    return table, threshold


def build_neighbor_task_table(model, adata):
    """Score, call, and annotate every receiver-neighbor pair for the neighbor task."""
    scores = get_amici_neighbor_interaction_scores(model, adata)
    truth = get_interaction_gt_neighbor_classes(adata, GT_INTERACTIONS, LABELS_KEY)

    nn_idx = np.asarray(adata.obsm["_nn_idx"])
    nn_dist = np.asarray(adata.obsm["_nn_dist"])
    pair_distances = pd.DataFrame(
        {
            "cell_idx": np.repeat(adata.obs_names.to_numpy(), nn_idx.shape[1]),
            "neighbor_idx": adata.obs_names.to_numpy()[nn_idx].ravel(),
            "distance": nn_dist.ravel(),
        }
    )

    table = truth.merge(scores[["cell_idx", "neighbor_idx", "amici_scores"]], on=["cell_idx", "neighbor_idx"])
    table = table.merge(pair_distances, on=["cell_idx", "neighbor_idx"], how="left")
    receiver_cells = adata.obs_names[(adata.obs[LABELS_KEY].astype(str) == RECEIVER_TYPE).to_numpy()]
    table = table[table["cell_idx"].isin(set(receiver_cells))].reset_index(drop=True)

    threshold = best_f1_threshold(table["class"].to_numpy(), table["amici_scores"].to_numpy())
    table["call"] = (table["amici_scores"] >= threshold).astype(int)
    table["outcome"] = label_outcomes(table["class"].to_numpy(), table["call"].to_numpy())
    table["outcome_class"] = collapse_outcomes(table["class"].to_numpy(), table["call"].to_numpy())
    return table, threshold


def summarize_by_distance(table, distance_col, label_col):
    """Accuracy and recall per distance bin."""
    bins = np.arange(0, MAX_PLOT_DISTANCE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH)
    binned = table.assign(distance_bin=pd.cut(table[distance_col], bins=bins, right=False))
    summary = binned.groupby("distance_bin", observed=True).apply(
        lambda group: pd.Series(
            {
                "n": len(group),
                "n_positive": int(group[label_col].sum()),
                "accuracy": float((group[label_col] == group["call"]).mean()),
                "recall": float(group.loc[group[label_col] == 1, "call"].mean())
                if (group[label_col] == 1).any()
                else np.nan,
            }
        ),
        include_groups=False,
    )
    return summary.reset_index()


# %% Plotting
def plot_task_errors(receiver_table, neighbor_table, receiver_summary, neighbor_summary):
    """Plot spatial error maps and distance profiles for both benchmark tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0][0]
    for outcome_class, color in OUTCOME_CLASS_COLORS.items():
        subset = receiver_table[receiver_table["outcome_class"] == outcome_class]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=4,
            c=color,
            label=f"{outcome_class} (n={len(subset)})",
            alpha=0.8 if outcome_class == "incorrect" else 0.3,
            linewidths=0,
        )
    ax.set_title("Receiver subtype task: correct vs incorrect calls")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7, markerscale=2.5)

    ax = axes[0][1]
    per_cell = (
        neighbor_table[neighbor_table["class"] == 1]
        .groupby("cell_idx")
        .agg(recall=("call", "mean"), n_positive_pairs=("call", "size"))
        .reset_index()
        .merge(receiver_table[["cell_idx", "x", "y"]], on="cell_idx", how="inner")
    )
    missing = receiver_table[~receiver_table["cell_idx"].isin(set(per_cell["cell_idx"]))]
    ax.scatter(
        missing["x"], missing["y"], s=4, c="#EEEEEE", label=f"no positive pairs (n={len(missing)})", linewidths=0
    )
    scatter = ax.scatter(per_cell["x"], per_cell["y"], s=5, c=per_cell["recall"], cmap="RdYlBu", vmin=0, vmax=1)
    fig.colorbar(scatter, ax=ax, label="fraction of true neighbor pairs recovered")
    ax.set_title("Neighbor interaction task: per-receiver recall")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7, markerscale=2.5)

    ax = axes[1][0]
    bins = np.arange(0, MAX_PLOT_DISTANCE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH)
    for outcome_class, color in OUTCOME_CLASS_COLORS.items():
        subset = receiver_table[receiver_table["outcome_class"] == outcome_class]
        ax.hist(
            subset["nearest_sender_distance"].clip(upper=MAX_PLOT_DISTANCE),
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{outcome_class} (n={len(subset)})",
        )
    for position, style_label in [
        (INTERACTION_START, "start"),
        (INTERACTION_PEAK, "peak"),
        (INTERACTION_END, "end"),
    ]:
        ax.axvline(position, color="black", linestyle="--", linewidth=1)
        ax.text(position, ax.get_ylim()[1] * 0.95, style_label, rotation=90, fontsize=7, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from receiver to nearest sender")
    ax.set_ylabel("Receiver cells (log)")
    ax.set_title("Receiver subtype task: outcome by distance to nearest sender")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    for outcome_class, color in OUTCOME_CLASS_COLORS.items():
        subset = neighbor_table[neighbor_table["outcome_class"] == outcome_class]
        ax.hist(
            subset["distance"].clip(upper=MAX_PLOT_DISTANCE),
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{outcome_class} (n={len(subset)})",
        )
    for position in (INTERACTION_START, INTERACTION_PEAK, INTERACTION_END):
        ax.axvline(position, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Receiver-neighbor pair distance")
    ax.set_ylabel("Pairs (log)")
    ax.set_title("Neighbor interaction task: outcome by pair distance")
    ax.legend(fontsize=7)

    fig.suptitle(
        "AMICI errors on the unimodal dataset (true effect: start 10, peak 20, end 30)",
        fontsize=14,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"unimodal_two_phase_task_error_analysis.{ext}"), dpi=300)
    plt.close(fig)


# %% Load model and run the analysis
best = json.load(open(BEST_PARAMS_PATH))
adata = sc.read_h5ad(DATA_PATH)
adata.obs_names_make_unique()
AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=int(best["n_neighbors"]))
model = AMICI.load(MODEL_PATH, adata=adata)

receiver_table, receiver_threshold = build_receiver_task_table(model, adata)
neighbor_table, neighbor_threshold = build_neighbor_task_table(model, adata)

receiver_summary = summarize_by_distance(receiver_table, "nearest_sender_distance", "label")
neighbor_summary = summarize_by_distance(neighbor_table, "distance", "class")

receiver_table.to_csv(os.path.join(RESULTS_DIR, "unimodal_receiver_task_outcomes.csv"), index=False)
neighbor_table.to_csv(os.path.join(RESULTS_DIR, "unimodal_neighbor_task_outcomes.csv"), index=False)
receiver_summary.to_csv(os.path.join(RESULTS_DIR, "unimodal_receiver_task_by_distance.csv"), index=False)
neighbor_summary.to_csv(os.path.join(RESULTS_DIR, "unimodal_neighbor_task_by_distance.csv"), index=False)

print(f"Receiver task: threshold={receiver_threshold:.4f}, n_cells={len(receiver_table)}")
print(receiver_table["outcome"].value_counts().to_string())
print(f"\nNeighbor task: threshold={neighbor_threshold:.4f}, n_pairs={len(neighbor_table)}")
print(neighbor_table["outcome"].value_counts().to_string())

print("\n=== Receiver task by distance to nearest sender ===")
print(receiver_summary.to_string(index=False))
print("\n=== Neighbor task by pair distance ===")
print(neighbor_summary.to_string(index=False))

plot_task_errors(receiver_table, neighbor_table, receiver_summary, neighbor_summary)
print(f"\nFigure written to {FIGURE_DIR}")
