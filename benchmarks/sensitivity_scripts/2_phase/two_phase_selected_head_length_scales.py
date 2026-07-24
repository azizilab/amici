# %% Imports
import json
import os
import sys

import matplotlib.pyplot as plt
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
ATTENTION_THRESHOLD = 0.1

INTERACTION_HEADS = {
    "10-unit phase": {"head_idx": 3, "gt_length_scale": 10, "interaction_subtype": "0_sub1"},
    "25-unit phase": {"head_idx": 2, "gt_length_scale": 25, "interaction_subtype": "0_sub2"},
}

DATA_PATH = "data/3ct_dataset_1int_2phase.h5ad"
BEST_PARAMS_PATH = "results/two_phase_amici_sweep/best_model_params.json"
MODEL_PATH = "saved_models/two_phase_amici_sweep/best_model"
FIGURE_DIR = "figures/two_phase_amici_sweep"
RESULTS_DIR = "results/two_phase_amici_sweep"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# %% Plotting
def plot_selected_head_length_scales(length_df):
    """Plot selected-head length-scale distributions for two-phase interactions."""
    order = ["10-unit phase", "25-unit phase"]
    label_map = {
        interaction: (
            f"{SENDER_TYPE} -> {RECEIVER_TYPE} "
            f"({INTERACTION_HEADS[interaction]['interaction_subtype']}, "
            f"head {INTERACTION_HEADS[interaction]['head_idx']})"
        )
        for interaction in order
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.violinplot(
        data=length_df,
        y="interaction",
        x="length_scale",
        order=order,
        color="#9ecae1",
        inner="quartile",
        cut=0,
        ax=ax,
    )
    for y_pos, interaction in enumerate(order):
        gt = INTERACTION_HEADS[interaction]["gt_length_scale"]
        ax.scatter(gt, y_pos, marker="x", color="black", s=150, linewidths=2.8, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_map[interaction] for interaction in order])
    ax.set_xlabel("Length scale at attention threshold 0.1")
    ax.set_ylabel("")
    ax.set_title("Two-phase selected-head length-scale distributions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGURE_DIR, f"two_phase_selected_head_length_scale_distributions.{ext}"),
            dpi=300,
            bbox_inches="tight",
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

records = []
for interaction, config in INTERACTION_HEADS.items():
    print(f"Computing {interaction} with head {config['head_idx']}", flush=True)
    length_df = counterfactual._calculate_length_scales(
        head_idxs=[config["head_idx"]],
        sender_types=[SENDER_TYPE],
        attention_threshold=ATTENTION_THRESHOLD,
    )
    length_df["interaction"] = interaction
    length_df["receiver_type"] = RECEIVER_TYPE
    length_df["interaction_subtype"] = config["interaction_subtype"]
    length_df["gt_length_scale"] = config["gt_length_scale"]
    length_df["attention_threshold"] = ATTENTION_THRESHOLD
    records.append(length_df)

selected_length_df = pd.concat(records, ignore_index=True)
selected_length_df.to_csv(
    os.path.join(RESULTS_DIR, "two_phase_selected_head_length_scale_samples.csv"),
    index=False,
)
plot_selected_head_length_scales(selected_length_df)
print(f"Saved selected-head length-scale plot to {FIGURE_DIR}", flush=True)
