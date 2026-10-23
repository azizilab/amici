"""Seed-sweep uncertainty for interaction length scales on the full Xenium dataset.

Trains ~20 AMICI models at the exact fixed hyperparameters used for the main Xenium figure
(reproducibility/xenium_full_config.yaml, model seed 18), varying only the random seed, and
reports the distribution of median length scales across seeds for the sender -> receiver
(head) pairs shown in the length-scale-distribution figure. There is no ground truth for
real data, so seed-to-seed variability of the median length scale is the uncertainty
quantity of interest here, rather than a bootstrap CI computed within one trained model.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../benchmarks"))

from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATA_DATE = "2025-05-01"
LABELS_KEY = "celltype_train_grouped"
COORD_KEY = "spatial"
DATA_PATH = os.path.join(SCRIPT_DIR, f"data/xenium_sample1/xenium_sample1_filtered_{DATA_DATE}.h5ad")

RUN_NAME = "xenium_length_scale_seed_sweep"
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models", RUN_NAME)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", RUN_NAME)
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures", RUN_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# 20 seeds, distinct from the seeds used in the original hyperparameter sweep (4, 7, 13, 18,
# 21, 22, 33, 38, 40, 42).
SEEDS = list(range(101, 121))

# Fixed hyperparameters, matching reproducibility/xenium_full_config.yaml exactly (the
# config used to train xenium_18_sweep_g3mucw4s_te7pkv3z, the model behind the main figure).
N_NEIGHBORS = 50
MODEL_PARAMS = {
    "n_heads": 8,
    "n_query_dim": 128,
    "n_head_size": 32,
    "n_nn_embed": 256,
    "n_nn_embed_hidden": 512,
    "attention_dummy_score": 3.0,
    "neighbor_dropout": 0.1,
    "attention_penalty_coef": 1e-6,
    "value_l1_penalty_coef": 0.001,
}
EXP_PARAMS = {
    "lr": 0.001,
    "epochs": 400,
    "batch_size": 512,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 10,
}
PENALTY_SCHEDULE = {"epoch_start": 5, "epoch_end": 15, "start_val": 1e-6, "end_val": 0.001, "flavor": "linear"}

ATTENTION_THRESHOLD = 0.1
# (sender, receiver, head_idx, sample_threshold): the exact rows shown in the reference
# Interaction Length Scale Distributions figure. sample_threshold is keyed by receiver, as in
# xenium_analysis.py's `interactions` dict.
INTERACTION_TUPLES = [
    ("CD4+_T_Cells", "CD8+_T_Cells", 5, 0.02),
    ("CD4+_T_Cells", "CD8+_T_Cells", 4, 0.02),
    ("CD4+_T_Cells", "B_Cells", 4, 0.02),
    ("CD8+_T_Cells", "Macrophages_1", 4, 0.02),
    ("B_Cells", "CD4+_T_Cells", 5, 0.02),
    ("CD8+_T_Cells", "Invasive_Tumor", 5, 0.001),
    ("CD8+_T_Cells", "Macrophages_1", 5, 0.02),
    ("Macrophages_2", "CD4+_T_Cells", 5, 0.02),
    ("CD4+_T_Cells", "B_Cells", 5, 0.02),
    ("Macrophages_2", "CD8+_T_Cells", 5, 0.02),
    ("DCIS_1", "Macrophages_1", 5, 0.02),
    ("Invasive_Tumor", "Macrophages_1", 5, 0.02),
    ("Macrophages_2", "Macrophages_1", 5, 0.02),
]


def row_label(sender, receiver, head):
    """Format one interaction's row label for the plot."""
    return f"{sender} → {receiver} (head {head})"


ROW_ORDER = [row_label(s, r, h) for s, r, h, _ in INTERACTION_TUPLES]


def train_or_load_seed_model(adata_train, adata, seed):
    """Train one seed's model at the fixed hyperparameters, or load it if already cached.

    Returns a model whose registered anndata is `adata` (the full dataset), ready for
    get_counterfactual_attention_patterns.
    """
    run_dir = os.path.join(MODEL_DIR, f"seed_{seed}")
    if os.path.exists(os.path.join(run_dir, "model.pt")):
        model = AMICI.load(run_dir, adata=adata)
    else:
        pl.seed_everything(seed)
        AMICI.setup_anndata(adata_train, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
        model = AMICI(adata_train, **MODEL_PARAMS)
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
                    PENALTY_SCHEDULE["epoch_start"],
                    PENALTY_SCHEDULE["epoch_end"],
                    PENALTY_SCHEDULE["start_val"],
                    PENALTY_SCHEDULE["end_val"],
                    PENALTY_SCHEDULE["flavor"],
                ),
            ],
        )
        model.save(run_dir, overwrite=True)

    AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    return model


def compute_seed_medians(model, adata, seed):
    """Median length scale per (sender, receiver, head) tuple for one trained model."""
    records = []
    receivers = sorted({receiver for _, receiver, _, _ in INTERACTION_TUPLES})
    for receiver in receivers:
        counterfactual = model.get_counterfactual_attention_patterns(cell_type=receiver, adata=adata)
        rows = [t for t in INTERACTION_TUPLES if t[1] == receiver]
        for sender, _, head, sample_threshold in rows:
            length_scale_df = counterfactual._calculate_length_scales(
                head_idxs=[head],
                sender_types=[sender],
                attention_threshold=ATTENTION_THRESHOLD,
                sample_threshold=sample_threshold,
            )
            records.append(
                {
                    "seed": seed,
                    "sender": sender,
                    "receiver": receiver,
                    "head_idx": head,
                    "row_label": row_label(sender, receiver, head),
                    "median_length_scale": length_scale_df["length_scale"].median(),
                    "n_samples": len(length_scale_df),
                }
            )
    return pd.DataFrame(records)


def plot_seed_violin(all_medians):
    """Violin per row: distribution of that row's median length scale across seeds."""
    order = ROW_ORDER[::-1]
    positions = np.arange(len(order))
    data = [
        all_medians.loc[all_medians["row_label"] == label, "median_length_scale"].dropna().to_numpy() for label in order
    ]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.5))
    ax.violinplot(data, positions=positions, vert=False, showmedians=True, widths=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("Median length scale across seeds (µm)")
    ax.set_title(f"Xenium: seed-to-seed variability of median length scale (n={len(SEEDS)} seeds)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGURE_DIR, f"xenium_length_scale_seed_variability.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def main():
    """Train one model per seed at the fixed hyperparameters and collect the median length scales across seeds."""
    select_gpu()
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()

    all_medians = []
    for seed in SEEDS:
        seed_csv = os.path.join(RESULTS_DIR, f"seed_{seed}_medians.csv")
        if os.path.exists(seed_csv):
            print(f"[skip] seed {seed} already computed", flush=True)
            all_medians.append(pd.read_csv(seed_csv))
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        model = train_or_load_seed_model(adata_train, adata, seed)
        seed_df = compute_seed_medians(model, adata, seed)
        seed_df.to_csv(seed_csv, index=False)
        all_medians.append(seed_df)
        print(seed_df.to_string(index=False), flush=True)

    all_medians_df = pd.concat(all_medians, ignore_index=True)
    all_medians_df.to_csv(os.path.join(RESULTS_DIR, "xenium_length_scale_seed_medians.csv"), index=False)
    plot_seed_violin(all_medians_df)

    print("\n=== Summary across seeds ===", flush=True)
    print(all_medians_df.groupby("row_label")["median_length_scale"].describe().to_string(), flush=True)
    print(f"\nWrote results to {RESULTS_DIR}\nWrote figure to {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    main()
