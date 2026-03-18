# %% Import libraries
import os
import random
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
from generate_dataset import _create_interaction_df, generate_synthetic_dataset  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
DATASET_SEEDS = [40, 123, 6, 23, 25, 88, 72, 58, 22, 31]
TRAIN_SEED = 42
END_PENALTY_VALUES = [1e-2, 1e-3, 1e-4, 1e-5]
N_NEIGHBORS = 50
N_HEADS = 8
LABELS_KEY = "leiden"
SUBTYPE_KEY = "subtype"
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
PENALTY_PARAMS = {
    "start_val": 1e-6,
    "epoch_start": 10,
    "epoch_end": 40,
    "value_l1_penalty_coef": 1e-4,
}
EXP_PARAMS = {
    "lr": 1e-3,
    "epochs": 400,
    "batch_size": 128,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
}

# %% Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "saved_models"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)

interaction_df = _create_interaction_df(GT_INTERACTIONS)

# %% Train and evaluate models across dataset seeds and end_penalty values
gene_pr_results = {p: [] for p in END_PENALTY_VALUES}
neighbor_pr_results = {p: [] for p in END_PENALTY_VALUES}
receiver_pr_results = {p: [] for p in END_PENALTY_VALUES}

gene_pr_curves = {}
neighbor_pr_curves = {}
receiver_pr_curves = {}

for dataset_seed_idx, dataset_seed in enumerate(DATASET_SEEDS):
    adata_path = os.path.join(base_dir, "data", f"semisyn_{dataset_seed}.h5ad")

    random.seed(dataset_seed)
    np.random.seed(dataset_seed)
    torch.random.manual_seed(dataset_seed)
    scvi.settings.seed = dataset_seed

    generate_synthetic_dataset(interaction_df, adata_path)
    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()

    all_gt_gene_scores = []
    for interaction_name, interaction_config in GT_INTERACTIONS.items():
        gt_gene_scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            SUBTYPE_KEY,
        )
        gt_gene_scores["interaction"] = interaction_name
        all_gt_gene_scores.append(gt_gene_scores)
    gt_gene_scores_df = pd.concat(all_gt_gene_scores, ignore_index=True)

    combined_interaction_subtype_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in GT_INTERACTIONS.values():
        combined_interaction_subtype_mask |= adata.obs[SUBTYPE_KEY] == interaction_config["interaction_subtype"]
    gt_receiver_classes_df = pd.DataFrame(
        {
            "cell_idx": adata.obs_names,
            "class": combined_interaction_subtype_mask.astype(float),
        }
    )

    AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=N_NEIGHBORS)

    for end_penalty in END_PENALTY_VALUES:
        end_penalty_str = f"{end_penalty:.0e}"
        model_path = os.path.join(
            base_dir, "saved_models", f"amici_attention_penalty_analysis_{dataset_seed}_{end_penalty_str}"
        )

        if not os.path.exists(os.path.join(model_path, "model.pt")):
            pl.seed_everything(TRAIN_SEED)
            adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
            AMICI.setup_anndata(
                adata_train,
                labels_key=LABELS_KEY,
                coord_obsm_key="spatial",
                n_neighbors=N_NEIGHBORS,
            )
            model = AMICI(
                adata_train,
                n_heads=N_HEADS,
                value_l1_penalty_coef=PENALTY_PARAMS["value_l1_penalty_coef"],
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
                        start_val=PENALTY_PARAMS["start_val"],
                        end_val=end_penalty,
                        epoch_start=PENALTY_PARAMS["epoch_start"],
                        epoch_end=PENALTY_PARAMS["epoch_end"],
                    ),
                ],
            )
            model.save(model_path, overwrite=True)

        model = AMICI.load(model_path, adata=adata)

        # Gene task
        all_gene_scores = []
        for interaction_name, interaction_config in GT_INTERACTIONS.items():
            gene_scores = get_amici_gene_task_scores(
                model, adata, interaction_config["sender"], interaction_config["receiver"]
            )
            gene_scores["interaction"] = interaction_name
            all_gene_scores.append(gene_scores)
        gene_scores_df = pd.concat(all_gene_scores, ignore_index=True)
        precision, recall, auprc = get_model_precision_recall_auc(
            gene_scores_df,
            gt_gene_scores_df,
            merge_cols=["gene", "interaction"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        gene_pr_results[end_penalty].append(auprc)
        if dataset_seed_idx == 0:
            gene_pr_curves[end_penalty] = {"precision": precision, "recall": recall, "auprc": auprc}

        # Neighbor interaction task
        gt_neighbor_classes_df = get_interaction_gt_neighbor_classes(adata, GT_INTERACTIONS, LABELS_KEY)
        neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata)
        precision, recall, auprc = get_model_precision_recall_auc(
            neighbor_scores_df,
            gt_neighbor_classes_df,
            merge_cols=["cell_idx", "neighbor_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        neighbor_pr_results[end_penalty].append(auprc)
        if dataset_seed_idx == 0:
            neighbor_pr_curves[end_penalty] = {"precision": precision, "recall": recall, "auprc": auprc}

        # Receiver subtype task
        receiver_scores_df = get_amici_receiver_subtype_scores(model, adata)
        precision, recall, auprc = get_model_precision_recall_auc(
            receiver_scores_df,
            gt_receiver_classes_df,
            merge_cols=["cell_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        receiver_pr_results[end_penalty].append(auprc)
        if dataset_seed_idx == 0:
            receiver_pr_curves[end_penalty] = {"precision": precision, "recall": recall, "auprc": auprc}

# %% Plot PR curves (first dataset seed)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(END_PENALTY_VALUES)))
task_curves = [
    (gene_pr_curves, "Gene Task"),
    (neighbor_pr_curves, "Neighbor Interaction Task"),
    (receiver_pr_curves, "Receiver Subtype Task"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (curves, title) in zip(axes, task_curves, strict=False):
    for (end_penalty, pr), color in zip(curves.items(), colors, strict=False):
        PrecisionRecallDisplay(recall=pr["recall"], precision=pr["precision"]).plot(
            ax=ax, name=f"p={end_penalty:.0e} (AUPRC={pr['auprc']:.2f})", color=color
        )
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("AMICI Sensitivity to Attention End Penalty", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "figures", "attention_penalty_analysis_pr_curves.png"), dpi=300, bbox_inches="tight")
plt.show()

# %% Plot AUPRC boxplots across dataset seeds
task_results = [
    (gene_pr_results, "Gene Task"),
    (neighbor_pr_results, "Neighbor Interaction Task"),
    (receiver_pr_results, "Receiver Subtype Task"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (results, title) in zip(axes, task_results, strict=False):
    data = [results[p] for p in END_PENALTY_VALUES]
    bp = ax.boxplot(data, labels=[f"{p:.0e}" for p in END_PENALTY_VALUES], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("End Attention Penalty")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("AUPRC Robustness to Attention End Penalty\n(across 10 dataset seeds)", fontsize=13)
plt.tight_layout()
plt.savefig(
    os.path.join(base_dir, "figures", "attention_penalty_analysis_auprc_boxplots.png"), dpi=300, bbox_inches="tight"
)
plt.show()
