# Notebook exploring the sensitivity of the length scales on semi-synthetic data with respect to the attention threshold.
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from amici import AMICI

# %%
# Paths to hardcoded data and model (copied from semisyn)
base_dir = os.path.dirname(os.path.abspath(__file__))
adata_path = os.path.join(base_dir, "data", "3ct_dataset_6.h5ad")
model_path = os.path.join(base_dir, "saved_models", "amici_3ct_dataset_2way_6_best_model")
interaction_config_path = os.path.join(base_dir, "data", "3ct_sweep_interaction_csv.csv")

adata = sc.read_h5ad(adata_path)
labels_key = "leiden"
model = AMICI.load(model_path, adata=adata)

# %%
interaction_df = pd.read_csv(interaction_config_path)
interaction_df

# %%
# Ground truth length scales from benchmark config
gt_length_scales = {
    (str(int(interaction_df.loc[idx, "sender_cell"])), str(int(interaction_df.loc[idx, "receptor_cell"]))): interaction_df.loc[
        idx, "radius_of_effect"
    ] for idx in interaction_df.index if interaction_df.loc[idx, "interaction_type"] == "interaction"
}

thresholds = np.arange(0.1, 1.0, 0.05).round(2).tolist()

# %%
# For each interaction, compute length scales across all thresholds
for idx, row in interaction_df.iterrows():
    if row["interaction_type"] != "interaction":
        continue

    receiver_type = str(row["receptor_cell"])
    sender_type = str(int(row["sender_cell"]))
    gt_ls = gt_length_scales.get((sender_type, receiver_type))

    print(f"Computing for {sender_type} -> {receiver_type}")

    counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
        adata=adata,
        cell_type=receiver_type,
    )

    # Find the best head via explained variance
    explained_variance = model.get_expl_variance_scores(adata=adata)
    max_expl_variance_head = explained_variance.compute_max_explained_variance_head(
        cell_type=receiver_type,
    )
    print(f"  Best head (max explained variance): {max_expl_variance_head}")

    # Collect length scales for each threshold (only compute for the best head)
    all_length_scale_records = []
    for threshold in thresholds:
        length_scale_df = counterfactual_attention_patterns._calculate_length_scales(
            head_idxs=[max_expl_variance_head],
            sender_types=[sender_type],
            attention_threshold=threshold,
        )
        length_scale_df = length_scale_df.copy()
        length_scale_df["attention_threshold"] = threshold
        all_length_scale_records.append(length_scale_df)

    combined_df = pd.concat(all_length_scale_records, ignore_index=True)

    # Plot box-and-whisker for this interaction
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=combined_df,
        x="attention_threshold",
        y="length_scale",
        ax=ax,
        fliersize=2,
        color="steelblue",
    )

    if gt_ls is not None:
        ax.axhline(y=gt_ls, color="red", linestyle="--", linewidth=1.5, label=f"GT length scale = {gt_ls}")
        ax.legend()

    ax.set_title(f"Length scale sensitivity to attention threshold\n({sender_type} -> {receiver_type}, head {max_expl_variance_head})")
    ax.set_xlabel("Attention Threshold")
    ax.set_ylabel("Length Scale")
    plt.tight_layout()
    plt.show()

# %%
