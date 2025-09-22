import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from amici import AMICI


def main():
    """Plot the KDE plot for the length scale task."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    model_path = os.path.dirname(snakemake.input.model_path)  # noqa: F821

    model = AMICI.load(
        model_path,
        adata=adata,
    )

    AMICI.setup_anndata(
        adata,
        labels_key=labels_key,
        coord_obsm_key="spatial",
        n_neighbors=50,
    )

    length_scale_dfs = []
    for interaction_name in dataset_config["gt_interactions"]:
        interaction_config = dataset_config["gt_interactions"][interaction_name]
        receiver_type = interaction_config["receiver"]
        sender_type = interaction_config["sender"]

        counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
            adata=adata,
            cell_type=receiver_type,
        )

        length_scale_df = counterfactual_attention_patterns._calculate_length_scales(
            head_idxs=np.arange(model.module.n_heads),
            sender_types=[sender_type],
            attention_threshold=0.1,
        )

        explained_variance = model.get_expl_variance_scores(
            adata=adata,
        )
        max_expl_variance_head = explained_variance.compute_max_explained_variance_head(cell_type=receiver_type)
        length_scale_df = length_scale_df[length_scale_df["head_idx"] == max_expl_variance_head]
        length_scale_df["interaction_name"] = f"{sender_type} -> {receiver_type}"
        length_scale_dfs.append(length_scale_df)

    length_scale_df = pd.concat(length_scale_dfs)

    sns.kdeplot(
        data=length_scale_df,
        x="length_scale",
        hue="interaction_name",
    )
    plt.title("Length scale distributions for all interactions")
    plt.xlabel("Length scale")
    plt.ylabel("Density")
    plt.savefig(snakemake.output[0])  # noqa: F821
    plt.savefig(snakemake.output[0].replace("png", "svg"))  # noqa: F821
    plt.close()

    with open(snakemake.output[1], "w") as f:  # noqa: F821
        f.write("Done")


if __name__ == "__main__":
    main()
