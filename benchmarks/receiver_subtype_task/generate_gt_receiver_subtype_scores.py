import numpy as np
import pandas as pd
import scanpy as sc


def main():
    """Generate the ground truth receiver subtype scores."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    subtype_labels_key = dataset_config["subtype_labels_key"]
    interactions_config = dataset_config["gt_interactions"]

    combined_interaction_subtype_mask = np.zeros(len(adata.obs_names))
    for interaction_name in interactions_config:
        interaction_config = interactions_config[interaction_name]
        interaction_subtype_mask = np.array(adata.obs[subtype_labels_key]) == interaction_config["interaction_subtype"]
        assert combined_interaction_subtype_mask.shape == interaction_subtype_mask.shape
        combined_interaction_subtype_mask = np.logical_or(combined_interaction_subtype_mask, interaction_subtype_mask)

    gt_receiver_classes_df = pd.DataFrame({"cell_idx": adata.obs_names, "class": np.zeros(len(adata.obs_names))})
    gt_receiver_classes_df.loc[combined_interaction_subtype_mask, "class"] = 1.0

    gt_receiver_classes_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
