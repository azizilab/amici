import pandas as pd
import scanpy as sc
from benchmark_utils import get_receiver_gt_ranked_genes


def main():
    """Generate the ground truth ranked genes for the gene task."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821
    adata.obs_names_make_unique()

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    subtype_labels_key = dataset_config["subtype_labels_key"]
    interactions_config = dataset_config["gt_interactions"]

    # Create a list to store all dataframes
    all_gt_ranked_genes = []

    for interaction_name in interactions_config:
        interaction_config = interactions_config[interaction_name]
        receiver_type = interaction_config["receiver"]
        interaction_subtype = interaction_config["interaction_subtype"]
        neutral_subtype = interaction_config["neutral_subtype"]

        gt_ranked_genes_df = get_receiver_gt_ranked_genes(
            adata,
            receiver_type,
            interaction_subtype,
            neutral_subtype,
            subtype_labels_key,
        )

        # Add a column to identify the interaction
        gt_ranked_genes_df["interaction"] = interaction_name

        # Append to our list
        all_gt_ranked_genes.append(gt_ranked_genes_df)

    # Combine all dataframes and save to a combined output file
    if all_gt_ranked_genes:
        all_gt_ranked_genes_df = pd.concat(all_gt_ranked_genes, ignore_index=True)
        all_gt_ranked_genes_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
