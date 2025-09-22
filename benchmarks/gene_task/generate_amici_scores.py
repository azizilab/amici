import os

import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from amici_benchmark_utils import get_amici_gene_task_scores

from amici import AMICI


def main():
    """Generate the AMICI scores for the gene task."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821
    adata.obs_names_make_unique()

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821

    model_path = os.path.dirname(snakemake.input.model_path)  # noqa: F821

    with open(snakemake.input.best_seed_path) as f:  # noqa: F821
        best_seed = int(f.read())

    pl.seed_everything(best_seed)

    model = AMICI.load(
        model_path,
        adata=adata,
    )
    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=50,
    )

    interactions_config = dataset_config["gt_interactions"]
    all_amici_gene_scores_df = []
    for interaction_name in interactions_config:
        interaction_config = interactions_config[interaction_name]
        receiver_type = interaction_config["receiver"]
        sender_type = interaction_config["sender"]

        amici_gene_scores_df = get_amici_gene_task_scores(
            model,
            adata,
            sender_type,
            receiver_type,
        )

        amici_gene_scores_df["interaction"] = interaction_name

        all_amici_gene_scores_df.append(amici_gene_scores_df)

    # Combine all dataframes and save to a combined output file
    if all_amici_gene_scores_df:
        all_amici_gene_scores_df = pd.concat(all_amici_gene_scores_df, ignore_index=True)
        all_amici_gene_scores_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
