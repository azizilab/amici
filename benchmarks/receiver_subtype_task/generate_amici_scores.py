import os

import pytorch_lightning as pl
import scanpy as sc
from amici_benchmark_utils import get_amici_receiver_subtype_scores

from amici import AMICI


def main():
    """Generate the AMICI scores for the receiver subtype task."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821

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

    receiver_subtype_attention_scores_df = get_amici_receiver_subtype_scores(
        model,
        adata,
    )

    receiver_subtype_attention_scores_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
