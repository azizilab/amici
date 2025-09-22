import scanpy as sc
from benchmark_utils import get_interaction_gt_neighbor_classes

from amici import AMICI


def main():
    """Generate the ground truth neighbor interaction scores."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    interactions_config = dataset_config["gt_interactions"]

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=50,
    )

    gt_neighbor_classes_df = get_interaction_gt_neighbor_classes(
        adata,
        interactions_config,
        dataset_config["labels_key"],
    )

    gt_neighbor_classes_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
