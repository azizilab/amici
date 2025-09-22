import os
import random

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

from amici.tools import is_count_data


def generate_synthetic_dataset(interaction_df, output_path):
    """
    Generate a synthetic dataset.

    Args:
        interaction_df: The interaction DataFrame.
        output_path: The path to save the dataset.

    """
    if not os.path.exists(output_path):
        # Load or download data
        if os.path.exists("/tmp/pbmc_ad.h5ad"):
            pbmc_ad = ad.read_h5ad("/tmp/pbmc_ad.h5ad")
        else:
            pbmc_ad = scvi.data.dataset_10x(dataset_name="fresh_68k_pbmc_donor_a")
            pbmc_ad.write_h5ad("/tmp/pbmc_ad.h5ad")

        positive_deg_check = False
        while not positive_deg_check:
            adata_pp = _preprocess(pbmc_ad, n_hvgs=500)
            adata_pp = _subcluster(adata_pp, n_subcluster_per_cluster=3, n_genes_for_subclustering=50)

            # Check if there are enough positive DEGs between the subclusters
            adata_pp, positive_deg_check = _check_positive_degs(adata_pp, interaction_df)
            if not positive_deg_check:
                scvi.settings.seed = np.random.randint(0, 1000)

        spatial_ad = _generate_2d_triangular_gradient_data(
            adata_pp,
            "3",
            "0",
            "2",
            interaction_df,
            num_cells=20000,
            rect_length=2000,
            rect_width=1000,
        )
        spatial_ad = _create_train_test_split(spatial_ad)
        spatial_ad.obs_names_make_unique()
        spatial_ad.write_h5ad(output_path)


def _check_positive_degs(
    adata,
    interaction_df,
):
    """
    Check if there are enough positive DEGs between the subclusters. If not, try swapping the neutral and interacting subtypes.

    Args:
        adata: The adata object.
        interaction_df: The interaction DataFrame.

    Returns
    -------
        adata: The possibly modified adata object
        positive_deg_check: Boolean indicating if there are enough positive DEGs
    """
    positive_deg_check = True

    adata_original = adata.copy()

    for receptor_cell in interaction_df["receptor_cell"].unique():
        # Get the neutral subtype for this receptor cell
        neutral_subtypes = interaction_df[
            (interaction_df["receptor_cell"] == receptor_cell) & (interaction_df["interaction_type"] == "neutral")
        ]["receptor_subtype"].tolist()

        # Get the interacting subtype for this receptor cell if it exists
        interacting_subtypes = interaction_df[
            (interaction_df["receptor_cell"] == receptor_cell) & (interaction_df["interaction_type"] == "interaction")
        ]["receptor_subtype"].tolist()

        # Skip if no interacting subtype for this receptor cell
        if not interacting_subtypes:
            continue

        # There should be exactly one neutral subtype per receptor cell
        neutral_subtype = neutral_subtypes[0]
        interacting_subtype = interacting_subtypes[0]

        # Create a temporary subset of adata for the two subtypes
        subtype_mask = adata.obs["subtype"].isin([neutral_subtype, interacting_subtype])
        adata_subtypes = adata[subtype_mask].copy()

        # Get the adjusted p-values and log fold changes from DEG analysis
        sc.tl.rank_genes_groups(adata_subtypes, "subtype", method="wilcoxon")
        pvals_adj = adata_subtypes.uns["rank_genes_groups"]["pvals_adj"][interacting_subtype]
        logfoldchanges = adata_subtypes.uns["rank_genes_groups"]["logfoldchanges"][interacting_subtype]

        # Filter by adjusted p-value and log fold change and store the count
        num_positive_degs = np.sum((pvals_adj < 0.05) & (logfoldchanges > 0.2))

        # If we have less than 10 positive DEGs, try swapping the labels
        if num_positive_degs < 10:
            positive_deg_check = False
            return adata_original, positive_deg_check

    return adata, positive_deg_check


def _create_train_test_split(adata):
    """
    Create a train and test split of the spatial adata object based on the spatial coordinates.

    The test split is created by selecting cells that are within a certain
    distance of the border of the spatial domain.

    Args:
        adata: The spatial adata object.

    Returns
    -------
        The adata object with the train and test split.
    """
    test_indices = np.where((adata.obsm["spatial"].iloc[:, 0] > 900) & (adata.obsm["spatial"].iloc[:, 0] < 1100))[0]
    train_indices = np.setdiff1d(np.arange(adata.shape[0]), test_indices)

    adata.obs["train_test_split"] = "unassigned"
    adata.obs.iloc[train_indices, adata.obs.columns.get_loc("train_test_split")] = "train"
    adata.obs.iloc[test_indices, adata.obs.columns.get_loc("train_test_split")] = "test"
    return adata


def _preprocess(adata, n_hvgs=2000):
    """
    Preprocess the adata object, compute the SCVI latent embeddings and perform subclustering.

    Args:
        adata: The adata object to preprocess.
        n_hvgs: The number of highly variable genes to use for the SCVI model.

    Returns
    -------
        The preprocessed adata object.
    """
    adata_log = adata.copy()
    if is_count_data(adata_log.X):
        if "counts" not in adata_log.layers:
            adata_log.layers["counts"] = adata_log.X.copy()
        sc.pp.normalize_total(adata_log)
        sc.pp.log1p(adata_log)
    sc.pp.highly_variable_genes(adata_log, flavor="seurat_v3", layer="counts", n_top_genes=n_hvgs, subset=True)
    scvi.model.SCVI.setup_anndata(adata_log, layer="counts", batch_key=None)
    scvi_model = scvi.model.SCVI(adata_log)
    scvi_model.train()
    z = scvi_model.get_latent_representation()
    adata_log.obsm["X_scvi"] = z

    sc.pp.neighbors(adata_log, use_rep="X_scvi")
    sc.tl.leiden(adata_log, resolution=0.3, key_added="leiden")

    # Prune small clusters
    cluster_counts = adata_log.obs["leiden"].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts >= 1000].index
    adata_log = adata_log[adata_log.obs["leiden"].isin(clusters_to_keep)]
    return adata_log


def _subcluster(adata, n_subcluster_per_cluster=3, n_genes_for_subclustering=200):
    """
    Subcluster the adata object based on the leiden clusters.

    Args:
        adata: The adata object to subcluster.
        n_subcluster_per_cluster: The number of subclusters to create per cluster.
        n_genes_for_subclustering: The number of genes to use for the subclustering.

    Returns
    -------
        The adata object with the subclusters.
    """
    print("Number of subclusters per cluster:", n_subcluster_per_cluster)

    gene_ratio = 0.5

    adata.obs["subtype"] = "unassigned"
    for ct in adata.obs["leiden"].unique():
        gene_avg_expression = np.ravel(adata.X.mean(axis=0))
        sorted_genes_by_expression = np.argsort(gene_avg_expression)
        n_top_genes = int(gene_ratio * n_genes_for_subclustering)
        n_remaining_genes = n_genes_for_subclustering - n_top_genes
        top_genes_indices, remaining_genes_indices = (
            sorted_genes_by_expression[-n_top_genes:],
            sorted_genes_by_expression[:-n_top_genes],
        )
        selected_genes_for_subclustering = adata.var_names[top_genes_indices]
        selected_genes_for_subclustering = np.concatenate(
            [
                selected_genes_for_subclustering,
                np.random.choice(
                    adata.var_names[remaining_genes_indices],
                    n_remaining_genes,
                    replace=False,
                ),
            ]
        )
        adata.var[f"is_gene_for_subclustering_{ct}"] = False
        adata.var.loc[selected_genes_for_subclustering, f"is_gene_for_subclustering_{ct}"] = True
        adata_sub = adata[:, adata.var[f"is_gene_for_subclustering_{ct}"]].copy()

        sc.pp.pca(adata_sub, n_comps=20)
        adata.obsm[f"X_rep_subclustering_{ct}"] = adata_sub.obsm["X_pca"]
        _leiden_subclustering_binary_search(adata, ct, n_subcluster_per_cluster)
    adata.obs["subtype"] = adata.obs["subtype"].astype("category")
    return adata


def _generate_2d_triangular_gradient_data(
    adata,
    ct1,
    ct2,
    ct3,
    interaction_df,
    num_cells=6000,
    rect_length=2000,
    rect_width=1000,
):
    """
    Generate a 2D triangular gradient data.

    Args:
        adata: The adata object to generate the data for.
        ct1: The first cell type.
        ct2: The second cell type.
        ct3: The third cell type.
        interaction_df: The interaction DataFrame.
        num_cells: The number of cells to generate.
        rect_length: The length of the rectangle.
        rect_width: The width of the rectangle.

    Returns
    -------
        The adata object with the generated data.
    """
    # rect_length is the dimension of the gradient
    gradient_width = 700

    # Generate uniform spatial positions
    positions = np.random.uniform(0, (rect_length, rect_width), size=(num_cells, 2))

    # Calculate midpoints for the lines
    mid_x = rect_length / 2
    mid_y = rect_width / 2

    # Calculate gradient zones around the lines
    gradient_x_start = mid_x - gradient_width / 2
    gradient_x_end = mid_x + gradient_width / 2
    gradient_y_start = mid_y - gradient_width / 2
    gradient_y_end = mid_y + gradient_width / 2

    # Assign cell types based on the position and gradient probabilities
    cell_types = []
    for pos in positions:
        x_gradient_prob = (
            np.clip((pos[0] - gradient_x_start) / gradient_width, 0, 1)
            if gradient_x_start <= pos[0] <= gradient_x_end
            else 1
        )
        y_gradient_prob = (
            np.clip((pos[1] - gradient_y_start) / gradient_width, 0, 1)
            if gradient_y_start <= pos[1] <= gradient_y_end
            else 1
        )

        if pos[0] < mid_x:  # Left half of the rectangle
            if pos[1] < mid_y:  # Bottom left quadrant
                cell_types.append(ct1 if np.random.rand() < x_gradient_prob else ct3)
            else:  # Top left quadrant, with gradient to ct2
                cell_types.append(ct2 if np.random.rand() < y_gradient_prob else ct1)
        else:  # Right half of the rectangle
            if pos[1] < mid_y:  # Bottom right quadrant, with gradient to ct2
                cell_types.append(ct3 if np.random.rand() < x_gradient_prob else ct1)
            else:  # Top right quadrant
                cell_types.append(ct2 if np.random.rand() < y_gradient_prob else ct3)

    # Create a DataFrame with cell positions and types
    spatial_data = pd.DataFrame(
        {
            "Cell_ID": range(1, num_cells + 1),
            "X": positions[:, 0],
            "Y": positions[:, 1],
            "Cell_Type": cell_types,
        }
    )

    rule_set = interaction_df[interaction_df["interaction_type"] == "interaction"].to_dict(orient="records")
    neutral_types = dict(
        zip(
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_cell"],
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_subtype"],
        )
    )
    assert ct1 in neutral_types
    assert ct2 in neutral_types
    assert ct3 in neutral_types

    spatial_data["Subtype"] = spatial_data["Cell_Type"].map(neutral_types)
    for rule in rule_set:
        receptor_cells = spatial_data[
            (spatial_data["Cell_Type"] == rule["receptor_cell"])
            & (spatial_data["Subtype"] == neutral_types[rule["receptor_cell"]])
        ]
        sender_cells = spatial_data[spatial_data["Cell_Type"] == rule["sender_cell"]]

        for _, receptor_cell in receptor_cells.iterrows():
            distances = np.sqrt(
                (sender_cells["X"] - receptor_cell["X"]) ** 2 + (sender_cells["Y"] - receptor_cell["Y"]) ** 2
            )
            if (distances <= rule["radius_of_effect"]).any():
                spatial_data.loc[spatial_data["Cell_ID"] == receptor_cell["Cell_ID"], "Subtype"] = rule[
                    "receptor_subtype"
                ]

    sampled_adatas = []
    for subtype in spatial_data["Subtype"].unique():
        subtype_cells = adata.obs[adata.obs["subtype"] == subtype].index
        num_samples = (spatial_data["Subtype"] == subtype).sum()

        sampled_indices = np.random.choice(subtype_cells, num_samples, replace=True)
        sampled_spatial = spatial_data.loc[spatial_data["Subtype"] == subtype, ["X", "Y"]]
        sampled_adata = adata[sampled_indices].copy()
        sampled_spatial.index = sampled_adata.obs_names
        sampled_adata.obsm["spatial"] = sampled_spatial
        sampled_adatas.append(sampled_adata)
    semisyn_spatial_adata = sc.concat(sampled_adatas, axis=0)
    return semisyn_spatial_adata


def _leiden_subclustering_binary_search(adata, cluster_id, target_subclusters):
    """
    Perform a binary search to find the optimal resolution for the subclustering.

    Args:
        adata: The adata object to perform the subclustering on.
        cluster_id: The cluster to perform the subclustering on.
        target_subclusters: The number of subclusters to create.

    Returns
    -------
        The adata object with the subclusters.
    """
    cluster_data = adata[adata.obs["leiden"] == cluster_id]
    sc.pp.neighbors(cluster_data, use_rep=f"X_rep_subclustering_{cluster_id}")

    min_resolution = 0.01
    max_resolution = 5.0

    while min_resolution <= max_resolution:
        resolution = (min_resolution + max_resolution) / 2
        sc.tl.leiden(cluster_data, resolution=resolution, key_added=f"leiden_sub_{cluster_id}")
        subcluster_labels = cluster_data.obs[f"leiden_sub_{cluster_id}"].unique()

        if len(subcluster_labels) < target_subclusters:
            min_resolution = resolution + 0.01
        elif len(subcluster_labels) > target_subclusters:
            max_resolution = resolution - 0.01
        else:
            break

    print(f"Cluster {cluster_id}: arrived at {len(subcluster_labels)} subclusters at resolution {resolution}.")
    subcluster_labels = cluster_data.obs[f"leiden_sub_{cluster_id}"].astype(str)
    subcluster_labels = [f"{cluster_id}_sub{label}" for label in subcluster_labels]
    adata.obs.loc[cluster_data.obs.index, "subtype"] = subcluster_labels
    return adata


def _create_interaction_df(gt_interactions):
    """
    Create a DataFrame for the interactions using the provided ground truth interactions config.

    Args:
        gt_interactions: A dictionary of ground truth interactions.

    Returns
    -------
        A DataFrame of the interactions.
    """
    ct1, ct2, ct3 = "3", "0", "2"
    interaction_dicts = [
        {
            "receptor_cell": ct1,
            "receptor_subtype": f"{ct1}_sub0",
            "interaction_type": "neutral",
        },
        {
            "receptor_cell": ct2,
            "receptor_subtype": f"{ct2}_sub0",
            "interaction_type": "neutral",
        },
        {
            "receptor_cell": ct3,
            "receptor_subtype": f"{ct3}_sub0",
            "interaction_type": "neutral",
        },
    ]

    for interaction in gt_interactions:
        interaction_config = gt_interactions[interaction]
        interaction_dicts.append(
            {
                "receptor_cell": interaction_config["receiver"],
                "sender_cell": interaction_config["sender"],
                "receptor_subtype": f"{interaction_config['receiver']}_sub1",
                "radius_of_effect": interaction_config["length_scale"],
                "interaction_type": "interaction",
            },
        )

    return pd.DataFrame(interaction_dicts)


def main():
    """Generate a synthetic dataset."""
    seed = int(snakemake.wildcards.seed)  # noqa: F821
    output_path = snakemake.output[0]  # noqa: F821

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    gt_interactions = dataset_config["gt_interactions"]

    interaction_df = _create_interaction_df(gt_interactions)

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    scvi.settings.seed = seed

    generate_synthetic_dataset(interaction_df, output_path)
    print(f"Successfully wrote output to {output_path}")


if __name__ == "__main__":
    main()
