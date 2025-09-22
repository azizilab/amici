import os
import random

import gitiii
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from einops import rearrange
from gitiii_benchmark_utils import convert_adata_to_csv, setup_gitiii_model
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def main():
    """Generate the GITIII scores for the gene task."""
    # Load the dataset
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821

    adata = sc.read_h5ad(dataset_path)

    converted_df_path = f"../../../data/{dataset_path.split('/')[-1].split('.')[0]}_converted.csv"
    models_dir = f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/saved_models"  # noqa: F821
    gene_names = adata.var_names.tolist()

    convert_adata_to_csv(
        adata,
        labels_key,
        models_dir,
        converted_df_path,
    )

    _ = setup_gitiii_model(
        converted_df_path,
        gene_names,
    )

    # Network analysis
    network_analyzer = gitiii.network_analyzer.Network_analyzer()
    results_matrix = network_analyzer.determine_network_sample(sample="slide1")
    results_matrix = rearrange(
        results_matrix,
        "(s r) g -> s r g",
        s=len(adata.obs[labels_key].unique()),
        r=len(adata.obs[labels_key].unique()),
    )

    # Convert the results into a DataFrame with scores
    # Rows are the sender cells, columns are the receiver cells
    cell_type_pair_sequence = network_analyzer.cell_type_pair_sequence

    # Create lists to store the data
    senders = []
    receivers = []
    genes = []
    z_scores = []

    # Iterate through the results matrix
    for sender_idx in range(results_matrix.shape[0]):
        for receiver_idx in range(results_matrix.shape[1]):
            if sender_idx == receiver_idx:
                continue
            # Extract sender and receiver from the pair sequence
            pair = cell_type_pair_sequence[sender_idx * results_matrix.shape[1] + receiver_idx]
            sender, receiver = pair.split("__")
            # Remove the string prefix (ct_)
            sender = sender.replace("ct_", "")
            receiver = receiver.replace("ct_", "")

            # For each gene
            for gene_idx, gene in enumerate(gene_names):
                senders.append(sender)
                receivers.append(receiver)
                genes.append(gene)
                z_scores.append(results_matrix[sender_idx, receiver_idx, gene_idx])

    # Create the DataFrame
    gitiii_scores_df = pd.DataFrame(
        {
            "sender": senders,
            "receiver": receivers,
            "gene": genes,
            "z_score": z_scores,
        }
    )

    # Calculate p-values using the formula: p = 2(1 - Φ(|z|)) from the paper
    # https://github.com/lugia-xiao/GITIII
    gitiii_scores_df["p_value_adj"] = 2 * (1 - norm.cdf(np.abs(gitiii_scores_df["z_score"])))

    # Group by sender-receiver pairs and apply BH correction within each group
    for _, group in gitiii_scores_df.groupby(["sender", "receiver"]):
        # Get the indices for this group
        indices = group.index
        # Apply BH correction to the p-values in this group
        _, corrected_pvals, _, _ = multipletests(gitiii_scores_df.loc[indices, "p_value_adj"].values, method="fdr_bh")
        # Update the DataFrame with corrected p-values
        gitiii_scores_df.loc[indices, "p_value_adj"] = corrected_pvals

    # Save the DataFrame to a CSV file
    os.chdir("../../..")
    gitiii_scores_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    # Seed everything
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
