import os
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from gitiii_benchmark_utils import convert_adata_to_csv, normalize_gitiii_scores, setup_gitiii_model


def main():
    """Generate the GITIII scores for the receiver subtype task."""
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

    setup_gitiii_model(
        converted_df_path,
        gene_names,
    )

    # Read the influence tensor and normalize the scores according to the GITIII code
    # Source: https://github.com/lugia-xiao/GITIII/blob/main/gitiii/subtyping_analyzer.py
    influence_tensor_path = os.path.join(os.getcwd(), "influence_tensor")
    if influence_tensor_path[-1] != "/":
        influence_tensor_path = influence_tensor_path + "/"

    influence_tensor = torch.load(influence_tensor_path + "edges_" + "slide1" + ".pth", weights_only=False)

    attention_scores = influence_tensor["attention_score"]

    avg_attention_scores = normalize_gitiii_scores(attention_scores)

    # Take the max attention score across the neighbors for all receivers
    receiver_attention_scores = avg_attention_scores.max(axis=1)

    gitiii_scores_df = pd.DataFrame(
        {
            "cell_idx": adata.obs_names,
            "gitiii_scores": receiver_attention_scores,
        }
    )

    # Save the scores
    os.chdir("../../../")
    gitiii_scores_df.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    # Seed everything
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
