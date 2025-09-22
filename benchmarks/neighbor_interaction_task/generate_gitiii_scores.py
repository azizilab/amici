import os
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from einops import repeat
from gitiii_benchmark_utils import convert_adata_to_csv, setup_gitiii_model


def main():
    """Generate the GITIII scores for the neighbor interaction task."""
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

    # Read the influence tensor and normalize the scores according to the GITIII code
    # Source: https://github.com/lugia-xiao/GITIII/blob/main/gitiii/subtyping_analyzer.py
    influence_tensor_path = os.path.join(os.getcwd(), "influence_tensor")
    if influence_tensor_path[-1] != "/":
        influence_tensor_path = influence_tensor_path + "/"

    influence_tensor = torch.load(influence_tensor_path + "edges_" + "slide1" + ".pth", weights_only=False)

    attention_scores = influence_tensor["attention_score"]

    proportion = torch.abs(attention_scores)
    proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)
    noise_threshold = 1e-5
    attention_scores[proportion < noise_threshold] = 0

    avg_attention_scores = np.mean(np.abs(attention_scores.detach().numpy()), axis=2)

    norm_attention_scores = avg_attention_scores / np.sum(avg_attention_scores, axis=1, keepdims=True)
    avg_attention_scores = np.where(
        np.sum(avg_attention_scores, axis=1, keepdims=True) == 0,
        np.zeros_like(avg_attention_scores),
        norm_attention_scores,
    )

    nn_idxs = influence_tensor["NN"][:, 1:]  # batch x n_neighbors

    obs_names = repeat(adata.obs_names.values, "b -> b n", n=nn_idxs.shape[1])
    nn_obs_names = adata.obs_names.values[nn_idxs]  # batch x n_neighbors

    assert obs_names.shape == nn_obs_names.shape == avg_attention_scores.shape

    gitiii_scores_df = pd.DataFrame(
        {
            "cell_idx": obs_names.flatten(),
            "neighbor_idx": nn_obs_names.flatten(),
            "gitiii_scores": avg_attention_scores.flatten(),
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
