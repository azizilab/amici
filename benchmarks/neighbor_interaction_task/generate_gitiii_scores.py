import os
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from benchmark_utils import plot_interaction_graph, plot_interaction_matrix
from einops import repeat
from gitiii_benchmark_utils import convert_adata_to_csv, setup_gitiii_model
from gpu_utils import select_gpu


def main():
    """Generate the GITIII scores for the neighbor interaction task."""
    select_gpu()
    # Load the dataset
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821

    project_root = os.path.abspath(os.getcwd())
    figures_dir = os.path.join(project_root, f"results/{dataset}_{seed}/figures")

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

    # Build cell-type interaction matrix using the same normalized attention scores
    cell_type_labels = adata.obs[labels_key].values
    cell_type_list = sorted(adata.obs[labels_key].unique().tolist())
    n_types = len(cell_type_list)
    ct_to_idx = {ct: i for i, ct in enumerate(cell_type_list)}

    it_scores = np.zeros((n_types, n_types))
    it_counts = np.zeros_like(it_scores)
    for i in range(avg_attention_scores.shape[0]):
        c_idx = ct_to_idx[cell_type_labels[i]]
        for j in range(nn_idxs.shape[1]):
            n_cell_idx = nn_idxs[i, j]
            if 0 <= n_cell_idx < len(cell_type_labels):
                n_idx = ct_to_idx[cell_type_labels[n_cell_idx]]
                it_scores[n_idx, c_idx] += avg_attention_scores[i, j]
                it_counts[n_idx, c_idx] += 1

    mean_scores = np.divide(it_scores, it_counts, where=it_counts > 0)
    matrix = pd.DataFrame(mean_scores, index=cell_type_list, columns=cell_type_list)

    os.makedirs(figures_dir, exist_ok=True)
    plot_interaction_matrix(
        matrix,
        figures_dir,
        title="GITIII cell-type interaction matrix",
        filename="gitiii_interaction_matrix",
        colorbar_label="Mean attention strength",
    )
    plot_interaction_graph(
        matrix,
        figures_dir,
        title="GITIII cell-type interaction graph",
        filename="gitiii_interaction_graph",
    )

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
