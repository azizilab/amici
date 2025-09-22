import os

import gitiii
import numpy as np
import pandas as pd
import torch


def convert_adata_to_csv(
    adata,
    labels_key,
    models_dir,
    save_df_path,
):
    """
    Convert an AnnData object to a CSV file based on GITIII's format specifications.

    Args:
        adata: AnnData object
        labels_key: str, key in adata.obs to use for labels
        models_dir: str, directory to save the model weights
        save_df_path: str, path to save the CSV file
    """
    # Get gene expression matrix
    expr_matrix = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)

    # Reformat labels as strings
    adata_sub = adata.copy()
    adata_sub.obs[labels_key] = adata_sub.obs[labels_key].apply(lambda x: f"ct_{x}")

    # Get observation columns
    observation_columns = pd.DataFrame(
        {
            "centerx": adata_sub.obsm["spatial"]["X"],
            "centery": adata_sub.obsm["spatial"]["Y"],
            "subclass": adata_sub.obs[labels_key],
            "section": ["slide1"] * adata_sub.n_obs,
        },
        index=adata_sub.obs_names,
    )

    # Combine all dataframes
    result_df = pd.concat([expr_matrix, observation_columns], axis=1)

    # Convert and save the DataFrame
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Change the working directory to the saved_models directory to avoid saving the model weights in the current directory
    os.chdir(models_dir)

    result_df.to_csv(save_df_path, index=False, quoting=1)
    print(f"Converted data saved to: {save_df_path}")


def setup_gitiii_model(
    converted_df_path,
    gene_names,
):
    """
    Setup the GITIII model and preprocess the data. Compute the influence tensor for further analysis.

    Args:
        converted_df_path: str, path to the converted CSV file
        gene_names: list, list of gene names

    Returns
    -------
        estimator: GITIII_estimator object
    """
    # Create the estimator and preprocess the data
    estimator = gitiii.estimator.GITIII_estimator(
        df_path=converted_df_path,
        genes=gene_names,
        use_log_normalize=False,
        species="human",
        use_nichenetv2=True,
        visualize_when_preprocessing=False,
        distance_threshold=80,
        process_num_neighbors=50,
        num_neighbors=50,
        batch_size_train=256,
        lr=1e-4,
        epochs=100,
        node_dim=256,
        edge_dim=48,
        att_dim=8,
        batch_size_val=256,
    )
    estimator.preprocess_dataset()

    # Calculate the influence tensor
    estimator.calculate_influence_tensor()

    return estimator


def normalize_gitiii_scores(attention_scores):
    """
    Normalize the GITIII scores according to the GITIII code

    Source: https://github.com/lugia-xiao/GITIII/blob/main/gitiii/subtyping_analyzer.py

    Args:
        attention_scores: torch.Tensor, attention scores

    Returns
    -------
        normalized_attention_scores: torch.Tensor, normalized attention scores
    """
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
    return avg_attention_scores
