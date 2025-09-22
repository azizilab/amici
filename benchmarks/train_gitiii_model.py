# %% Import the necessary libraries
import random

import gitiii
import numpy as np
import scanpy as sc
import torch
from gitiii_benchmark_utils import convert_adata_to_csv


def main():
    """Train the GITIII model."""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Load and reformat the dataset for GITIII
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821

    adata = sc.read_h5ad(dataset_path)

    models_dir = f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/saved_models"  # noqa: F821
    converted_df_path = f"../../../data/{dataset_path.split('/')[-1].split('.')[0]}_converted.csv"

    convert_adata_to_csv(adata, labels_key, models_dir, converted_df_path)

    # Create the estimator and preprocess the data
    # Change the working directory to the saved_models directory to avoid saving the model weights in the current directory
    estimator = gitiii.estimator.GITIII_estimator(
        df_path=converted_df_path,
        genes=adata.var_names.to_list(),
        use_log_normalize=False,
        species="human",
        use_nichenetv2=True,
        visualize_when_preprocessing=False,
        distance_threshold=80,
        process_num_neighbors=50,
        num_neighbors=50,
        batch_size_train=128,
        lr=1e-4,
        epochs=100,
        node_dim=256,
        edge_dim=48,
        att_dim=8,
        batch_size_val=128,
    )
    estimator.preprocess_dataset()

    # Train the model and calculate the influence tensor
    estimator.train()


if __name__ == "__main__":
    main()
