import os
import random
import sys

import numpy as np
import scanpy as sc
import tensorflow as tf
from ncem_benchmark_utils import get_model_parameters, plot_ncem_loss_curves, train_ncem


def main(input_path, labels_key, dataset, seed, niche_size):
    """
    Train the NCEM model.

    Args:
        input_path: The path to the input adata object.
        labels_key: The key to the labels in the adata object.
        dataset: The dataset to train the model on.
        seed: The seed to use for the model.
    """
    adata = sc.read_h5ad(input_path)
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = adata.obsm["spatial"].copy()

    model_dir = f"results/{dataset}_{seed}/saved_models"
    model_path = os.path.join(model_dir, f"ncem_{niche_size}_checkpoint_{dataset}_{seed}")
    model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_{dataset}_{seed}.pickle")

    if os.path.exists(model_path) and os.path.exists(model_args_path):
        print(f"Model path {model_path} and model args path {model_args_path} already exist, skipping")
        return

    exp_params, model_params, train_params = get_model_parameters(niche_size)

    model_history, _ = train_ncem(
        adata,
        labels_key,
        exp_params,
        model_params,
        train_params,
        model_path,
        model_args_path,
    )

    # Plot the loss curves and save them
    plot_ncem_loss_curves(
        model_history,
        niche_size,
        f"results/{dataset}_{seed}/figures",
    )


if __name__ == "__main__":
    print("Started training NCEM")
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Parse the arguments in the command line
    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]
    labels_key = args[2]
    dataset = args[3]
    seed = args[4]
    niche_size = args[5]

    try:
        niche_size = int(niche_size)
        main(input_path, labels_key, dataset, seed, niche_size)
        # Write completion file
        with open(output_path, "w") as f:
            f.write("Training done")
            f.flush()
            os.fsync(f.fileno())
        print("Finished training NCEM")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
