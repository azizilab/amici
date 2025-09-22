import os

import matplotlib.pyplot as plt
import torch
from cgcom.scripts import train_model
from cgcom.utils import get_exp_params, get_model_params


def main():
    """Main function to train the CGCom model."""
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    model_path = snakemake.output[0]  # noqa: F821

    # Get the hyperparameters and the model parameters for the CGCom model
    exp_params = get_exp_params(lr=0.001, num_epochs=50, neighbor_threshold_ratio=0.01)
    model_params = get_model_params(
        fc_hidden_channels_2=500,
        fc_hidden_channels_3=512,
        fc_hidden_channels_4=64,
        num_classes=10,
        device=torch.device("cuda"),
        ligand_channel=500,
        receptor_channel=500,
        TF_channel=500,
        mask_indexes=None,
        disable_lr_masking=True,
    )

    # Train the model
    model, model_path, train_losses, val_losses = train_model(
        exp_params,
        model_params,
        dataset_path=dataset_path,
        model_path=model_path,
        labels_key=labels_key,
        disable_lr_masking=True,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.legend()
    plt.savefig(
        os.path.join(f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/figures/cgcom_loss_curve.png")  # noqa: F821
    )
    plt.savefig(
        os.path.join(f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/figures/cgcom_loss_curve.svg")  # noqa: F821
    )
    plt.close()


if __name__ == "__main__":
    main()
