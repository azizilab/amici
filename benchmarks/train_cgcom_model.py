import json
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from cgcom.scripts import train_model
from cgcom.utils import get_exp_params, get_model_params


def train_single_run(
    dataset_path, labels_key, lr, neighbor_threshold_ratio, run_seed, run_model_path, run_results_path
):
    """Train a single CGCom run and return the best validation loss, loading from cache if available."""
    if os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: val_loss={cached['val_loss']:.6f}")
        return cached["val_loss"]

    os.makedirs(os.path.dirname(run_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(run_results_path), exist_ok=True)

    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    exp_params = get_exp_params(lr=lr, num_epochs=50, neighbor_threshold_ratio=neighbor_threshold_ratio)
    model_params = get_model_params(
        fc_hidden_channels_2=500,
        fc_hidden_channels_3=512,
        fc_hidden_channels_4=64,
        num_classes=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ligand_channel=500,
        receptor_channel=500,
        TF_channel=500,
        mask_indexes=None,
        disable_lr_masking=True,
    )

    _model, _model_path, train_losses, val_losses = train_model(
        exp_params,
        model_params,
        dataset_path=dataset_path,
        model_path=run_model_path,
        labels_key=labels_key,
        disable_lr_masking=True,
    )

    best_val_loss = float(min(val_losses))
    print(f"  best val_loss={best_val_loss:.6f}")

    with open(run_results_path, "w") as f:
        json.dump({"val_loss": best_val_loss, "train_losses": train_losses, "val_losses": val_losses}, f)

    return best_val_loss


def main():
    """Train the CGCom model with a hyperparameter sweep over lr, neighbor_threshold_ratio, and seed, keeping the best run."""
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821
    final_model_path = os.path.abspath(snakemake.output[0])  # noqa: F821

    models_dir = os.path.dirname(final_model_path)
    os.makedirs(models_dir, exist_ok=True)

    learning_rates = [1e-3, 1e-4, 1e-5]
    neighbor_threshold_ratios = [0.005, 0.01, 0.02]
    run_seeds = [42, 22, 38]
    all_runs = [(lr, ntr, s) for lr in learning_rates for ntr in neighbor_threshold_ratios for s in run_seeds]

    best_val_loss = float("inf")
    best_run_id = None

    for run_id, (lr, neighbor_threshold_ratio, run_seed) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"cgcom_sweep_run_{run_id}")
        run_model_path = os.path.join(run_dir, "cgcom_model.pt")
        run_results_path = os.path.join(models_dir, f"cgcom_sweep_run_{run_id}_results.json")

        print(
            f"Run {run_id + 1}/{len(all_runs)}: lr={lr}, neighbor_threshold_ratio={neighbor_threshold_ratio}, seed={run_seed}"
        )

        # Skip runs that already have both model and results artifacts.
        if os.path.exists(run_model_path) and os.path.exists(run_results_path):
            with open(run_results_path) as f:
                cached = json.load(f)
            val_loss = float(cached["val_loss"])
            print(f"  Skipping existing run: val_loss={val_loss:.6f}")
        else:
            val_loss = train_single_run(
                dataset_path, labels_key, lr, neighbor_threshold_ratio, run_seed, run_model_path, run_results_path
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_run_id = run_id

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    best_lr, best_ntr, best_seed_val = all_runs[best_run_id]
    best_run_dir = os.path.join(models_dir, f"cgcom_sweep_run_{best_run_id}")

    print(
        f"Best run {best_run_id}: lr={best_lr}, neighbor_threshold_ratio={best_ntr}, seed={best_seed_val}, val_loss={best_val_loss:.6f}"
    )

    for fname in os.listdir(best_run_dir):
        src = os.path.join(best_run_dir, fname)
        dst = os.path.join(models_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    best_run_model = os.path.join(models_dir, "cgcom_model.pt")
    if best_run_model != final_model_path:
        shutil.copy2(best_run_model, final_model_path)

    best_params_path = os.path.join(models_dir, "cgcom_best_params.json")
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "learning_rate": best_lr,
                "neighbor_threshold_ratio": best_ntr,
                "seed": int(best_seed_val),
                "val_loss": best_val_loss,
            },
            f,
        )

    with open(os.path.join(models_dir, f"cgcom_sweep_run_{best_run_id}_results.json")) as f:
        best_results = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(best_results["train_losses"], label="Train Loss")
    plt.plot(best_results["val_losses"], label="Validation Loss", linestyle="--")
    plt.legend()
    figures_dir = os.path.join(f"results/{dataset}_{seed}/figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "cgcom_loss_curve.png"))
    plt.savefig(os.path.join(figures_dir, "cgcom_loss_curve.svg"))
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
