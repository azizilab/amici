# %%
# Import libraries
import os
import json
import torch
import numpy as np
import anndata as ad
import pandas as pd
import pytorch_lightning as pl

from datetime import date
from itertools import product
from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

# %%
# Load data
data_date = "2025-05-01"
model_date = date.today()
adata_full = ad.read_h5ad(f"data/xenium_sample1_filtered_{data_date}.h5ad")

# Use only training cells for cross-validation
adata_train_full = adata_full[adata_full.obs["train_test_split"] == "train"].copy()
print(f"Total training cells for cross-validation: {len(adata_train_full)}")

results_dir = f"crossval_results/xenium_sample1_crossval_{data_date}_model_{model_date}"
os.makedirs(results_dir, exist_ok=True)

# %%
# Cross-validation settings
SEEDS = [13, 18, 21, 22, 33]
N_FOLDS = 3
LABELS_KEY = "celltype_train_grouped"

# Fixed hyperparameters
FIXED_PARAMS = {
    "lr": 1e-3,
    "batch_size": 512,
    "n_neighbors": 50,
    "penalty_flavor_params": "linear",
}

# Grid hyperparameters
param_grid = {
    "end_attention_penalty": [1e-3, 1e-2],
    "attention_penalty_schedule": [(5, 15), (10, 20)],
    "value_l1_penalty_coef": [1e-5, 1e-3],
}

# %%
def create_spatial_folds(adata, n_folds, axis="X"):
    """
    Split cells into n_folds contiguous spatial strips along the given axis.
    Returns a list of index arrays (one per fold).
    """
    coords = adata.obsm["spatial"][axis].values
    sorted_positions = np.argsort(coords)
    fold_indices = np.array_split(sorted_positions, n_folds)
    return fold_indices


def train_and_eval_fold(adata_cv_train, adata_eval, val_idx, params, seed):
    """
    Train AMICI on adata_cv_train, then evaluate reconstruction loss on
    val_idx cells using adata_eval (the full training set) for neighbor context.

    Returns the scalar reconstruction loss on the validation fold.
    """
    torch.cuda.empty_cache()
    pl.seed_everything(seed)

    penalty_schedule_params = {
        "start_attention_penalty": 1e-6,
        "end_attention_penalty": params["end_attention_penalty"],
        "epoch_start": params["attention_penalty_schedule"][0],
        "epoch_end": params["attention_penalty_schedule"][1],
        "flavor": params["penalty_flavor_params"],
    }
    model_params = {
        "n_heads": 8,
        "n_query_dim": 128,
        "n_head_size": 32,
        "n_nn_embed": 256,
        "n_nn_embed_hidden": 512,
        "attention_dummy_score": 3.0,
        "neighbor_dropout": 0.1,
        "attention_penalty_coef": penalty_schedule_params["start_attention_penalty"],
        "value_l1_penalty_coef": params["value_l1_penalty_coef"],
    }

    # Train on the fold's training subset
    AMICI.setup_anndata(
        adata_cv_train,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=params["n_neighbors"],
    )
    model = AMICI(adata_cv_train, **model_params)

    model.train(
        max_epochs=400,
        batch_size=int(params["batch_size"]),
        plan_kwargs={"lr": params["lr"]},
        early_stopping=True,
        early_stopping_monitor="elbo_validation",
        early_stopping_patience=10,
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                penalty_schedule_params["epoch_start"],
                penalty_schedule_params["epoch_end"],
                penalty_schedule_params["start_attention_penalty"],
                penalty_schedule_params["end_attention_penalty"],
                penalty_schedule_params["flavor"],
            ),
        ],
    )

    # Evaluate on val fold using the full training adata for neighbor context
    AMICI.setup_anndata(
        adata_eval,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=params["n_neighbors"],
    )
    val_reconstruction_loss = model.get_reconstruction_error(
        adata_eval, indices=val_idx, batch_size=128
    )["reconstruction_loss"]
    if hasattr(val_reconstruction_loss, "item"):
        val_reconstruction_loss = val_reconstruction_loss.item()

    return val_reconstruction_loss


# %%
# Build all hyperparameter combinations (grid params merged with fixed params)
param_keys = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = [
    {**FIXED_PARAMS, **dict(zip(param_keys, combo))}
    for combo in product(*param_values)
]

print(f"Total hyperparameter combinations: {len(param_combinations)}")
print(f"Seeds: {SEEDS}")
print(f"Folds: {N_FOLDS}")
print(f"Total training runs: {len(param_combinations) * len(SEEDS) * N_FOLDS}")

# %%
# Run cross-validation sweep
all_results = []

for combo_idx, params in enumerate(param_combinations):
    print(f"\n=== Combination {combo_idx + 1}/{len(param_combinations)} ===")
    print(f"Params: {params}")

    all_fold_losses = []

    for seed in SEEDS:
        fold_indices = create_spatial_folds(adata_train_full, N_FOLDS)

        for fold_idx, val_idx in enumerate(fold_indices):
            train_idx = np.concatenate(
                [fold_indices[i] for i in range(N_FOLDS) if i != fold_idx]
            )
            adata_cv_train = adata_train_full[train_idx].copy()

            print(
                f"  Seed {seed}, Fold {fold_idx + 1}/{N_FOLDS}: "
                f"train={len(adata_cv_train)}, val={len(val_idx)}"
            )

            try:
                loss = train_and_eval_fold(
                    adata_cv_train, adata_train_full, val_idx, params, seed
                )
                all_fold_losses.append(loss)
                print(f"    Val reconstruction loss: {loss:.4f}")
            except Exception as e:
                print(f"    Failed: {e}")
                all_fold_losses.append(float("nan"))

    avg_loss = float(np.nanmean(all_fold_losses))
    print(f"  => Average val reconstruction loss: {avg_loss:.4f}")

    result = {
        **params,
        "avg_val_reconstruction_loss": avg_loss,
        "all_fold_losses": json.dumps(all_fold_losses),
    }
    # Convert tuple params to strings for CSV serialisation
    result["attention_penalty_schedule"] = str(params["attention_penalty_schedule"])
    all_results.append(result)

    # Save intermediate results after each combination
    pd.DataFrame(all_results).to_csv(
        os.path.join(results_dir, "crossval_results.csv"), index=False
    )

# %%
# Select best hyperparameters
results_df = pd.DataFrame(all_results)
best_idx = results_df["avg_val_reconstruction_loss"].idxmin()
best_row = results_df.iloc[best_idx]

print("\n=== Best Hyperparameters ===")
print(best_row.to_string())

results_df.to_csv(os.path.join(results_dir, "crossval_results.csv"), index=False)

# Save best params as JSON (excluding bookkeeping columns)
skip_cols = {"avg_val_reconstruction_loss", "all_fold_losses"}
best_params_dict = {
    k: v for k, v in best_row.items() if k not in skip_cols
}
with open(os.path.join(results_dir, "best_params.json"), "w") as f:
    json.dump(best_params_dict, f, indent=2)

print(f"\nResults saved to: {results_dir}/")
# %%
