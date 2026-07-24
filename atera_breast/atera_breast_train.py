# %% Import libraries
import os
from datetime import date

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

# %% Load data and model dir
DATA_DATE = "2026-07-22"
LABELS_KEY = "celltype_train_grouped"
COORD_KEY = "spatial"
DATA_DIR = "data"
PROJECT_NAME = f"atera_breast_sweep_{DATA_DATE}_model_{date.today()}"
ENTITY_NAME = "stattention"
SAVED_MODELS_DIR = f"saved_models/atera_breast_sweep_{DATA_DATE}_model_{date.today()}"

adata = ad.read_h5ad(f"{DATA_DIR}/atera_breast_filtered_{DATA_DATE}.h5ad")
adata_train = ad.read_h5ad(f"{DATA_DIR}/atera_breast_filtered_train_{DATA_DATE}.h5ad")
adata_test = ad.read_h5ad(f"{DATA_DIR}/atera_breast_filtered_test_{DATA_DATE}.h5ad")

CELL_TYPE_PALETTE = {
    "High_Grade_DCIS": "#D55E00",
    "Luminal_DCIS": "#E69F00",
    "Basal_DCIS": "#CC79A7",
    "CAFs": "#6A3D9A",
    "Fibroblasts": "#A6761D",
    "Myoepithelial": "#0072B2",
    "Endothelial": "#009E73",
    "Pericytes": "#66A61E",
    "T_Cells": "#56B4E9",
    "Dendritic Cells": "#1B9E77",
    "Macrophages": "#7570B3",
    "Myeloid": "#E7298A",
    "Mast_Cells": "#8DD3C7",
    "Plasma_Cells": "#FDB462",
    "Plasma_Mast_Mixture": "#B3B3B3",
    "Apocrine Cells": "#BC80BD",
}


def get_config_value(name, default):
    return getattr(wandb.config, name, default)


# %% Train model
def train():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    run = None
    try:
        run = wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.4, device=0)

        seed = int(get_config_value("seed", 40))
        pl.seed_everything(seed)

        print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
        print(f"Train cells: {adata_train.n_obs}; test cells: {adata_test.n_obs}")
        print(adata.obs[LABELS_KEY].value_counts().to_string())

        penalty_schedule = get_config_value("attention_penalty_schedule", [5, 15])
        penalty_schedule_params = {
            "start_attention_penalty": 1e-6,
            "end_attention_penalty": float(get_config_value("end_attention_penalty", 1.0e-3)),
            "epoch_start": int(penalty_schedule[0]),
            "epoch_end": int(penalty_schedule[1]),
            "flavor": get_config_value("penalty_flavor_params", "linear"),
        }
        model_params = {
            "n_heads": int(get_config_value("n_heads", 8)),
            "n_query_dim": 128,
            "n_head_size": 32,
            "n_nn_embed": 256,
            "n_nn_embed_hidden": 512,
            "attention_dummy_score": 3.0,
            "neighbor_dropout": 0.1,
            "attention_penalty_coef": penalty_schedule_params["start_attention_penalty"],
            "value_l1_penalty_coef": float(get_config_value("value_l1_penalty_coef", 1.0e-3)),
        }
        exp_params = {
            "lr": float(get_config_value("lr", 1.0e-3)),
            "epochs": int(get_config_value("epochs", 400)),
            "batch_size": int(get_config_value("batch_size", 512)),
            "early_stopping": True,
            "early_stopping_monitor": "elbo_validation",
            "early_stopping_patience": int(get_config_value("early_stopping_patience", 10)),
        }
        data_params = {
            "data_date": DATA_DATE,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_train_cells": adata_train.n_obs,
            "n_test_cells": adata_test.n_obs,
            "labels_key": LABELS_KEY,
            "coord_key": COORD_KEY,
        }
        wandb.config.update(model_params, allow_val_change=True)
        wandb.config.update(exp_params, allow_val_change=True)
        wandb.config.update(penalty_schedule_params, allow_val_change=True)
        wandb.config.update(data_params, allow_val_change=True)

        AMICI.setup_anndata(
            adata_train,
            labels_key=LABELS_KEY,
            coord_obsm_key=COORD_KEY,
            n_neighbors=int(get_config_value("n_neighbors", 50)),
        )
        model = AMICI(adata_train, **model_params)

        model_path = os.path.join(
            SAVED_MODELS_DIR,
            f"atera_breast_{seed}_sweep_{run.sweep_id}_{run.id}_params_{date.today()}",
        )
        if not os.path.exists(model_path):
            model.train(
                max_epochs=exp_params["epochs"],
                batch_size=exp_params["batch_size"],
                plan_kwargs={"lr": exp_params["lr"]},
                early_stopping=exp_params["early_stopping"],
                early_stopping_monitor=exp_params["early_stopping_monitor"],
                early_stopping_patience=exp_params["early_stopping_patience"],
                check_val_every_n_epoch=1,
                use_wandb=True,
                wandb_project=PROJECT_NAME,
                wandb_entity=ENTITY_NAME,
                wandb_run_name=f"atera_breast_{seed}_sweep_{run.sweep_id}_{run.id}",
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
            model.save(model_path, overwrite=True)
        else:
            print(f"Model already exists at {model_path}; skipping training")

        # %% Evaluate test set
        AMICI.setup_anndata(
            adata_test,
            labels_key=LABELS_KEY,
            coord_obsm_key=COORD_KEY,
            n_neighbors=int(get_config_value("n_neighbors", 50)),
        )
        test_elbo = model.get_elbo(adata_test, batch_size=exp_params["batch_size"]).item()
        test_reconstruction_loss = model.get_reconstruction_error(
            adata_test,
            batch_size=exp_params["batch_size"],
        )["reconstruction_loss"]
        print(f"Test ELBO: {test_elbo}")
        print(f"Test reconstruction loss: {test_reconstruction_loss}")
        wandb.log(
            {
                "test_elbo": test_elbo,
                "test_reconstruction_loss": test_reconstruction_loss,
                "model_path": model_path,
            }
        )

        # %% Log interpretation plots
        AMICI.setup_anndata(
            adata,
            labels_key=LABELS_KEY,
            coord_obsm_key=COORD_KEY,
            n_neighbors=int(get_config_value("n_neighbors", 50)),
        )
        expl_variance_scores = model.get_expl_variance_scores(
            adata,
            run_permutation_test=False,
        )
        expl_variance_scores.plot_explained_variance_barplot(
            palette=CELL_TYPE_PALETTE,
            wandb_log=True,
            show=False,
        )

        attention_patterns = model.get_attention_patterns(
            adata,
            batch_size=32,
        )
        attention_patterns.plot_attention_summary(
            palette=CELL_TYPE_PALETTE,
            wandb_log=True,
            show=False,
            max_distance=150,
            bin_size=10,
            min_bin_count=50,
        )

    except KeyboardInterrupt:
        print("Training interrupted")
        raise
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    train()
# %%
