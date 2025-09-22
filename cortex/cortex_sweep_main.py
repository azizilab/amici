# %%
# Import libraries
import os
import wandb
import torch
import numpy as np
import anndata as ad
import pytorch_lightning as pl

from datetime import date
from amici import AMICI
from multiprocessing import Process
from amici.callbacks import AttentionPenaltyMonitor

# %%
# Load data and model dir
data_date = "2025-04-28"
model_date = date.today()
adata = ad.read_h5ad(f"data/cortex_processed_{data_date}.h5ad")
adata.obsm["spatial"] = adata.obs[["centroid_x", "centroid_y"]].values
adata_train = adata[adata.obs["in_test"] == False].copy()

saved_models_dir = f"saved_models/cortex_sweep_{data_date}_model_{model_date}"
project_name = f"cortex_sweep_{data_date}_model_{model_date}"
entity_name = "stattention"


def train():
    run = None
    try:
        run = wandb.init(project=project_name, entity=entity_name)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
        seed = wandb.config.seed
        pl.seed_everything(seed)

        penalty_schedule_params = {
            "start_attention_penalty": 1e-6,
            "end_attention_penalty": wandb.config.end_attention_penalty,
            "epoch_start": wandb.config.attention_penalty_schedule[0],
            "epoch_end": wandb.config.attention_penalty_schedule[1],
            "flavor": wandb.config.penalty_flavor_params,
        }

        log_params = {
            "use_wandb": True,
            "wandb_project": project_name,
            "wandb_entity": entity_name,
            "wandb_run": f"cortex_{seed}_sweep_{run.sweep_id}_{run.id}_params",
        }
        model_params = {
            "n_heads": 8,
            "n_query_dim": 128,
            "n_head_size": 32,
            "n_nn_embed": 256,
            "n_nn_embed_hidden": 512,
            "attention_dummy_score": 3.0,
            "neighbor_dropout": 0.1,
            "attention_penalty_coef": penalty_schedule_params[
                "start_attention_penalty"
            ],
            "value_l1_penalty_coef": wandb.config.value_l1_penalty_coef,
        }
        exp_params = {
            "lr": wandb.config.lr,
            "epochs": 400,
            "batch_size": wandb.config.batch_size,
            "early_stopping": True,
            "early_stopping_monitor": "elbo_validation",
            "early_stopping_patience": 20,
            "learning_rate_monitor": True,
        }

        # Define model and setup anndata
        AMICI.setup_anndata(
            adata_train,
            labels_key="subclass",
            coord_obsm_key="spatial",
            n_neighbors=wandb.config.n_neighbors,
        )
        model = AMICI(adata_train, **model_params)

        # Train model and save best model using early stopping
        model_path = os.path.join(
            saved_models_dir,
            f"cortex_{seed}_sweep_{run.sweep_id}_{run.id}_params_{model_date}",
        )

        plan_kwargs = {}
        if "lr" in exp_params:
            plan_kwargs["lr"] = exp_params["lr"]

        model.train(
            max_epochs=int(exp_params.get("epochs")),
            batch_size=int(exp_params.get("batch_size", 128)),
            plan_kwargs=plan_kwargs,
            early_stopping=exp_params.get("early_stopping", False),
            early_stopping_monitor=exp_params.get("early_stopping_monitor"),
            early_stopping_patience=exp_params.get("early_stopping_patience", 5),
            check_val_every_n_epoch=1,
            use_wandb=log_params.get("use_wandb"),
            wandb_project=log_params.get("wandb_project"),
            wandb_entity=log_params.get("wandb_entity"),
            wandb_run_name=log_params.get("wandb_run"),
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
        if log_params.get("use_wandb"):
            wandb.config.update(exp_params)
            wandb.config.update(model_params)
            wandb.config.update(penalty_schedule_params)

        model.save(model_path, overwrite=True)

        # Evaluate test set
        AMICI.setup_anndata(
            adata,
            labels_key="subclass",
            coord_obsm_key="spatial",
            n_neighbors=wandb.config.n_neighbors,
        )

        CELL_TYPE_PALETTE = {
            # Excitatory Neurons
            "L2/3 IT": "#e41a1c",  # 4532 cells
            "L4/5 IT": "#ff7f00",  # 4617 cells
            "L5 IT": "#fdbf6f",  # 2319 cells
            "L5 ET": "#e31a1c",  # 846 cells
            "L6 IT": "#6a3d9a",  # 1941 cells
            "L6 IT Car3": "#cab2d6",  # 391 cells
            "L6 CT": "#fb9a99",  # 3109 cells
            "L5/6 NP": "#a6cee3",  # 345 cells
            "L6b": "#1f78b4",  # 499 cells
            # Inhibitory Neurons
            "Pvalb": "#8dd3c7",  # 880 cells
            "Sst": "#80b1d3",  # 479 cells
            "Lamp5": "#33a02c",  # 334 cells
            "Vip": "#b2df8a",  # 257 cells
            "Sncg": "#bc80bd",  # 35 cells
            # Glial Cells
            "Astro": "#bebada",  # 2560 cells
            "Oligo": "#fb8072",  # 2786 cells
            "OPC": "#b3de69",  # 660 cells
            "Micro": "#fccde5",  # 981 cells
            "VLMC": "#d9d9d9",  # 764 cells
            # Vascular Cells
            "Endo": "#ffff33",  # 2478 cells
            "Peri": "#ffffb3",  # 904 cells
            "PVM": "#fdb462",  # 570 cells
            "SMC": "#8dd3c7",  # 507 cells
            # Other
            "other": "#999999",  # 1037 cells
        }

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
        )

        # Get test set metrics
        test_elbo = model.get_elbo(
            adata, indices=np.where(adata.obs["in_test"])[0], batch_size=128
        ).item()
        test_reconstruction_loss = model.get_reconstruction_error(
            adata, indices=np.where(adata.obs["in_test"])[0], batch_size=128
        )["reconstruction_loss"]
        wandb.log(
            {
                "test_elbo": test_elbo,
                "test_reconstruction_loss": test_reconstruction_loss,
            }
        )
        run.finish()
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
