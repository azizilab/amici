# %% Import libraries
import os
import wandb

import anndata as ad
import pytorch_lightning as pl

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

# %% Seed everything
run_num = 1
seed = 58
pl.seed_everything(seed)

# %% Load data and model dir
labels_key = "celltype_train_grouped"

model_date = "2024-11-04"
adata = ad.read_h5ad(f"data/xenium_proseg/xenium_proseg_filtered_{model_date}.h5ad")
adata_train = ad.read_h5ad(
    f"data/xenium_proseg/xenium_proseg_filtered_train_{model_date}.h5ad"
)
adata_test = ad.read_h5ad(
    f"data/xenium_proseg/xenium_proseg_filtered_test_{model_date}.h5ad"
)

saved_models_dir = "./saved_models"

# %% Set up model and training parameters
penalty_schedule_params = {
    "start_attention_penalty": 1e-5,
    "end_attention_penalty": 1e-4,
    "epoch_start": 40,
    "epoch_end": 80,
}

log_params = {
    "use_wandb": False,
    "wandb_project": "stattention-logging",
    "wandb_entity": "stattention",
    "wandb_run": f"xenium_breast_{seed}_{run_num}",
}
model_params = {
    "n_heads": 8,
    "n_query_dim": 256,
    "n_head_size": 32,
    "n_nn_embed": 256,
    "n_nn_embed_hidden": 512,
    "n_pe_label_hidden": 512,
    "n_pe_label_embed": 256,
    "n_pe_dim": 512,
    "neighbor_dropout": 0.1,
    "attention_penalty_coef": penalty_schedule_params["start_attention_penalty"],
    "value_l1_penalty_coef": 1e-4,
}
exp_params = {
    "lr": 1e-5,
    "epochs": 400,
    "batch_size": 128,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "learning_rate_monitor": True,
}

AMICI.setup_anndata(
    adata_train,
    labels_key=labels_key,
    coord_obsm_key="spatial",
)
model = AMICI(adata_train, **model_params)

# %% Train model if it does not exist
model_path = os.path.join(saved_models_dir, f"xenium_{seed}_khushi_params_{run_num}")

if not os.path.exists(model_path):
    plan_kwargs = {}
    if "lr" in exp_params:
        plan_kwargs["lr"] = exp_params["lr"]

    model.train(
        max_epochs=int(exp_params.get("epochs")),
        batch_size=int(exp_params.get("batch_size", 128)),
        plan_kwargs=plan_kwargs,
        early_stopping=exp_params.get("early_stopping", False),
        early_stopping_monitor=exp_params.get("early_stopping_monitor"),
        early_stopping_patience=exp_params.get("early_stopping_patience", 10),
        check_val_every_n_epoch=1,
        use_wandb=log_params.get("use_wandb"),
        wandb_project=log_params.get("wandb_project"),
        wandb_entity=log_params.get("wandb_entity"),
        wandb_run_name=log_params.get("wandb_run"),
        enable_checkpointing=True,
        callbacks=[
            AttentionPenaltyMonitor(
                penalty_schedule_params["epoch_start"],
                penalty_schedule_params["epoch_end"],
                penalty_schedule_params["start_attention_penalty"],
                penalty_schedule_params["end_attention_penalty"],
            ),
            # ModelInterpretationLogging()
        ],
    )
    if log_params.get("use_wandb"):
        wandb.config.update(exp_params)
        wandb.config.update(model_params)
        wandb.config.update(penalty_schedule_params)

    model.save(model_path, overwrite=True)

# %% Evaluate test set
AMICI.setup_anndata(adata_test, labels_key=labels_key, coord_obsm_key="spatial")
test_elbo = model.get_elbo(adata_test, batch_size=128).item()
test_reconstruction_loss = model.get_reconstruction_error(adata_test, batch_size=128)[
    "reconstruction_loss"
]
print(f"Test ELBO: {test_elbo}")
print(f"Test reconstruction loss: {test_reconstruction_loss}")
if log_params.get("use_wandb"):
    wandb.log(
        {"test_elbo": test_elbo, "test_reconstruction_loss": test_reconstruction_loss}
    )
    wandb.finish()
# %%
