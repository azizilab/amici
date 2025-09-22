# %%
import os

import scanpy as sc
import wandb

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

import torch

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# %%
saved_models_dir = "./saved_models"

# %%
# load adata
adata_train = sc.read_h5ad(
    "/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_train.h5ad"
)
adata_test = sc.read_h5ad(
    "/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_test.h5ad"
)
print(adata_train)
print(adata_test)
# %%
# run wandb sweep
sweep_config = {
    "method": "bayes",
    "name": "cosmx_liver_cancerous_subset",
    "metric": {"name": "elbo_validation", "goal": "minimize"},
    "parameters": {
        "n_heads": {"min": 1, "max": 32},
        "start_attention_penalty": {
            "min": 1e-6,
            "max": 1e-5,
            "distribution": "log_uniform_values",
        },
        "end_attention_penalty": {
            "min": 1e-5,
            "max": 1e-4,
            "distribution": "log_uniform_values",
        },
        "penalty_epoch_end": {"min": 10, "max": 80},
        "n_query_dim": {"min": 128, "max": 512, "distribution": "q_log_uniform_values"},
        "n_head_size": {"min": 128, "max": 512, "distribution": "q_log_uniform_values"},
        "n_pe_dim": {"values": [128, 256, 512]},
        "lr": {"min": 1e-3, "max": 1e-1, "distribution": "log_uniform_values"},
        "batch_size": {"min": 128, "max": 512, "distribution": "q_log_uniform_values"},
        "use_empirical_ct_means": {"values": [True, False]},
        "norm_first": {"values": [True, False]},
        "neighbor_dropout": {"min": 0.0, "max": 0.3, "distribution": "uniform"},
        "value_l1_penalty_coef": {
            "min": 1e-4,
            "max": 1e-3,
            "distribution": "log_uniform_values",
        },
        "residual_l2_penalty_coef": {
            "min": 1e-3,
            "max": 1.0,
            "distribution": "log_uniform_values",
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="cosmx_liver_cancer_sub_sweep")


# %%
def main():
    run = wandb.init()

    torch.cuda.empty_cache()

    penalty_schedule_params = {
        "start_attention_penalty": wandb.config.start_attention_penalty,
        "end_attention_penalty": wandb.config.end_attention_penalty,
        "epoch_start": 10,
        "epoch_end": wandb.config.penalty_epoch_end,
    }

    log_params = {
        "use_wandb": True,
        "wandb_project": "cosmx",
        "wandb_entity": "stattention",
        "wandb_run": "cosmx_liver_cancerous_liver",
    }
    model_params = {
        "n_heads": wandb.config.n_heads,
        "n_layers": 1,
        "n_query_dim": wandb.config.n_query_dim,
        "n_query_len": 1,
        "n_head_size": wandb.config.n_head_size,
        "n_nn_embed": 256,
        "n_nn_embed_hidden": 512,
        "n_pe_label_hidden": 512,
        "n_pe_label_embed": 256,
        "n_pe_dim": wandb.config.n_pe_dim,
        "add_dummy_dim": True,
        "attention_temp_coef": 1.0,
        "attention_dummy_score": 3.0,
        "neighbor_dropout": wandb.config.neighbor_dropout,
        "attention_penalty_coef": penalty_schedule_params["start_attention_penalty"],
        "value_l1_penalty_coef": wandb.config.value_l1_penalty_coef,
    }
    exp_params = {
        "lr": wandb.config.lr,
        "epochs": 400,
        "batch_size": wandb.config.batch_size,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_validation",
        "early_stopping_patience": 10,
        "learning_rate_monitor": True,
    }

    AMICI.setup_anndata(
        adata_train,
        labels_key="cellType",
        coord_obsm_key="spatial",
    )
    model = AMICI(adata_train, **model_params)

    plan_kwargs = {}
    if "lr" in exp_params:
        plan_kwargs["lr"] = exp_params["lr"]

    model.train(
        max_epochs=int(exp_params.get("epochs")),
        batch_size=int(exp_params.get("batch_size", 128)),
        plan_kwargs=plan_kwargs,
        early_stopping=exp_params.get("early_stopping", False),
        early_stopping_monitor=exp_params.get("early_stopping_monitor"),
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

    model_path = os.path.join(
        saved_models_dir, f"cosmx_liver_cancer_sub_sweep_{run.sweep_id}_{run.id}"
    )
    model.save(model_path, overwrite=True)

    # # evaluate test set
    AMICI.setup_anndata(adata_test, labels_key="cellType", coord_obsm_key="spatial")
    test_elbo = model.get_elbo(adata_test, batch_size=128).item()
    test_reconstruction_loss = model.get_reconstruction_error(
        adata_test, batch_size=128
    )["reconstruction_loss"]
    wandb.log(
        {"test_elbo": test_elbo, "test_reconstruction_loss": test_reconstruction_loss}
    )


# %%
wandb.agent(sweep_id, function=main)

# %%
