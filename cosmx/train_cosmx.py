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
def main():
    penalty_schedule_params = {
        "start_attention_penalty": 1e-5,
        "end_attention_penalty": 1e-4,
        "epoch_start": 40,
        "epoch_end": 80,
    }

    log_params = {
        "use_wandb": True,
        "wandb_project": "cosmx",
        "wandb_entity": "stattention",
        "wandb_run": "cosmx_liver_cancerous_liver_khushi_params",
    }
    model_params = {
        "n_heads": 8,
        "n_layers": 1,
        "n_query_dim": 256,
        "n_query_len": 1,
        "n_head_size": 32,
        "n_nn_embed": 256,
        "n_nn_embed_hidden": 512,
        "n_pe_label_hidden": 512,
        "n_pe_label_embed": 256,
        "n_pe_dim": 512,
        "add_dummy_dim": True,
        "attention_temp_coef": 1.0,
        "attention_dummy_score": 3.0,
        "neighbor_dropout": 0.1,
        "attention_penalty_coef": penalty_schedule_params["start_attention_penalty"],
        "value_l1_penalty_coef": 1e-4,
        "residual_l2_penalty_coef": 0.5,
        "use_empirical_ct_means": False,
        "norm_first": True,
    }
    exp_params = {
        "lr": 1e-5,
        "epochs": 400,
        "batch_size": 128,
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

    model_path = os.path.join(saved_models_dir, f"cosmx_liver_cancer_sub_khushi_params")
    model.save(model_path, overwrite=True)

    # # evaluate test set
    AMICI.setup_anndata(adata_test, labels_key="cellType", coord_obsm_key="spatial")
    test_elbo = model.get_elbo(adata_test, batch_size=128).item()
    test_reconstruction_loss = model.get_reconstruction_error(
        adata_test, batch_size=128
    )["reconstruction_loss"]
    print(f"Test ELBO: {test_elbo}")
    print(f"Test reconstruction loss: {test_reconstruction_loss}")
    wandb.log(
        {"test_elbo": test_elbo, "test_reconstruction_loss": test_reconstruction_loss}
    )
    wandb.finish()


# %%
main()

# %%
