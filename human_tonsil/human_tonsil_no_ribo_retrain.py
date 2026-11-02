"""Retrain the main tonsil AMICI model without ribosomal-protein genes and MALAT1.

The CosMx 1000-plex panel carries five RPL genes, UBA52 (RPL40) and FAU (RPS30), and MALAT1:
about 12% of all counts, and the Xenium and Atera gene sets have none of them. They filled
top-gene slots across the tonsil interactions far above chance, and because each cell was
scaled to a fixed total that included them, their field-of-view variation reached every other
gene's normalized value. This drops them from the counts, renormalizes exactly as the original
data (log1p of counts scaled to the median total), and trains one model at the fixed
hyperparameters of the main figure (wandb sweep 0f9rk0na, run 4303v0zu, seed 40), so it can be
compared with that model before deciding whether to switch.

Run from the human_tonsil directory:  python human_tonsil_no_ribo_retrain.py
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy import sparse

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

# %% Config
DATA_DATE = "2026-07-25"
LABELS_KEY = "celltype_manual_fine"
COORD_KEY = "spatial"
DATA_DIR = "data"
SOURCE_PATH = os.path.join(DATA_DIR, f"human_tonsil_filtered_{DATA_DATE}.h5ad")
OUT_PATHS = {
    split: os.path.join(DATA_DIR, f"human_tonsil_filtered_no_ribo{suffix}_{DATA_DATE}.h5ad")
    for split, suffix in [("all", ""), ("train", "_train"), ("test", "_test")]
}
REMOVED_GENES = ["RPL21", "RPL22", "RPL32", "RPL34", "RPL37", "UBA52", "FAU", "MALAT1"]

RUN_NAME = "human_tonsil_no_ribo_retrain"
SEED = 40
MODEL_DIR = os.path.join("saved_models", RUN_NAME, f"seed_{SEED}")
RESULTS_DIR = os.path.join("results", RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed hyperparameters, matching wandb/sweep-0f9rk0na/config-4303v0zu.yaml (as in
# human_tonsil_length_scale_seed_sweep.py).
N_NEIGHBORS = 50
MODEL_PARAMS = {
    "n_heads": 12,
    "n_query_dim": 128,
    "n_head_size": 32,
    "n_nn_embed": 256,
    "n_nn_embed_hidden": 512,
    "attention_dummy_score": 3.0,
    "neighbor_dropout": 0.1,
    "attention_penalty_coef": 1e-6,
    "value_l1_penalty_coef": 1e-5,
}
EXP_PARAMS = {
    "lr": 0.001,
    "epochs": 400,
    "batch_size": 512,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 10,
}
PENALTY_SCHEDULE = {"epoch_start": 5, "epoch_end": 15, "start_val": 1e-6, "end_val": 1e-4, "flavor": "linear"}


def build_data():
    """Drop REMOVED_GENES from the counts and renormalize; same cells and train/test split as the source."""
    adata = ad.read_h5ad(SOURCE_PATH)
    missing = [g for g in REMOVED_GENES if g not in adata.var_names]
    assert not missing, f"not in panel: {missing}"
    removed = adata.var_names.isin(REMOVED_GENES)
    all_counts = adata.layers["counts"]
    share = all_counts[:, removed].sum() / all_counts.sum()
    print(f"Removing {removed.sum()} genes, {100 * share:.1f}% of all counts")
    adata = adata[:, ~removed].copy()

    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    assert totals.min() > 0, "a cell has no counts left"
    target_sum = float(np.median(totals))
    X = sparse.diags(target_sum / totals) @ counts
    X.data = np.log1p(X.data)
    adata.X = X.tocsr().astype(np.float32)
    adata.uns["normalize"] = {"target_sum": target_sum, "median_counts_before": target_sum}
    adata.uns["removed_genes"] = REMOVED_GENES
    print(f"{adata.n_obs} cells x {adata.n_vars} genes; renormalized to the median total, {target_sum:.0f} counts")

    adata.write_h5ad(OUT_PATHS["all"])
    for split in ["train", "test"]:
        adata[adata.obs["train_test_split"] == split].copy().write_h5ad(OUT_PATHS[split])


def train(adata_train):
    """Train the seed-40 model at the fixed hyperparameters and save its loss history."""
    pl.seed_everything(SEED)
    AMICI.setup_anndata(adata_train, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    model = AMICI(adata_train, **MODEL_PARAMS)
    model.train(
        max_epochs=EXP_PARAMS["epochs"],
        batch_size=EXP_PARAMS["batch_size"],
        plan_kwargs={"lr": EXP_PARAMS["lr"]},
        early_stopping=EXP_PARAMS["early_stopping"],
        early_stopping_monitor=EXP_PARAMS["early_stopping_monitor"],
        early_stopping_patience=EXP_PARAMS["early_stopping_patience"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                PENALTY_SCHEDULE["epoch_start"],
                PENALTY_SCHEDULE["epoch_end"],
                PENALTY_SCHEDULE["start_val"],
                PENALTY_SCHEDULE["end_val"],
                PENALTY_SCHEDULE["flavor"],
            ),
        ],
    )
    model.save(MODEL_DIR, overwrite=True)
    history = pd.concat([v for v in model.history.values() if isinstance(v, pd.DataFrame)], axis=1)
    history.to_csv(os.path.join(RESULTS_DIR, f"seed_{SEED}_history.csv"))
    return model


def main():
    """Build the data if missing, train or load the model, and score it on the test region."""
    if not all(os.path.exists(p) for p in OUT_PATHS.values()):
        build_data()
    adata_train = ad.read_h5ad(OUT_PATHS["train"])
    adata_test = ad.read_h5ad(OUT_PATHS["test"])
    if os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
        model = AMICI.load(MODEL_DIR, adata=adata_train)
        print(f"Loaded existing model from {MODEL_DIR}")
    else:
        model = train(adata_train)

    AMICI.setup_anndata(adata_test, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    test_elbo = model.get_elbo(adata_test, batch_size=EXP_PARAMS["batch_size"]).item()
    test_reconstruction_loss = model.get_reconstruction_error(adata_test, batch_size=EXP_PARAMS["batch_size"])[
        "reconstruction_loss"
    ]
    print(f"Test ELBO: {test_elbo}")
    print(f"Test reconstruction loss: {test_reconstruction_loss}")
    pd.Series({"test_elbo": test_elbo, "test_reconstruction_loss": test_reconstruction_loss}).to_csv(
        os.path.join(RESULTS_DIR, f"seed_{SEED}_test_metrics.csv")
    )


if __name__ == "__main__":
    main()
