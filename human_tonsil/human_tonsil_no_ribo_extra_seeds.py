"""Train extra seeds of the no-ribosomal tonsil model, to check that head-specific programs replicate.

Same data, hyperparameters and penalty schedule as human_tonsil_no_ribo_retrain.py (seed 40); only
the seed changes. Seeds come from the 20-seed length-scale sweep (101-120), so the models are
comparable with that sweep's rank-matched heads.

Run from the human_tonsil directory:  python human_tonsil_no_ribo_extra_seeds.py 101 [102 ...]
"""

import os
import sys

import anndata as ad

import human_tonsil_no_ribo_retrain as retrain


def main():
    """Train each seed given on the command line, skipping seeds whose model already exists."""
    adata_train = ad.read_h5ad(retrain.OUT_PATHS["train"])
    for seed in [int(s) for s in sys.argv[1:]]:
        retrain.SEED = seed
        retrain.MODEL_DIR = os.path.join("saved_models", retrain.RUN_NAME, f"seed_{seed}")
        if os.path.exists(os.path.join(retrain.MODEL_DIR, "model.pt")):
            print(f"seed {seed}: model exists, skipping", flush=True)
            continue
        print(f"seed {seed}: training", flush=True)
        retrain.train(adata_train.copy())


if __name__ == "__main__":
    main()
