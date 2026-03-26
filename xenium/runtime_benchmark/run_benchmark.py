#!/usr/bin/env python
"""Benchmark AMICI training runtime and peak memory on Xenium dataset subsets."""

import argparse
import gc
import json
import os
import time
import tracemalloc

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import read_h5ad

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor

MODEL_PARAMS = {
    "n_heads": 8,
    "n_query_dim": 256,
    "n_head_size": 32,
    "n_nn_embed": 256,
    "n_nn_embed_hidden": 512,
    "neighbor_dropout": 0.1,
    "attention_penalty_coef": 1e-5,
    "value_l1_penalty_coef": 1e-4,
}

BATCH_SIZE = 128
LR = 1e-5
LABELS_KEY = "celltype_train_grouped"
COORD_KEY = "spatial"

PENALTY_PARAMS = {
    "epoch_start": 40,
    "epoch_end": 80,
    "start_val": 1e-5,
    "end_val": 1e-4,
}


def default_subset_sizes(n_obs):
    """Generate geometric progression from 1000 up to n_obs."""
    steps = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    sizes = [s for s in steps if s < n_obs]
    if not sizes or sizes[-1] != n_obs:
        sizes.append(n_obs)
    return sizes


def subset_adata(adata, n_cells, seed):
    """Create a reproducible random subset of the AnnData object."""
    if n_cells >= adata.n_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(adata.n_obs, size=n_cells, replace=False)
    return adata[indices].copy()


def run_single_benchmark(adata_subset, accelerator, max_epochs):
    """Run a single benchmark and return timing + memory results."""
    gc.collect()

    if accelerator == "gpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()

    # Time setup_anndata (includes NN computation)
    t_setup_start = time.perf_counter()
    AMICI.setup_anndata(
        adata_subset,
        labels_key=LABELS_KEY,
        coord_obsm_key=COORD_KEY,
    )
    t_setup_end = time.perf_counter()

    # Time model creation + training
    t_train_start = time.perf_counter()
    model = AMICI(adata_subset, **MODEL_PARAMS)
    model.train(
        max_epochs=max_epochs,
        batch_size=BATCH_SIZE,
        plan_kwargs={"lr": LR},
        early_stopping=False,
        check_val_every_n_epoch=1,
        accelerator=accelerator,
        devices="auto",
        use_wandb=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            AttentionPenaltyMonitor(
                PENALTY_PARAMS["epoch_start"],
                PENALTY_PARAMS["epoch_end"],
                PENALTY_PARAMS["start_val"],
                PENALTY_PARAMS["end_val"],
            ),
        ],
    )
    t_train_end = time.perf_counter()

    if accelerator == "gpu":
        peak_memory_bytes = torch.cuda.max_memory_allocated()
    else:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "n_cells": adata_subset.n_obs,
        "runtime_setup_s": t_setup_end - t_setup_start,
        "runtime_train_s": t_train_end - t_train_start,
        "runtime_total_s": (t_setup_end - t_setup_start) + (t_train_end - t_train_start),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": peak_memory_bytes / (1024**3),
        "accelerator": accelerator,
    }


def run_benchmarks(adata, subset_sizes, accelerator, max_epochs, seed):
    """Run benchmarks for all subset sizes on a given accelerator."""
    results = []
    for n_cells in subset_sizes:
        print(f"Benchmarking n_cells={n_cells} on {accelerator}...")
        adata_subset = subset_adata(adata, n_cells, seed)

        try:
            result = run_single_benchmark(adata_subset, accelerator, max_epochs)
            results.append(result)
            print(
                f"  Runtime: {result['runtime_total_s']:.2f}s "
                f"(setup={result['runtime_setup_s']:.2f}s, train={result['runtime_train_s']:.2f}s), "
                f"Peak memory: {result['peak_memory_gb']:.3f} GB"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  CUDA OOM at n_cells={n_cells}, skipping.")
            else:
                raise

        del adata_subset
        gc.collect()
        if accelerator == "gpu":
            torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="AMICI Runtime Benchmark")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the training h5ad file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save results JSON files"
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=None,
        help="List of subset sizes (default: geometric progression up to dataset size)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50, help="Number of training epochs per run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsetting")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="both",
        choices=["cpu", "gpu", "both"],
        help="Which accelerator(s) to benchmark",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    adata = read_h5ad(args.data_path)
    print(f"Loaded dataset with {adata.n_obs} cells, {adata.n_vars} genes")

    subset_sizes = args.subset_sizes or default_subset_sizes(adata.n_obs)
    print(f"Subset sizes: {subset_sizes}")

    accelerators = []
    if args.accelerator in ("cpu", "both"):
        accelerators.append("cpu")
    if args.accelerator in ("gpu", "both"):
        if torch.cuda.is_available():
            accelerators.append("gpu")
        else:
            print("WARNING: GPU requested but CUDA not available. Skipping GPU benchmark.")

    os.makedirs(args.output_dir, exist_ok=True)

    for accel in accelerators:
        print(f"\n{'='*60}")
        print(f"Running benchmarks on {accel.upper()}")
        print(f"{'='*60}")

        results = run_benchmarks(adata, subset_sizes, accel, args.max_epochs, args.seed)

        output_path = os.path.join(args.output_dir, f"benchmark_results_{accel}.json")
        payload = {
            "metadata": {
                "model_params": MODEL_PARAMS,
                "train_params": {
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "max_epochs": args.max_epochs,
                },
                "penalty_params": PENALTY_PARAMS,
                "seed": args.seed,
                "data_path": args.data_path,
                "total_dataset_cells": adata.n_obs,
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
