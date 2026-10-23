"""Extend a finished (non-cross-validated) AMICI sweep with extra seeds.

The benchmark sweep keeps only its best model, chosen by the lowest test reconstruction loss. Training the same
hyperparameter grid for additional seeds and comparing each new run against the recorded best loss is equivalent
to re-running the whole sweep with the longer seed list, without retraining the finished runs. Model selection
stays on test reconstruction loss, as in train_amici_model.py.

The new runs are trained with train_amici_model.py's own functions, every new model is kept, and each run's
hyperparameters and test loss are written to <out_dir>/<dataset>_<seed>/sweep_runs_seeds_<seeds>.json. If a new
run beats the previous best, it is saved as <out_dir>/<dataset>_<seed>/saved_models/amici_<dataset>_<seed>_best_model.

Usage (from benchmarks/, in the amici benchmark environment), for a finished sweep whose best test loss is in
results/breast_cancer_42/amici_breast_cancer_42_mse_loss.txt:
    python extend_amici_sweep.py --config <the sweep's config> \
        --dataset breast_cancer --data data/breast_cancer_42.h5ad \
        --previous-best-loss results/breast_cancer_42/amici_breast_cancer_42_mse_loss.txt \
        --seeds 101 102 --out-dir results_extra_seeds --base-run-id 1000
A file holding "inf" as --previous-best-loss runs a sweep from scratch.
Several calls can train in parallel with --no-select and distinct --base-run-id; a final call with --select-only
compares every logged run against the previous best.
"""

import argparse
import json
import os
from types import SimpleNamespace

import scanpy as sc
import train_amici_model as T
import yaml


def main():
    """Train the extra seeds and/or pick the sweep's best model over all logged runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--previous-best-loss", required=True, help="file holding the finished sweep's best test loss")
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--no-select", action="store_true", help="train and log only; select later with --select-only")
    parser.add_argument("--select-only", action="store_true", help="pick the best of all logged runs and save it")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-run-id", type=int, default=0, help="offset so run ids do not collide across calls")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    dataset_config = config["datasets"][args.dataset]
    assert not dataset_config.get("use_cross_validation", False), "only non-CV sweeps select on test loss"
    dataset_seed = str(dataset_config["seeds"][0])
    sweep_seeds = list(dataset_config["sweep_params"]["seed"])
    assert not set(args.seeds) & set(sweep_seeds), f"seeds {set(args.seeds) & set(sweep_seeds)} already in the sweep"

    # train_amici_model.py reads paths from the Snakemake object; point it at the output directory, with the same
    # output layout as the train_amici_model rule.
    name = f"{args.dataset}_{dataset_seed}"
    models_dir = os.path.join(args.out_dir, name, "saved_models")
    os.makedirs(os.path.join(models_dir, f"amici_{name}_best_model"), exist_ok=True)
    T.snakemake = SimpleNamespace(
        config={"results_path": args.out_dir},
        wildcards=SimpleNamespace(dataset=args.dataset, seed=dataset_seed),
        output=[
            os.path.join(models_dir, f"amici_{name}_best_model", "model.pt"),
            os.path.join(models_dir, f"amici_{name}_best_seed.txt"),
            os.path.join(args.out_dir, name, f"amici_{name}_mse_loss.txt"),
        ],
    )
    run_dir = os.path.join(args.out_dir, name)
    adata = sc.read_h5ad(args.data)
    adata.obs_names_make_unique()

    if not args.select_only:
        assert args.seeds, "--seeds is required unless --select-only"
        extended = {**dataset_config, "sweep_params": {**dataset_config["sweep_params"], "seed": args.seeds}}
        runs = T._build_run_configs(extended)
        print(f"Training {len(runs)} runs for seeds {args.seeds}")
        results = T._run_sweep_parallel(
            adata, extended, runs, eval_indices=None, save_model=True, base_run_id=args.base_run_id
        )
        records = []
        for i, run in enumerate(runs):
            run_id = f"run_{args.base_run_id + i}"
            records.append(
                {
                    "run_id": run_id,
                    "penalty_params": run["penalty_params"],
                    "exp_params": run["exp_params"],
                    "test_recons": float(results[run_id]["test_recons"]),
                    "model_path": results[run_id]["model_path"],
                }
            )
        log_path = os.path.join(run_dir, f"sweep_runs_seeds_{'_'.join(map(str, args.seeds))}.json")
        with open(log_path, "w") as handle:
            json.dump(records, handle, indent=2)
        print(f"Wrote {log_path}")
        if args.no_select:
            return

    records = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("sweep_runs_seeds_") and fname.endswith(".json"):
            records.extend(json.load(open(os.path.join(run_dir, fname))))
    previous_best = float(open(args.previous_best_loss).read().strip())
    best = min(records, key=lambda r: r["test_recons"])
    print(
        f"{len(records)} new runs; best {best['run_id']} (seed {best['exp_params']['seed']}) test loss "
        f"{best['test_recons']:.6f}; previous best {previous_best:.6f}"
    )
    if best["test_recons"] < previous_best:
        model_config = {"penalty_params": best["penalty_params"], "exp_params": best["exp_params"]}
        T._save_best_model(adata, model_config, best["model_path"], best["test_recons"])
        print(f"New best model saved under {run_dir}/saved_models/")
    else:
        print("The previous best model remains the sweep's best.")


if __name__ == "__main__":
    main()
