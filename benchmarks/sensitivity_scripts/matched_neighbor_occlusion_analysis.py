"""Occlude high-attention neighbors against distance- and cell-type-matched low-attention neighbors.

Expression is not matched: the attention logit is a deterministic function of neighbor expression,
distance, and receiver label, so matching on expression would make the two arms identical.
"""

# %% Import libraries
import itertools
import json
import os
import random
import shutil
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scvi
import torch
from scipy.stats import wilcoxon
from scvi import REGISTRY_KEYS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from generate_dataset import _create_interaction_df, generate_synthetic_dataset  # noqa: E402
from generate_realistic_dataset import generate_realistic_dataset  # noqa: E402
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici._constants import NN_REGISTRY_KEYS  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402
from amici.interpretation._attention_module import AMICIAttentionModule  # noqa: E402

# %% Config
RUN_NAME = "matched_neighbor_occlusion_analysis"
DEFAULT_RANDOM_SEED = 42
N_DISTANCE_BINS = 10
MATCHED_OCCLUSION_FRACTION = 0.5
MATCHED_ARM_SEED = 0
CHUNK_SIZE = 20000
INFERENCE_BATCH_SIZE = 128

INTERVENTIONS = ["postsoftmax"]

EXP_DEFAULTS = {
    "epochs": 400,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
}

SYNTHETIC_INTERACTIONS = {
    "interaction_1": {
        "sender": "0",
        "receiver": "2",
        "interaction_subtype": "2_sub1",
        "neutral_subtype": "2_sub0",
        "length_scale": 10,
    },
    "interaction_2": {
        "sender": "3",
        "receiver": "0",
        "interaction_subtype": "0_sub1",
        "neutral_subtype": "0_sub0",
        "length_scale": 20,
    },
}
REALISTIC_INTERACTIONS = {
    "interaction_1": {
        "sender": "Macrophages",
        "receiver": "DCIS",
        "interaction_subtype": "DCIS_sub1",
        "neutral_subtype": "DCIS_sub0",
        "length_scale": 40,
    },
    "interaction_2": {
        "sender": "T_Cells",
        "receiver": "Endothelial",
        "interaction_subtype": "Endothelial_sub1",
        "neutral_subtype": "Endothelial_sub0",
        "length_scale": 25,
    },
    "interaction_3": {
        "sender": "Invasive_Tumor",
        "receiver": "Myoepi",
        "interaction_subtype": "Myoepi_sub1",
        "neutral_subtype": "Myoepi_sub0",
        "length_scale": 15,
    },
}

DATASET_CONFIGS = {
    "3ct_dataset_2way": {
        "kind": "synthetic",
        "dataset_seed": 58,
        "labels_key": "leiden",
        "subtype_key": "subtype",
        "gt_interactions": SYNTHETIC_INTERACTIONS,
        "source_h5ad_path": "sensitivity_scripts/data/semisyn_58.h5ad",
        "sweep_params": {
            "end_attention_penalty": [3e-4],
            "attention_penalty_schedule": [[10, 40]],
            "seed": [21, 33, 88],
            "value_l1_penalty_coef": [3e-5],
            "batch_size": [128],
            "lr": [1e-3],
            "n_neighbors": [50],
            "penalty_flavor_params": ["linear"],
            "n_heads": [8],
        },
    },
    "breast_cancer": {
        "kind": "realistic",
        "dataset_seed": 2,
        "labels_key": "cell_type",
        "subtype_key": "subtype",
        "gt_interactions": REALISTIC_INTERACTIONS,
        "sweep_params": {
            "end_attention_penalty": [1e-5],
            "attention_penalty_schedule": [[15, 30]],
            "seed": [21, 22, 33],
            "value_l1_penalty_coef": [1e-5],
            "batch_size": [256],
            "lr": [1e-3],
            "n_neighbors": [50],
            "penalty_flavor_params": ["linear"],
            "n_heads": [10],
        },
        "flex_h5_path": "data/GSM7782698_count_raw_feature_bc_matrix.h5",
        "annot_path": "data/41467_2023_43458_MOESM4_ESM.xlsx",
        "xenium_path": "data/xenium_rep1_io.h5ad",
        "base_h5ad_path": "sensitivity_scripts/data/realistic_length_scale_dataset_bootstrap_ci/breast_cancer_2.h5ad",
        "scvi_model_dir": "data/scvi_model",
    },
}

# %% Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
saved_models_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figure_dir = os.path.join(base_dir, "figures", RUN_NAME)
results_dir = os.path.join(base_dir, "results", RUN_NAME)

os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# %% Dataset and model helpers
def benchmark_path(path):
    """Resolve benchmark config paths relative to the benchmarks directory."""
    return path if os.path.isabs(path) else os.path.join(benchmark_dir, path)


def dataset_path(dataset_name, dataset_config):
    """Return the cached h5ad path for this analysis."""
    return os.path.join(data_dir, f"{dataset_name}_{dataset_config['dataset_seed']}.h5ad")


def ensure_dataset(dataset_name, dataset_config):
    """Generate the semisynthetic dataset if it is not already cached."""
    dataset_seed = int(dataset_config["dataset_seed"])
    adata_path = dataset_path(dataset_name, dataset_config)
    if os.path.exists(adata_path):
        return

    source_h5ad_path = dataset_config.get("source_h5ad_path")
    if source_h5ad_path and os.path.exists(benchmark_path(source_h5ad_path)):
        shutil.copyfile(benchmark_path(source_h5ad_path), adata_path)
        return

    if dataset_config["kind"] == "synthetic":
        interaction_df = _create_interaction_df(dataset_config["gt_interactions"])
        random.seed(dataset_seed)
        np.random.seed(dataset_seed)
        torch.random.manual_seed(dataset_seed)
        scvi.settings.seed = dataset_seed
        generate_synthetic_dataset(interaction_df, adata_path)
        return

    base_h5ad_path = benchmark_path(dataset_config["base_h5ad_path"])
    if os.path.exists(base_h5ad_path):
        shutil.copyfile(base_h5ad_path, adata_path)
        return

    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    scvi.settings.seed = dataset_seed
    generate_realistic_dataset(
        benchmark_path(dataset_config["flex_h5_path"]),
        benchmark_path(dataset_config["annot_path"]),
        benchmark_path(dataset_config["xenium_path"]),
        adata_path,
        scvi_model_dir=benchmark_path(dataset_config["scvi_model_dir"]),
    )


def build_sweep_runs(dataset_config):
    """Build the AMICI candidates used to select the occlusion model."""
    sweep = dataset_config["sweep_params"]
    run_configs = []
    for run_idx, (end_val, schedule, train_seed, value_l1, batch_size, lr, n_neighbors, flavor, n_heads) in enumerate(
        itertools.product(
            sweep["end_attention_penalty"],
            sweep["attention_penalty_schedule"],
            sweep["seed"],
            sweep["value_l1_penalty_coef"],
            sweep["batch_size"],
            sweep["lr"],
            sweep["n_neighbors"],
            sweep["penalty_flavor_params"],
            sweep["n_heads"],
        )
    ):
        run_configs.append(
            {
                "end_val": end_val,
                "flavor": flavor,
                "value_l1": value_l1,
                "train_seed": train_seed,
                "epoch_start": schedule[0],
                "epoch_end": schedule[1],
                "batch_size": batch_size,
                "n_heads": n_heads,
                "lr": lr,
                "n_neighbors": n_neighbors,
                "run_idx": run_idx,
            }
        )
    return run_configs


def train_or_load_run(dataset_name, adata, dataset_config, run):
    """Train or load one AMICI candidate model."""
    run_idx = int(run["run_idx"])
    run_dir = os.path.join(saved_models_dir, dataset_name, "candidate_runs", f"run_{run_idx}_seed_{run['train_seed']}")
    model_path = os.path.join(run_dir, "model")
    result_path = os.path.join(run_dir, "result.json")

    if os.path.exists(os.path.join(model_path, "model.pt")) and os.path.exists(result_path):
        return json.load(open(result_path))

    os.makedirs(run_dir, exist_ok=True)
    pl.seed_everything(run["train_seed"])
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    AMICI.setup_anndata(
        adata_train,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model = AMICI(adata_train, n_heads=int(run["n_heads"]), value_l1_penalty_coef=float(run["value_l1"]))
    model.train(
        max_epochs=EXP_DEFAULTS["epochs"],
        batch_size=int(run["batch_size"]),
        plan_kwargs={"lr": float(run["lr"])},
        early_stopping=EXP_DEFAULTS["early_stopping"],
        early_stopping_monitor=EXP_DEFAULTS["early_stopping_monitor"],
        early_stopping_patience=EXP_DEFAULTS["early_stopping_patience"],
        check_val_every_n_epoch=1,
        use_wandb=False,
        callbacks=[
            AttentionPenaltyMonitor(
                start_val=1e-6,
                end_val=float(run["end_val"]),
                epoch_start=int(run["epoch_start"]),
                epoch_end=int(run["epoch_end"]),
                flavor=run["flavor"],
            ),
        ],
    )

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model.save(model_path, overwrite=True)
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]
    test_loss = (
        model.get_reconstruction_error(adata, indices=test_indices, batch_size=int(run["batch_size"]))[
            "reconstruction_loss"
        ]
        .detach()
        .cpu()
        .numpy()
        .item()
    )
    result = {**run, "test_loss": test_loss, "model_path": model_path}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def train_or_load_model(dataset_name, adata, dataset_config):
    """Train/load candidates and return the best model by held-out reconstruction loss."""
    default_run = build_sweep_runs(dataset_config)[0]
    model_path = os.path.join(saved_models_dir, dataset_name, "model")
    result_path = os.path.join(saved_models_dir, dataset_name, "result.json")

    if os.path.exists(os.path.join(model_path, "model.pt")) and os.path.exists(result_path):
        selected_run = json.load(open(result_path))
        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(selected_run.get("n_neighbors", default_run["n_neighbors"])),
        )
        return AMICI.load(model_path, adata=adata)

    candidate_results = [
        train_or_load_run(dataset_name, adata, dataset_config, run) for run in build_sweep_runs(dataset_config)
    ]
    selected_run = min(candidate_results, key=lambda result: result["test_loss"])

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    shutil.copytree(selected_run["model_path"], model_path)
    with open(result_path, "w") as f:
        json.dump({**selected_run, "model_path": model_path, "selection_metric": "test_loss"}, f, indent=2)

    AMICI.setup_anndata(
        adata,
        labels_key=dataset_config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(selected_run["n_neighbors"]),
    )
    return AMICI.load(model_path, adata=adata)


# %% Attention extraction
def get_attention_matrix(model, adata):
    """Max-over-head attention weight for every (cell, neighbor slot) pair.

    Matches the scoring used by the neighbor-interaction benchmark task. Rows returned by
    AMICIAttentionModule.compute are stacked head-major, so reshaping recovers the head axis.

    Returns
    -------
        np.ndarray: n_cells x n_neighbors array of attention weights.
    """
    n_neighbors = np.asarray(adata.obsm["_nn_idx"]).shape[1]
    neighbor_cols = [f"neighbor_{i}" for i in range(n_neighbors)]
    attention_chunks = []

    for chunk_start in range(0, adata.n_obs, CHUNK_SIZE):
        chunk_indices = np.arange(chunk_start, min(chunk_start + CHUNK_SIZE, adata.n_obs))
        attention_patterns = AMICIAttentionModule.compute(
            model,
            adata=adata,
            indices=chunk_indices,
            batch_size=INFERENCE_BATCH_SIZE,
            prog_bar=False,
        )
        attention = attention_patterns._attention_patterns_df[neighbor_cols].to_numpy(dtype=np.float32)
        attention_chunks.append(attention.reshape(-1, len(chunk_indices), n_neighbors).max(axis=0))

    return np.vstack(attention_chunks)


# %% Matched occlusion masks
def build_matched_masks(adata, attention, label_codes, n_distance_bins, fraction, seed):
    """Build matched high-/low-attention occlusion masks plus a within-stratum random arm.

    Neighbors of each receiver are grouped into (neighbor cell type, distance bin) strata. Within
    every stratum holding at least two neighbors, the top-attention and bottom-attention halves are
    occluded in separate arms, so the two arms remove the same number of neighbors of the same cell
    types from the same distance bins of the same receivers. The per-stratum count is clamped to
    half the stratum size so the two arms never share a neighbor.

    Args:
        adata: AnnData registered with AMICI nearest-neighbor fields.
        attention: n_cells x n_neighbors attention weights.
        label_codes: Integer cell type code per cell.
        n_distance_bins: Number of quantile distance bins used for matching.
        fraction: Fraction of each stratum occluded in each arm.
        seed: Seed for the random within-stratum arm.

    Returns
    -------
        tuple[dict[str, np.ndarray], pd.DataFrame]: Keep/drop masks per arm and a per-occluded-pair
        diagnostic frame recording the arm, cell type, and distance of every removed neighbor.
    """
    nn_dist = np.asarray(adata.obsm["_nn_dist"])
    nn_codes = label_codes[np.asarray(adata.obsm["_nn_idx"])]
    n_cells, n_neighbors = nn_dist.shape

    bin_edges = np.quantile(nn_dist, np.linspace(0, 1, n_distance_bins + 1)[1:-1])
    dist_bins = np.digitize(nn_dist, bin_edges)

    cell_ids = np.repeat(np.arange(n_cells), n_neighbors)
    stratum_ids = (nn_codes * (n_distance_bins + 1) + dist_bins).ravel()
    flat_attention = attention.ravel()

    order = np.lexsort((flat_attention, stratum_ids, cell_ids))
    sorted_cells = cell_ids[order]
    sorted_strata = stratum_ids[order]

    group_start = np.flatnonzero(
        np.concatenate(([True], (sorted_cells[1:] != sorted_cells[:-1]) | (sorted_strata[1:] != sorted_strata[:-1])))
    )
    group_sizes = np.diff(np.append(group_start, len(order)))
    group_of_entry = np.repeat(np.arange(len(group_start)), group_sizes)
    rank_in_group = np.arange(len(order)) - group_start[group_of_entry]
    size_of_entry = group_sizes[group_of_entry]

    k_per_group = np.maximum(1, np.floor(fraction * group_sizes).astype(int))
    k_per_group = np.minimum(k_per_group, group_sizes // 2)
    k_of_entry = k_per_group[group_of_entry]

    eligible = size_of_entry >= 2
    is_low = eligible & (rank_in_group < k_of_entry)
    is_high = eligible & (rank_in_group >= size_of_entry - k_of_entry)

    rng = np.random.default_rng(seed)
    random_key = rng.random(len(order))
    random_order = np.lexsort((random_key, sorted_strata, sorted_cells))
    random_rank = np.empty(len(order), dtype=int)
    random_rank[random_order] = np.arange(len(order))
    random_rank = random_rank - group_start[group_of_entry]
    is_random = eligible & (random_rank < k_of_entry)

    masks = {}
    diagnostics = []
    for arm_name, selection in [("high_attention", is_high), ("low_attention", is_low), ("random", is_random)]:
        drop_flat = np.zeros(len(order), dtype=bool)
        drop_flat[order[selection]] = True
        mask = np.ones((n_cells, n_neighbors), dtype=int)
        mask[drop_flat.reshape(n_cells, n_neighbors)] = 0
        masks[arm_name] = mask

        dropped = drop_flat.reshape(n_cells, n_neighbors)
        diagnostics.append(
            pd.DataFrame(
                {
                    "arm": arm_name,
                    "distance": nn_dist[dropped],
                    "neighbor_code": nn_codes[dropped],
                    "attention": attention[dropped],
                }
            )
        )

    return masks, pd.concat(diagnostics, ignore_index=True)


# %% Masked prediction
def _postsoftmax_zero_hook(attn_result, hook, neighbor_mask):
    """Zero the post-softmax attention of occluded neighbors without renormalizing survivors."""
    mask = torch.cat([neighbor_mask, torch.ones_like(neighbor_mask[:, :1])], dim=1)
    attn_result *= mask.unsqueeze(1).unsqueeze(2).expand_as(attn_result)
    return attn_result


def get_masked_predictions(model, adata, indices, attention_mask, intervention, batch_size):
    """Predict expression for `indices` under an occlusion applied via the chosen intervention.

    Args:
        model: Trained AMICI model.
        adata: AnnData registered with AMICI nearest-neighbor fields.
        indices: Cell indices to predict.
        attention_mask: n_cells x n_neighbors keep/drop mask, or None for the unoccluded baseline.
        intervention: "presoftmax" (mask before softmax, survivors renormalize) or
            "postsoftmax" (zero the softmax output, freed mass stays on the empty token).
        batch_size: Inference batch size.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: Predictions and observed expression for `indices`.
    """
    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
    predictions = []
    observed = []
    batch_start = 0

    for tensors in scdl:
        true_X = tensors[REGISTRY_KEYS.X_KEY].cpu().numpy()
        tensors = {k: v.to(model.device) for k, v in tensors.items()}
        n_batch = tensors[NN_REGISTRY_KEYS.NN_IDX_KEY].shape[0]

        batch_mask = None
        if attention_mask is not None:
            batch_mask = torch.tensor(
                attention_mask[indices[batch_start : batch_start + n_batch]],
                device=model.device,
                dtype=torch.int,
            )

        model.module.reset_hooks()
        if batch_mask is None:
            _, outputs = model.module(tensors, compute_loss=False)
        elif intervention == "presoftmax":
            _, outputs = model.module(
                tensors,
                generative_kwargs={"attention_mask": batch_mask},
                compute_loss=False,
            )
        else:
            hook_fn = partial(_postsoftmax_zero_hook, neighbor_mask=batch_mask)
            outputs = model.module.run_with_hooks(
                tensors,
                fwd_hooks=[("attention_layer.hook_pattern", hook_fn)],
                compute_loss=False,
            )[1]

        predictions.append(outputs["prediction"].detach().cpu().numpy())
        observed.append(true_X)
        batch_start += n_batch

    return np.vstack(predictions), np.vstack(observed)


def per_cell_error(predictions, observed):
    """Mean squared reconstruction error per cell."""
    return ((predictions - observed) ** 2).mean(axis=1)


def measure_empty_token_mass(model, adata, indices, batch_size):
    """Fraction of attention mass the model places on real neighbors vs. the empty token.

    Occlusion can only move as much probability mass as the real neighbors hold, so this bounds
    the effect size any neighbor-occlusion experiment can produce for a given model.

    Returns
    -------
        dict[str, float]: Mean attention mass on real neighbors and on the empty token.
    """
    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
    real_mass = []
    dummy_mass = []

    for tensors in scdl:
        tensors = {k: v.to(model.device) for k, v in tensors.items()}
        model.module.reset_hooks()
        with torch.no_grad():
            _, outputs = model.module(
                tensors,
                generative_kwargs={"return_attention_patterns": True},
                compute_loss=False,
            )
        patterns = outputs["attention_patterns"].detach()  # batch x heads x (neighbors + dummy)
        real_mass.append(patterns[..., :-1].sum(-1).mean().item())
        dummy_mass.append(patterns[..., -1].mean().item())

    return {
        "attention_mass_real_neighbors": float(np.mean(real_mass)),
        "attention_mass_empty_token": float(np.mean(dummy_mass)),
    }


# %% Analysis
def run_matched_occlusion(dataset_name, dataset_config, model, adata, label_codes):
    """Run the matched occlusion comparison for one dataset and return tidy per-cell results."""
    attention = get_attention_matrix(model, adata)
    masks, diagnostics = build_matched_masks(
        adata,
        attention,
        label_codes,
        N_DISTANCE_BINS,
        MATCHED_OCCLUSION_FRACTION,
        MATCHED_ARM_SEED,
    )
    diagnostics["dataset"] = dataset_name

    receiver_subtypes = {config["interaction_subtype"] for config in dataset_config["gt_interactions"].values()}
    receiver_types = {config["receiver"] for config in dataset_config["gt_interactions"].values()}
    receiver_mask = adata.obs[dataset_config["labels_key"]].astype(str).isin(receiver_types).to_numpy()
    indices = np.flatnonzero(receiver_mask)
    is_interacting = adata.obs[dataset_config["subtype_key"]].astype(str).isin(receiver_subtypes).to_numpy()[indices]

    batch_size = int(build_sweep_runs(dataset_config)[0]["batch_size"])
    mass = measure_empty_token_mass(model, adata, indices, batch_size)
    print(
        f"  attention mass on real neighbors: {mass['attention_mass_real_neighbors']:.4f} "
        f"| on empty token: {mass['attention_mass_empty_token']:.4f}"
    )

    records = []
    for intervention in INTERVENTIONS:
        baseline_predictions, observed = get_masked_predictions(model, adata, indices, None, intervention, batch_size)
        baseline_error = per_cell_error(baseline_predictions, observed)

        for arm_name, mask in masks.items():
            arm_predictions, _ = get_masked_predictions(model, adata, indices, mask, intervention, batch_size)
            arm_error = per_cell_error(arm_predictions, observed)
            records.append(
                pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "intervention": intervention,
                        "arm": arm_name,
                        "cell_idx": adata.obs_names[indices],
                        "is_interacting_receiver": is_interacting,
                        "baseline_error": baseline_error,
                        "occluded_error": arm_error,
                        "delta_error": arm_error - baseline_error,
                        "n_occluded": (mask[indices] == 0).sum(axis=1),
                        **mass,
                    }
                )
            )

    return pd.concat(records, ignore_index=True), diagnostics


def summarize_matched_occlusion(results_df):
    """Paired high- vs low-attention comparison per dataset, intervention, and receiver group."""
    summaries = []
    group_cols = ["dataset", "intervention", "is_interacting_receiver"]
    for (dataset_name, intervention, interacting), group in results_df.groupby(group_cols):
        wide = group.pivot_table(index="cell_idx", columns="arm", values="delta_error")
        if not {"high_attention", "low_attention"}.issubset(wide.columns):
            continue
        paired = wide.dropna(subset=["high_attention", "low_attention"])
        if len(paired) < 10:
            continue

        difference = paired["high_attention"] - paired["low_attention"]
        statistic, pvalue = wilcoxon(paired["high_attention"], paired["low_attention"])
        summaries.append(
            {
                "dataset": dataset_name,
                "intervention": intervention,
                "is_interacting_receiver": interacting,
                "n_cells": len(paired),
                "mean_delta_high": paired["high_attention"].mean(),
                "mean_delta_low": paired["low_attention"].mean(),
                "mean_delta_random": paired["random"].mean() if "random" in paired else np.nan,
                "mean_high_minus_low": difference.mean(),
                "median_high_minus_low": difference.median(),
                "frac_cells_high_worse": (difference > 0).mean(),
                "wilcoxon_stat": statistic,
                "wilcoxon_pvalue": pvalue,
            }
        )
    return pd.DataFrame(summaries)


# %% Plotting
def plot_matched_occlusion(results_df, summary_df, diagnostics_df):
    """Plot matched occlusion effects and the matching-quality diagnostics."""
    dataset_names = list(results_df["dataset"].unique())
    arm_order = ["high_attention", "random", "low_attention"]
    arm_colors = {"high_attention": "#D55E00", "random": "#999999", "low_attention": "#0072B2"}

    fig, axes = plt.subplots(len(dataset_names), 3, figsize=(17, 4.6 * len(dataset_names)), squeeze=False)
    for row_idx, dataset_name in enumerate(dataset_names):
        dataset_results = results_df[results_df["dataset"] == dataset_name]

        ax = axes[row_idx][0]
        interacting_labels = [True, False]
        bar_width = 0.25
        positions = np.arange(len(interacting_labels))
        for arm_offset, arm_name in enumerate(arm_order):
            arm_results = dataset_results[
                (dataset_results["arm"] == arm_name) & (dataset_results["intervention"] == INTERVENTIONS[0])
            ]
            means = [
                arm_results[arm_results["is_interacting_receiver"] == flag]["delta_error"].mean()
                for flag in interacting_labels
            ]
            errors = [
                arm_results[arm_results["is_interacting_receiver"] == flag]["delta_error"].sem()
                for flag in interacting_labels
            ]
            ax.bar(
                positions + (arm_offset - 1) * bar_width,
                means,
                bar_width,
                yerr=errors,
                label=arm_name.replace("_", " "),
                color=arm_colors[arm_name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(["interacting\nreceivers", "neutral\nreceivers"])
        ax.set_ylabel("Δ reconstruction error vs. baseline")
        ax.set_title(f"{dataset_name}\nmatched occlusion effect ({INTERVENTIONS[0]})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[row_idx][1]
        group_labels = []
        for group_offset, (flag, label) in enumerate([(True, "interacting"), (False, "neutral")]):
            subset = dataset_results[
                (dataset_results["intervention"] == INTERVENTIONS[0])
                & (dataset_results["is_interacting_receiver"] == flag)
            ]
            wide = subset.pivot_table(index="cell_idx", columns="arm", values="delta_error").dropna(
                subset=["high_attention", "low_attention"]
            )
            if wide.empty:
                continue
            difference = wide["high_attention"] - wide["low_attention"]
            ax.violinplot(difference.to_numpy(), positions=[group_offset], showmedians=True, widths=0.7)
            group_labels.append((group_offset, f"{label}\n(n={len(difference)})"))
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks([position for position, _ in group_labels])
        ax.set_xticklabels([label for _, label in group_labels])
        ax.set_ylabel("Δerror(high) − Δerror(low), per cell")
        ax.set_title(f"{dataset_name}\npaired difference by receiver group")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[row_idx][2]
        dataset_diagnostics = diagnostics_df[diagnostics_df["dataset"] == dataset_name]
        for arm_name in arm_order:
            distances = dataset_diagnostics[dataset_diagnostics["arm"] == arm_name]["distance"]
            ax.hist(
                distances,
                bins=40,
                histtype="step",
                density=True,
                label=f"{arm_name.replace('_', ' ')} (mean={distances.mean():.1f})",
                color=arm_colors[arm_name],
            )
        ax.set_xlabel("Distance of occluded neighbor")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset_name}\nmatching check: occluded-neighbor distances")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Occlusion of high- vs. distance- and cell-type-matched low-attention neighbors",
        fontsize=14,
    )
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(figure_dir, f"matched_neighbor_occlusion.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% Main
def main():
    """Train/load one model per dataset and run the matched occlusion comparison."""
    select_gpu()
    random.seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    scvi.settings.seed = DEFAULT_RANDOM_SEED

    all_results = []
    all_diagnostics = []
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"=== {dataset_name} ===")
        ensure_dataset(dataset_name, dataset_config)
        adata = sc.read_h5ad(dataset_path(dataset_name, dataset_config))
        adata.obs_names_make_unique()

        model = train_or_load_model(dataset_name, adata, dataset_config)
        label_codes = pd.Categorical(adata.obs[dataset_config["labels_key"]]).codes.astype(np.int64)

        results_df, diagnostics_df = run_matched_occlusion(dataset_name, dataset_config, model, adata, label_codes)
        all_results.append(results_df)
        all_diagnostics.append(diagnostics_df)

    results_df = pd.concat(all_results, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)
    summary_df = summarize_matched_occlusion(results_df)

    results_df.to_csv(os.path.join(results_dir, "matched_occlusion_per_cell.csv"), index=False)
    diagnostics_df.to_csv(os.path.join(results_dir, "matched_occlusion_diagnostics.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "matched_occlusion_summary.csv"), index=False)

    print("\n=== Matched occlusion summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Matching quality (occluded neighbors per arm) ===")
    print(
        diagnostics_df.groupby(["dataset", "arm"])
        .agg(n_occluded=("distance", "size"), mean_distance=("distance", "mean"), mean_attention=("attention", "mean"))
        .to_string()
    )

    plot_matched_occlusion(results_df, summary_df, diagnostics_df)
    print(f"\nFigures written to {figure_dir}")


if __name__ == "__main__":
    main()
