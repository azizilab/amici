# %% Import libraries
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scvi
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENSITIVITY_DIR = os.path.dirname(SCRIPT_DIR)
BENCHMARK_DIR = os.path.abspath(os.path.join(SENSITIVITY_DIR, ".."))
sys.path.insert(0, BENCHMARK_DIR)
sys.path.insert(0, SENSITIVITY_DIR)

from amici_benchmark_utils import (  # noqa: E402
    get_amici_gene_task_scores,
    get_amici_neighbor_interaction_scores,
    get_amici_receiver_subtype_scores,
)
from benchmark_utils import (  # noqa: E402
    get_interaction_gt_neighbor_classes,
    get_model_precision_recall_auc,
    get_receiver_gt_ranked_genes,
)
from generate_dataset import (  # noqa: E402
    _check_positive_degs,
    _create_train_test_split,
    _preprocess,
    _subcluster,
)
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402

# %% Config
RUN_NAME = "two_phase_amici_sweep"
DATASET_SEED = 42
DATASET_NAME = "3ct_dataset_1int_2phase"
GCS_DATASET_URI = "gs://justin-amici/semisyn/3ct_1int_2phase/3ct_dataset_1int_2phase.h5ad"

LABELS_KEY = "leiden"
SUBTYPE_KEY = "subtype"
INNER_RADIUS = 10
OUTER_RADIUS = 25
ATTENTION_THRESHOLD = 0.1

GT_INTERACTIONS = {
    "interaction_inner": {
        "sender": "3",
        "receiver": "0",
        "interaction_subtype": "0_sub1",
        "neutral_subtype": "0_sub0",
        "length_scale": INNER_RADIUS,
    },
    "interaction_outer": {
        "sender": "3",
        "receiver": "0",
        "interaction_subtype": "0_sub2",
        "neutral_subtype": "0_sub0",
        "length_scale": OUTER_RADIUS,
    },
}

SWEEP_PARAMS = {
    "end_attention_penalty": [1e-3, 1e-4, 1e-5],
    "attention_penalty_schedule": [[10, 30]],
    "seed": [21, 22, 33, 38],
    "value_l1_penalty_coef": [1e-6, 1e-5],
    "batch_size": [256],
    "lr": [1e-3],
    "n_neighbors": [50],
    "penalty_flavor_params": ["linear"],
    "n_heads": [8],
}

EXP_DEFAULTS = {
    "epochs": 400,
    "early_stopping": True,
    "early_stopping_monitor": "elbo_validation",
    "early_stopping_patience": 20,
}

# %% Paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models", RUN_NAME)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", RUN_NAME)
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures", RUN_NAME)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# %% Dataset generation
def set_all_seeds(seed):
    """Set all random seeds used by dataset generation and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def build_two_phase_interaction_df():
    """Build interaction metadata for one sender-receiver pair with two distance phases."""
    return pd.DataFrame(
        [
            {"receptor_cell": "3", "receptor_subtype": "3_sub0", "interaction_type": "neutral"},
            {"receptor_cell": "0", "receptor_subtype": "0_sub0", "interaction_type": "neutral"},
            {"receptor_cell": "2", "receptor_subtype": "2_sub0", "interaction_type": "neutral"},
            {
                "receptor_cell": "0",
                "sender_cell": "3",
                "receptor_subtype": "0_sub1",
                "radius_of_effect": INNER_RADIUS,
                "interaction_type": "interaction",
            },
            {
                "receptor_cell": "0",
                "sender_cell": "3",
                "receptor_subtype": "0_sub2",
                "radius_of_effect": OUTER_RADIUS,
                "interaction_type": "interaction",
            },
        ]
    )


def generate_two_phase_spatial_data(adata, num_cells=20000, rect_length=2000, rect_width=1000):
    """Generate triangular 3-cell-type data with two receiver distance phases.

    Args:
        adata: Preprocessed AnnData with `leiden` and `subtype`.
        num_cells: Number of semi-synthetic cells to sample.
        rect_length: Width of the spatial domain.
        rect_width: Height of the spatial domain.

    Returns
    -------
        AnnData with semi-synthetic spatial coordinates and two-phase subtypes.
    """
    ct1, ct2, ct3 = "3", "0", "2"
    gradient_width = 700
    positions = np.random.uniform(0, (rect_length, rect_width), size=(num_cells, 2))

    mid_x = rect_length / 2
    mid_y = rect_width / 2
    gradient_x_start = mid_x - gradient_width / 2
    gradient_x_end = mid_x + gradient_width / 2
    gradient_y_start = mid_y - gradient_width / 2
    gradient_y_end = mid_y + gradient_width / 2

    cell_types = []
    for x, y in positions:
        x_prob = (
            np.clip((x - gradient_x_start) / gradient_width, 0, 1) if gradient_x_start <= x <= gradient_x_end else 1
        )
        y_prob = (
            np.clip((y - gradient_y_start) / gradient_width, 0, 1) if gradient_y_start <= y <= gradient_y_end else 1
        )

        if x < mid_x:
            if y < mid_y:
                cell_types.append(ct1 if np.random.rand() < x_prob else ct3)
            else:
                cell_types.append(ct2 if np.random.rand() < y_prob else ct1)
        elif y < mid_y:
            cell_types.append(ct3 if np.random.rand() < x_prob else ct1)
        else:
            cell_types.append(ct2 if np.random.rand() < y_prob else ct3)

    spatial_data = pd.DataFrame(
        {"Cell_ID": range(1, num_cells + 1), "X": positions[:, 0], "Y": positions[:, 1], "Cell_Type": cell_types}
    )
    neutral_types = {"0": "0_sub0", "2": "2_sub0", "3": "3_sub0"}
    spatial_data["Subtype"] = spatial_data["Cell_Type"].map(neutral_types)

    receiver_mask = spatial_data["Cell_Type"] == "0"
    sender_cells = spatial_data[spatial_data["Cell_Type"] == "3"]
    sender_xy = sender_cells[["X", "Y"]].to_numpy()
    receiver_xy = spatial_data.loc[receiver_mask, ["X", "Y"]].to_numpy()
    min_dist = np.sqrt(((receiver_xy[:, None, :] - sender_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

    receiver_indices = spatial_data.index[receiver_mask]
    spatial_data.loc[receiver_indices[min_dist <= INNER_RADIUS], "Subtype"] = "0_sub1"
    spatial_data.loc[receiver_indices[(min_dist > INNER_RADIUS) & (min_dist <= OUTER_RADIUS)], "Subtype"] = "0_sub2"

    sampled_adatas = []
    for subtype in spatial_data["Subtype"].unique():
        subtype_cells = adata.obs[adata.obs["subtype"] == subtype].index
        num_samples = (spatial_data["Subtype"] == subtype).sum()
        sampled_indices = np.random.choice(subtype_cells, num_samples, replace=True)
        sampled_spatial = spatial_data.loc[spatial_data["Subtype"] == subtype, ["X", "Y"]]
        sampled_adata = adata[sampled_indices].copy()
        sampled_spatial.index = sampled_adata.obs_names
        sampled_adata.obsm["spatial"] = sampled_spatial
        sampled_adatas.append(sampled_adata)

    semisyn_adata = sc.concat(sampled_adatas, axis=0)
    semisyn_adata.obs["leiden"] = semisyn_adata.obs["leiden"].astype("category")
    semisyn_adata.obs["subtype"] = semisyn_adata.obs["subtype"].astype("category")
    return semisyn_adata


def generate_two_phase_dataset(output_path):
    """Generate the two-phase synthetic dataset from PBMC source data."""
    if os.path.exists("/tmp/pbmc_ad.h5ad"):
        pbmc_ad = sc.read_h5ad("/tmp/pbmc_ad.h5ad")
    else:
        pbmc_ad = scvi.data.dataset_10x(dataset_name="fresh_68k_pbmc_donor_a")
        pbmc_ad.write_h5ad("/tmp/pbmc_ad.h5ad")

    interaction_df = build_two_phase_interaction_df()
    positive_deg_check = False
    while not positive_deg_check:
        adata_pp = _preprocess(pbmc_ad, n_hvgs=500)
        adata_pp = _subcluster(adata_pp, n_subcluster_per_cluster=3, n_genes_for_subclustering=50)
        adata_pp, positive_deg_check = _check_positive_degs(adata_pp, interaction_df)
        if not positive_deg_check:
            scvi.settings.seed = np.random.randint(0, 1000)

    adata = generate_two_phase_spatial_data(adata_pp)
    adata = _create_train_test_split(adata)
    adata.obs_names_make_unique()
    adata.write_h5ad(output_path)


def ensure_two_phase_dataset():
    """Load the archived two-phase dataset, copying or generating it if needed."""
    adata_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.h5ad")
    if not os.path.exists(adata_path):
        try:
            subprocess.run(["gsutil", "cp", GCS_DATASET_URI, adata_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            set_all_seeds(DATASET_SEED)
            generate_two_phase_dataset(adata_path)

    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()
    return adata, adata_path


# %% Sweep helpers
def build_run_configs():
    """Build AMICI candidate training configs."""
    run_configs = []
    for end_val, schedule, train_seed, value_l1, batch_size, lr, n_neighbors, flavor, n_heads in itertools.product(
        SWEEP_PARAMS["end_attention_penalty"],
        SWEEP_PARAMS["attention_penalty_schedule"],
        SWEEP_PARAMS["seed"],
        SWEEP_PARAMS["value_l1_penalty_coef"],
        SWEEP_PARAMS["batch_size"],
        SWEEP_PARAMS["lr"],
        SWEEP_PARAMS["n_neighbors"],
        SWEEP_PARAMS["penalty_flavor_params"],
        SWEEP_PARAMS["n_heads"],
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
            }
        )
    return run_configs


def cache_matches_run(result, run):
    """Return whether a cached result matches this sweep config."""
    return all(result.get(key) == run.get(key) for key in run)


def build_task_ground_truth(adata):
    """Compute task ground-truth labels for the two-phase dataset."""
    gt_gene_scores = []
    for interaction_name, interaction_config in GT_INTERACTIONS.items():
        scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            SUBTYPE_KEY,
        )
        scores["interaction"] = interaction_name
        gt_gene_scores.append(scores)

    combined_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in GT_INTERACTIONS.values():
        combined_mask |= adata.obs[SUBTYPE_KEY] == interaction_config["interaction_subtype"]

    gt_receiver_classes_df = pd.DataFrame({"cell_idx": adata.obs_names, "class": combined_mask.astype(float)})
    gt_neighbor_classes_df = get_interaction_gt_neighbor_classes(adata, GT_INTERACTIONS, LABELS_KEY)
    return pd.concat(gt_gene_scores, ignore_index=True), gt_neighbor_classes_df, gt_receiver_classes_df


def evaluate_tasks(model, adata):
    """Evaluate gene, neighbor-interaction, and receiver-subtype AUPRC tasks."""
    gt_gene_scores_df, gt_neighbor_classes_df, gt_receiver_classes_df = build_task_ground_truth(adata)
    records = []

    try:
        all_gene_scores = []
        for interaction_name, interaction_config in GT_INTERACTIONS.items():
            gene_scores = get_amici_gene_task_scores(
                model, adata, interaction_config["sender"], interaction_config["receiver"]
            )
            gene_scores["interaction"] = interaction_name
            all_gene_scores.append(gene_scores)
        gene_scores_df = pd.concat(all_gene_scores, ignore_index=True)
        _, _, auprc = get_model_precision_recall_auc(
            gene_scores_df,
            gt_gene_scores_df,
            merge_cols=["gene", "interaction"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        records.append({"task": "Gene", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        records.append({"task": "Gene", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    try:
        neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata)
        _, _, auprc = get_model_precision_recall_auc(
            neighbor_scores_df,
            gt_neighbor_classes_df,
            merge_cols=["cell_idx", "neighbor_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        records.append({"task": "Neighbor interaction", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        records.append({"task": "Neighbor interaction", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    try:
        receiver_scores_df = get_amici_receiver_subtype_scores(model, adata)
        _, _, auprc = get_model_precision_recall_auc(
            receiver_scores_df,
            gt_receiver_classes_df,
            merge_cols=["cell_idx"],
            scores_col="amici_scores",
            gt_class_col="class",
        )
        records.append({"task": "Receiver subtype", "auprc": auprc, "status": "success", "error": None})
    except Exception as exc:  # noqa: BLE001
        records.append({"task": "Receiver subtype", "auprc": np.nan, "status": "failed", "error": repr(exc)})

    return pd.DataFrame(records)


def train_or_load_run(adata, run, run_idx):
    """Train one AMICI run or load its cached result."""
    run_dir = os.path.join(MODEL_DIR, f"run_{run_idx}_seed_{run['train_seed']}")
    result_path = os.path.join(MODEL_DIR, f"run_{run_idx}_seed_{run['train_seed']}_results.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        if cache_matches_run(result, {**run, "run_idx": run_idx}):
            return result

    result = {**run, "run_idx": run_idx, "model_path": run_dir}
    try:
        pl.seed_everything(run["train_seed"])
        adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
        AMICI.setup_anndata(
            adata_train, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=int(run["n_neighbors"])
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

        AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=int(run["n_neighbors"]))
        model.save(run_dir, overwrite=True)
        eval_indices = np.where(adata.obs["train_test_split"] == "test")[0]
        test_loss = (
            model.get_reconstruction_error(adata, indices=eval_indices, batch_size=int(run["batch_size"]))[
                "reconstruction_loss"
            ]
            .detach()
            .cpu()
            .numpy()
            .item()
        )
        if not np.isfinite(test_loss):
            raise ValueError(f"Non-finite test loss: {test_loss}")
        result.update({"status": "success", "test_loss": test_loss, "error": None})
    except Exception as exc:  # noqa: BLE001
        result.update(
            {"status": "failed", "test_loss": np.nan, "error": repr(exc), "traceback": traceback.format_exc()}
        )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# %% Plotting
def plot_task_scores(task_scores_df):
    """Save a compact task AUPRC bar plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(task_scores_df["task"], task_scores_df["auprc"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUPRC")
    ax.set_title("AMICI on Two-Phase Synthetic Dataset")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "two_phase_task_auprc.png"), dpi=300)
    plt.close(fig)


def compute_length_scale_estimates(adata, model, best_result):
    """Compute best-head AMICI length scales for the two distance phases."""
    explained_variance = model.get_expl_variance_scores(adata=adata)
    length_scale_cache = {}
    estimate_records = []
    sample_records = []

    for interaction_name, interaction_config in GT_INTERACTIONS.items():
        receiver_type = interaction_config["receiver"]
        sender_type = interaction_config["sender"]
        interaction_label = f"{sender_type} -> {receiver_type}: {interaction_config['interaction_subtype']}"
        cache_key = (receiver_type, sender_type)
        if cache_key not in length_scale_cache:
            print(f"Computing length scales for {sender_type} -> {receiver_type}", flush=True)
            counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
                adata=adata,
                cell_type=receiver_type,
            )
            best_head = explained_variance.compute_max_explained_variance_head(cell_type=receiver_type)
            length_scale_df = counterfactual_attention_patterns._calculate_length_scales(
                head_idxs=[best_head],
                sender_types=[sender_type],
                attention_threshold=ATTENTION_THRESHOLD,
            )
            length_scale_cache[cache_key] = (best_head, length_scale_df)
        best_head, cached_length_scale_df = length_scale_cache[cache_key]
        length_scale_df = cached_length_scale_df.copy()

        estimate_records.append(
            {
                "interaction": interaction_name,
                "interaction_label": interaction_label,
                "receiver_type": receiver_type,
                "sender_type": sender_type,
                "interaction_subtype": interaction_config["interaction_subtype"],
                "head_idx": best_head,
                "mean_length_scale": length_scale_df["length_scale"].mean(),
                "median_length_scale": length_scale_df["length_scale"].median(),
                "n_length_scale_samples": len(length_scale_df),
                "gt_length_scale": interaction_config["length_scale"],
                "attention_threshold": ATTENTION_THRESHOLD,
                "best_end_val": best_result["end_val"],
                "best_value_l1": best_result["value_l1"],
                "best_train_seed": best_result["train_seed"],
                "best_test_loss": best_result["test_loss"],
                "best_model_path": best_result["model_path"],
            }
        )

        length_scale_df["interaction"] = interaction_name
        length_scale_df["interaction_label"] = interaction_label
        length_scale_df["receiver_type"] = receiver_type
        length_scale_df["sender_type"] = sender_type
        length_scale_df["interaction_subtype"] = interaction_config["interaction_subtype"]
        length_scale_df["gt_length_scale"] = interaction_config["length_scale"]
        length_scale_df["attention_threshold"] = ATTENTION_THRESHOLD
        sample_records.append(length_scale_df)

    return pd.DataFrame(estimate_records), pd.concat(sample_records, ignore_index=True)


def plot_length_scale_distributions(samples_df, estimates_df):
    """Save length-scale distributions for the two-phase synthetic dataset."""
    import matplotlib.pyplot as plt

    plot_interactions = (
        estimates_df[["interaction", "interaction_label", "gt_length_scale", "mean_length_scale"]]
        .drop_duplicates()
        .sort_values("gt_length_scale")
    )
    positions = np.arange(1, len(plot_interactions) + 1)
    violin_data = [
        samples_df.loc[samples_df["interaction"] == interaction, "length_scale"].dropna().to_numpy()
        for interaction in plot_interactions["interaction"]
    ]
    gt_values = plot_interactions["gt_length_scale"].astype(float).to_numpy()
    mean_values = plot_interactions["mean_length_scale"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    violins = ax.violinplot(
        violin_data,
        positions=positions,
        vert=False,
        widths=0.75,
        showmeans=False,
        showextrema=True,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#4C78A8")
        body.set_alpha(0.45)
    for part in ("cmins", "cmaxes", "cbars"):
        violins[part].set_color("#4C78A8")
        violins[part].set_linewidth(1.4)

    ax.scatter(mean_values, positions, marker="o", color="#4C78A8", s=70, label="Mean inferred")
    ax.scatter(gt_values, positions, marker="x", color="black", s=120, linewidths=2.4, label="Ground truth")
    axis_values = np.concatenate([samples_df["length_scale"].dropna().to_numpy(), gt_values, mean_values])
    axis_min = np.nanmin(axis_values)
    axis_max = np.nanmax(axis_values)
    padding = 0.06 * (axis_max - axis_min) if axis_max > axis_min else 1.0
    ax.set_xlim(axis_min - padding, axis_max + padding)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_interactions["interaction_label"])
    ax.set_xlabel("Length scale at attention threshold 0.1")
    ax.set_ylabel("Interaction phase")
    ax.set_title("Two-phase synthetic AMICI length-scale recovery")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGURE_DIR, f"two_phase_length_scale_distributions.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


# %% Main
def main():
    """Run the two-phase AMICI sweep when this notebook-style script is executed."""
    select_gpu()
    adata, adata_path = ensure_two_phase_dataset()
    run_configs = build_run_configs()
    print(f"Loaded {adata_path}: {adata.shape}")
    print(f"Running {len(run_configs)} candidate AMICI configs")

    run_results = []
    for run_idx, run_config in enumerate(run_configs):
        print(f"Run {run_idx + 1}/{len(run_configs)}: {run_config}")
        run_results.append(train_or_load_run(adata, run_config, run_idx))

    run_results_df = pd.DataFrame(run_results)
    run_results_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_run_results.csv"), index=False)

    successful_runs = [result for result in run_results if result.get("status") == "success"]
    if not successful_runs:
        raise RuntimeError("No successful AMICI runs completed.")

    best_result = min(successful_runs, key=lambda result: result["test_loss"])
    best_model_path = os.path.join(MODEL_DIR, "best_model")
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(best_result["model_path"], best_model_path)
    best_result = {**best_result, "selection_metric": "test_loss", "n_successful_runs": len(successful_runs)}
    with open(os.path.join(RESULTS_DIR, "best_model_params.json"), "w") as f:
        json.dump(best_result, f, indent=2)

    AMICI.setup_anndata(
        adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=int(best_result["n_neighbors"])
    )
    model = AMICI.load(best_model_path, adata=adata)
    task_scores_df = evaluate_tasks(model, adata)
    task_scores_df["dataset"] = DATASET_NAME
    task_scores_df["dataset_path"] = adata_path
    task_scores_df["best_test_loss"] = best_result["test_loss"]
    task_scores_df["best_train_seed"] = best_result["train_seed"]
    task_scores_df["best_end_val"] = best_result["end_val"]
    task_scores_df["best_value_l1"] = best_result["value_l1"]
    task_scores_df["n_successful_runs"] = len(successful_runs)
    task_scores_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_task_scores.csv"), index=False)
    plot_task_scores(task_scores_df)

    length_scale_estimates_df, length_scale_samples_df = compute_length_scale_estimates(adata, model, best_result)
    length_scale_estimates_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_length_scale_estimates.csv"), index=False)
    length_scale_samples_df.to_csv(os.path.join(RESULTS_DIR, "two_phase_length_scale_samples.csv"), index=False)
    plot_length_scale_distributions(length_scale_samples_df, length_scale_estimates_df)


if __name__ == "__main__":
    main()
