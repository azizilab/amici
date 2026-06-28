# %% Import libraries
import itertools
import os
import random
import shutil
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp_sparse
import scvi
import torch
from sklearn.metrics import PrecisionRecallDisplay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
    _preprocess,
    _subcluster,
)
from generate_dataset import (  # noqa: E402
    _create_interaction_df as create_synthetic_interaction_df,
)
from generate_dataset import (  # noqa: E402
    _create_train_test_split as create_synthetic_train_test_split,
)
from generate_realistic_dataset import (  # noqa: E402
    GT_INTERACTIONS as REALISTIC_GT_INTERACTION_LIST,
)
from generate_realistic_dataset import (  # noqa: E402
    _create_interaction_df as create_realistic_interaction_df,
)
from generate_realistic_dataset import (  # noqa: E402
    _create_train_test_split as create_realistic_train_test_split,
)
from generate_realistic_dataset import (  # noqa: E402
    _determine_spatial_subtypes,
    _load_flex_data,
    _load_xenium_spatial,
    _preprocess_flex,
    _subcluster_by_celltype,
)
from gpu_utils import select_gpu  # noqa: E402

from amici import AMICI  # noqa: E402
from amici.callbacks import AttentionPenaltyMonitor  # noqa: E402
from amici.tools import is_count_data  # noqa: E402

# %% Config
DATASET_SEED = 42
RUN_NAME = "neutral_sampling_negative_control"

SYNTHETIC_SWEEP_PARAMS = {
    "end_attention_penalty": [1e-3, 1e-4],
    "value_l1_penalty_coef": [1e-5, 1e-4],
    "attention_penalty_schedule": [[10, 30]],
    "seed": [22, 42],
    "batch_size": [128],
    "lr": [1e-3],
    "n_neighbors": [50],
    "penalty_flavor_params": ["linear"],
    "n_heads": [8],
}

REALISTIC_SWEEP_PARAMS = {
    "end_attention_penalty": [1e-5, 1e-4],
    "value_l1_penalty_coef": [1e-5, 1e-4],
    "attention_penalty_schedule": [[15, 30]],
    "seed": [22, 42],
    "batch_size": [256],
    "lr": [1e-3],
    "n_neighbors": [50],
    "penalty_flavor_params": ["linear"],
    "n_heads": [8, 10],
}
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
    "3ct_dataset_2way_neutral_control": {
        "kind": "synthetic",
        "labels_key": "leiden",
        "subtype_key": "subtype",
        "gt_interactions": SYNTHETIC_INTERACTIONS,
        "sweep_params": SYNTHETIC_SWEEP_PARAMS,
    },
    "breast_cancer_neutral_control": {
        "kind": "realistic",
        "labels_key": "cell_type",
        "subtype_key": "subtype",
        "gt_interactions": REALISTIC_INTERACTIONS,
        "sweep_params": REALISTIC_SWEEP_PARAMS,
        "flex_h5_path": "data/GSM7782698_count_raw_feature_bc_matrix.h5",
        "annot_path": "data/41467_2023_43458_MOESM4_ESM.xlsx",
        "xenium_path": "data/xenium_rep1_io.h5ad",
        "base_h5ad_path": "data/breast_cancer_42.h5ad",
        "scvi_model_dir": "data/scvi_model_neutral_control",
        "spatial_bounds": ((0, 3200), (0, 2800)),
        "n_cv_folds": 3,
    },
}

# %% Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
saved_models_dir = os.path.join(base_dir, "saved_models", RUN_NAME)
figure_dir = os.path.join(base_dir, "figures")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)


# %% Helper functions
def benchmark_path(path):
    """Resolve benchmark config paths relative to the benchmarks directory."""
    return path if os.path.isabs(path) else os.path.join(benchmark_dir, path)


def dataset_path(dataset_name):
    """Return the cached h5ad path for a negative-control dataset."""
    return os.path.join(data_dir, f"{dataset_name}_{DATASET_SEED}.h5ad")


def build_run_configs(sweep_params):
    """Build a small AMICI sweep from Snakemake-style parameter lists."""
    keys = [
        "end_attention_penalty",
        "value_l1_penalty_coef",
        "seed",
        "attention_penalty_schedule",
        "batch_size",
        "n_heads",
        "lr",
        "n_neighbors",
        "penalty_flavor_params",
    ]
    runs = []
    for end_val, value_l1, seed, schedule, batch_size, n_heads, lr, n_neighbors, flavor in itertools.product(
        *(sweep_params[k] for k in keys)
    ):
        runs.append(
            {
                "end_val": end_val,
                "value_l1": value_l1,
                "seed": seed,
                "epoch_start": schedule[0],
                "epoch_end": schedule[1],
                "batch_size": batch_size,
                "n_heads": n_heads,
                "lr": lr,
                "n_neighbors": n_neighbors,
                "flavor": flavor,
            }
        )
    return runs


def neutral_source_map(interaction_df):
    """Map interaction subtype labels back to the matching neutral subtype."""
    neutral_types = dict(
        zip(
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_cell"],
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_subtype"],
            strict=False,
        )
    )
    return {
        row["receptor_subtype"]: neutral_types[row["receptor_cell"]]
        for _, row in interaction_df[interaction_df["interaction_type"] == "interaction"].iterrows()
    }


def apply_spatial_rules(spatial_data, interaction_df):
    """Label rule-positive receivers as interaction subtypes while preserving neutral defaults."""
    neutral_types = dict(
        zip(
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_cell"],
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_subtype"],
            strict=False,
        )
    )
    spatial_data["Subtype"] = spatial_data["Cell_Type"].map(neutral_types)

    for rule in interaction_df[interaction_df["interaction_type"] == "interaction"].to_dict(orient="records"):
        receptor_cells = spatial_data[
            (spatial_data["Cell_Type"] == rule["receptor_cell"])
            & (spatial_data["Subtype"] == neutral_types[rule["receptor_cell"]])
        ]
        sender_cells = spatial_data[spatial_data["Cell_Type"] == rule["sender_cell"]]
        for _, receptor_cell in receptor_cells.iterrows():
            distances = np.sqrt(
                (sender_cells["X"] - receptor_cell["X"]) ** 2 + (sender_cells["Y"] - receptor_cell["Y"]) ** 2
            )
            if (distances <= rule["radius_of_effect"]).any():
                spatial_data.loc[spatial_data["Cell_ID"] == receptor_cell["Cell_ID"], "Subtype"] = rule[
                    "receptor_subtype"
                ]
    return spatial_data


def synthetic_layout(interaction_df, num_cells=20000, rect_length=2000, rect_width=1000):
    """Create the original three-cell-type semisynthetic spatial layout."""
    ct1, ct2, ct3 = "3", "0", "2"
    gradient_width = 700
    positions = np.random.uniform(0, (rect_length, rect_width), size=(num_cells, 2))
    mid_x, mid_y = rect_length / 2, rect_width / 2
    gx0, gx1 = mid_x - gradient_width / 2, mid_x + gradient_width / 2
    gy0, gy1 = mid_y - gradient_width / 2, mid_y + gradient_width / 2

    cell_types = []
    for x, y in positions:
        x_prob = np.clip((x - gx0) / gradient_width, 0, 1) if gx0 <= x <= gx1 else 1
        y_prob = np.clip((y - gy0) / gradient_width, 0, 1) if gy0 <= y <= gy1 else 1
        if x < mid_x and y < mid_y:
            cell_types.append(ct1 if np.random.rand() < x_prob else ct3)
        elif x < mid_x:
            cell_types.append(ct2 if np.random.rand() < y_prob else ct1)
        elif y < mid_y:
            cell_types.append(ct3 if np.random.rand() < x_prob else ct1)
        else:
            cell_types.append(ct2 if np.random.rand() < y_prob else ct3)

    spatial_data = pd.DataFrame(
        {
            "Cell_ID": range(1, num_cells + 1),
            "X": positions[:, 0],
            "Y": positions[:, 1],
            "Cell_Type": cell_types,
        }
    )
    return apply_spatial_rules(spatial_data, interaction_df)


def sample_existing_cells_by_neutral_subtype(adata, spatial_data, subtype_source_map):
    """Sample existing single-cell profiles, forcing interaction labels to neutral source pools."""
    sampled_adatas = []
    for target_subtype in spatial_data["Subtype"].unique():
        source_subtype = subtype_source_map.get(target_subtype, target_subtype)
        source_cells = adata.obs[adata.obs["subtype"] == source_subtype].index
        subtype_mask = spatial_data["Subtype"] == target_subtype

        sampled_indices = np.random.choice(source_cells, subtype_mask.sum(), replace=True)
        sampled_adata = adata[sampled_indices].copy()
        sampled_adata.obs["source_subtype"] = source_subtype
        sampled_adata.obs["subtype"] = target_subtype
        sampled_spatial = spatial_data.loc[subtype_mask, ["X", "Y"]].copy()
        sampled_spatial.index = sampled_adata.obs_names
        sampled_adata.obsm["spatial"] = sampled_spatial
        sampled_adatas.append(sampled_adata)

    sampled = sc.concat(sampled_adatas, axis=0)
    sampled.obs_names_make_unique()
    return sampled


def sample_scvi_by_neutral_subtype(flex_adata, spatial_data, scvi_model, rng):
    """Generate realistic profiles, forcing interaction labels to neutral Flex subtype pools."""
    pools = {
        flex_subtype: np.where(flex_adata.obs["subtype"] == flex_subtype)[0]
        for flex_subtype in flex_adata.obs["subtype"].unique()
    }
    sampled_adatas = []

    for target_subtype in spatial_data["Subtype"].unique():
        cell_type = target_subtype.rsplit("_sub", 1)[0]
        source_subtype = f"{cell_type}_sub0" if target_subtype.endswith("_sub1") else target_subtype
        if source_subtype not in pools:
            print(f"Warning: No Flex pool for {source_subtype}; skipping {target_subtype}.")
            continue

        subtype_mask = spatial_data["Subtype"] == target_subtype
        source_adata = flex_adata[rng.choice(pools[source_subtype], subtype_mask.sum(), replace=True)].copy()
        generated_counts = scvi_model.posterior_predictive_sample(source_adata, n_samples=1)
        if hasattr(generated_counts, "todense"):
            generated_counts = np.array(generated_counts.todense())
        if generated_counts.ndim == 3:
            generated_counts = generated_counts[:, 0, :]

        sampled_adata = ad.AnnData(X=sp_sparse.csr_matrix(generated_counts), var=flex_adata.var.copy())
        if is_count_data(sampled_adata.X):
            sampled_adata.layers["counts"] = sampled_adata.X.copy()
            sc.pp.normalize_total(sampled_adata)
            sc.pp.log1p(sampled_adata)

        sampled_adata.obsm["spatial"] = spatial_data.loc[subtype_mask, ["X", "Y"]].values
        sampled_adata.obs["spatial_id"] = spatial_data.loc[subtype_mask].index.values
        sampled_adata.obs["cell_type"] = spatial_data.loc[subtype_mask, "Cell_Type"].values
        sampled_adata.obs["subtype"] = target_subtype
        sampled_adata.obs["source_subtype"] = source_subtype
        sampled_adatas.append(sampled_adata)

    sampled = sc.concat(sampled_adatas, axis=0)
    sampled.obs_names_make_unique()
    return sampled


def sample_cached_realistic_by_neutral_subtype(base_adata, rng):
    """Use a cached realistic semisynthetic dataset as the source pool for a neutral control."""
    source_pools = {
        subtype: np.where(base_adata.obs["subtype"].astype(str).values == subtype)[0]
        for subtype in base_adata.obs["subtype"].astype(str).unique()
    }
    sampled_adatas = []

    for target_subtype in base_adata.obs["subtype"].astype(str).unique():
        cell_type = target_subtype.rsplit("_sub", 1)[0]
        source_subtype = f"{cell_type}_sub0" if target_subtype.endswith("_sub1") else target_subtype
        if source_subtype not in source_pools:
            print(f"Warning: No cached source pool for {source_subtype}; skipping {target_subtype}.")
            continue

        target_mask = base_adata.obs["subtype"].astype(str).values == target_subtype
        target_obs = base_adata.obs.loc[target_mask].copy()
        sampled_indices = rng.choice(source_pools[source_subtype], target_mask.sum(), replace=True)
        sampled_adata = base_adata[sampled_indices].copy()

        sampled_adata.obs["cell_type"] = target_obs["cell_type"].values
        sampled_adata.obs["subtype"] = target_subtype
        sampled_adata.obs["source_subtype"] = source_subtype
        if "train_test_split" in target_obs:
            sampled_adata.obs["train_test_split"] = target_obs["train_test_split"].values
        if "cv_fold" in target_obs:
            sampled_adata.obs["cv_fold"] = target_obs["cv_fold"].values
        sampled_adata.obsm["spatial"] = np.asarray(base_adata.obsm["spatial"])[target_mask]
        sampled_adatas.append(sampled_adata)

    sampled = sc.concat(sampled_adatas, axis=0)
    sampled.obs_names_make_unique()
    return sampled


def generate_synthetic_control(output_path, config):
    """Generate one semisynthetic negative-control dataset."""
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}, skipping.")
        return

    if os.path.exists("/tmp/pbmc_ad.h5ad"):
        pbmc_ad = ad.read_h5ad("/tmp/pbmc_ad.h5ad")
    else:
        pbmc_ad = scvi.data.dataset_10x(dataset_name="fresh_68k_pbmc_donor_a")
        pbmc_ad.write_h5ad("/tmp/pbmc_ad.h5ad")

    interaction_df = create_synthetic_interaction_df(config["gt_interactions"])
    positive_deg_check = False
    while not positive_deg_check:
        adata_pp = _preprocess(pbmc_ad, n_hvgs=500)
        adata_pp = _subcluster(adata_pp, n_subcluster_per_cluster=3, n_genes_for_subclustering=50)
        adata_pp, positive_deg_check = _check_positive_degs(adata_pp, interaction_df)
        if not positive_deg_check:
            scvi.settings.seed = np.random.randint(0, 1000)

    spatial_data = synthetic_layout(interaction_df)
    sampled = sample_existing_cells_by_neutral_subtype(adata_pp, spatial_data, neutral_source_map(interaction_df))
    sampled = create_synthetic_train_test_split(sampled)
    sampled.obs_names_make_unique()
    sampled.write_h5ad(output_path)


def generate_realistic_control(output_path, config):
    """Generate one realistic semisynthetic negative-control dataset."""
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}, skipping.")
        return

    raw_input_paths = [
        benchmark_path(config["flex_h5_path"]),
        benchmark_path(config["annot_path"]),
        benchmark_path(config["xenium_path"]),
    ]
    if not all(os.path.exists(path) for path in raw_input_paths):
        base_h5ad_path = benchmark_path(config["base_h5ad_path"])
        if not os.path.exists(base_h5ad_path):
            missing_paths = [path for path in raw_input_paths if not os.path.exists(path)]
            raise FileNotFoundError(
                "Missing raw realistic inputs and cached base h5ad. "
                f"Missing raw inputs: {missing_paths}; cached base path: {base_h5ad_path}"
            )

        print(f"Raw realistic inputs not found. Building neutral control from cached {base_h5ad_path}.")
        base_adata = sc.read_h5ad(base_h5ad_path)
        sampled = sample_cached_realistic_by_neutral_subtype(base_adata, np.random.default_rng(DATASET_SEED))
        sampled.write_h5ad(output_path)
        return

    flex_ad = _load_flex_data(benchmark_path(config["flex_h5_path"]), benchmark_path(config["annot_path"]))
    flex_pp, scvi_model = _preprocess_flex(
        flex_ad,
        n_hvgs=500,
        model_save_dir=benchmark_path(config["scvi_model_dir"]),
    )
    flex_pp = _subcluster_by_celltype(flex_pp, n_subcluster_per_cluster=2, n_genes_for_subclustering=200)
    spatial_subset = _load_xenium_spatial(benchmark_path(config["xenium_path"]), config["spatial_bounds"])

    present_cell_types = set(spatial_subset.obs["Cluster"].unique())
    active_interactions = [
        i
        for i in REALISTIC_GT_INTERACTION_LIST
        if i["receiver"] in present_cell_types and i["sender"] in present_cell_types
    ]
    receptor_clusters = sorted({i["receiver"] for i in active_interactions})
    interaction_df = create_realistic_interaction_df(active_interactions, receptor_clusters)
    spatial_data = _determine_spatial_subtypes(spatial_subset, interaction_df)

    sampled = sample_scvi_by_neutral_subtype(flex_pp, spatial_data, scvi_model, np.random.default_rng(DATASET_SEED))
    sampled = create_realistic_train_test_split(sampled, n_cv_folds=config.get("n_cv_folds", 3))
    sampled.obs_names_make_unique()
    sampled.write_h5ad(output_path)


def generate_dataset(dataset_name, config):
    """Generate the configured negative-control dataset if needed."""
    if config["kind"] == "synthetic":
        generate_synthetic_control(dataset_path(dataset_name), config)
    elif config["kind"] == "realistic":
        generate_realistic_control(dataset_path(dataset_name), config)
    else:
        raise ValueError(f"Unknown dataset kind: {config['kind']}")


def train_or_load_model(dataset_name, adata, config, run, run_id):
    """Train one AMICI run, or load its cached reconstruction loss."""
    run_path = os.path.join(saved_models_dir, dataset_name, f"run_{run_id}")
    loss_path = os.path.join(run_path, "test_recons.txt")
    if os.path.exists(os.path.join(run_path, "model.pt")) and os.path.exists(loss_path):
        with open(loss_path) as f:
            return run_path, float(f.read())

    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(os.path.dirname(run_path), exist_ok=True)

    pl.seed_everything(run["seed"])
    adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
    AMICI.setup_anndata(
        adata_train,
        labels_key=config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model = AMICI(
        adata_train,
        n_heads=int(run["n_heads"]),
        value_l1_penalty_coef=float(run["value_l1"]),
    )
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
        labels_key=config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(run["n_neighbors"]),
    )
    model.save(run_path, overwrite=True)
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]
    test_recons = (
        model.get_reconstruction_error(adata, indices=test_indices, batch_size=int(run["batch_size"]))[
            "reconstruction_loss"
        ]
        .detach()
        .cpu()
        .numpy()
    )
    with open(loss_path, "w") as f:
        f.write(str(float(test_recons)))
    return run_path, float(test_recons)


def best_model_for_dataset(dataset_name, adata, config):
    """Run the compact sweep and return the best cached AMICI model."""
    results = []
    for run_id, run in enumerate(build_run_configs(config["sweep_params"])):
        model_path, test_loss = train_or_load_model(dataset_name, adata, config, run, run_id)
        results.append((test_loss, model_path, run))
        print(f"{dataset_name} run_{run_id}: test reconstruction loss={test_loss:.6f}")

    _, best_model_path, best_run = min(results, key=lambda x: x[0])
    print(f"{dataset_name} best run: {best_run}")
    AMICI.setup_anndata(
        adata,
        labels_key=config["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=int(best_run["n_neighbors"]),
    )
    return AMICI.load(best_model_path, adata=adata)


def compute_pr_curves(model, adata, config):
    """Compute AMICI PR curves for all three benchmark tasks."""
    all_gt_gene_scores = []
    all_gene_scores = []
    for interaction_name, interaction_config in config["gt_interactions"].items():
        gt_gene_scores = get_receiver_gt_ranked_genes(
            adata,
            interaction_config["receiver"],
            interaction_config["interaction_subtype"],
            interaction_config["neutral_subtype"],
            config["subtype_key"],
        )
        gt_gene_scores["interaction"] = interaction_name
        all_gt_gene_scores.append(gt_gene_scores)

        gene_scores = get_amici_gene_task_scores(
            model,
            adata,
            interaction_config["sender"],
            interaction_config["receiver"],
        )
        gene_scores["interaction"] = interaction_name
        all_gene_scores.append(gene_scores)

    gt_gene_scores_df = pd.concat(all_gt_gene_scores, ignore_index=True)
    gene_scores_df = pd.concat(all_gene_scores, ignore_index=True)
    gene_precision, gene_recall, gene_auprc = get_model_precision_recall_auc(
        gene_scores_df,
        gt_gene_scores_df,
        merge_cols=["gene", "interaction"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    gt_neighbor_classes_df = get_interaction_gt_neighbor_classes(
        adata,
        config["gt_interactions"],
        config["labels_key"],
    )
    neighbor_scores_df = get_amici_neighbor_interaction_scores(model, adata)
    neighbor_precision, neighbor_recall, neighbor_auprc = get_model_precision_recall_auc(
        neighbor_scores_df,
        gt_neighbor_classes_df,
        merge_cols=["cell_idx", "neighbor_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    receiver_mask = np.zeros(len(adata.obs_names), dtype=bool)
    for interaction_config in config["gt_interactions"].values():
        receiver_mask |= adata.obs[config["subtype_key"]] == interaction_config["interaction_subtype"]
    gt_receiver_classes_df = pd.DataFrame({"cell_idx": adata.obs_names, "class": receiver_mask.astype(float)})
    receiver_scores_df = get_amici_receiver_subtype_scores(model, adata)
    receiver_precision, receiver_recall, receiver_auprc = get_model_precision_recall_auc(
        receiver_scores_df,
        gt_receiver_classes_df,
        merge_cols=["cell_idx"],
        scores_col="amici_scores",
        gt_class_col="class",
    )

    return {
        "Gene Task": (gene_precision, gene_recall, gene_auprc, int(gt_gene_scores_df["class"].sum())),
        "Neighbor Interaction Task": (
            neighbor_precision,
            neighbor_recall,
            neighbor_auprc,
            int(gt_neighbor_classes_df["class"].sum()),
        ),
        "Receiver Subtype Task": (receiver_precision, receiver_recall, receiver_auprc, int(receiver_mask.sum())),
    }


def plot_all_pr_curves(all_pr_curves):
    """Save one figure with both datasets and all three AMICI PR tasks."""
    task_names = ["Gene Task", "Neighbor Interaction Task", "Receiver Subtype Task"]
    fig, axes = plt.subplots(len(all_pr_curves), len(task_names), figsize=(18, 9), squeeze=False)

    for row_idx, (dataset_name, pr_curves) in enumerate(all_pr_curves.items()):
        for col_idx, task_name in enumerate(task_names):
            precision, recall, auprc, n_pos = pr_curves[task_name]
            ax = axes[row_idx][col_idx]
            PrecisionRecallDisplay(recall=recall, precision=precision).plot(
                ax=ax,
                name=f"AMICI (AUPRC={auprc:.2f})",
                color="#3568a8",
            )
            ax.set_title(f"{dataset_name}\n{task_name}", fontsize=10)
            ax.text(0.05, 0.08, f"n={n_pos} positives", transform=ax.transAxes, fontsize=9)
            ax.legend(fontsize=8)

    fig.suptitle("AMICI neutral-sampled negative-control PR curves", fontsize=14)
    plt.tight_layout()
    figure_path = os.path.join(figure_dir, f"{RUN_NAME}_amici_pr_curves.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved PR curves to {figure_path}")


# %% Generate datasets, run compact sweeps, and compute PR curves
select_gpu()
random.seed(DATASET_SEED)
np.random.seed(DATASET_SEED)
torch.random.manual_seed(DATASET_SEED)
torch.manual_seed(DATASET_SEED)
scvi.settings.seed = DATASET_SEED

all_pr_curves = {}
for dataset_name, dataset_config in DATASET_CONFIGS.items():
    generate_dataset(dataset_name, dataset_config)
    adata = sc.read_h5ad(dataset_path(dataset_name))
    adata.obs_names_make_unique()

    model = best_model_for_dataset(dataset_name, adata, dataset_config)
    all_pr_curves[dataset_name] = compute_pr_curves(model, adata, dataset_config)

# %% Plot PR curves
plot_all_pr_curves(all_pr_curves)
