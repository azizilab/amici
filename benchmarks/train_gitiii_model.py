import json
import os
import random
import shutil

import gitiii
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from gitiii.calculate_PCC import Calculate_PCC
from gitiii.dataloader import GITIII_dataset
from gitiii.model import GITIII, Loss_function
from gitiii_benchmark_utils import convert_adata_to_csv
from gpu_utils import select_gpu
from torch.utils.data import DataLoader, Subset

GITIII_SPLIT_VERSION = 2


def _results_path():
    return snakemake.config.get("results_path", "results/").rstrip("/")  # noqa: F821


def _mark_excluded_cells_in_processed_csv(run_dir, excluded_cell_coords):
    """Set flag=False for excluded cells in GITIII's processed CSV, matched by (centerx, centery)."""
    processed_csv_path = os.path.join(run_dir, "data", "processed", "slide1.csv")
    df = pd.read_csv(processed_csv_path)
    if "flag" not in df.columns:
        df["flag"] = True
    is_excluded = df.apply(lambda r: (r["centerx"], r["centery"]) in excluded_cell_coords, axis=1)
    df.loc[is_excluded, "flag"] = False
    df.to_csv(processed_csv_path, index=False)


def _expected_cache_metadata(split_mode):
    return {"split_mode": split_mode, "split_version": GITIII_SPLIT_VERSION}


def _cache_matches(cached, split_mode):
    if split_mode == "random_internal" and "split_mode" not in cached:
        return True
    expected = _expected_cache_metadata(split_mode)
    return all(cached.get(key) == value for key, value in expected.items())


def _dataset_center_coords(dataset):
    coords = []
    previous_count = 0
    for sample_id in range(len(dataset.samples)):
        original_indices = dataset.arg_meta[sample_id].cpu().numpy()
        centerx = dataset.centerx[sample_id][original_indices].cpu().numpy()
        centery = dataset.centery[sample_id][original_indices].cpu().numpy()
        for local_idx, coord in enumerate(zip(centerx, centery, strict=False)):
            coords.append((previous_count + local_idx, coord))
        previous_count += len(original_indices)
    return coords


def _dataset_center_index_lookup(dataset):
    lookup = {}
    previous_count = 0
    for sample_id in range(len(dataset.samples)):
        original_indices = dataset.arg_meta[sample_id].cpu().numpy()
        for local_idx, original_idx in enumerate(original_indices):
            lookup[previous_count + local_idx] = (sample_id, int(original_idx))
        previous_count += len(original_indices)
    return lookup


def _coord_index_sets(dataset, coord_sets):
    index_sets = [set() for _ in coord_sets]
    for sample_id in range(len(dataset.samples)):
        centerx = dataset.centerx[sample_id].cpu().numpy()
        centery = dataset.centery[sample_id].cpu().numpy()
        for original_idx, coord in enumerate(zip(centerx, centery, strict=False)):
            for set_idx, coords in enumerate(coord_sets):
                if coords is not None and coord in coords:
                    index_sets[set_idx].add((sample_id, original_idx))
    return index_sets


class _ScrubbedGITIIISubset(Subset):
    def __init__(self, dataset, indices, scrub_original_indices):
        super().__init__(dataset, indices)
        self.center_lookup = _dataset_center_index_lookup(dataset)
        self.scrub_original_indices = scrub_original_indices

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        item = self.dataset[dataset_idx]
        sample_id, original_idx = self.center_lookup[dataset_idx]
        neighbor_indices = self.dataset.indexes[sample_id][original_idx].cpu().numpy()
        scrub_mask = np.array(
            [(sample_id, int(neighbor_idx)) in self.scrub_original_indices for neighbor_idx in neighbor_indices]
        )
        scrub_mask[0] = False
        if not np.any(scrub_mask):
            return item

        clean_item = dict(item)
        for key in ("x", "type_exp", "cell_types", "position_x", "position_y"):
            value = clean_item[key].clone()
            if value.ndim == 1:
                value[scrub_mask] = value[0]
            else:
                value[scrub_mask, ...] = value[0]
            clean_item[key] = value
        return clean_item


def _make_explicit_loaders(data_dir, num_neighbors, batch_size, val_cell_coords, scrub_cell_coords):
    dataset = GITIII_dataset(processed_dir=data_dir, num_neighbors=num_neighbors)
    train_indices = []
    val_indices = []
    use_validation = val_cell_coords is not None

    for dataset_idx, coord in _dataset_center_coords(dataset):
        if use_validation and coord in val_cell_coords:
            val_indices.append(dataset_idx)
        else:
            train_indices.append(dataset_idx)

    if not train_indices:
        raise ValueError("GITIII explicit split produced no training cells")

    (scrub_original_indices,) = _coord_index_sets(dataset, [scrub_cell_coords])
    train_dataset = _ScrubbedGITIIISubset(dataset, train_indices, scrub_original_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if use_validation:
        if not val_indices:
            raise ValueError("GITIII explicit split produced no validation cells")
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    print(f"Explicit GITIII split: {len(train_indices)} train cells, {len(val_indices)} validation cells")
    return train_loader, val_loader


def _train_gitiii_with_explicit_split(
    run_dir,
    val_cell_coords,
    scrub_cell_coords,
    num_neighbors=50,
    batch_size=128,
    lr=1e-4,
    epochs=50,
    node_dim=256,
    edge_dim=48,
    att_dim=8,
    use_cell_type_embedding=True,
):
    data_dir = os.path.join(run_dir, "data", "processed")
    train_loader, val_loader = _make_explicit_loaders(
        data_dir,
        num_neighbors,
        batch_size,
        val_cell_coords,
        scrub_cell_coords,
    )

    torch.cuda.empty_cache()
    ligands_info = torch.load(os.path.join(run_dir, "data", "ligands.pth"))
    genes = torch.load(os.path.join(run_dir, "data", "genes.pth"))
    model = GITIII(
        genes,
        ligands_info,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_heads=2,
        n_layers=1,
        node_dim_small=16,
        att_dim=att_dim,
        use_cell_type_embedding=use_cell_type_embedding,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = Loss_function(genes, ligands_info).cuda()
    evaluator = Calculate_PCC(genes, ligands_info)

    records = []
    best_val = float("inf")
    best_metric = float("inf")
    checkpoint_path = os.path.join(run_dir, "GRIT.pth")
    best_model_path = os.path.join(run_dir, "GRIT_best.pth")

    print("Start explicit-split GITIII training")
    for epoch in range(epochs):
        model.train()
        loss_train1 = 0
        loss_train2 = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.cuda() for key, value in batch.items()}
            y_pred = model(batch)
            y = batch["y"]
            loss1, loss2 = loss_func(y_pred, y)
            evaluator.add_input(y_pred, y)
            loss1.backward()
            optimizer.step()
            loss_train1 += loss1.cpu().item()
            loss_train2 += loss2.cpu().item()

        pcc1_train, pcc2_train = evaluator.calculate_pcc(clear=True)
        train_loss1 = loss_train1 / len(train_loader)
        train_loss2 = loss_train2 / len(train_loader)

        val_loss1 = np.nan
        val_loss2 = np.nan
        pcc1_val = np.nan
        pcc2_val = np.nan
        metric = train_loss1

        if val_loader is not None:
            model.eval()
            loss_val1 = 0
            loss_val2 = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.cuda() for key, value in batch.items()}
                    y_pred = model(batch)
                    y = batch["y"]
                    evaluator.add_input(y_pred, y)
                    loss1, loss2 = loss_func(y_pred, y)
                    loss_val1 += loss1.cpu().item()
                    loss_val2 += loss2.cpu().item()
                    torch.cuda.empty_cache()
            pcc1_val, pcc2_val = evaluator.calculate_pcc(clear=True)
            val_loss1 = loss_val1 / len(val_loader)
            val_loss2 = loss_val2 / len(val_loader)
            metric = val_loss1

        records.append(
            [
                epoch,
                train_loss1,
                train_loss2,
                pcc1_train,
                pcc2_train,
                val_loss1,
                val_loss2,
                pcc1_val,
                pcc2_val,
            ]
        )
        pd.DataFrame(
            data=records,
            columns=[
                "epoch",
                "train_loss_interaction",
                "train_loss_downstream",
                "PCC1_train",
                "PCC2_train",
                "val_loss_interaction",
                "val_loss_downstream",
                "PCC1_val",
                "PCC2_val",
            ],
        ).to_csv(os.path.join(run_dir, "record_GRIT.csv"))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "records": records,
            "best_val": best_val,
        }
        torch.save(checkpoint, checkpoint_path)
        if metric < best_metric:
            best_metric = metric
            best_val = metric
            torch.save(model.state_dict(), best_model_path)

        if val_loader is None:
            print(f"Finish epoch {epoch}: train_loss_interaction={train_loss1:.6f}")
        else:
            print(
                f"Finish epoch {epoch}: train_loss_interaction={train_loss1:.6f}; "
                f"val_loss_interaction={val_loss1:.6f}"
            )

    return float(best_metric)


def train_single_run(
    converted_df_path,
    gene_names,
    lr,
    distance_threshold,
    run_dir,
    run_results_path,
    test_cell_coords,
    val_cell_coords=None,
    split_mode="random_internal",
):
    """Train a single GITIII run and return the best validation MSE, loading from cache if available."""
    if os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        if _cache_matches(cached, split_mode):
            print(f"  Loaded cached result: val_mse={cached['val_mse']:.6f}")
            return cached["val_mse"]
        print("  Ignoring cached result from an older GITIII split mode")
        os.remove(run_results_path)

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    estimator = gitiii.estimator.GITIII_estimator(
        df_path=converted_df_path,
        genes=gene_names,
        use_log_normalize=False,
        species="human",
        use_nichenetv2=True,
        visualize_when_preprocessing=False,
        distance_threshold=distance_threshold,
        process_num_neighbors=50,
        num_neighbors=50,
        batch_size_train=128,
        lr=lr,
        epochs=50,
        node_dim=256,
        edge_dim=48,
        att_dim=8,
        batch_size_val=128,
    )
    estimator.preprocess_dataset()
    excluded_center_coords = (
        test_cell_coords if split_mode in {"random_internal", "explicit_cv", "all_train"} else set()
    )
    _mark_excluded_cells_in_processed_csv(run_dir, excluded_center_coords)

    if split_mode == "random_internal":
        estimator.train()
        records_path = os.path.join(run_dir, "record_GRIT.csv")
        val_mse = pd.read_csv(records_path)["val_loss_interaction"].min()
    elif split_mode == "explicit_cv":
        val_mse = _train_gitiii_with_explicit_split(
            run_dir,
            val_cell_coords=val_cell_coords,
            scrub_cell_coords=test_cell_coords | val_cell_coords,
            lr=lr,
        )
    elif split_mode == "explicit_test":
        val_mse = _train_gitiii_with_explicit_split(
            run_dir,
            val_cell_coords=val_cell_coords,
            scrub_cell_coords=test_cell_coords,
            lr=lr,
        )
    elif split_mode == "all_train":
        val_mse = _train_gitiii_with_explicit_split(
            run_dir,
            val_cell_coords=None,
            scrub_cell_coords=test_cell_coords,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown GITIII split mode: {split_mode}")

    print(f"  best val_mse={val_mse:.6f}")

    return float(val_mse)


def _get_spatial_coords(adata, mask):
    spatial = adata.obsm["spatial"]
    if hasattr(spatial, "iloc"):
        spatial_x = spatial["X"].values
        spatial_y = spatial["Y"].values
    else:
        spatial_x = spatial[:, 0]
        spatial_y = spatial[:, 1]
    return set(zip(spatial_x[mask], spatial_y[mask], strict=False))


def _run_sweep(
    converted_df_path,
    gene_names,
    all_runs,
    models_dir,
    test_cell_coords,
    run_prefix,
    project_root,
    val_cell_coords=None,
    split_mode="random_internal",
):
    best_val_mse = float("inf")
    best_run_id = None
    scores = {}

    for run_id, (lr, distance_threshold) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"{run_prefix}_run_{run_id}")
        run_results_path = os.path.join(models_dir, f"{run_prefix}_run_{run_id}_results.json")

        print(f"Run {run_id + 1}/{len(all_runs)}: lr={lr}, distance_threshold={distance_threshold}")

        val_mse = train_single_run(
            converted_df_path,
            gene_names,
            lr,
            distance_threshold,
            run_dir,
            run_results_path,
            test_cell_coords,
            val_cell_coords=val_cell_coords,
            split_mode=split_mode,
        )

        if not os.path.exists(run_results_path):
            with open(run_results_path, "w") as f:
                json.dump({"val_mse": val_mse, **_expected_cache_metadata(split_mode)}, f)

        os.chdir(project_root)
        scores[run_id] = val_mse

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_run_id = run_id

    return best_run_id, best_val_mse, scores


def main():
    """Train the GITIII model with a hyperparameter sweep, keeping the best run."""
    select_gpu()
    project_root = os.path.abspath(os.getcwd())

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821

    adata = sc.read_h5ad(dataset_path)

    models_dir = os.path.abspath(f"{_results_path()}/{dataset}_{seed}/saved_models")
    converted_df_path = f"../../../data/{dataset_path.split('/')[-1].split('.')[0]}_converted.csv"

    convert_adata_to_csv(adata, labels_key, models_dir, converted_df_path)
    abs_converted_df_path = os.path.abspath(converted_df_path)
    os.chdir(project_root)

    test_mask = np.array(adata.obs["train_test_split"] == "test")
    test_cell_coords = _get_spatial_coords(adata, test_mask)

    gene_names = adata.var_names.tolist()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    learning_rates = [1e-3, 1e-4]
    distance_thresholds = [50, 80, 120]
    all_runs = [(lr, dt) for lr in learning_rates for dt in distance_thresholds]

    use_cv = dataset_config.get("use_cross_validation", False)

    if use_cv:
        if "cv_fold" not in adata.obs:
            raise ValueError("use_cross_validation is true, but adata.obs['cv_fold'] is missing")

        fold_values = sorted(int(fold) for fold in adata.obs.loc[adata.obs["cv_fold"] >= 0, "cv_fold"].unique())
        cv_scores = {run_id: [] for run_id in range(len(all_runs))}

        for fold_id in fold_values:
            print(f"\n=== GITIII CV fold {fold_id + 1}/{len(fold_values)} ===")
            fold_mask = np.array((adata.obs["train_test_split"] == "train") & (adata.obs["cv_fold"] == fold_id))
            fold_cell_coords = _get_spatial_coords(adata, fold_mask)
            _, _, fold_scores = _run_sweep(
                abs_converted_df_path,
                gene_names,
                all_runs,
                models_dir,
                test_cell_coords,
                f"gitiii_cv_fold_{fold_id}",
                project_root,
                val_cell_coords=fold_cell_coords,
                split_mode="explicit_cv",
            )
            for run_id, score in fold_scores.items():
                cv_scores[run_id].append(score)

        avg_cv_scores = {run_id: float(np.mean(scores)) for run_id, scores in cv_scores.items()}
        best_cv_run_id = min(avg_cv_scores, key=avg_cv_scores.get)
        best_lr, best_distance_threshold = all_runs[best_cv_run_id]
        print(
            f"\nBest GITIII CV config {best_cv_run_id}: lr={best_lr}, "
            f"distance_threshold={best_distance_threshold}, avg_val_mse={avg_cv_scores[best_cv_run_id]:.6f}"
        )

        best_run_id, best_val_mse, _ = _run_sweep(
            abs_converted_df_path,
            gene_names,
            [(best_lr, best_distance_threshold)],
            models_dir,
            test_cell_coords,
            "gitiii_final",
            project_root,
            split_mode="all_train",
        )
        final_run_prefix = "gitiii_final"
    else:
        best_run_id, best_val_mse, _ = _run_sweep(
            abs_converted_df_path,
            gene_names,
            all_runs,
            models_dir,
            test_cell_coords,
            "gitiii_final",
            project_root,
            val_cell_coords=test_cell_coords,
            split_mode="explicit_test",
        )
        best_lr, best_distance_threshold = all_runs[best_run_id]
        final_run_prefix = "gitiii_final"

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    best_run_dir = os.path.join(models_dir, f"{final_run_prefix}_run_{best_run_id}")

    print(
        f"Best run {best_run_id}: lr={best_lr}, distance_threshold={best_distance_threshold}, "
        f"val_mse={best_val_mse:.6f}"
    )

    # If the best run directory was cleaned up by a prior run that failed after the copy step,
    # re-run training for that configuration to regenerate the model files.
    if not os.path.exists(best_run_dir):
        run_results_path_best = os.path.join(models_dir, f"{final_run_prefix}_run_{best_run_id}_results.json")
        if os.path.exists(run_results_path_best):
            os.remove(run_results_path_best)
        print(f"Best run directory missing — re-training run {best_run_id} to regenerate model files")
        retry_split_mode = "all_train" if use_cv else "explicit_test"
        retry_val_cell_coords = None if use_cv else test_cell_coords
        train_single_run(
            abs_converted_df_path,
            gene_names,
            best_lr,
            best_distance_threshold,
            best_run_dir,
            run_results_path_best,
            test_cell_coords,
            val_cell_coords=retry_val_cell_coords,
            split_mode=retry_split_mode,
        )
        os.chdir(project_root)

    for fname in os.listdir(best_run_dir):
        src = os.path.join(best_run_dir, fname)
        dst = os.path.join(models_dir, fname)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src) and not fname.endswith("_results.json"):
            shutil.copy2(src, dst)

    with open(os.path.join(models_dir, "gitiii_best_params.json"), "w") as f:
        json.dump(
            {
                "learning_rate": best_lr,
                "distance_threshold": best_distance_threshold,
                "seed": 42,
                "val_mse": best_val_mse,
            },
            f,
        )

    # Clean up all sweep run directories (best model already copied to models_dir)
    for run_id_cleanup in range(len(all_runs)):
        run_dir_cleanup = os.path.join(models_dir, f"gitiii_final_run_{run_id_cleanup}")
        if os.path.exists(run_dir_cleanup):
            shutil.rmtree(run_dir_cleanup)
    figures_dir = f"{_results_path()}/{dataset}_{seed}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    records_df = pd.read_csv(os.path.join(models_dir, "record_GRIT.csv"))
    plt.figure(figsize=(10, 5))
    plt.plot(records_df["train_loss_interaction"].values, label="Train Loss")
    plt.plot(records_df["val_loss_interaction"].values, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.png"))
    plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.svg"))
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
