# %%
"""Check held-out region cell-type composition against full tissue.

This diagnostic helps assess whether a spatially contiguous test region remains
compositionally comparable to the full tissue. Jensen-Shannon divergence is
computed on per-cell-type fractions using base-2 logs, so values range from 0 to 1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial.distance import jensenshannon

# %%
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
FIGURES_DIR = SCRIPT_DIR / "figures" / "test_region_composition_diagnostic"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PASS_JSD_THRESHOLD = 0.05
WARN_JSD_THRESHOLD = 0.10

DATASETS = [
    {
        "name": "Xenium fine",
        "path": REPO_DIR / "xenium" / "data" / "xenium_sample1" / "xenium_sample1_filtered_2025-05-01.h5ad",
        "label_col": "celltype_train_grouped",
        "split_col": "train_test_split",
    },
    {
        "name": "Xenium low-res",
        "path": REPO_DIR / "xenium" / "data" / "xenium_sample1_filtered_lowres_2026-03-27.h5ad",
        "label_col": "celltype_lowres",
        "split_col": "train_test_split",
    },
    {
        "name": "Xenium proseg/stromal",
        "path": REPO_DIR / "xenium" / "data" / "xenium_proseg_filtered_2025-04-08.h5ad",
        "label_col": "celltype_train_grouped",
        "split_col": "train_test_split",
    },
    {
        "name": "Atera breast",
        "path": REPO_DIR / "atera_breast" / "data" / "atera_breast_filtered_2026-07-22.h5ad",
        "label_col": "celltype_train_grouped",
        "split_col": "train_test_split",
    },
    {
        "name": "Cortex",
        "path": REPO_DIR / "cortex" / "data" / "cortex_processed_2025-04-28.h5ad",
        "label_col": "subclass",
        "split_col": "in_test",
    },
]


# %%
def parse_test_mask(obs, split_col):
    """Return a boolean mask for held-out test cells."""
    split_values = obs[split_col]
    if split_values.dtype == bool:
        return split_values.to_numpy()

    values = split_values.astype(str).str.lower()
    for token in ["test", "test_set", "heldout", "held_out"]:
        if (values == token).any():
            return (values == token).to_numpy()
    if values.isin(["0"]).any() and values.isin(["1"]).any():
        return (values == "1").to_numpy()
    raise ValueError(f"Could not parse test split from {split_col}: {values.value_counts().head().to_dict()}")


def classify_jsd(jsd):
    """Classify composition agreement using heuristic JSD thresholds."""
    if jsd < PASS_JSD_THRESHOLD:
        return "pass"
    if jsd < WARN_JSD_THRESHOLD:
        return "caution"
    return "mismatch"


def compute_composition(dataset):
    """Compute cell-type composition summaries for one AnnData object."""
    adata = sc.read_h5ad(dataset["path"])
    obs = adata.obs
    label_col = dataset["label_col"]
    split_col = dataset["split_col"]

    if label_col not in obs.columns:
        raise KeyError(f"{label_col} not found in {dataset['path']}")
    if split_col not in obs.columns:
        raise KeyError(f"{split_col} not found in {dataset['path']}")

    test_mask = parse_test_mask(obs, split_col)
    labels = obs[label_col].astype(str)

    full_counts = labels.value_counts().sort_index()
    test_counts = labels[test_mask].value_counts().reindex(full_counts.index, fill_value=0).sort_index()
    train_counts = labels[~test_mask].value_counts().reindex(full_counts.index, fill_value=0).sort_index()

    full_frac = full_counts / full_counts.sum()
    test_frac = test_counts / max(test_counts.sum(), 1)
    train_frac = train_counts / max(train_counts.sum(), 1)

    jsd_test_full = float(jensenshannon(test_frac.to_numpy(), full_frac.to_numpy(), base=2.0) ** 2)
    jsd_test_train = float(jensenshannon(test_frac.to_numpy(), train_frac.to_numpy(), base=2.0) ** 2)

    composition = pd.DataFrame(
        {
            "dataset": dataset["name"],
            "cell_type": full_counts.index,
            "full_count": full_counts.values,
            "test_count": test_counts.values,
            "train_count": train_counts.values,
            "full_frac": full_frac.values,
            "test_frac": test_frac.values,
            "train_frac": train_frac.values,
        }
    )
    composition["test_minus_full"] = composition["test_frac"] - composition["full_frac"]
    composition["abs_test_minus_full"] = composition["test_minus_full"].abs()

    summary = {
        "dataset": dataset["name"],
        "path": str(dataset["path"]),
        "label_col": label_col,
        "split_col": split_col,
        "n_cells": int(adata.n_obs),
        "test_cells": int(test_mask.sum()),
        "test_fraction": float(test_mask.mean()),
        "cell_types": int(len(full_counts)),
        "missing_cell_types_in_test": int((test_counts == 0).sum()),
        "jsd_test_full": jsd_test_full,
        "jsd_test_train": jsd_test_train,
        "status": classify_jsd(jsd_test_full),
    }
    return summary, composition


# %%
summaries = []
compositions = []
missing = []

for dataset in DATASETS:
    if not dataset["path"].exists():
        missing.append({"dataset": dataset["name"], "path": str(dataset["path"])})
        continue
    summary, composition = compute_composition(dataset)
    summaries.append(summary)
    compositions.append(composition)

summary_df = pd.DataFrame(summaries)
composition_df = pd.concat(compositions, ignore_index=True) if compositions else pd.DataFrame()
missing_df = pd.DataFrame(missing)

summary_df.to_csv(FIGURES_DIR / "test_region_composition_jsd_summary.csv", index=False)
composition_df.to_csv(FIGURES_DIR / "test_region_cell_type_composition.csv", index=False)
missing_df.to_csv(FIGURES_DIR / "missing_test_region_datasets.csv", index=False)

# %%
if not summary_df.empty:
    plt.figure(figsize=(7, 3.8))
    palette = {"pass": "#4C78A8", "caution": "#F58518", "mismatch": "#E45756"}
    ax = sns.barplot(data=summary_df, x="dataset", y="jsd_test_full", hue="status", dodge=False, palette=palette)
    ax.axhline(
        PASS_JSD_THRESHOLD, color="0.25", linestyle="--", linewidth=1, label=f"pass threshold ({PASS_JSD_THRESHOLD})"
    )
    ax.axhline(
        WARN_JSD_THRESHOLD, color="0.45", linestyle=":", linewidth=1, label=f"caution threshold ({WARN_JSD_THRESHOLD})"
    )
    ax.set_xlabel("")
    ax.set_ylabel("JSD(test || full)")
    ax.set_title("Held-out region cell-type composition agreement")
    ax.set_ylim(0, max(WARN_JSD_THRESHOLD * 1.15, summary_df["jsd_test_full"].max() * 1.25))
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_region_jsd_barplot.png", dpi=300)
    plt.savefig(FIGURES_DIR / "test_region_jsd_barplot.svg")
    plt.close()

# %%
if not composition_df.empty:
    top_shifts = (
        composition_df.sort_values("abs_test_minus_full", ascending=False)
        .groupby("dataset", group_keys=False)
        .head(8)
        .copy()
    )
    top_shifts["label"] = top_shifts["dataset"] + " | " + top_shifts["cell_type"]

    plt.figure(figsize=(6.2, max(4.0, 0.24 * len(top_shifts))))
    ax = sns.barplot(data=top_shifts, y="label", x="test_minus_full", color="#4C78A8")
    ax.axvline(0, color="0.2", linewidth=1)
    ax.set_xlabel("Test fraction - full tissue fraction")
    ax.set_ylabel("")
    ax.set_title("Largest cell-type composition shifts in held-out regions")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_region_largest_composition_shifts.png", dpi=300)
    plt.savefig(FIGURES_DIR / "test_region_largest_composition_shifts.svg")
    plt.close()

# %%
print(summary_df.to_string(index=False))
if not missing_df.empty:
    print("\nMissing datasets skipped:")
    print(missing_df.to_string(index=False))
