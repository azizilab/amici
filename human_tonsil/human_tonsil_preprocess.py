# %% Import libraries
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# %% Paths and parameters
DATA_DIR = Path("human_tonsil/data")
FIGURE_DIR = Path("human_tonsil/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_DIR / "HumanTonsil_annotated_clean.h5ad"
LABELS_KEY = "celltype_manual_fine"
COORD_KEY = "spatial"
TEST_FRACTION = 0.10

# %% Load annotated Human Tonsil data
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing Human Tonsil AnnData file: {INPUT_PATH}")

adata = sc.read_h5ad(INPUT_PATH)
adata.obs_names_make_unique()

print(f"Number of cells before filtering: {adata.n_obs}")
print(f"Number of genes before filtering: {adata.n_vars}")
print("Fine annotations before filtering:")
print(adata.obs[LABELS_KEY].value_counts(dropna=False).to_string())

# %% Filter unlabeled annotations
labels = adata.obs[LABELS_KEY].astype("string")
is_labeled = labels.notna() & ~labels.str.strip().str.lower().isin({"", "unassigned", "unlabeled", "unknown"})
adata = adata[is_labeled.to_numpy()].copy()
adata.obs[LABELS_KEY] = adata.obs[LABELS_KEY].astype(str)

print(f"Number of cells after annotation filter: {adata.n_obs}")
print(f"Number of genes retained: {adata.n_vars}")
print("Fine annotations used for training:")
print(adata.obs[LABELS_KEY].value_counts().to_string())

# %% Add cell radius and train/test split
if "Area.um2" in adata.obs:
    adata.obs["cell_radius"] = np.sqrt(adata.obs["Area.um2"].astype(float) / np.pi)
elif "Area" in adata.obs:
    adata.obs["cell_radius"] = np.sqrt(adata.obs["Area"].astype(float) / np.pi)
else:
    adata.obs["cell_radius"] = 0.0

spatial = np.asarray(adata.obsm[COORD_KEY])
y_cutoff = np.quantile(spatial[:, 1], TEST_FRACTION)
is_test = spatial[:, 1] <= y_cutoff

adata.obs["train_test_split"] = "train"
adata.obs.loc[adata.obs_names[is_test], "train_test_split"] = "test"

adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
adata_test = adata[adata.obs["train_test_split"] == "test"].copy()

print(f"Test slice y <= {y_cutoff:.2f}")
print(f"Number of total cells: {adata.n_obs}")
print(f"Number of training cells: {adata_train.n_obs}")
print(f"Number of test cells: {adata_test.n_obs}")
print(f"Test fraction: {adata_test.n_obs / adata.n_obs:.3f}")

# %% Visualize annotation and split
palette = dict(
    zip(
        sorted(adata.obs[LABELS_KEY].unique()),
        sns.color_palette("tab20", n_colors=adata.obs[LABELS_KEY].nunique()).as_hex(),
    )
)


def plot_spatial(adata_obj, color_key, title, output_name, show_test_border=False):
    plot_df = pd.DataFrame(
        {
            "X": np.asarray(adata_obj.obsm[COORD_KEY])[:, 0],
            "Y": np.asarray(adata_obj.obsm[COORD_KEY])[:, 1],
            color_key: adata_obj.obs[color_key].values,
        },
        index=adata_obj.obs_names,
    )
    plt.figure(figsize=(14, 9))
    sns.scatterplot(
        data=plot_df,
        x="X",
        y="Y",
        hue=color_key,
        s=0.35,
        alpha=0.7,
        linewidth=0,
        palette=palette if color_key == LABELS_KEY else None,
    )
    if show_test_border:
        plt.axhline(y_cutoff, color="black", linestyle="--", linewidth=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, markerscale=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=300)
    plt.close()


plot_spatial(adata, LABELS_KEY, "Human Tonsil Fine Annotations", "human_tonsil_fine_annotations.png")
plot_spatial(adata, "train_test_split", "Human Tonsil Train/Test Split", "human_tonsil_train_test_split.png", True)

# %% Save filtered datasets
today = date.today()
adata.write_h5ad(DATA_DIR / f"human_tonsil_filtered_{today}.h5ad")
adata_train.write_h5ad(DATA_DIR / f"human_tonsil_filtered_train_{today}.h5ad")
adata_test.write_h5ad(DATA_DIR / f"human_tonsil_filtered_test_{today}.h5ad")
