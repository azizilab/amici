# %% Import libraries
import os

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from matplotlib.colors import Normalize
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from omnipath.interactions import import_intercell_network
from omnipath.requests import Annotations

from amici import AMICI

# %% Configuration
seed = 18
pl.seed_everything(seed)

labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"

SIGNIFICANCE_THRESHOLD = 2  # -log10(pval_adj) > 2 means p < 0.01
DIFF_THRESHOLD = 0  # diff > 0 means positive neighbor contribution
SEGMENTATION_NEAR_THRESHOLD = 20.0  # microns — receivers within this distance of a sender
SEGMENTATION_FAR_THRESHOLD = 100.0  # microns — senders at least this far from all receivers
SEGMENTATION_PVAL_THRESHOLD = 0.1  # Mann-Whitney U test threshold

# %% Load data and model
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")

saved_models_dir = f"saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
wandb_run_id = "te7pkv3z"
wandb_sweep_id = "g3mucw4s"
model_path = os.path.join(
    saved_models_dir,
    f"xenium_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)

AMICI.setup_anndata(adata, labels_key=labels_key, coord_obsm_key="spatial", n_neighbors=50)
model = AMICI.load(model_path, adata=adata)
print(f"Loaded model from {model_path}")
print(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes")

# %% Query OmniPath for ligand-receptor pairs
print("Fetching OmniPath ligand-receptor database...")
lr_df = import_intercell_network(
    interactions_params={"datasets": ["ligrecextra", "omnipath"]},
    transmitter_params={"categories": ["ligand"]},
    receiver_params={"categories": ["receptor"]},
)

ligand_col = "genesymbol_intercell_source"
receptor_col = "genesymbol_intercell_target"

ligand_genes = set(lr_df[ligand_col])
receptor_genes = set(lr_df[receptor_col])
all_lr_genes = ligand_genes | receptor_genes

# Map gene -> role(s)
gene_roles = {}
for _, row in lr_df.iterrows():
    lig = row[ligand_col]
    rec = row[receptor_col]
    gene_roles.setdefault(lig, set()).add("ligand")
    gene_roles.setdefault(rec, set()).add("receptor")

panel_lr_genes = all_lr_genes & set(adata.var_names)
print(f"OmniPath LR genes: {len(all_lr_genes)}")
print(f"Panel genes: {adata.n_vars}")
print(f"Panel genes in LR database: {len(panel_lr_genes)}")

# Fetch pathway/category annotations from OmniPath
print("Fetching pathway annotations...")
annot_df = Annotations.get(
    proteins=list(panel_lr_genes),
    resources=["SIGNOR", "SignaLink_pathway", "Matrisome"],
)
pathway_annot = annot_df[annot_df["label"].isin(["pathway", "subclass"])][
    ["genesymbol", "value"]
].drop_duplicates()
gene_pathway = (
    pathway_annot.groupby("genesymbol")["value"]
    .apply(lambda x: "; ".join(sorted(set(x))))
    .to_dict()
)
print(f"Genes with pathway annotations: {len(gene_pathway)}")

# %% Compute ablation scores for all receiver cell types (cached)
CACHE_DIR = "figures/.lr_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
sig_cache_path = os.path.join(CACHE_DIR, "sig_df.parquet")

cell_types = sorted(adata.obs[labels_key].unique())

if os.path.exists(sig_cache_path):
    print(f"\nLoading cached ablation results from {sig_cache_path}")
    sig_df = pd.read_parquet(sig_cache_path)
else:
    print(f"\nComputing ablation scores for {len(cell_types)} cell types...")
    all_significant = []

    for receiver_ct in tqdm(cell_types, desc="Ablation scores"):
        ablation_scores = model.get_neighbor_ablation_scores(
            adata=adata,
            cell_type=receiver_ct,
            compute_z_value=True,
        )
        abl_df = ablation_scores._ablation_scores_df

        diff_cols = [c for c in abl_df.columns if c.endswith("_diff")]
        sender_types = [c.replace("_diff", "") for c in diff_cols]

        n_sig = 0
        for sender in sender_types:
            if sender == receiver_ct:
                continue
            diff_col = f"{sender}_diff"
            pval_col = f"{sender}_nl10_pval_adj"
            z_col = f"{sender}_z_value"

            if pval_col not in abl_df.columns:
                continue

            sig_mask = (abl_df[diff_col] > DIFF_THRESHOLD) & (abl_df[pval_col] > SIGNIFICANCE_THRESHOLD)
            sig_genes = abl_df[sig_mask]

            for _, row in sig_genes.iterrows():
                gene = row["gene"]
                all_significant.append({
                    "sender": sender,
                    "receiver": receiver_ct,
                    "gene": gene,
                    "diff": row[diff_col],
                    "z_value": row[z_col],
                    "nl10_pval_adj": row[pval_col],
                    "is_lr_gene": gene in all_lr_genes,
                })
                n_sig += 1

    sig_df = pd.DataFrame(all_significant)
    sig_df.to_parquet(sig_cache_path)
    print(f"Cached ablation results to {sig_cache_path}")

print(f"\nTotal significant gene-pair entries: {len(sig_df)}")
print(f"Entries with LR genes: {sig_df['is_lr_gene'].sum()}")

# %% Filter to LR genes and annotate
lr_sig_df = sig_df[sig_df["is_lr_gene"]].copy()

lr_sig_df["role"] = lr_sig_df["gene"].map(
    lambda g: "/".join(sorted(gene_roles.get(g, set())))
)
lr_sig_df["pathway"] = lr_sig_df["gene"].map(
    lambda g: gene_pathway.get(g, "")
)
lr_sig_df["interaction"] = lr_sig_df["sender"] + " → " + lr_sig_df["receiver"]

print(f"\nLR-annotated significant genes: {len(lr_sig_df)}")
print(f"Unique interactions with LR genes: {lr_sig_df['interaction'].nunique()}")
print(f"Unique LR genes found: {lr_sig_df['gene'].nunique()}")

# %% Save CSV table
os.makedirs("figures", exist_ok=True)
csv_path = "figures/lr_genes_table.csv"
lr_sig_df.sort_values(["interaction", "nl10_pval_adj"], ascending=[True, False]).to_csv(
    csv_path, index=False
)
print(f"Saved {csv_path}")

# %% Segmentation artifact test (Mann-Whitney U) — cached
# For each (sender, receiver, gene) triple, test whether receiver expression near
# senders is greater than sender expression far from receivers.  Genes that fail
# to reject the null may reflect segmentation leakage rather than true interactions.
seg_cache_path = os.path.join(CACHE_DIR, "lr_sig_df_with_seg.parquet")

if os.path.exists(seg_cache_path):
    print(f"\nLoading cached segmentation results from {seg_cache_path}")
    lr_sig_df = pd.read_parquet(seg_cache_path)
else:
    print("\nRunning segmentation artifact test...")

    X_dense = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    spatial = adata.obsm["spatial"]

    seg_results = {}  # (sender, receiver, gene) -> mannwhitney p-value
    tested_pairs_cache = {}  # (sender, receiver) -> {near_receiver_idx, far_sender_idx}

    for _, row in tqdm(lr_sig_df.iterrows(), total=len(lr_sig_df), desc="Segmentation test"):
        sender, receiver, gene = row["sender"], row["receiver"], row["gene"]
        pair_key = (sender, receiver)

        if pair_key not in tested_pairs_cache:
            sender_mask = adata.obs[labels_key] == sender
            receiver_mask = adata.obs[labels_key] == receiver
            sender_coords = spatial[sender_mask].values
            receiver_coords = spatial[receiver_mask].values

            # Near receivers: receivers within threshold of any sender
            dist_r2s = cdist(receiver_coords, sender_coords)
            near_mask = dist_r2s.min(axis=1) <= SEGMENTATION_NEAR_THRESHOLD
            near_idx = np.where(receiver_mask)[0][near_mask]

            # Far senders: senders at least threshold from all receivers
            dist_s2r = cdist(sender_coords, receiver_coords)
            far_mask = dist_s2r.min(axis=1) >= SEGMENTATION_FAR_THRESHOLD
            far_idx = np.where(sender_mask)[0][far_mask]

            # Subsample to the smaller group for balanced comparison
            n = min(len(near_idx), len(far_idx))
            if n > 0:
                rng = np.random.RandomState(42)
                near_idx = rng.choice(near_idx, n, replace=False) if len(near_idx) > n else near_idx
                far_idx = rng.choice(far_idx, n, replace=False) if len(far_idx) > n else far_idx

            tested_pairs_cache[pair_key] = {"near": near_idx, "far": far_idx}

        cache = tested_pairs_cache[pair_key]
        near_idx, far_idx = cache["near"], cache["far"]

        if len(near_idx) < 2 or len(far_idx) < 2:
            seg_results[(sender, receiver, gene)] = np.nan
            continue

        gene_idx = adata.var_names.get_loc(gene)
        near_vals = X_dense[near_idx, gene_idx]
        far_vals = X_dense[far_idx, gene_idx]

        _, pval = mannwhitneyu(near_vals, far_vals, alternative="greater")
        seg_results[(sender, receiver, gene)] = pval

    lr_sig_df["seg_pval"] = lr_sig_df.apply(
        lambda r: seg_results.get((r["sender"], r["receiver"], r["gene"]), np.nan), axis=1
    )
    lr_sig_df["passes_seg_test"] = lr_sig_df["seg_pval"] < SEGMENTATION_PVAL_THRESHOLD
    lr_sig_df.to_parquet(seg_cache_path)
    print(f"Cached segmentation results to {seg_cache_path}")

n_pass = lr_sig_df["passes_seg_test"].sum()
n_total = lr_sig_df["passes_seg_test"].notna().sum()
print(f"Segmentation test: {n_pass}/{n_total} gene-pair entries pass (p < {SEGMENTATION_PVAL_THRESHOLD})")

# Save CSV with segmentation results
lr_sig_df.sort_values(["interaction", "nl10_pval_adj"], ascending=[True, False]).to_csv(
    csv_path, index=False
)
print(f"Saved {csv_path} with segmentation test results")

# %% Generate dot plot
# Filter to curated interactions from xenium_analysis.py
curated_interactions = {
    "CD8+_T_Cells": ["Macrophages_2", "CD4+_T_Cells"],
    "Macrophages_1": ["Invasive_Tumor", "CD8+_T_Cells", "DCIS_1", "Macrophages_2"],
    "B_Cells": ["CD4+_T_Cells"],
    "Invasive_Tumor": ["CD8+_T_Cells"],
    "CD4+_T_Cells": ["B_Cells", "Macrophages_2"],
}
curated_pairs = set()
for receiver, senders in curated_interactions.items():
    for sender in senders:
        curated_pairs.add(f"{sender} → {receiver}")

lr_sig_df_curated = lr_sig_df[lr_sig_df["interaction"].isin(curated_pairs)]

# Filter out genes that fail the segmentation test
lr_sig_df_curated = lr_sig_df_curated[lr_sig_df_curated["passes_seg_test"] == True]  # noqa: E712
print(f"\nCurated interactions with LR genes (after seg. filter): {lr_sig_df_curated['interaction'].nunique()}")
print(f"LR gene entries in curated set: {len(lr_sig_df_curated)}")

# For readability, limit to top N genes per interaction by significance
TOP_N_GENES_PER_INTERACTION = 15
plot_rows = []
for interaction in lr_sig_df_curated["interaction"].unique():
    sub = lr_sig_df_curated[lr_sig_df_curated["interaction"] == interaction].nlargest(
        TOP_N_GENES_PER_INTERACTION, "nl10_pval_adj"
    )
    plot_rows.append(sub)
plot_df = pd.concat(plot_rows)

# Get unique genes and interactions for the plot
unique_genes = plot_df["gene"].unique()
unique_interactions = sorted(plot_df["interaction"].unique())

n_genes = len(unique_genes)
n_interactions = len(unique_interactions)
fig, ax = plt.subplots(figsize=(max(14, n_genes * 0.55), max(6, n_interactions * 0.6)))

# Normalization for dot size and color
size_norm = Normalize(
    vmin=plot_df["nl10_pval_adj"].min(),
    vmax=plot_df["nl10_pval_adj"].max(),
)
color_norm = Normalize(
    vmin=plot_df["diff"].min(),
    vmax=0.5,
)
cmap = plt.cm.Reds

# Map interactions and genes to positions
interaction_to_y = {inter: i for i, inter in enumerate(unique_interactions)}
gene_to_x = {gene: i for i, gene in enumerate(sorted(unique_genes))}

# Plot dots
for _, row in plot_df.iterrows():
    x = gene_to_x[row["gene"]]
    y = interaction_to_y[row["interaction"]]
    size = 30 + 200 * size_norm(row["nl10_pval_adj"])
    color = cmap(color_norm(row["diff"]))
    marker = "^" if "receptor" in row["role"] else "o"

    ax.scatter(x, y, s=size, c=[color], marker=marker, edgecolors="black", linewidths=0.5, zorder=3)

ax.set_xticks(range(len(gene_to_x)))
ax.set_xticklabels(sorted(unique_genes), rotation=90, fontstyle="italic")
ax.set_yticks(range(len(interaction_to_y)))
ax.set_yticklabels(unique_interactions)
ax.set_xlabel("Gene")
ax.set_ylabel("Cell Type Interaction")
ax.set_title("Significant Downstream LR Genes per Cell Type Interaction")
ax.set_xlim(-0.5, len(gene_to_x) - 0.5)
ax.set_ylim(-0.5, len(interaction_to_y) - 0.5)
ax.grid(True, alpha=0.15)

# Color bar for diff
sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
cbar.set_label("Neighbor Contribution")
cbar.ax.tick_params(labelsize=14)

# Size legend
size_values = [2, 5, 8]
size_labels = ["2", "5", "8"]
size_handles = [
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=np.sqrt(30 + 200 * size_norm(v)), label=l)
    for v, l in zip(size_values, size_labels)
]

# Marker legend for ligand/receptor
ligand_handle = mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=10, label="Ligand")
receptor_handle = mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=10, label="Receptor")

all_handles = [ligand_handle, receptor_handle] + size_handles
legend_labels = ["Ligand", "Receptor"] + [f"-log10(padj)={l}" for l in size_labels]
ax.legend(all_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.15, 1.0), fontsize=12, title="Marker / Size", title_fontsize=13, labelspacing=1.2)

plt.tight_layout()

for ext in ("png", "svg"):
    save_path = f"figures/lr_dotplot_supplementary.{ext}"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
plt.close(fig)

# %% Print summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total cell type pairs analyzed: {len(cell_types) * (len(cell_types) - 1)}")
print(f"Total significant gene-pair entries: {len(sig_df)}")
print(f"Entries matching LR database: {len(lr_sig_df)} ({100*len(lr_sig_df)/len(sig_df):.1f}%)")
print(f"Unique LR genes: {lr_sig_df['gene'].nunique()}")
print(f"Unique interactions with LR genes: {lr_sig_df['interaction'].nunique()}")
print(f"\nTop LR genes by frequency across interactions:")
print(lr_sig_df["gene"].value_counts().head(15).to_string())
# %%
