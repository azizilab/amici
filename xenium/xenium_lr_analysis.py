# %% Import libraries
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from matplotlib.colors import Normalize
from omnipath.interactions import import_intercell_network

from amici import AMICI

# %% Configuration
seed = 18
pl.seed_everything(seed)

labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"

SIGNIFICANCE_THRESHOLD = 2  # -log10(pval_adj) > 2 means p < 0.01
DIFF_THRESHOLD = 0  # diff > 0 means positive neighbor contribution

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

# Map gene -> role(s) and LR partners
gene_roles = {}
gene_partners = {}
for _, row in lr_df.iterrows():
    lig = row[ligand_col]
    rec = row[receptor_col]
    gene_roles.setdefault(lig, set()).add("ligand")
    gene_roles.setdefault(rec, set()).add("receptor")
    gene_partners.setdefault(lig, set()).add(rec)
    gene_partners.setdefault(rec, set()).add(lig)

panel_lr_genes = all_lr_genes & set(adata.var_names)
print(f"OmniPath LR genes: {len(all_lr_genes)}")
print(f"Panel genes: {adata.n_vars}")
print(f"Panel genes in LR database: {len(panel_lr_genes)}")

# %% Compute ablation scores for all receiver cell types
cell_types = sorted(adata.obs[labels_key].unique())
print(f"\nComputing ablation scores for {len(cell_types)} cell types...")

all_significant = []

for receiver_ct in cell_types:
    print(f"  {receiver_ct}...", end=" ", flush=True)
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

    print(f"{n_sig} significant gene-sender pairs")

sig_df = pd.DataFrame(all_significant)
print(f"\nTotal significant gene-pair entries: {len(sig_df)}")
print(f"Entries with LR genes: {sig_df['is_lr_gene'].sum()}")

# %% Filter to LR genes and annotate
lr_sig_df = sig_df[sig_df["is_lr_gene"]].copy()

lr_sig_df["role"] = lr_sig_df["gene"].map(
    lambda g: "/".join(sorted(gene_roles.get(g, set())))
)
lr_sig_df["lr_partners"] = lr_sig_df["gene"].map(
    lambda g: ", ".join(sorted(list(gene_partners.get(g, set()))[:5]))
    + ("..." if len(gene_partners.get(g, set())) > 5 else "")
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
print(f"\nCurated interactions with LR genes: {lr_sig_df_curated['interaction'].nunique()}")
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

fig, ax = plt.subplots(figsize=(max(12, len(unique_genes) * 0.5), max(6, len(unique_interactions) * 0.45)))

# Normalization for dot size and color
size_norm = Normalize(
    vmin=plot_df["nl10_pval_adj"].min(),
    vmax=plot_df["nl10_pval_adj"].max(),
)
color_norm = Normalize(
    vmin=plot_df["diff"].min(),
    vmax=plot_df["diff"].max(),
)
cmap = plt.cm.RdYlBu_r

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
ax.set_xticklabels(sorted(unique_genes), rotation=90, fontsize=8)
ax.set_yticks(range(len(interaction_to_y)))
ax.set_yticklabels(unique_interactions, fontsize=9)
ax.set_xlabel("Gene", fontsize=11)
ax.set_ylabel("Cell Type Interaction", fontsize=11)
ax.set_title("Significant Downstream LR Genes per Cell Type Interaction", fontsize=13)
ax.set_xlim(-0.5, len(gene_to_x) - 0.5)
ax.set_ylim(-0.5, len(interaction_to_y) - 0.5)
ax.grid(True, alpha=0.15)

# Color bar for diff
sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
cbar.set_label("Neighbor Contribution", fontsize=10)

# Size legend
size_values = [2, 5, 8]
size_labels = ["2", "5", "8"]
size_handles = [
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=np.sqrt(30 + 200 * size_norm(v)), label=l)
    for v, l in zip(size_values, size_labels)
]

# Marker legend for ligand/receptor
ligand_handle = mlines.Line2D([], [], color="gray", marker="o", linestyle="None", markersize=8, label="Ligand")
receptor_handle = mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=8, label="Receptor")

all_handles = [ligand_handle, receptor_handle] + size_handles
legend_labels = ["Ligand", "Receptor"] + [f"-log10(padj)={l}" for l in size_labels]
ax.legend(all_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.15, 1.0), fontsize=8, title="Marker / Size", labelspacing=1.2)

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
