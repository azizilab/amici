# %% Imports
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy import sparse
from scipy.stats import f_oneway, wilcoxon
from statsmodels.stats.multitest import multipletests

# %% Constants, palettes, and grid search parameter ranges
labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"
seed = 18
MIN_CELLS_PER_GROUP = 30

CELL_TYPE_PALETTE = {
    "CD8+_T_Cells": "#56B4E9",
    "CD4+_T_Cells": "#009E4E",
    "DCIS_1": "#E69F00",
    "DCIS_2": "#1a476e",
    "IRF7+_DCs": "#7f7f7f",
    "LAMP3+_DCs": "#305738",
    "Macrophages_1": "#e0a4dc",
    "Macrophages_2": "#de692a",
    "Myoepi_ACTA2+": "#823960",
    "Myoepi_KRT15+": "#575396",
    "Invasive_Tumor": "#cf4242",
    "Stromal": "#968253",
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Perivascular-Like": "#872727",
    "Endothelial": "#277987",
}

quantile_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
n_clusters_range = [4, 6, 8, 10, 12, 14, 16]

hub_cache_dir = "data/xenium_sample1/grid_search_cache/hub_results"
comp_cache_dir = "data/xenium_sample1/grid_search_cache/composition_results"

os.makedirs("figures/hub_validation_anova", exist_ok=True)

# %% Load adata and cluster assignments (q=0.9, k=10)
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
adata = adata[adata.obs["sample"] == "0"]  # Rep1 only, ~119k cells

# Primary hub params
primary_q = 0.9
primary_k = 10

hub_cache_path = os.path.join(hub_cache_dir, f"hub_q{primary_q}_k{primary_k}.csv")
comp_cache_path = os.path.join(comp_cache_dir, f"composition_k{primary_k}.csv")

hub_df = pd.read_csv(hub_cache_path, index_col=0)
adata.obs["hub_cluster"] = hub_df.loc[adata.obs_names, "hub_cluster"].astype(int).astype("category")

comp_df = pd.read_csv(comp_cache_path, index_col=0)
adata.obs["composition_cluster"] = comp_df.loc[adata.obs_names, "composition_cluster"].astype(int).astype("category")

print(f"Loaded adata: {adata.shape}")
print(f"Hub clusters (q={primary_q}, k={primary_k}): {adata.obs['hub_cluster'].nunique()} unique")
print(f"Composition clusters (k={primary_k}): {adata.obs['composition_cluster'].nunique()} unique")

# %% Helper: extract dense expression DataFrame for an adata subset
def get_expression_df(adata_subset):
    """Return a DataFrame of shape (n_cells, n_genes) with dense expression values."""
    X = adata_subset.X
    if sparse.issparse(X):
        X = X.toarray()
    return pd.DataFrame(X, index=adata_subset.obs_names, columns=adata_subset.var_names)

# %% ANOVA function: within-cell-type, per-gene
def run_within_ct_anova(adata, cluster_key, cell_type, labels_key, min_cells=30):
    """
    For a given cell type, run one-way ANOVA per gene across clusters.

    Returns a DataFrame with columns: gene, F_stat, pval, pval_adj, significant.
    Returns None if fewer than 2 valid groups (each with >= min_cells).
    """
    # Subset to cell type
    mask = adata.obs[labels_key] == cell_type
    adata_ct = adata[mask]

    # Get cluster groups and filter by min_cells
    cluster_counts = adata_ct.obs[cluster_key].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cells].index.tolist()

    if len(valid_clusters) < 2:
        return None

    # Further subset to only cells in valid clusters
    adata_ct = adata_ct[adata_ct.obs[cluster_key].isin(valid_clusters)]

    # Get expression matrix
    expr_df = get_expression_df(adata_ct)
    clusters = adata_ct.obs[cluster_key].values

    genes = expr_df.columns.tolist()
    results = []

    for gene in genes:
        gene_vals = expr_df[gene].values
        groups = [gene_vals[clusters == c] for c in valid_clusters]

        # Check that at least one group has nonzero variance
        if all(np.var(g) == 0 for g in groups):
            results.append({"gene": gene, "F_stat": 0.0, "pval": 1.0})
            continue

        try:
            f_stat, pval = f_oneway(*groups)
            if np.isnan(f_stat):
                f_stat = 0.0
                pval = 1.0
        except Exception:
            f_stat = 0.0
            pval = 1.0

        results.append({"gene": gene, "F_stat": f_stat, "pval": pval})

    results_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    reject, pval_adj, _, _ = multipletests(results_df["pval"], method="fdr_bh", alpha=0.05)
    results_df["pval_adj"] = pval_adj
    results_df["significant"] = reject

    return results_df

# %% Run ANOVA across all cell types for hub and composition clusters
cell_types = sorted(adata.obs[labels_key].unique().tolist())

hub_results = {}
comp_results = {}

print("Running within-cell-type ANOVA for hub clusters...")
for ct in tqdm(cell_types):
    res = run_within_ct_anova(adata, "hub_cluster", ct, labels_key, min_cells=MIN_CELLS_PER_GROUP)
    if res is not None:
        hub_results[ct] = res

print("\nRunning within-cell-type ANOVA for composition clusters...")
for ct in tqdm(cell_types):
    res = run_within_ct_anova(adata, "composition_cluster", ct, labels_key, min_cells=MIN_CELLS_PER_GROUP)
    if res is not None:
        comp_results[ct] = res

# Cell types that have results for both
common_cts = sorted(set(hub_results.keys()) & set(comp_results.keys()))
print(f"\nCell types with valid ANOVA results for both methods: {len(common_cts)}")
for ct in common_cts:
    print(f"  {ct}: hub {hub_results[ct]['significant'].sum()} sig, comp {comp_results[ct]['significant'].sum()} sig")

# %% Summary table
summary_rows = []
for ct in common_cts:
    h = hub_results[ct]
    c = comp_results[ct]
    hub_n_sig = int(h["significant"].sum())
    comp_n_sig = int(c["significant"].sum())
    hub_median_F = float(h["F_stat"].median())
    comp_median_F = float(c["F_stat"].median())
    hub_excess = hub_n_sig - comp_n_sig
    summary_rows.append({
        "cell_type": ct,
        "hub_n_sig": hub_n_sig,
        "comp_n_sig": comp_n_sig,
        "hub_median_F": round(hub_median_F, 2),
        "comp_median_F": round(comp_median_F, 2),
        "hub_excess": hub_excess,
    })

summary_df = pd.DataFrame(summary_rows).sort_values("hub_excess", ascending=False).reset_index(drop=True)
print("\n=== Summary: Within-Cell-Type ANOVA (Hub vs Composition) ===")
print(f"Parameters: q={primary_q}, k={primary_k}, MIN_CELLS_PER_GROUP={MIN_CELLS_PER_GROUP}")
print(summary_df.to_string(index=False))

# %% Grouped bar chart: # significant genes per cell type
fig, ax = plt.subplots(figsize=(8, max(6, len(common_cts) * 0.5)))

# Sort by hub_excess descending — ascending=True so highest hub_excess appears at top
plot_df = summary_df.sort_values("hub_excess", ascending=True)

y_pos = np.arange(len(plot_df))
bar_height = 0.35

ax.barh(y_pos + bar_height / 2, plot_df["hub_n_sig"], bar_height,
        label="Hub clusters", color="#56B4E9", edgecolor="white", linewidth=0.5)
ax.barh(y_pos - bar_height / 2, plot_df["comp_n_sig"], bar_height,
        label="Composition clusters", color="#E69F00", edgecolor="white", linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["cell_type"], fontsize=10)
ax.set_xlabel("Number of significant genes (FDR < 0.05)", fontsize=11)
ax.set_title(
    f"Within-cell-type ANOVA: significant genes\n(q={primary_q}, k={primary_k})",
    fontsize=12,
)
ax.legend(loc="lower right", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("figures/hub_validation_anova/anova_sig_genes_barplot.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/hub_validation_anova/anova_sig_genes_barplot.svg", bbox_inches="tight")
plt.show()

# %% F-statistic scatter plots for focus cell types
focus_cts = ["Invasive_Tumor", "DCIS_1", "DCIS_2"]
focus_cts = [ct for ct in focus_cts if ct in common_cts]

fig, axes = plt.subplots(1, len(focus_cts), figsize=(5 * len(focus_cts), 5))
if len(focus_cts) == 1:
    axes = [axes]

for ax, ct in zip(axes, focus_cts):
    h = hub_results[ct]
    c = comp_results[ct]

    # Merge on gene for paired comparison
    merged = h[["gene", "F_stat"]].merge(
        c[["gene", "F_stat"]], on="gene", suffixes=("_hub", "_comp")
    )

    ax.scatter(
        merged["F_stat_comp"], merged["F_stat_hub"],
        s=20, alpha=0.6, color=CELL_TYPE_PALETTE.get(ct, "#333333"), edgecolors="none",
    )

    # Diagonal reference line
    max_val = max(merged["F_stat_hub"].max(), merged["F_stat_comp"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)

    # Wilcoxon signed-rank test on paired F-statistics
    try:
        stat, wpval = wilcoxon(merged["F_stat_hub"], merged["F_stat_comp"])
        pval_str = f"p = {wpval:.2e}" if wpval < 0.01 else f"p = {wpval:.3f}"
    except Exception:
        pval_str = "p = N/A"

    ax.text(
        0.05, 0.92, f"Wilcoxon {pval_str}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.set_xlabel("F-statistic (Composition)", fontsize=10)
    ax.set_ylabel("F-statistic (Hub)", fontsize=10)
    ax.set_title(ct, fontsize=12, color=CELL_TYPE_PALETTE.get(ct, "#333333"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle(
    f"Per-gene F-statistic: Hub vs Composition (q={primary_q}, k={primary_k})",
    fontsize=13, y=1.03,
)
plt.tight_layout()
fig.savefig("figures/hub_validation_anova/anova_fstat_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/hub_validation_anova/anova_fstat_scatter.svg", bbox_inches="tight")
plt.show()

# %% Sensitivity heatmap: sweep over all cached (q, k) hub params, per cell type
print("Running sensitivity analysis across all (q, k) combinations...")

# Store per-cell-type excess: {cell_type: np.array of shape (n_k, n_q)}
per_ct_heatmap = {ct: np.full((len(n_clusters_range), len(quantile_thresholds)), np.nan)
                  for ct in cell_types}

for i, k in enumerate(tqdm(n_clusters_range, desc="k")):
    # Load composition clusters for this k (matched)
    comp_path = os.path.join(comp_cache_dir, f"composition_k{k}.csv")
    if not os.path.exists(comp_path):
        continue
    comp_df_k = pd.read_csv(comp_path, index_col=0)

    for j, q in enumerate(quantile_thresholds):
        hub_path = os.path.join(hub_cache_dir, f"hub_q{q}_k{k}.csv")
        if not os.path.exists(hub_path):
            continue

        # Load cluster assignments
        hub_df_qk = pd.read_csv(hub_path, index_col=0)

        # Assign clusters to temporary obs columns
        adata.obs["_hub_tmp"] = hub_df_qk.loc[adata.obs_names, "hub_cluster"].astype(int).astype("category")
        adata.obs["_comp_tmp"] = comp_df_k.loc[adata.obs_names, "composition_cluster"].astype(int).astype("category")

        for ct in cell_types:
            h_res = run_within_ct_anova(adata, "_hub_tmp", ct, labels_key, min_cells=MIN_CELLS_PER_GROUP)
            c_res = run_within_ct_anova(adata, "_comp_tmp", ct, labels_key, min_cells=MIN_CELLS_PER_GROUP)

            if h_res is not None and c_res is not None:
                per_ct_heatmap[ct][i, j] = int(h_res["significant"].sum()) - int(c_res["significant"].sum())

# Clean up temporary columns
adata.obs.drop(columns=["_hub_tmp", "_comp_tmp"], inplace=True, errors="ignore")

# Filter to cell types that have at least one non-NaN entry
valid_cts = [ct for ct in cell_types if not np.all(np.isnan(per_ct_heatmap[ct]))]
print(f"Cell types with valid sensitivity data: {len(valid_cts)}")

# Plot one heatmap per cell type
n_cols = 4
n_rows = int(np.ceil(len(valid_cts) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

# Determine a shared symmetric color range across all cell types
all_vals = np.concatenate([per_ct_heatmap[ct].ravel() for ct in valid_cts])
all_vals = all_vals[np.isfinite(all_vals)]
vabs = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) > 0 else 1

for idx, ct in enumerate(valid_cts):
    ax = axes[idx // n_cols, idx % n_cols]
    hm_df = pd.DataFrame(
        per_ct_heatmap[ct],
        index=[str(kk) for kk in n_clusters_range],
        columns=[str(qq) for qq in quantile_thresholds],
    )
    sns.heatmap(
        hm_df.astype(float),
        annot=True,
        fmt=".0f",
        cmap="RdBu_r",
        center=0,
        vmin=-vabs,
        vmax=vabs,
        linewidths=0.5,
        ax=ax,
        cbar=False,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(ct.replace("_", " "), fontsize=10, fontweight="bold",
                 color=CELL_TYPE_PALETTE.get(ct, "#333333"))
    ax.set_xlabel("q", fontsize=9)
    ax.set_ylabel("k", fontsize=9)
    ax.tick_params(labelsize=8)

# Hide unused axes
for idx in range(len(valid_cts), n_rows * n_cols):
    axes[idx // n_cols, idx % n_cols].set_visible(False)

# Add a shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vabs, vmax=vabs))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Hub excess sig. genes")

fig.suptitle("Sensitivity: Hub excess significant genes per cell type across (q, k)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 0.92, 0.98])
fig.savefig("figures/hub_validation_anova/anova_sensitivity_per_celltype.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/hub_validation_anova/anova_sensitivity_per_celltype.svg", bbox_inches="tight")
plt.show()

# Also plot the aggregate (sum across cell types) heatmap
agg_data = np.nansum([per_ct_heatmap[ct] for ct in valid_cts], axis=0)
agg_df = pd.DataFrame(
    agg_data,
    index=[str(kk) for kk in n_clusters_range],
    columns=[str(qq) for qq in quantile_thresholds],
)

fig, ax = plt.subplots(figsize=(8, 6))
vabs_agg = max(abs(np.nanmin(agg_data)), abs(np.nanmax(agg_data)))

sns.heatmap(
    agg_df.astype(float),
    annot=True,
    fmt=".0f",
    cmap="RdBu_r",
    center=0,
    vmin=-vabs_agg,
    vmax=vabs_agg,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Hub excess significant genes\n(sum across cell types)"},
)

ax.set_xlabel("Quantile threshold (q)", fontsize=11)
ax.set_ylabel("Number of clusters (k)", fontsize=11)
ax.set_title("Sensitivity: Hub excess significant genes across (q, k)", fontsize=12)

plt.tight_layout()
fig.savefig("figures/hub_validation_anova/anova_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/hub_validation_anova/anova_sensitivity_heatmap.svg", bbox_inches="tight")
plt.show()

print("\nDone. All figures saved to figures/hub_validation_anova/")

# %% Save summary to markdown
md_path = "figures/hub_validation_anova/anova_summary.md"
with open(md_path, "w") as f:
    f.write("# Within-Cell-Type ANOVA: Hub vs Composition Clusters\n\n")
    f.write(f"**Parameters**: q={primary_q}, k={primary_k}, MIN_CELLS_PER_GROUP={MIN_CELLS_PER_GROUP}\n\n")

    # Summary table
    f.write("## Significant Genes per Cell Type (FDR < 0.05)\n\n")
    f.write("| Cell Type | Hub Sig | Comp Sig | Hub Median F | Comp Median F | Hub Excess |\n")
    f.write("|-----------|---------|----------|--------------|---------------|------------|\n")
    for _, row in summary_df.iterrows():
        f.write(f"| {row['cell_type']} | {row['hub_n_sig']} | {row['comp_n_sig']} "
                f"| {row['hub_median_F']:.2f} | {row['comp_median_F']:.2f} "
                f"| {row['hub_excess']:+d} |\n")

    # Totals
    total_hub = summary_df["hub_n_sig"].sum()
    total_comp = summary_df["comp_n_sig"].sum()
    total_excess = summary_df["hub_excess"].sum()
    f.write(f"| **Total** | **{total_hub}** | **{total_comp}** | | | **{total_excess:+d}** |\n\n")

    # Per-cell-type results
    f.write("## Per-Cell-Type Details\n\n")
    for ct in common_cts:
        h = hub_results[ct]
        c = comp_results[ct]
        hub_sig_genes = sorted(h.loc[h["significant"], "gene"].tolist())
        comp_sig_genes = sorted(c.loc[c["significant"], "gene"].tolist())
        hub_only = sorted(set(hub_sig_genes) - set(comp_sig_genes))
        comp_only = sorted(set(comp_sig_genes) - set(hub_sig_genes))
        shared = sorted(set(hub_sig_genes) & set(comp_sig_genes))

        f.write(f"### {ct}\n\n")
        f.write(f"- Hub significant: {len(hub_sig_genes)} genes\n")
        f.write(f"- Composition significant: {len(comp_sig_genes)} genes\n")
        f.write(f"- Shared: {len(shared)} genes\n")
        f.write(f"- Hub-only: {len(hub_only)} genes\n")
        f.write(f"- Comp-only: {len(comp_only)} genes\n\n")
        if len(hub_only) > 0:
            f.write(f"**Hub-only genes**: {', '.join(hub_only)}\n\n")
        if len(comp_only) > 0:
            f.write(f"**Comp-only genes**: {', '.join(comp_only)}\n\n")

print(f"Summary saved to {md_path}")
