# %% Imports
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import gseapy as gp
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% Constants, palettes, paths, focus cell types
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

FOCUS_CELL_TYPES = ["Invasive_Tumor", "DCIS_1", "DCIS_2"]

# Matched parameters: same k for hub and composition
k = 10
q = 0.9
seed = 18
MIN_CELLS_PER_CLUSTER = 30
FDR_THRESHOLD = 0.25

labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"

hub_path = f"data/xenium_sample1/grid_search_cache/hub_results/hub_q{q}_k{k}.csv"
comp_path = f"data/xenium_sample1/grid_search_cache/composition_results/composition_k{k}.csv"

fig_dir = "figures/hub_validation_gsea"
os.makedirs(fig_dir, exist_ok=True)

# Gene set libraries for GSEA
PRIMARY_GENE_SETS = "MSigDB_Hallmark_2020"
FALLBACK_GENE_SETS = "GO_Biological_Process_2023"

# Method display names and colors
METHOD_CONFIG = {
    "hub_cluster": {"label": "Hub clusters", "color": "#1f77b4"},
    "composition_cluster": {"label": "Composition clusters", "color": "#ff7f0e"},
}

print(f"Parameters: k={k}, q={q}, seed={seed}")
print(f"Hub path:  {hub_path}")
print(f"Comp path: {comp_path}")
print(f"Figures:   {fig_dir}/")

# %% Load adata and cluster assignments
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
adata = adata[adata.obs["sample"] == "0"]  # Rep1 only, ~119k cells
print(f"Loaded adata: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Load hub clusters
hub_df = pd.read_csv(hub_path, index_col=0)
print(f"Hub clusters loaded: {hub_df.shape[0]} cells, {hub_df['hub_cluster'].nunique()} clusters")

# Load composition clusters
comp_df = pd.read_csv(comp_path, index_col=0)
print(f"Composition clusters loaded: {comp_df.shape[0]} cells, {comp_df['composition_cluster'].nunique()} clusters")

# Align and assign to adata.obs
adata.obs["hub_cluster"] = hub_df["hub_cluster"].loc[adata.obs_names].astype(str)
adata.obs["composition_cluster"] = comp_df["composition_cluster"].loc[adata.obs_names].astype(str)

# Check for NaN after alignment
n_hub_na = adata.obs["hub_cluster"].isna().sum()
n_comp_na = adata.obs["composition_cluster"].isna().sum()
print(f"Hub NaN: {n_hub_na}, Composition NaN: {n_comp_na}")

# Print cell counts per focus cell type
for ct in FOCUS_CELL_TYPES:
    n = (adata.obs[labels_key] == ct).sum()
    print(f"  {ct}: {n} cells")

# %% Compute DEGs via scanpy rank_genes_groups for each focus cell type x method
deg_results = {}  # (cell_type, method, cluster_id) -> DataFrame

for ct in FOCUS_CELL_TYPES:
    ct_mask = adata.obs[labels_key] == ct
    ct_adata = adata[ct_mask].copy()
    print(f"\n{'='*60}")
    print(f"Cell type: {ct} ({ct_adata.shape[0]} cells)")
    print(f"{'='*60}")

    for method in ["hub_cluster", "composition_cluster"]:
        cluster_col = method
        cluster_counts = ct_adata.obs[cluster_col].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= MIN_CELLS_PER_CLUSTER].index.tolist()

        print(f"\n  {METHOD_CONFIG[method]['label']}:")
        print(f"    Total clusters present: {ct_adata.obs[cluster_col].nunique()}")
        print(f"    Clusters with >= {MIN_CELLS_PER_CLUSTER} cells: {len(valid_clusters)}")

        if len(valid_clusters) < 2:
            print(f"    SKIPPING: need at least 2 valid clusters for DEG analysis")
            continue

        # Subset to cells in valid clusters only
        valid_mask = ct_adata.obs[cluster_col].isin(valid_clusters)
        ct_sub = ct_adata[valid_mask].copy()

        # Run rank_genes_groups
        sc.tl.rank_genes_groups(ct_sub, groupby=cluster_col, method="wilcoxon")

        # Extract DEGs per cluster
        for clust in valid_clusters:
            deg_df = sc.get.rank_genes_groups_df(ct_sub, group=str(clust))
            n_sig = (deg_df["pvals_adj"] < 0.05).sum()
            deg_results[(ct, method, str(clust))] = deg_df
            print(f"    Cluster {clust}: {cluster_counts[clust]} cells, "
                  f"{n_sig} significant DEGs (adj.p < 0.05)")

print(f"\nTotal DEG result sets: {len(deg_results)}")

# %% Run GSEA with gseapy.prerank for each (cell_type, method, cluster)
gsea_all_results = []  # list of dicts for summary DataFrame

for (ct, method, clust), deg_df in tqdm(deg_results.items(), desc="Running GSEA"):
    # Build ranked gene list: gene names as index, Wilcoxon scores as values
    ranked = deg_df.set_index("names")["scores"].dropna()
    ranked = ranked[~ranked.index.duplicated(keep="first")]
    ranked = ranked.sort_values(ascending=False)

    if len(ranked) < 5:
        print(f"  Skipping {ct}/{method}/cluster {clust}: only {len(ranked)} genes")
        continue

    # Try primary gene sets (Hallmark), then fallback (GO BP) if too sparse
    for gene_set_name in [PRIMARY_GENE_SETS, FALLBACK_GENE_SETS]:
        try:
            pre_res = gp.prerank(
                rnk=ranked,
                gene_sets=gene_set_name,
                min_size=5,
                max_size=500,
                permutation_num=1000,
                seed=seed,
                no_plot=True,
                verbose=False,
            )
            res_df = pre_res.res2d
            if res_df is None or len(res_df) == 0:
                continue

            # Ensure numeric columns
            for col in ["NES", "FDR q-val", "FWER p-val", "NOM p-val"]:
                if col in res_df.columns:
                    res_df[col] = pd.to_numeric(res_df[col], errors="coerce")

            # Parse matched_size from "Tag %" column (format: "hits/matched_size")
            def parse_matched_size(tag_pct):
                try:
                    return int(str(tag_pct).split("/")[1])
                except (IndexError, ValueError):
                    return np.nan

            for _, row in res_df.iterrows():
                matched_size = parse_matched_size(row.get("Tag %", ""))
                gsea_all_results.append({
                    "cell_type": ct,
                    "method": method,
                    "cluster": clust,
                    "gene_set_library": gene_set_name,
                    "pathway": row.get("Term", row.name),
                    "nes": row.get("NES", np.nan),
                    "fdr": row.get("FDR q-val", np.nan),
                    "matched_size": matched_size,
                    "lead_genes": row.get("Lead_genes", ""),
                })

        except Exception as e:
            print(f"  GSEA failed for {ct}/{method}/cluster {clust} "
                  f"({gene_set_name}): {type(e).__name__}: {e}")
            continue

# Build summary DataFrame
gsea_df = pd.DataFrame(gsea_all_results)
print(f"\nTotal GSEA result rows: {len(gsea_df)}")

if len(gsea_df) > 0:
    gsea_df["fdr"] = pd.to_numeric(gsea_df["fdr"], errors="coerce")
    gsea_df["nes"] = pd.to_numeric(gsea_df["nes"], errors="coerce")
    gsea_df["matched_size"] = pd.to_numeric(gsea_df["matched_size"], errors="coerce")
    gsea_df["significant"] = gsea_df["fdr"] < FDR_THRESHOLD

    # Report overlap stats
    for lib in gsea_df["gene_set_library"].unique():
        lib_df = gsea_df[gsea_df["gene_set_library"] == lib]
        avg_matched = lib_df["matched_size"].mean()
        print(f"  {lib}: {len(lib_df)} pathway-cluster tests, "
              f"avg matched_size = {avg_matched:.1f}")

# %% Summary table: significant pathways per cell type x method
print("\n" + "=" * 70)
print("GSEA SUMMARY: Significant pathways (FDR < 0.25) per cell type x method")
print("=" * 70)

summary_rows = []

for gene_set_lib in gsea_df["gene_set_library"].unique():
    lib_df = gsea_df[gsea_df["gene_set_library"] == gene_set_lib]
    print(f"\n--- Gene set library: {gene_set_lib} ---")

    for ct in FOCUS_CELL_TYPES:
        for method in ["hub_cluster", "composition_cluster"]:
            subset = lib_df[(lib_df["cell_type"] == ct) & (lib_df["method"] == method)]
            n_sig = subset["significant"].sum()
            n_total = len(subset)
            # Count unique significant pathways (across clusters)
            unique_sig_pathways = subset.loc[subset["significant"], "pathway"].nunique()

            summary_rows.append({
                "gene_set_library": gene_set_lib,
                "cell_type": ct,
                "method": method,
                "method_label": METHOD_CONFIG[method]["label"],
                "n_significant_tests": int(n_sig),
                "n_total_tests": n_total,
                "n_unique_sig_pathways": unique_sig_pathways,
            })

            print(f"  {ct:20s} | {METHOD_CONFIG[method]['label']:25s} | "
                  f"{unique_sig_pathways:3d} unique sig pathways "
                  f"({n_sig}/{n_total} tests)")

summary_df = pd.DataFrame(summary_rows)

# Print head-to-head comparison
print("\n--- Head-to-head comparison (unique significant pathways) ---")
for gene_set_lib in summary_df["gene_set_library"].unique():
    lib_summary = summary_df[summary_df["gene_set_library"] == gene_set_lib]
    print(f"\n  {gene_set_lib}:")
    for ct in FOCUS_CELL_TYPES:
        hub_n = lib_summary.loc[
            (lib_summary["cell_type"] == ct) & (lib_summary["method"] == "hub_cluster"),
            "n_unique_sig_pathways"
        ].values
        comp_n = lib_summary.loc[
            (lib_summary["cell_type"] == ct) & (lib_summary["method"] == "composition_cluster"),
            "n_unique_sig_pathways"
        ].values
        hub_n = hub_n[0] if len(hub_n) > 0 else 0
        comp_n = comp_n[0] if len(comp_n) > 0 else 0
        winner = "HUB" if hub_n > comp_n else ("COMP" if comp_n > hub_n else "TIE")
        print(f"    {ct:20s}: Hub={hub_n:3d}  Comp={comp_n:3d}  -> {winner}")

# %% Stacked bar chart: two bars per cell type (hub, comp), each split into unique + shared
fig, axes = plt.subplots(1, len(gsea_df["gene_set_library"].unique()),
                         figsize=(7 * len(gsea_df["gene_set_library"].unique()), 5),
                         squeeze=False)

hub_color = METHOD_CONFIG["hub_cluster"]["color"]
comp_color = METHOD_CONFIG["composition_cluster"]["color"]
# Lighter versions for the "shared" segment
hub_shared_color = "#7fbce6"
comp_shared_color = "#ffbf7f"

for ax_idx, gene_set_lib in enumerate(gsea_df["gene_set_library"].unique()):
    ax = axes[0, ax_idx]
    lib_df = gsea_df[gsea_df["gene_set_library"] == gene_set_lib]

    hub_unique_vals = []
    hub_shared_vals = []
    comp_unique_vals = []
    comp_shared_vals = []

    for ct in FOCUS_CELL_TYPES:
        ct_lib = lib_df[lib_df["cell_type"] == ct]
        hub_sig = set(ct_lib[
            (ct_lib["method"] == "hub_cluster") & (ct_lib["significant"])
        ]["pathway"].unique())
        comp_sig = set(ct_lib[
            (ct_lib["method"] == "composition_cluster") & (ct_lib["significant"])
        ]["pathway"].unique())

        shared = hub_sig & comp_sig
        hub_unique_vals.append(len(hub_sig - comp_sig))
        hub_shared_vals.append(len(shared))
        comp_unique_vals.append(len(comp_sig - hub_sig))
        comp_shared_vals.append(len(shared))

    x = np.arange(len(FOCUS_CELL_TYPES))
    width = 0.35

    # Hub bar: unique (bottom, dark) + shared (top, light)
    ax.bar(x - width / 2, hub_unique_vals, width,
           label="Hub unique", color=hub_color, edgecolor="black", linewidth=0.5)
    ax.bar(x - width / 2, hub_shared_vals, width, bottom=hub_unique_vals,
           label="Shared", color=hub_shared_color, edgecolor="black", linewidth=0.5)

    # Comp bar: unique (bottom, dark) + shared (top, light)
    ax.bar(x + width / 2, comp_unique_vals, width,
           label="Comp unique", color=comp_color, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, comp_shared_vals, width, bottom=comp_unique_vals,
           label="Shared ", color=comp_shared_color, edgecolor="black", linewidth=0.5)

    # Add count labels inside each segment
    for x_off, unique_vals, shared_vals_list in [
        (-width / 2, hub_unique_vals, hub_shared_vals),
        (width / 2, comp_unique_vals, comp_shared_vals),
    ]:
        for i, (u, s) in enumerate(zip(unique_vals, shared_vals_list)):
            if u > 0:
                ax.text(x[i] + x_off, u / 2, str(u),
                        ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            if s > 0:
                ax.text(x[i] + x_off, u + s / 2, str(s),
                        ha="center", va="center", fontsize=9, fontweight="bold", color="#333333")

    # Total label on top of each bar
    for i in range(len(FOCUS_CELL_TYPES)):
        hub_total = hub_unique_vals[i] + hub_shared_vals[i]
        comp_total = comp_unique_vals[i] + comp_shared_vals[i]
        if hub_total > 0:
            ax.text(x[i] - width / 2, hub_total + 0.2, str(hub_total),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        if comp_total > 0:
            ax.text(x[i] + width / 2, comp_total + 0.2, str(comp_total),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Cell type", fontsize=12)
    ax.set_ylabel("Significant pathways\n(FDR < 0.25)", fontsize=12)
    ax.set_title(gene_set_lib.replace("_", " "), fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ct_labels = [ct.replace("_", " ") for ct in FOCUS_CELL_TYPES]
    ax.set_xticklabels(ct_labels, fontsize=11)
    # Deduplicate "Shared" in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Rename to cleaner labels
    clean_labels = {"Hub unique": "Hub unique", "Shared": "Shared",
                    "Comp unique": "Comp unique", "Shared ": "Shared"}
    deduped = {}
    for lbl, hdl in by_label.items():
        clean = clean_labels.get(lbl, lbl)
        if clean not in deduped:
            deduped[clean] = hdl
    ax.legend(deduped.values(), deduped.keys(), fontsize=10, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

fig.suptitle("GSEA: Hub vs Composition Cluster DEGs", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

fig.savefig(os.path.join(fig_dir, "gsea_sig_pathways_barplot.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "gsea_sig_pathways_barplot.svg"),
            bbox_inches="tight")
print(f"Saved stacked bar chart to {fig_dir}/gsea_sig_pathways_barplot.png/.svg")
plt.show()

# %% Butterfly/mirror bar charts for all focus cell types
from matplotlib.patches import Patch


def clean_pathway_name(name):
    """Remove common prefixes and clean up pathway names for display."""
    name = name.replace("Hallmark ", "").replace("HALLMARK_", "").replace("HALLMARK ", "")
    name = name.replace("_", " ").title()
    if len(name) > 45:
        name = name[:42] + "..."
    return name


for focus_ct in FOCUS_CELL_TYPES:
    print(f"\n{'='*60}")
    print(f"Butterfly chart for: {focus_ct}")
    print(f"{'='*60}")

    # Use Hallmark results preferentially; fall back to GO BP if too sparse
    hallmark_ct = gsea_df[
        (gsea_df["cell_type"] == focus_ct) & (gsea_df["gene_set_library"] == PRIMARY_GENE_SETS)
    ]
    if len(hallmark_ct) > 0 and hallmark_ct["significant"].sum() > 0:
        butterfly_lib = PRIMARY_GENE_SETS
    else:
        butterfly_lib = FALLBACK_GENE_SETS
        print(f"  Hallmark too sparse for {focus_ct}, using {FALLBACK_GENE_SETS}")

    ct_gsea = gsea_df[
        (gsea_df["cell_type"] == focus_ct) & (gsea_df["gene_set_library"] == butterfly_lib)
    ]

    # Aggregate: for each pathway x method, take the cluster with the lowest FDR
    # (most significant result), so significance counts match the bar chart.
    butterfly_data = []
    for method in ["hub_cluster", "composition_cluster"]:
        method_df = ct_gsea[ct_gsea["method"] == method]
        if len(method_df) == 0:
            continue
        for pathway in method_df["pathway"].unique():
            pw_df = method_df[method_df["pathway"] == pathway]
            best_idx = pw_df["fdr"].idxmin()
            best_row = pw_df.loc[best_idx]
            butterfly_data.append({
                "pathway": pathway,
                "method": method,
                "nes": best_row["nes"],
                "fdr": best_row["fdr"],
                "matched_size": best_row["matched_size"],
                "significant": best_row["fdr"] < FDR_THRESHOLD,
            })

    butterfly_df = pd.DataFrame(butterfly_data)

    # Filter to pathways significant in at least one method
    sig_pathways = butterfly_df.loc[butterfly_df["significant"], "pathway"].unique()
    butterfly_df = butterfly_df[butterfly_df["pathway"].isin(sig_pathways)]
    print(f"  Pathways significant in at least one method: {len(sig_pathways)}")

    if len(butterfly_df) == 0:
        print(f"  No significant pathways for {focus_ct}, skipping butterfly chart.")
        continue

    # Get union of all pathways
    all_pathways = butterfly_df["pathway"].unique()

    # Create a pivot for plotting: pathway x method
    hub_data = butterfly_df[butterfly_df["method"] == "hub_cluster"].set_index("pathway")
    comp_data = butterfly_df[butterfly_df["method"] == "composition_cluster"].set_index("pathway")

    # Sort pathways by hub |NES| (descending)
    pathway_order = []
    for pw in all_pathways:
        hub_nes = hub_data.loc[pw, "nes"] if pw in hub_data.index else 0
        pathway_order.append((pw, abs(hub_nes)))
    pathway_order.sort(key=lambda x: x[1], reverse=True)
    pathways_sorted = [p[0] for p in pathway_order]

    fig, ax = plt.subplots(figsize=(14, max(6, len(pathways_sorted) * 0.45)))

    y_positions = np.arange(len(pathways_sorted))

    hub_color = METHOD_CONFIG["hub_cluster"]["color"]
    comp_color = METHOD_CONFIG["composition_cluster"]["color"]
    hub_faded = mcolors.to_rgba(hub_color, alpha=0.3)
    comp_faded = mcolors.to_rgba(comp_color, alpha=0.3)

    for i, pw in enumerate(pathways_sorted):
        # Hub bars extend to the LEFT (negative x)
        if pw in hub_data.index:
            h_nes = hub_data.loc[pw, "nes"]
            h_fdr = hub_data.loc[pw, "fdr"]
            h_sig = hub_data.loc[pw, "significant"]
            color = hub_color if h_sig else hub_faded
            ax.barh(i, -abs(h_nes), height=0.7, color=color,
                    edgecolor="black" if h_sig else "gray",
                    linewidth=0.8 if h_sig else 0.4)
            sign_str = "+" if h_nes > 0 else "-" if h_nes < 0 else ""
            label = f"{sign_str}{abs(h_nes):.2f} (q={h_fdr:.2f})"
            # Place annotation inside the bar to avoid overlapping y-axis labels
            ax.text(-abs(h_nes) / 2, i, label,
                    ha="center", va="center", fontsize=6.5,
                    color="white" if h_sig else "#444444", fontweight="bold")

        # Composition bars extend to the RIGHT (positive x)
        if pw in comp_data.index:
            c_nes = comp_data.loc[pw, "nes"]
            c_fdr = comp_data.loc[pw, "fdr"]
            c_sig = comp_data.loc[pw, "significant"]
            color = comp_color if c_sig else comp_faded
            ax.barh(i, abs(c_nes), height=0.7, color=color,
                    edgecolor="black" if c_sig else "gray",
                    linewidth=0.8 if c_sig else 0.4)
            sign_str = "+" if c_nes > 0 else "-" if c_nes < 0 else ""
            label = f"{sign_str}{abs(c_nes):.2f} (q={c_fdr:.2f})"
            # Place annotation inside the bar
            ax.text(abs(c_nes) / 2, i, label,
                    ha="center", va="center", fontsize=6.5,
                    color="white" if c_sig else "#444444", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([clean_pathway_name(pw) for pw in pathways_sorted], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|NES|", fontsize=12)
    ax.set_title(f"GSEA Butterfly Chart: {focus_ct.replace('_', ' ')}\n"
                 f"({butterfly_lib.replace('_', ' ')})",
                 fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor=hub_color, edgecolor="black", label="Hub (FDR < 0.25)"),
        Patch(facecolor=hub_faded, edgecolor="gray", label="Hub (n.s.)"),
        Patch(facecolor=comp_color, edgecolor="black", label="Composition (FDR < 0.25)"),
        Patch(facecolor=comp_faded, edgecolor="gray", label="Composition (n.s.)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)

    ax.text(-0.02, 1.02, "Hub clusters", transform=ax.transAxes,
            ha="right", fontsize=11, fontweight="bold", color=hub_color)
    ax.text(1.02, 1.02, "Composition clusters", transform=ax.transAxes,
            ha="left", fontsize=11, fontweight="bold", color=comp_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    ct_slug = focus_ct.lower().replace("+", "").replace("-", "_")
    fig.savefig(os.path.join(fig_dir, f"gsea_butterfly_{ct_slug}.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"gsea_butterfly_{ct_slug}.svg"),
                bbox_inches="tight")
    print(f"  Saved to {fig_dir}/gsea_butterfly_{ct_slug}.png/.svg")
    plt.show()

# %% Detailed pathway table: pathways significant in hub but NOT composition
print("\n" + "=" * 70)
print("PATHWAYS SIGNIFICANT IN HUB CLUSTERS BUT NOT COMPOSITION CLUSTERS")
print("(Unique biological signals captured by attention-based interaction)")
print("=" * 70)

for gene_set_lib in gsea_df["gene_set_library"].unique():
    lib_df = gsea_df[gsea_df["gene_set_library"] == gene_set_lib]
    print(f"\n--- {gene_set_lib} ---")

    for ct in FOCUS_CELL_TYPES:
        ct_lib = lib_df[lib_df["cell_type"] == ct]

        # Get significant pathways for hub
        hub_sig = ct_lib[
            (ct_lib["method"] == "hub_cluster") & (ct_lib["significant"])
        ]["pathway"].unique()

        # Get significant pathways for composition
        comp_sig = ct_lib[
            (ct_lib["method"] == "composition_cluster") & (ct_lib["significant"])
        ]["pathway"].unique()

        # Unique to hub
        hub_only = set(hub_sig) - set(comp_sig)
        # Unique to composition
        comp_only = set(comp_sig) - set(hub_sig)
        # Shared
        shared = set(hub_sig) & set(comp_sig)

        print(f"\n  {ct}:")
        print(f"    Hub-significant:  {len(hub_sig)} pathways")
        print(f"    Comp-significant: {len(comp_sig)} pathways")
        print(f"    Shared:           {len(shared)} pathways")
        print(f"    Hub-only:         {len(hub_only)} pathways")
        print(f"    Comp-only:        {len(comp_only)} pathways")

        if len(hub_only) > 0:
            print(f"\n    Hub-only pathways (unique to attention-based clustering):")
            # Get full details for hub-only pathways
            hub_details = ct_lib[
                (ct_lib["method"] == "hub_cluster") &
                (ct_lib["pathway"].isin(hub_only))
            ].sort_values("fdr")

            # Aggregate: best result per pathway
            for pw in sorted(hub_only):
                pw_rows = hub_details[hub_details["pathway"] == pw]
                best_idx = pw_rows["fdr"].idxmin()
                row = pw_rows.loc[best_idx]
                # Check if composition tested this pathway at all
                comp_pw = ct_lib[
                    (ct_lib["method"] == "composition_cluster") &
                    (ct_lib["pathway"] == pw)
                ]
                if len(comp_pw) > 0:
                    comp_best = comp_pw.loc[comp_pw["fdr"].idxmin()]
                    comp_info = f"comp NES={comp_best['nes']:.2f}, FDR={comp_best['fdr']:.3f}"
                else:
                    comp_info = "not tested in composition"

                matched_str = f"{int(row['matched_size'])}" if pd.notna(row['matched_size']) else "?"
                print(f"      {pw}")
                print(f"        Hub:  NES={row['nes']:.2f}, FDR={row['fdr']:.3f}, "
                      f"matched={matched_str} genes "
                      f"(cluster {row['cluster']})")
                print(f"        Comp: {comp_info}")

        if len(shared) > 0:
            print(f"\n    Shared significant pathways:")
            for pw in sorted(shared):
                hub_pw = ct_lib[
                    (ct_lib["method"] == "hub_cluster") & (ct_lib["pathway"] == pw)
                ]
                comp_pw = ct_lib[
                    (ct_lib["method"] == "composition_cluster") & (ct_lib["pathway"] == pw)
                ]
                h_best = hub_pw.loc[hub_pw["fdr"].idxmin()]
                c_best = comp_pw.loc[comp_pw["fdr"].idxmin()]
                print(f"      {pw}")
                print(f"        Hub:  NES={h_best['nes']:.2f}, FDR={h_best['fdr']:.3f}")
                print(f"        Comp: NES={c_best['nes']:.2f}, FDR={c_best['fdr']:.3f}")

print("\n" + "=" * 70)
print("NOTE: 250-gene Xenium panel limits overlap with hallmark gene sets.")
print("Small matched_size values indicate sparse overlap; interpret with caution.")
print("=" * 70)

# %% Save summary to markdown
md_path = os.path.join(fig_dir, "gsea_summary.md")
with open(md_path, "w") as f:
    f.write("# GSEA: Hub vs Composition Cluster DEGs\n\n")
    f.write(f"**Parameters**: q={q}, k={k}, seed={seed}\n\n")
    f.write(f"**FDR threshold**: {FDR_THRESHOLD}\n\n")
    f.write(f"**Gene set libraries**: {PRIMARY_GENE_SETS}, {FALLBACK_GENE_SETS}\n\n")

    # Summary table per gene set library
    f.write("## Significant Pathways per Cell Type (FDR < 0.25)\n\n")
    for gene_set_lib in summary_df["gene_set_library"].unique():
        lib_summary = summary_df[summary_df["gene_set_library"] == gene_set_lib]
        f.write(f"### {gene_set_lib}\n\n")
        f.write("| Cell Type | Method | Unique Sig Pathways | Sig Tests / Total |\n")
        f.write("|-----------|--------|--------------------:|------------------:|\n")
        for _, row in lib_summary.iterrows():
            f.write(f"| {row['cell_type']} | {row['method_label']} "
                    f"| {row['n_unique_sig_pathways']} "
                    f"| {row['n_significant_tests']}/{row['n_total_tests']} |\n")
        f.write("\n")

    # Head-to-head
    f.write("## Head-to-Head Comparison\n\n")
    for gene_set_lib in summary_df["gene_set_library"].unique():
        lib_summary = summary_df[summary_df["gene_set_library"] == gene_set_lib]
        f.write(f"### {gene_set_lib}\n\n")
        f.write("| Cell Type | Hub | Comp | Winner |\n")
        f.write("|-----------|----:|-----:|--------|\n")
        for ct in FOCUS_CELL_TYPES:
            hub_n = lib_summary.loc[
                (lib_summary["cell_type"] == ct) & (lib_summary["method"] == "hub_cluster"),
                "n_unique_sig_pathways"
            ].values
            comp_n = lib_summary.loc[
                (lib_summary["cell_type"] == ct) & (lib_summary["method"] == "composition_cluster"),
                "n_unique_sig_pathways"
            ].values
            hub_n = hub_n[0] if len(hub_n) > 0 else 0
            comp_n = comp_n[0] if len(comp_n) > 0 else 0
            winner = "Hub" if hub_n > comp_n else ("Comp" if comp_n > hub_n else "Tie")
            f.write(f"| {ct} | {hub_n} | {comp_n} | **{winner}** |\n")
        f.write("\n")

    # Detailed pathway comparison
    f.write("## Pathway Details per Cell Type\n\n")
    for gene_set_lib in gsea_df["gene_set_library"].unique():
        lib_df = gsea_df[gsea_df["gene_set_library"] == gene_set_lib]
        f.write(f"### {gene_set_lib}\n\n")

        for ct in FOCUS_CELL_TYPES:
            ct_lib = lib_df[lib_df["cell_type"] == ct]
            hub_sig = set(ct_lib[
                (ct_lib["method"] == "hub_cluster") & (ct_lib["significant"])
            ]["pathway"].unique())
            comp_sig = set(ct_lib[
                (ct_lib["method"] == "composition_cluster") & (ct_lib["significant"])
            ]["pathway"].unique())

            hub_only = sorted(hub_sig - comp_sig)
            comp_only = sorted(comp_sig - hub_sig)
            shared = sorted(hub_sig & comp_sig)

            f.write(f"#### {ct}\n\n")
            f.write(f"- Hub-significant: {len(hub_sig)} pathways\n")
            f.write(f"- Comp-significant: {len(comp_sig)} pathways\n")
            f.write(f"- Shared: {len(shared)}, Hub-only: {len(hub_only)}, Comp-only: {len(comp_only)}\n\n")

            if len(hub_only) > 0:
                f.write("**Hub-only pathways**:\n\n")
                f.write("| Pathway | Hub NES | Hub FDR | Comp NES | Comp FDR |\n")
                f.write("|---------|--------:|--------:|---------:|---------:|\n")
                for pw in hub_only:
                    hub_pw = ct_lib[(ct_lib["method"] == "hub_cluster") & (ct_lib["pathway"] == pw)]
                    h_best = hub_pw.loc[hub_pw["fdr"].idxmin()]
                    comp_pw = ct_lib[(ct_lib["method"] == "composition_cluster") & (ct_lib["pathway"] == pw)]
                    if len(comp_pw) > 0:
                        c_best = comp_pw.loc[comp_pw["fdr"].idxmin()]
                        f.write(f"| {pw} | {h_best['nes']:.2f} | {h_best['fdr']:.3f} "
                                f"| {c_best['nes']:.2f} | {c_best['fdr']:.3f} |\n")
                    else:
                        f.write(f"| {pw} | {h_best['nes']:.2f} | {h_best['fdr']:.3f} "
                                f"| - | - |\n")
                f.write("\n")

            if len(shared) > 0:
                f.write("**Shared pathways**:\n\n")
                f.write("| Pathway | Hub NES | Hub FDR | Comp NES | Comp FDR |\n")
                f.write("|---------|--------:|--------:|---------:|---------:|\n")
                for pw in shared:
                    hub_pw = ct_lib[(ct_lib["method"] == "hub_cluster") & (ct_lib["pathway"] == pw)]
                    comp_pw = ct_lib[(ct_lib["method"] == "composition_cluster") & (ct_lib["pathway"] == pw)]
                    h_best = hub_pw.loc[hub_pw["fdr"].idxmin()]
                    c_best = comp_pw.loc[comp_pw["fdr"].idxmin()]
                    f.write(f"| {pw} | {h_best['nes']:.2f} | {h_best['fdr']:.3f} "
                            f"| {c_best['nes']:.2f} | {c_best['fdr']:.3f} |\n")
                f.write("\n")

            if len(comp_only) > 0:
                f.write("**Comp-only pathways**:\n\n")
                f.write("| Pathway | Comp NES | Comp FDR | Hub NES | Hub FDR |\n")
                f.write("|---------|--------:|--------:|---------:|---------:|\n")
                for pw in comp_only:
                    comp_pw = ct_lib[(ct_lib["method"] == "composition_cluster") & (ct_lib["pathway"] == pw)]
                    c_best = comp_pw.loc[comp_pw["fdr"].idxmin()]
                    hub_pw = ct_lib[(ct_lib["method"] == "hub_cluster") & (ct_lib["pathway"] == pw)]
                    if len(hub_pw) > 0:
                        h_best = hub_pw.loc[hub_pw["fdr"].idxmin()]
                        f.write(f"| {pw} | {c_best['nes']:.2f} | {c_best['fdr']:.3f} "
                                f"| {h_best['nes']:.2f} | {h_best['fdr']:.3f} |\n")
                    else:
                        f.write(f"| {pw} | {c_best['nes']:.2f} | {c_best['fdr']:.3f} "
                                f"| - | - |\n")
                f.write("\n")

    f.write("---\n\n")
    f.write("**Note**: 250-gene Xenium panel limits overlap with hallmark gene sets. "
            "Small matched_size values indicate sparse overlap; interpret with caution.\n")

print(f"Summary saved to {md_path}")

# %%
