# %% Imports
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import gseapy as gp
import numpy as np
import pandas as pd
import os
import re
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% Constants, palettes, paths, cell type categories
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

# Cell type categories
TUMOR_CELL_TYPES = ["Invasive_Tumor", "DCIS_1", "DCIS_2"]
IMMUNE_CELL_TYPES = ["CD8+_T_Cells", "CD4+_T_Cells", "IRF7+_DCs", "LAMP3+_DCs",
                     "Macrophages_1", "Macrophages_2", "B_Cells", "Mast_Cells"]
STROMAL_CELL_TYPES = ["Stromal", "Myoepi_ACTA2+", "Myoepi_KRT15+",
                      "Perivascular-Like", "Endothelial"]
ALL_CELL_TYPES = TUMOR_CELL_TYPES + IMMUNE_CELL_TYPES + STROMAL_CELL_TYPES

# ---------------------------------------------------------------------------
# Gene set mode: "gsea", "kegg_signaling", or "reactome"
# ---------------------------------------------------------------------------
GENE_SET_MODE = "gsea"

def _kegg_signaling_filter(df):
    """Filter GSEA results to pathways containing 'signaling'."""
    mask = df["pathway"].str.contains("signaling", case=False, na=False)
    n_before = len(df)
    out = df[mask].copy().reset_index(drop=True)
    print(f"After filtering to 'signaling' pathways: {len(out)} rows "
          f"(removed {n_before - len(out)}), "
          f"{out['pathway'].nunique()} unique pathways")
    return out

_gsea_ct_sets = {}
for _ct in TUMOR_CELL_TYPES:
    _gsea_ct_sets[_ct] = "MSigDB_Hallmark_2020"
for _ct in IMMUNE_CELL_TYPES + STROMAL_CELL_TYPES:
    _gsea_ct_sets[_ct] = "Reactome_Pathways_2024"

GENE_SET_CONFIGS = {
    "gsea": {
        "ct_gene_sets": _gsea_ct_sets,
        "bar_chart_labels": ["Tumor (Hallmark)", "Immune (Reactome)", "Stromal (Reactome)"],
        "fig_dir_base": "hub_validation_gsea",
        "file_prefix": "gsea",
        "min_size": 5,
        "pathway_filter": None,
        "md_title": "GSEA",
        "md_gene_set_desc": (
            "**Gene set assignment**:\n"
            "- Tumor cell types: MSigDB_Hallmark_2020\n"
            "- Immune cell types: Reactome_Pathways_2024\n"
            "- Stromal cell types: Reactome_Pathways_2024\n"
        ),
        "md_note_extra": "",
        "display_name": "GSEA",
    },
    "kegg_signaling": {
        "ct_gene_sets": {ct: "KEGG_2021_Human" for ct in ALL_CELL_TYPES},
        "bar_chart_labels": ["Tumor (KEGG Signaling)", "Immune (KEGG Signaling)",
                             "Stromal (KEGG Signaling)"],
        "fig_dir_base": "hub_validation_kegg_signaling",
        "file_prefix": "kegg_signaling",
        "min_size": 3,
        "pathway_filter": _kegg_signaling_filter,
        "md_title": "KEGG Signaling",
        "md_gene_set_desc": "**Gene set library**: KEGG_2021_Human (filtered to signaling pathways)\n",
        "md_note_extra": " Only pathways containing 'signaling' in their name are included.",
        "display_name": "KEGG Signaling",
    },
    "reactome": {
        "ct_gene_sets": {ct: "Reactome_Pathways_2024" for ct in ALL_CELL_TYPES},
        "bar_chart_labels": ["Tumor (Reactome)", "Immune (Reactome)",
                             "Stromal (Reactome)"],
        "fig_dir_base": "hub_validation_reactome",
        "file_prefix": "reactome",
        "min_size": 5,
        "pathway_filter": None,
        "md_title": "Reactome",
        "md_gene_set_desc": "**Gene set library**: Reactome_Pathways_2024\n",
        "md_note_extra": "",
        "display_name": "Reactome",
    },
}

# Unpack active configuration
_cfg = GENE_SET_CONFIGS[GENE_SET_MODE]
CT_GENE_SETS = _cfg["ct_gene_sets"]
BAR_CHART_PANELS = list(zip(_cfg["bar_chart_labels"],
                            [TUMOR_CELL_TYPES, IMMUNE_CELL_TYPES, STROMAL_CELL_TYPES]))
MIN_SIZE = _cfg["min_size"]
FILE_PREFIX = _cfg["file_prefix"]

# Matched parameters: same k for hub and composition
k = 10
q = 0.9
seed = 18
MIN_CELLS_PER_CLUSTER = 30
FDR_THRESHOLD = 0.25
POSITIVE_ONLY = True  # True: butterfly charts show only positively enriched pathways

labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"

hub_path = f"data/xenium_sample1/grid_search_cache/hub_results/hub_q{q}_k{k}.csv"
comp_path = f"data/xenium_sample1/grid_search_cache/composition_results/composition_k{k}.csv"

fig_dir = f"figures/{_cfg['fig_dir_base']}_pos" if POSITIVE_ONLY else f"figures/{_cfg['fig_dir_base']}"
os.makedirs(fig_dir, exist_ok=True)

# Method display names and colors
METHOD_CONFIG = {
    "hub_cluster": {"label": "Hub clusters", "color": "#1f77b4"},
    "composition_cluster": {"label": "Composition clusters", "color": "#ff7f0e"},
}

print(f"Parameters: k={k}, q={q}, seed={seed}")
print(f"Hub path:  {hub_path}")
print(f"Comp path: {comp_path}")
print(f"Figures:   {fig_dir}/")
for ct in ALL_CELL_TYPES:
    print(f"  {ct}: {CT_GENE_SETS[ct]}")

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

# Print cell counts per cell type
for ct in ALL_CELL_TYPES:
    n = (adata.obs[labels_key] == ct).sum()
    print(f"  {ct}: {n} cells")

# %% Compute DEGs via scanpy rank_genes_groups for each cell type x method
deg_results = {}  # (cell_type, method, cluster_id) -> DataFrame

for ct in ALL_CELL_TYPES:
    ct_mask = adata.obs[labels_key] == ct
    ct_adata = adata[ct_mask].copy()
    print(f"\n{'='*60}")
    print(f"Cell type: {ct} ({ct_adata.shape[0]} cells) -> {CT_GENE_SETS[ct]}")
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


def parse_matched_size(tag_pct):
    try:
        return int(str(tag_pct).split("/")[1])
    except (IndexError, ValueError):
        return np.nan


for (ct, method, clust), deg_df in tqdm(deg_results.items(), desc="Running GSEA"):
    # Build ranked gene list: gene names as index, Wilcoxon scores as values
    ranked = deg_df.set_index("names")["scores"].dropna()
    ranked = ranked[~ranked.index.duplicated(keep="first")]
    ranked = ranked.sort_values(ascending=False)

    if len(ranked) < 5:
        print(f"  Skipping {ct}/{method}/cluster {clust}: only {len(ranked)} genes")
        continue

    # Use the gene set library assigned to this cell type
    gene_set_name = CT_GENE_SETS[ct]
    try:
        pre_res = gp.prerank(
            rnk=ranked,
            gene_sets=gene_set_name,
            min_size=MIN_SIZE,
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

# Apply optional pathway filter (e.g. KEGG signaling filter)
if len(gsea_df) > 0 and _cfg["pathway_filter"] is not None:
    gsea_df = _cfg["pathway_filter"](gsea_df)

if len(gsea_df) > 0:
    gsea_df["fdr"] = pd.to_numeric(gsea_df["fdr"], errors="coerce")
    gsea_df["nes"] = pd.to_numeric(gsea_df["nes"], errors="coerce")
    gsea_df["matched_size"] = pd.to_numeric(gsea_df["matched_size"], errors="coerce")
    if POSITIVE_ONLY:
        gsea_df["significant"] = (gsea_df["fdr"] < FDR_THRESHOLD) & (gsea_df["nes"] > 0)
    else:
        gsea_df["significant"] = gsea_df["fdr"] < FDR_THRESHOLD

    # Report overlap stats
    for lib in gsea_df["gene_set_library"].unique():
        lib_df = gsea_df[gsea_df["gene_set_library"] == lib]
        avg_matched = lib_df["matched_size"].mean()
        print(f"  {lib}: {len(lib_df)} pathway-cluster tests, "
              f"avg matched_size = {avg_matched:.1f}")

# %% Summary table: significant pathways per cell type x method
print("\n" + "=" * 70)
print(f"{_cfg['display_name'].upper()} SUMMARY: Significant pathways (FDR < 0.25) per cell type x method")
print("=" * 70)

summary_rows = []

for ct in ALL_CELL_TYPES:
    gene_set_lib = CT_GENE_SETS[ct]
    lib_df = gsea_df[gsea_df["gene_set_library"] == gene_set_lib]

    for method in ["hub_cluster", "composition_cluster"]:
        subset = lib_df[(lib_df["cell_type"] == ct) & (lib_df["method"] == method)]
        n_sig = subset["significant"].sum()
        n_total = len(subset)
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
              f"({n_sig}/{n_total} tests) [{gene_set_lib}]")

summary_df = pd.DataFrame(summary_rows)

# Print head-to-head comparison
print("\n--- Head-to-head comparison (unique significant pathways) ---")
for panel_title, ct_list in BAR_CHART_PANELS:
    print(f"\n  {panel_title}:")
    for ct in ct_list:
        hub_n = summary_df.loc[
            (summary_df["cell_type"] == ct) & (summary_df["method"] == "hub_cluster"),
            "n_unique_sig_pathways"
        ].values
        comp_n = summary_df.loc[
            (summary_df["cell_type"] == ct) & (summary_df["method"] == "composition_cluster"),
            "n_unique_sig_pathways"
        ].values
        hub_n = hub_n[0] if len(hub_n) > 0 else 0
        comp_n = comp_n[0] if len(comp_n) > 0 else 0
        winner = "HUB" if hub_n > comp_n else ("COMP" if comp_n > hub_n else "TIE")
        print(f"    {ct:20s}: Hub={hub_n:3d}  Comp={comp_n:3d}  -> {winner}")

# %% Stacked bar chart: two bars per cell type (hub, comp), each split into unique + shared
fig, axes = plt.subplots(1, len(BAR_CHART_PANELS),
                         figsize=(6 * len(BAR_CHART_PANELS), 6),
                         squeeze=False)

hub_color = METHOD_CONFIG["hub_cluster"]["color"]
comp_color = METHOD_CONFIG["composition_cluster"]["color"]
hub_shared_color = "#7fbce6"
comp_shared_color = "#ffbf7f"

for ax_idx, (panel_title, ct_list) in enumerate(BAR_CHART_PANELS):
    ax = axes[0, ax_idx]

    hub_unique_vals = []
    hub_shared_vals = []
    comp_unique_vals = []
    comp_shared_vals = []

    for ct in ct_list:
        ct_gsea_sub = gsea_df[gsea_df["cell_type"] == ct]
        hub_sig = set(ct_gsea_sub[
            (ct_gsea_sub["method"] == "hub_cluster") & (ct_gsea_sub["significant"])
        ]["pathway"].unique())
        comp_sig = set(ct_gsea_sub[
            (ct_gsea_sub["method"] == "composition_cluster") & (ct_gsea_sub["significant"])
        ]["pathway"].unique())

        shared = hub_sig & comp_sig
        hub_unique_vals.append(len(hub_sig - comp_sig))
        hub_shared_vals.append(len(shared))
        comp_unique_vals.append(len(comp_sig - hub_sig))
        comp_shared_vals.append(len(shared))

    x = np.arange(len(ct_list))
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
    for i in range(len(ct_list)):
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
    ax.set_title(panel_title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ct_labels = [ct.replace("_", " ") for ct in ct_list]
    ax.set_xticklabels(ct_labels, fontsize=9, rotation=45, ha="right")
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    clean_labels = {"Hub unique": "Hub unique", "Shared": "Shared",
                    "Comp unique": "Comp unique", "Shared ": "Shared"}
    deduped = {}
    for lbl, hdl in by_label.items():
        clean = clean_labels.get(lbl, lbl)
        if clean not in deduped:
            deduped[clean] = hdl
    ax.legend(deduped.values(), deduped.keys(), fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

fig.suptitle(f"{_cfg['display_name']}: Hub vs Composition Cluster DEGs", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

fig.savefig(os.path.join(fig_dir, f"{FILE_PREFIX}_barplot.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, f"{FILE_PREFIX}_barplot.svg"),
            bbox_inches="tight")
print(f"Saved stacked bar chart to {fig_dir}/{FILE_PREFIX}_barplot.png/.svg")
plt.show()

# %% Butterfly/mirror bar charts for all cell types
from matplotlib.patches import Patch


def clean_pathway_name(name):
    """Clean pathway names for display (handles Hallmark, KEGG, and Reactome formats)."""
    name = re.sub(r'\s*R-HSA-\d+$', '', name)  # Reactome IDs
    name = name.split(" Homo sapiens")[0]  # KEGG suffixes
    name = name.replace("Hallmark ", "").replace("HALLMARK_", "").replace("HALLMARK ", "")
    name = name.replace("_", " ").title()
    if len(name) > 50:
        name = name[:47] + "..."
    return name


saved_butterfly_cts = []

for focus_ct in ALL_CELL_TYPES:
    print(f"\n{'='*60}")
    print(f"Butterfly chart for: {focus_ct}")
    print(f"{'='*60}")

    # Use the gene set library assigned to this cell type
    butterfly_lib = CT_GENE_SETS[focus_ct]

    ct_gsea = gsea_df[
        (gsea_df["cell_type"] == focus_ct) & (gsea_df["gene_set_library"] == butterfly_lib)
    ]

    if len(ct_gsea) == 0:
        print(f"  No GSEA results for {focus_ct}, skipping.")
        continue

    # Aggregate: for each pathway x method, pick representative cluster result.
    # POSITIVE_ONLY mode: prefer positively significant (NES > 0 + FDR < threshold);
    #   if not significant, take cluster with highest NES (most positive direction).
    # Standard mode: take the cluster with the lowest FDR.
    butterfly_data = []
    for method in ["hub_cluster", "composition_cluster"]:
        method_df = ct_gsea[ct_gsea["method"] == method]
        if len(method_df) == 0:
            continue
        for pathway in method_df["pathway"].unique():
            pw_df = method_df[method_df["pathway"] == pathway]
            if POSITIVE_ONLY:
                pos_sig = pw_df[(pw_df["nes"] > 0) & (pw_df["fdr"] < FDR_THRESHOLD)]
                if len(pos_sig) > 0:
                    best_idx = pos_sig["fdr"].idxmin()
                else:
                    best_idx = pw_df["nes"].idxmax()
            else:
                best_idx = pw_df["fdr"].idxmin()
            best_row = pw_df.loc[best_idx]
            is_sig = (best_row["nes"] > 0 and best_row["fdr"] < FDR_THRESHOLD) \
                if POSITIVE_ONLY else (best_row["fdr"] < FDR_THRESHOLD)
            butterfly_data.append({
                "pathway": pathway,
                "method": method,
                "nes": best_row["nes"],
                "fdr": best_row["fdr"],
                "matched_size": best_row["matched_size"],
                "significant": is_sig,
            })

    butterfly_df = pd.DataFrame(butterfly_data)

    # Filter to pathways significant in at least one method
    sig_pathways = butterfly_df.loc[butterfly_df["significant"], "pathway"].unique()
    butterfly_df = butterfly_df[butterfly_df["pathway"].isin(sig_pathways)]
    mode_label = "positively significant" if POSITIVE_ONLY else "significant"
    print(f"  Pathways {mode_label} in at least one method: {len(sig_pathways)}")

    if len(butterfly_df) == 0:
        print(f"  No {mode_label} pathways for {focus_ct}, skipping butterfly chart.")
        continue

    # Get union of all pathways
    all_pathways = butterfly_df["pathway"].unique()

    # Create a pivot for plotting: pathway x method
    hub_data = butterfly_df[butterfly_df["method"] == "hub_cluster"].set_index("pathway")
    comp_data = butterfly_df[butterfly_df["method"] == "composition_cluster"].set_index("pathway")

    # Sort pathways by hub NES (descending); POSITIVE_ONLY uses raw NES, else |NES|
    pathway_order = []
    for pw in all_pathways:
        hub_nes = hub_data.loc[pw, "nes"] if pw in hub_data.index else 0
        sort_val = hub_nes if POSITIVE_ONLY else abs(hub_nes)
        pathway_order.append((pw, sort_val))
    pathway_order.sort(key=lambda x: x[1], reverse=True)
    pathways_sorted = [p[0] for p in pathway_order]

    fig, ax = plt.subplots(figsize=(16, max(6, len(pathways_sorted) * 0.65)))

    y_positions = np.arange(len(pathways_sorted))

    hub_color_bf = METHOD_CONFIG["hub_cluster"]["color"]
    comp_color_bf = METHOD_CONFIG["composition_cluster"]["color"]
    hub_faded = mcolors.to_rgba(hub_color_bf, alpha=0.3)
    comp_faded = mcolors.to_rgba(comp_color_bf, alpha=0.3)

    for i, pw in enumerate(pathways_sorted):
        # Hub bars extend to the LEFT (negative x)
        if pw in hub_data.index:
            h_nes = hub_data.loc[pw, "nes"]
            h_fdr = hub_data.loc[pw, "fdr"]
            h_sig = hub_data.loc[pw, "significant"]
            color = hub_color_bf if h_sig else hub_faded
            ax.barh(i, -abs(h_nes), height=0.7, color=color,
                    edgecolor="black" if h_sig else "gray",
                    linewidth=0.8 if h_sig else 0.4)
            sign_str = "+" if h_nes > 0 else "-" if h_nes < 0 else ""
            label = f"{sign_str}{abs(h_nes):.2f} (q={h_fdr:.2f})"
            ax.text(-abs(h_nes) / 2, i, label,
                    ha="center", va="center", fontsize=11,
                    color="white" if h_sig else "#444444", fontweight="bold")

        # Composition bars extend to the RIGHT (positive x)
        if pw in comp_data.index:
            c_nes = comp_data.loc[pw, "nes"]
            c_fdr = comp_data.loc[pw, "fdr"]
            c_sig = comp_data.loc[pw, "significant"]
            color = comp_color_bf if c_sig else comp_faded
            ax.barh(i, abs(c_nes), height=0.7, color=color,
                    edgecolor="black" if c_sig else "gray",
                    linewidth=0.8 if c_sig else 0.4)
            sign_str = "+" if c_nes > 0 else "-" if c_nes < 0 else ""
            label = f"{sign_str}{abs(c_nes):.2f} (q={c_fdr:.2f})"
            ax.text(abs(c_nes) / 2, i, label,
                    ha="center", va="center", fontsize=11,
                    color="white" if c_sig else "#444444", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([clean_pathway_name(pw) for pw in pathways_sorted], fontsize=14)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|NES|", fontsize=16)
    ax.set_title(f"{_cfg['display_name']} Butterfly Chart: {focus_ct.replace('_', ' ')}\n"
                 f"({butterfly_lib.replace('_', ' ')})",
                 fontsize=17, fontweight="bold")
    ax.tick_params(axis='x', labelsize=13)

    legend_elements = [
        Patch(facecolor=hub_color_bf, edgecolor="black", label="Hub (FDR < 0.25)"),
        Patch(facecolor=hub_faded, edgecolor="gray", label="Hub (n.s.)"),
        Patch(facecolor=comp_color_bf, edgecolor="black", label="Composition (FDR < 0.25)"),
        Patch(facecolor=comp_faded, edgecolor="gray", label="Composition (n.s.)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=12, frameon=True, framealpha=0.9)

    ax.text(-0.02, 1.02, "Hub clusters", transform=ax.transAxes,
            ha="right", fontsize=15, fontweight="bold", color=hub_color_bf)
    ax.text(1.02, 1.02, "Composition clusters", transform=ax.transAxes,
            ha="left", fontsize=15, fontweight="bold", color=comp_color_bf)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    ct_slug = focus_ct.lower().replace("+", "").replace("-", "_")
    fig.savefig(os.path.join(fig_dir, f"{FILE_PREFIX}_butterfly_{ct_slug}.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{FILE_PREFIX}_butterfly_{ct_slug}.svg"),
                bbox_inches="tight")
    saved_butterfly_cts.append(focus_ct)
    print(f"  Saved to {fig_dir}/{FILE_PREFIX}_butterfly_{ct_slug}.png/.svg")
    plt.show()

# %% Detailed pathway table: pathways significant in hub but NOT composition
print("\n" + "=" * 70)
print(f"{_cfg['display_name'].upper()} PATHWAYS SIGNIFICANT IN HUB CLUSTERS BUT NOT COMPOSITION CLUSTERS")
print("(Unique biological signals captured by attention-based interaction)")
print("=" * 70)

for ct in ALL_CELL_TYPES:
    gene_set_lib = CT_GENE_SETS[ct]
    ct_lib = gsea_df[(gsea_df["cell_type"] == ct) & (gsea_df["gene_set_library"] == gene_set_lib)]

    if len(ct_lib) == 0:
        continue

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

    print(f"\n  {ct} [{gene_set_lib}]:")
    print(f"    Hub-significant:  {len(hub_sig)} pathways")
    print(f"    Comp-significant: {len(comp_sig)} pathways")
    print(f"    Shared:           {len(shared)} pathways")
    print(f"    Hub-only:         {len(hub_only)} pathways")
    print(f"    Comp-only:        {len(comp_only)} pathways")

    if len(hub_only) > 0:
        print(f"\n    Hub-only pathways (unique to attention-based clustering):")
        hub_details = ct_lib[
            (ct_lib["method"] == "hub_cluster") &
            (ct_lib["pathway"].isin(hub_only))
        ].sort_values("fdr")

        for pw in sorted(hub_only):
            pw_rows = hub_details[hub_details["pathway"] == pw]
            best_idx = pw_rows["fdr"].idxmin()
            row = pw_rows.loc[best_idx]
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
print("NOTE: 250-gene Xenium panel limits overlap with gene sets.")
print("Small matched_size values indicate sparse overlap; interpret with caution.")
print("=" * 70)

# %% Save summary to markdown
md_path = os.path.join(fig_dir, f"{FILE_PREFIX}_summary.md")
with open(md_path, "w") as f:
    f.write(f"# {_cfg['md_title']}: Hub vs Composition Cluster DEGs\n\n")
    f.write(f"**Parameters**: q={q}, k={k}, seed={seed}\n\n")
    f.write(f"**FDR threshold**: {FDR_THRESHOLD}\n\n")
    f.write(f"{_cfg['md_gene_set_desc']}\n")

    # Summary table per panel group
    f.write("## Significant Pathways per Cell Type (FDR < 0.25)\n\n")
    for panel_title, ct_list in BAR_CHART_PANELS:
        f.write(f"### {panel_title}\n\n")
        f.write("| Cell Type | Method | Unique Sig Pathways | Sig Tests / Total |\n")
        f.write("|-----------|--------|--------------------:|------------------:|\n")
        for ct in ct_list:
            ct_summary = summary_df[summary_df["cell_type"] == ct]
            for _, row in ct_summary.iterrows():
                f.write(f"| {row['cell_type']} | {row['method_label']} "
                        f"| {row['n_unique_sig_pathways']} "
                        f"| {row['n_significant_tests']}/{row['n_total_tests']} |\n")
        f.write("\n")

    # Head-to-head
    f.write("## Head-to-Head Comparison\n\n")
    for panel_title, ct_list in BAR_CHART_PANELS:
        f.write(f"### {panel_title}\n\n")
        f.write("| Cell Type | Hub | Comp | Winner |\n")
        f.write("|-----------|----:|-----:|--------|\n")
        for ct in ct_list:
            hub_n = summary_df.loc[
                (summary_df["cell_type"] == ct) & (summary_df["method"] == "hub_cluster"),
                "n_unique_sig_pathways"
            ].values
            comp_n = summary_df.loc[
                (summary_df["cell_type"] == ct) & (summary_df["method"] == "composition_cluster"),
                "n_unique_sig_pathways"
            ].values
            hub_n = hub_n[0] if len(hub_n) > 0 else 0
            comp_n = comp_n[0] if len(comp_n) > 0 else 0
            winner = "Hub" if hub_n > comp_n else ("Comp" if comp_n > hub_n else "Tie")
            f.write(f"| {ct} | {hub_n} | {comp_n} | **{winner}** |\n")
        f.write("\n")

    # Detailed pathway comparison
    f.write("## Pathway Details per Cell Type\n\n")
    for ct in ALL_CELL_TYPES:
        gene_set_lib = CT_GENE_SETS[ct]
        ct_lib = gsea_df[(gsea_df["cell_type"] == ct) & (gsea_df["gene_set_library"] == gene_set_lib)]
        if len(ct_lib) == 0:
            continue

        hub_sig = set(ct_lib[
            (ct_lib["method"] == "hub_cluster") & (ct_lib["significant"])
        ]["pathway"].unique())
        comp_sig = set(ct_lib[
            (ct_lib["method"] == "composition_cluster") & (ct_lib["significant"])
        ]["pathway"].unique())

        hub_only = sorted(hub_sig - comp_sig)
        comp_only = sorted(comp_sig - hub_sig)
        shared = sorted(hub_sig & comp_sig)

        f.write(f"### {ct} ({gene_set_lib})\n\n")
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
    f.write("**Note**: 250-gene Xenium panel limits overlap with gene sets. "
            f"Small matched_size values indicate sparse overlap; interpret with caution.{_cfg['md_note_extra']}\n")

print(f"Summary saved to {md_path}")

# %% Combined butterfly figures — split into tumor vs immune/stromal
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec


def _make_combined_butterfly(img_list, save_name, fig_dir):
    """Create a 2-column combined figure from a list of (ct, img_array) tuples."""
    ncols = 2
    nrows = (len(img_list) + ncols - 1) // ncols

    col_w = 8
    row_heights = []
    for row_idx in range(nrows):
        max_h_inches = 0
        for col_idx in range(ncols):
            flat_idx = row_idx * ncols + col_idx
            if flat_idx < len(img_list):
                h, w = img_list[flat_idx][1].shape[:2]
                max_h_inches = max(max_h_inches, col_w * h / w)
        row_heights.append(max_h_inches)

    fig = plt.figure(figsize=(col_w * ncols, sum(row_heights)))
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=row_heights,
                  wspace=0.03, hspace=0.05)

    for idx, (ct, img) in enumerate(img_list):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.set_axis_off()

    for idx in range(len(img_list), nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_axis_off()

    combined_path = os.path.join(fig_dir, save_name)
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.savefig(combined_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.show()
    print(f"Saved combined butterfly chart ({len(img_list)} panels) to {combined_path}")


# Collect images per group
group_imgs = {"tumor": [], "immune": [], "stromal": []}
for ct in saved_butterfly_cts:
    ct_slug = ct.lower().replace("+", "").replace("-", "_").replace(" ", "_")
    path = os.path.join(fig_dir, f"{FILE_PREFIX}_butterfly_{ct_slug}.png")
    img = (ct, imread(path))
    if ct in TUMOR_CELL_TYPES:
        group_imgs["tumor"].append(img)
    elif ct in IMMUNE_CELL_TYPES:
        group_imgs["immune"].append(img)
    elif ct in STROMAL_CELL_TYPES:
        group_imgs["stromal"].append(img)

for group_name, imgs in group_imgs.items():
    if imgs:
        _make_combined_butterfly(imgs, f"{FILE_PREFIX}_butterfly_combined_{group_name}.png", fig_dir)

if not any(group_imgs.values()):
    print("No butterfly charts to combine.")

# %%
