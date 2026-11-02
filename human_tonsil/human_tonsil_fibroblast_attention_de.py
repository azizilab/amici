# %% High- vs low-attention fibroblast senders for CD4 T cell receivers in the human tonsil.
"""Split fibroblasts by the attention they receive from CD4 T cells and test what separates them.

Fig. 6d highlights CCR7 in CD4 T cell receivers, which AMICI places on head 4, the longer-range
head (~27 um median effective pairwise radius) rather than the ~6 um contact heads. CCR7 responds
to CCL19 and CCL21, which reticular fibroblasts secrete, so the fibroblasts AMICI attends to most
should be the ones producing those chemokines.

Following atera_breast_analysis.py (T cell senders to CAF receivers), each sender cell's score is
its attention summed over neighbor slots and over heads, thresholded at a quantile of the positive
scores, and the groups are compared with a Wilcoxon rank-sum test against an explicit reference
group. Two further scores are recorded: head 4 alone, and the total divided by the number of CD4 T
cells attending the fibroblast, which separates how strongly each receiver attends from how many
CD4 T cells happen to be nearby.

Because high-attention fibroblasts sit among more T cells, they also pick up more T cell transcripts
through segmentation bleed. Differentially expressed genes are therefore filtered with the
segmentation artifact test, with the roles of the two cell types exchanged: the bleed guarded
against here runs from the CD4 T cells into the fibroblasts, the opposite of the direction tested
when reporting downstream genes of the interaction, so the fibroblasts take the receiver role and
the CD4 T cells the sender role. A gene passes when expression in the fibroblasts bordering CD4 T
cells is significantly higher than in CD4 T cells away from fibroblasts, in fibroblasts away from
CD4 T cells, and in the CD4 T cells those fibroblasts border. Each gene is also annotated with the
log-ratio of its mean expression in CD4 T cells over fibroblasts, as a descriptive companion to the
test.

Spatial panels use plot_spatial_feature from atera_breast_analysis.py and the high/low palette from
the Xenium subtyping analysis.
"""

# %% Import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import ttest_ind

from amici import AMICI
from amici.interpretation._ablation_module import AMICIAblationModule

# %% Seed everything
seed = 40
pl.seed_everything(seed)

# %% Load data
labels_key = "celltype_manual_fine"
data_date = "2026-07-25"
cosmx_pixel_size_um = 0.12028
adata = sc.read_h5ad(f"./data/human_tonsil_filtered_no_ribo_{data_date}.h5ad")

figures_dir = "./figures/human_tonsil_fibroblast_attention"
os.makedirs(figures_dir, exist_ok=True)
results_dir = "./results/human_tonsil_fibroblast_attention"
os.makedirs(results_dir, exist_ok=True)

model_path = "saved_models/human_tonsil_no_ribo_retrain/seed_40"
model_path = os.environ.get("HUMAN_TONSIL_AMICI_MODEL_DIR", model_path)

receiver_ct = "CD4 T cell"
sender_ct = "Fibroblast"
# Head 4 carries CCR7, SELL, PLAC8 and NELL2 in CD4 T cell receivers; it is the longer-range of the
# heads retained for fibroblast senders (median effective pairwise radius ~27 um, against ~6 um for
# heads 6 and 7), so it is reported alongside the total.
ccr7_head_idx = 4
quantile = 0.9

# The high/low sender palette of xenium_subtyping_analysis.py, so the groups read the same way.
SUBTYPING_PALETTE = {
    "Low-interacting": "#73fb88",
    "High-interacting": "#009e1d",
}
CELL_TYPE_PALETTE = dict(
    zip(
        sorted(adata.obs[labels_key].astype(str).unique()),
        sns.color_palette("tab20", n_colors=adata.obs[labels_key].nunique()).as_hex(),
        strict=True,
    )
)
# Chemokines the CCR7 result predicts, plus the matrix genes of non-reticular fibroblasts.
MARKERS_OF_INTEREST = [
    "CCL19",
    "CCL21",
    "CXCL12",
    "CXCL14",
    "CCL2",
    "IL7",
    "PTGDS",
    "DCN",
    "LUM",
    "COL1A1",
    "COL1A2",
    "COL3A1",
]

print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
print(f"Loading model from {model_path}")


# %% Plot helpers
def _get_gene_expression(adata, gene):
    """Return a dense expression vector for a gene."""
    if gene not in adata.var_names:
        raise KeyError(f"{gene} not found in adata.var_names.")
    gene_expr = adata[:, gene].X
    if hasattr(gene_expr, "toarray"):
        gene_expr = gene_expr.toarray()
    return np.asarray(gene_expr).reshape(-1)


def spatial_coords(adata):
    """Spatial coordinates in micrometers, as a frame of X and Y like the Xenium analyses expect."""
    return pd.DataFrame(
        np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * cosmx_pixel_size_um,
        index=adata.obs_names,
        columns=["X", "Y"],
    )


def plot_spatial_feature(
    adata,
    values,
    title,
    colorbar_label,
    save_prefix,
    point_size=8,
    vmin=None,
    vmax=None,
    save_dir=None,
):
    """Plot a continuous feature on spatial coordinates, as in xenium_spatial_analysis.py."""
    save_dir = figures_dir if save_dir is None else save_dir
    plot_df = spatial_coords(adata)
    plot_df["value"] = np.asarray(values)
    vmin = float(plot_df["value"].min()) if vmin is None else vmin
    vmax = float(plot_df["value"].max()) if vmax is None else vmax
    palette = LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for low attention
            "#FF4444",  # Red for high attention
        ],
        N=100,
    )

    plt.figure(figsize=(15, 8))
    scatter = sns.scatterplot(
        plot_df,
        x="X",
        y="Y",
        hue="value",
        palette=palette,
        alpha=1.0,
        s=point_size,
        marker="o",
    )
    plt.title(title)
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")

    # The colorbar carries the real data range, so the scale of the score is readable.
    sm = plt.cm.ScalarMappable(cmap=palette, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", pad=0.01)
    cbar.set_label(colorbar_label)
    if scatter.get_legend() is not None:
        scatter.get_legend().remove()
    plt.gca().invert_yaxis()

    plt.savefig(f"{save_dir}/{save_prefix}.png", dpi=300, bbox_inches="tight")
    # Hiding the points keeps the svg small enough to edit, as in xenium_spatial_analysis.py.
    scatter.collections[0].set_visible(False)
    plt.savefig(f"{save_dir}/{save_prefix}.svg", dpi=300, bbox_inches="tight")
    scatter.collections[0].set_visible(True)
    plt.show()


def plot_spatial_groups(
    adata,
    groups,
    palette,
    title,
    save_prefix,
    point_size=6,
    save_dir=None,
):
    """Plot the high- and low-attention groups, as xenium_subtyping_analysis.py draws them."""
    save_dir = figures_dir if save_dir is None else save_dir
    plot_df = spatial_coords(adata)
    plot_df["group"] = pd.Series(groups).reindex(plot_df.index).astype(str)

    plt.figure(figsize=(15, 8))
    scatters, labels = [], []
    # Low-interacting cells are drawn first so the high-interacting ones stay visible on top.
    for group in ["Low-interacting", "High-interacting"]:
        group_df = plot_df[plot_df["group"] == group]
        if group_df.empty:
            continue
        scatters.append(
            sns.scatterplot(
                group_df,
                x="X",
                y="Y",
                facecolor=palette[group],
                alpha=1.0,
                s=point_size,
                marker="o",
            )
        )
        labels.append(f"{group} {sender_ct} (n = {len(group_df):,})")
    plt.title(title)
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.legend(
        labels,
        title="Neighbor Category",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.5,
        markerscale=3.0,
    )
    plt.gca().invert_yaxis()

    plt.savefig(f"{save_dir}/{save_prefix}.png", dpi=300, bbox_inches="tight")
    for scatter in scatters:
        for collection in scatter.collections:
            collection.set_visible(False)
    plt.savefig(f"{save_dir}/{save_prefix}.svg", dpi=300, bbox_inches="tight")
    for scatter in scatters:
        for collection in scatter.collections:
            collection.set_visible(True)
    plt.show()


# %% Load model
model = AMICI.load(
    model_path,
    adata=adata,
)
AMICI.setup_anndata(
    adata,
    labels_key=labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)
labels = adata.obs[labels_key].astype(str)

# %% Attention each fibroblast receives from CD4 T cell receivers
# get_total_sender_attention_to_receiver in atera_breast_analysis.py accumulates attention per
# receiver cell. Here the groups are the senders, so attention is accumulated onto the sender
# instead, over every CD4 T cell that holds the fibroblast among its 50 nearest neighbors.
receiver_indices = list(np.flatnonzero(labels.to_numpy() == receiver_ct))
attention_patterns = model.get_attention_patterns(
    adata,
    indices=receiver_indices,
    batch_size=512,
)
attention_df = attention_patterns._attention_patterns_df
nn_idxs_df = attention_patterns._nn_idxs_df
neighbor_cols = [col for col in attention_df.columns if col.startswith("neighbor_")]
heads = attention_df["head"].astype(int).to_numpy()

neighbor_positions = adata.obs_names.get_indexer(nn_idxs_df.to_numpy().ravel())
is_sender = (neighbor_positions >= 0) & (labels.to_numpy()[neighbor_positions] == sender_ct)

total_attention = np.zeros(adata.n_obs)
head_attention = np.zeros(adata.n_obs)
# Attention is a softmax over one receiver's 50 neighbors and the dummy slot, so any single value,
# and any sum within one receiver and head, stays in [0, 1]. Summing across receivers and heads, as
# atera_breast_analysis.py does for its CAF receivers, leaves that scale: the total then grows with
# how many CD4 T cells happen to lie nearby. xenium_spatial_analysis.py instead aggregates within a
# (receiver, head) and takes the maximum over them, which for a sender cell is the largest single
# attention any receiver pays it, since a sender appears once in a given receiver's neighbor list.
# Both are kept: the maximum as the bounded score and the total as the Atera-style one.
max_attention = np.zeros(adata.n_obs)
for head_idx in sorted(set(heads)):
    head_values = attention_df.loc[heads == head_idx, neighbor_cols].to_numpy(dtype=float).ravel()
    head_contribution = np.bincount(
        neighbor_positions[is_sender],
        weights=head_values[is_sender],
        minlength=adata.n_obs,
    )
    total_attention += head_contribution
    np.maximum.at(max_attention, neighbor_positions[is_sender], head_values[is_sender])
    if head_idx == ccr7_head_idx:
        head_attention = head_contribution
n_attending_receivers = np.bincount(neighbor_positions[is_sender], minlength=adata.n_obs)

sender_cells = adata.obs_names[labels.to_numpy() == sender_ct]
sender_attention = pd.DataFrame(
    {
        "total_attention": total_attention[labels.to_numpy() == sender_ct],
        "max_attention": max_attention[labels.to_numpy() == sender_ct],
        f"head_{ccr7_head_idx}_attention": head_attention[labels.to_numpy() == sender_ct],
        "n_attending_receivers": n_attending_receivers[labels.to_numpy() == sender_ct],
    },
    index=sender_cells,
)
sender_attention["attention_per_receiver"] = sender_attention["total_attention"] / sender_attention[
    "n_attending_receivers"
].where(sender_attention["n_attending_receivers"] > 0)
sender_attention.to_csv(os.path.join(results_dir, "human_tonsil_fibroblast_attention_scores.csv"))

attending = sender_attention[sender_attention["n_attending_receivers"] > 0]
print(
    f"{len(sender_attention)} {sender_ct} cells; {len(attending)} are a neighbor of at least one of "
    f"the {len(receiver_indices)} {receiver_ct} cells"
)
print(attending.describe().round(4).to_string())

# %% Segmentation artifact test for transcripts bleeding from CD4 T cells into fibroblasts
# The thresholds of the Methods are in micrometers, while adata.obsm["spatial"] holds CosMx global
# pixels, so they are converted as in human_tonsil_analysis.py.
segmentation_kwargs = {
    "receiver_type": sender_ct,
    "adata": adata,
    "near_threshold": 20.0 / cosmx_pixel_size_um,
    "far_threshold": 100.0 / cosmx_pixel_size_um,
    "pval_threshold": 0.05,
    "coord_obsm_key": "spatial",
    "compare_near_senders": True,
}
segmentation_probe = AMICIAblationModule(_labels_key=labels_key)


def segmentation_passing_genes(genes):
    """Genes whose expression in interface fibroblasts is not explained by bleed from CD4 T cells."""
    passing_pairs = segmentation_probe._get_segmentation_passing_gene_pairs(
        genes_by_sender={receiver_ct: list(genes)},
        **segmentation_kwargs,
    )
    return {gene for _, gene in passing_pairs}


# %% How much of each gene is carried by CD4 T cells rather than fibroblasts
# A descriptive companion to the test: a gene whose mean expression is far higher in CD4 T cells
# than in fibroblasts is a spillover candidate. It cannot resolve a gene that bleeds both ways,
# such as fibroblast CCL21 carried into the surrounding T cells, which the test handles.
receiver_mean = np.asarray(adata[labels.to_numpy() == receiver_ct].X.mean(axis=0)).reshape(-1)
sender_mean = np.asarray(adata[labels.to_numpy() == sender_ct].X.mean(axis=0)).reshape(-1)
spillover_df = pd.DataFrame(
    {
        "gene": adata.var_names.astype(str),
        f"mean_in_{receiver_ct}": receiver_mean,
        f"mean_in_{sender_ct}": sender_mean,
        "log2_receiver_over_sender": np.log2((receiver_mean + 1e-9) / (sender_mean + 1e-9)),
    }
).set_index("gene")

# %% High- vs low-attention groups and differential expression
# The maximum is the score the groups and the figures are built from, matching the sender side of
# xenium_spatial_analysis.py. The others are reported alongside it as checks.
primary_score_col = "max_attention"
score_cols = [primary_score_col, "total_attention", f"head_{ccr7_head_idx}_attention", "attention_per_receiver"]
de_results = {}
for score_col in score_cols:
    # The quantile is taken over the positive scores, as in atera_breast_analysis.py and
    # xenium_spatial_analysis.py, so fibroblasts no CD4 T cell attends do not set the threshold.
    positive_scores = attending.loc[attending[score_col] > 0, score_col]
    attention_threshold = positive_scores.quantile(quantile)
    group = pd.Series("Low-interacting", index=positive_scores.index)
    group[positive_scores > attention_threshold] = "High-interacting"

    de_adata = adata[group.index].copy()
    de_adata.obs["attention_group"] = pd.Categorical(group.reindex(de_adata.obs_names))
    # Both groups are named explicitly, so the test is high against low and nothing else.
    sc.tl.rank_genes_groups(
        de_adata,
        groupby="attention_group",
        groups=["High-interacting"],
        reference="Low-interacting",
        method="wilcoxon",
    )
    de_df = sc.get.rank_genes_groups_df(de_adata, group="High-interacting").rename(columns={"names": "gene"})
    de_df["rank"] = np.arange(1, len(de_df) + 1)
    de_df = de_df.join(spillover_df["log2_receiver_over_sender"], on="gene")
    # Only genes enriched in the high-attention group can be carried in by bleed from the T cells,
    # so the test is run on those; genes depleted there are marked as not applicable.
    enriched = (de_df["logfoldchanges"] > 0) & (de_df["pvals_adj"] < 0.05)
    passing = segmentation_passing_genes(de_df.loc[enriched, "gene"])
    de_df["passes_segmentation"] = np.where(enriched, de_df["gene"].isin(passing), pd.NA)
    de_df.to_csv(
        os.path.join(results_dir, f"human_tonsil_fibroblast_de_high_vs_low_{score_col}.csv"),
        index=False,
    )
    de_results[score_col] = (group, de_df, de_adata)

    print(
        f"\n=== {score_col}: {quantile:.0%} threshold {attention_threshold:.4f}; "
        f"{int((group == 'High-interacting').sum())} high vs {int((group == 'Low-interacting').sum())} low"
    )
    print(
        f"{int(enriched.sum())} genes enriched in high-attention fibroblasts, "
        f"{len(passing)} of them pass the segmentation artifact test"
    )
    print("top 15 enriched genes that pass:")
    print(de_df[de_df["passes_segmentation"] == True].head(15).round(4).to_string(index=False))  # noqa: E712
    print("top 15 enriched genes flagged as potential bleed from the CD4 T cells:")
    print(de_df[de_df["passes_segmentation"] == False].head(15).round(4).to_string(index=False))  # noqa: E712
    print("markers of interest:")
    print(de_df[de_df["gene"].isin(MARKERS_OF_INTEREST)].round(4).to_string(index=False))

# %% Spatial panels for the total-attention split
sender_adata = adata[sender_cells].copy()
# The maximum is a single softmax weight, so the colorbar spans the attention scale itself. The
# total is kept as a second panel, where the scale is a sum over attending receivers and heads.
plot_spatial_feature(
    sender_adata,
    sender_attention.loc[sender_adata.obs_names, "max_attention"],
    title=f"{sender_ct} senders by maximum attention from a {receiver_ct} receiver",
    colorbar_label=f"Maximum attention from a {receiver_ct}",
    save_prefix="human_tonsil_fibroblast_max_attention_spatial",
    vmin=0.0,
    vmax=1.0,
)
plot_spatial_feature(
    sender_adata,
    sender_attention.loc[sender_adata.obs_names, "total_attention"],
    title=f"{sender_ct} senders by total attention from {receiver_ct} receivers",
    colorbar_label=f"Total attention from {receiver_ct}\n(summed over receivers and heads)",
    save_prefix="human_tonsil_fibroblast_total_attention_spatial",
)

primary_group, primary_de, primary_de_adata = de_results[primary_score_col]
plot_spatial_groups(
    sender_adata[primary_group.index].copy(),
    primary_group,
    palette=SUBTYPING_PALETTE,
    title=(
        f"{sender_ct} senders split at the {quantile:.0%} quantile of the maximum attention "
        f"any {receiver_ct} pays them"
    ),
    save_prefix="human_tonsil_fibroblast_attention_groups_spatial",
)

for gene in ["CCL21", "CCL19"]:
    if gene in adata.var_names:
        plot_spatial_feature(
            sender_adata,
            _get_gene_expression(sender_adata, gene),
            title=f"{gene} in {sender_ct} cells",
            colorbar_label=f"{gene} (log-normalized)",
            save_prefix=f"human_tonsil_fibroblast_{gene}_spatial",
        )

# %% Expression of the top differentially expressed genes that pass the segmentation test
# Both groups are shown, so the size of the difference is visible rather than only its direction.
# Mean log-normalized expression is plotted as it is: scaling per gene across the groups would set
# the high group to 1 and the low group to 0 for every gene, since there are only two of them.
passing_de = primary_de[primary_de["passes_segmentation"] == True]  # noqa: E712
top_genes = passing_de.sort_values("pvals_adj").head(20).sort_values("logfoldchanges", ascending=False)["gene"].tolist()
primary_de_adata.obs["attention_group"] = primary_de_adata.obs["attention_group"].cat.reorder_categories(
    ["High-interacting", "Low-interacting"]
)
matrixplot = sc.pl.matrixplot(
    primary_de_adata,
    var_names=top_genes,
    groupby="attention_group",
    cmap="Reds",
    colorbar_title="Mean expression\n(log-normalized)",
    title=(
        f"Top 20 genes by adjusted p-value in high- vs low-attention {sender_ct} cells,\n"
        "excluding genes flagged as potential segmentation artifacts"
    ),
    figsize=(10, 2.4),
    show=False,
    return_fig=True,
)
matrixplot.savefig(
    f"{figures_dir}/human_tonsil_fibroblast_de_matrixplot_mean_expression.png", dpi=300, bbox_inches="tight"
)
matrixplot.savefig(
    f"{figures_dir}/human_tonsil_fibroblast_de_matrixplot_mean_expression.svg", dpi=300, bbox_inches="tight"
)
plt.close("all")

# The same panel on expression z-scored per gene over the fibroblasts of both groups. On the raw
# scale the highly expressed genes, such as VIM and PTGDS, take up the colour range and leave the
# smaller contrasts unreadable; z-scoring puts every gene on a comparable scale around its own mean.
scaled_adata = primary_de_adata.copy()
sc.pp.scale(scaled_adata, max_value=10)
scaled_matrixplot = sc.pl.matrixplot(
    scaled_adata,
    var_names=top_genes,
    groupby="attention_group",
    cmap="bwr",
    vmin=-0.6,
    vmax=0.6,
    colorbar_title="Mean z-scored\nexpression",
    title=(
        f"Top 20 genes by adjusted p-value in high- vs low-attention {sender_ct} cells,\n"
        "excluding genes flagged as potential segmentation artifacts"
    ),
    figsize=(10, 2.4),
    show=False,
    return_fig=True,
)
scaled_matrixplot.savefig(f"{figures_dir}/human_tonsil_fibroblast_de_matrixplot.png", dpi=300, bbox_inches="tight")
scaled_matrixplot.savefig(f"{figures_dir}/human_tonsil_fibroblast_de_matrixplot.svg", dpi=300, bbox_inches="tight")
plt.close("all")

# %% Attention score distributions of the two groups
# The groups are cut on the maximum, so that panel only shows where the threshold fell. The other
# three scores are not what the split was made on, so they show whether it carries over to the
# total, to head 4 alone and to the attention per attending receiver. Welch's t-test accompanies
# each panel, as in xenium_spatial_analysis.py; the leftmost one is descriptive by construction.
score_labels = {
    "max_attention": f"Maximum attention from a {receiver_ct}",
    "total_attention": "Total attention (summed over receivers and heads)",
    "attention_per_receiver": "Total attention per attending receiver",
}
# Every score here aggregates over all heads; head 4 on its own is left to the differential
# expression tables, where it serves as the check that the CCR7 head agrees with the rest.
boxplot_score_cols = ["max_attention", "total_attention", "attention_per_receiver"]
scores_by_group = attending.loc[primary_group.index].copy()
scores_by_group["attention_group"] = primary_group

fig, axes = plt.subplots(1, len(boxplot_score_cols), figsize=(4.2 * len(boxplot_score_cols), 4.6))
for ax, score_col in zip(axes, boxplot_score_cols, strict=True):
    sns.boxplot(
        data=scores_by_group,
        x="attention_group",
        y=score_col,
        order=["High-interacting", "Low-interacting"],
        hue="attention_group",
        hue_order=["High-interacting", "Low-interacting"],
        palette=SUBTYPING_PALETTE,
        legend=False,
        linewidth=0.9,
        fliersize=0.8,
        width=0.65,
        ax=ax,
    )
    high = scores_by_group.loc[scores_by_group["attention_group"] == "High-interacting", score_col]
    low = scores_by_group.loc[scores_by_group["attention_group"] == "Low-interacting", score_col]
    t_stat, p_val = ttest_ind(np.asarray(high, dtype=float), np.asarray(low, dtype=float), equal_var=False)
    ax.set_title(f"{score_labels[score_col]}\nWelch t = {t_stat:.0f}, p = {p_val:.2e}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(score_col.replace("_", " "))
    # The scores are strongly right-skewed, so the whiskers and outliers need a log scale to read.
    # symlog keeps the zeros, and the limit is clamped so the unused negative half is not drawn.
    linthresh = max(float(scores_by_group[score_col].quantile(0.01)), 1e-3)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(bottom=0, top=float(scores_by_group[score_col].max()) * 1.3)
fig.suptitle(f"Attention score distributions of high- and low-attention {sender_ct} cells", y=1.02)
fig.tight_layout()
fig.savefig(f"{figures_dir}/human_tonsil_fibroblast_attention_score_boxplots.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{figures_dir}/human_tonsil_fibroblast_attention_score_boxplots.svg", dpi=300, bbox_inches="tight")
plt.close("all")

print("\nattention scores by group (median [IQR]):")
print(
    scores_by_group.groupby("attention_group")[score_cols]
    .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    .round(4)
    .to_string()
)

# %%
