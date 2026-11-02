"""Extended Data plots: tonsil interactions agree between the 1000-gene model and the no-ribosomal retrain.

The main tonsil model (human_tonsil_40_sweep_0f9rk0na_4303v0zu, 1000 genes) and the seed-40 retrain without
RPL21/22/32/34/37, UBA52, FAU and MALAT1 (992 genes; human_tonsil_no_ribo_retrain.py) are compared on their
all-cell-type neighbour ablation scores, over the 992 genes both models carry:

  a. interaction strength per sender -> receiver pair: the number of genes with a significant positive neighbour
     contribution (adjusted p < 0.05 and a positive ablation difference), the Fig. 6b edge weight
  b. per-gene neighbour contribution (ablation score difference) for the three sender -> receiver pairs of Fig. 6c

Both read the ablation caches that human_tonsil_analysis.py writes, computing them from the model when absent.
The plotted values are written beside the figure as CSVs, with each pair's gene-level correlation.

Run from the human_tonsil directory:  python human_tonsil_no_ribo_interaction_correlation.py
"""

# %% Imports
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from amici import AMICI
from amici.interpretation._ablation_module import AMICIAblationModule

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 7

# %% Config
LABELS_KEY = "celltype_manual_fine"
COORD_KEY = "spatial"
N_NEIGHBORS = 50
P_ADJ = 0.05
REMOVED_GENES = ["RPL21", "RPL22", "RPL32", "RPL34", "RPL37", "UBA52", "FAU", "MALAT1"]
MODELS = {
    "original": {
        "data": "data/human_tonsil_filtered_2026-07-25.h5ad",
        "model": "saved_models/human_tonsil_sweep_2026-07-25_model_2026-07-26/human_tonsil_40_sweep_0f9rk0na_4303v0zu_params_2026-07-26",
        "ablation": "figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    },
    "retrained": {
        "data": "data/human_tonsil_filtered_no_ribo_2026-07-25.h5ad",
        "model": "saved_models/human_tonsil_no_ribo_retrain/seed_40",
        "ablation": "figures/human_tonsil_no_ribo_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    },
}
FIG6C_PAIRS = [  # (sender, receiver)
    ("Fibroblast", "CD4 T cell"),
    ("Tfh cell", "GC B cell (resting)"),
    ("Follicular dendritic cell", "GC B cell (resting)"),
]
SHORT_NAMES = {"Follicular dendritic cell": "FDC", "GC B cell (resting)": "GC B (resting)", "Tfh cell": "Tfh"}
OUT_DIR = "figures/human_tonsil_no_ribo_interaction_correlation"

# Reference data-viz palette: one accent, a recessive neutral, ink for text and reference lines.
ACCENT = "#2a78d6"
NEUTRAL = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


def ablation_table(cfg):
    """All-cell-type ablation scores indexed by (receiver, gene), without the eight removed genes."""
    if not os.path.exists(cfg["ablation"]):
        adata = sc.read_h5ad(cfg["data"])
        model = AMICI.load(cfg["model"], adata=adata)
        AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
        os.makedirs(os.path.dirname(cfg["ablation"]), exist_ok=True)
        model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True).save_object(cfg["ablation"])
    df = AMICIAblationModule.load_object(cfg["ablation"])._ablation_scores_df
    df = df.assign(gene=df["gene"].astype(str), cell_type=df["cell_type"].astype(str))
    return df[~df["gene"].isin(REMOVED_GENES)].set_index(["cell_type", "gene"]).sort_index()


def gene_scores(tables):
    """One row per sender, receiver and shared gene: each model's contribution and whether it is significant."""
    shared = tables["original"].index.intersection(tables["retrained"].index)
    cell_types = sorted(shared.get_level_values("cell_type").unique())
    parts = []
    for sender in cell_types:
        part = pd.DataFrame(index=shared)
        for name, df in tables.items():
            diff = df.loc[shared, f"{sender}_diff"].astype(float)
            part[f"{name}_diff"] = diff
            part[f"{name}_significant"] = (df.loc[shared, f"{sender}_nl10_pval_adj"] > -np.log10(P_ADJ)) & (diff > 0)
        parts.append(part.reset_index().rename(columns={"cell_type": "receiver"}).assign(sender=sender))
    out = pd.concat(parts, ignore_index=True)
    return out[out["sender"] != out["receiver"]]


def pearson(a, b):
    """Pearson r, or NaN when either side is constant (a sender that never neighbours the receiver)."""
    return stats.pearsonr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan


def pair_label(sender, receiver):
    """Short sender -> receiver label for the plots."""
    return f"{SHORT_NAMES.get(sender, sender)} → {SHORT_NAMES.get(receiver, receiver)}"


def identity_line(ax, lo, hi):
    """Dashed y = x reference behind the data."""
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.75, color=INK_SECONDARY, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def style(ax):
    """Recessive axes: no top or right spine, thin ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=2.5, labelsize=6)


# %% Load both models' ablation scores on the shared genes
tables = {name: ablation_table(cfg) for name, cfg in MODELS.items()}
genes = gene_scores(tables)
os.makedirs(OUT_DIR, exist_ok=True)

# %% Per-pair interaction strength and gene-level agreement
pairs = (
    genes.groupby(["sender", "receiver"])
    .apply(
        lambda g: pd.Series(
            {
                "original_n_significant": g["original_significant"].sum(),
                "retrained_n_significant": g["retrained_significant"].sum(),
                "gene_pearson": pearson(g["original_diff"], g["retrained_diff"]),
                "gene_spearman": stats.spearmanr(g["original_diff"], g["retrained_diff"])[0]
                if g["original_diff"].std() > 0 and g["retrained_diff"].std() > 0
                else np.nan,
            }
        ),
        include_groups=False,
    )
    .reset_index()
    .astype({"original_n_significant": int, "retrained_n_significant": int})
)
pairs["fig6c_pair"] = [(s, r) in FIG6C_PAIRS for s, r in zip(pairs["sender"], pairs["receiver"], strict=True)]
pairs.to_csv(os.path.join(OUT_DIR, "interaction_strength_by_pair.csv"), index=False)

x, y = pairs["original_n_significant"], pairs["retrained_n_significant"]
r_pairs, rho_pairs = stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]
gene_r = pairs["gene_pearson"].dropna()
strength = (x + y) / 2
strong = pairs.loc[strength >= strength.median(), "gene_pearson"].dropna()
print(f"interaction strength, {len(pairs)} sender -> receiver pairs: Pearson {r_pairs:.3f}, Spearman {rho_pairs:.3f}")
print(
    f"gene-level Pearson within a pair ({len(gene_r)} pairs with contributions): median {gene_r.median():.3f} "
    f"[IQR {gene_r.quantile(0.25):.3f}-{gene_r.quantile(0.75):.3f}]; "
    f"stronger half of pairs median {strong.median():.3f}, min {strong.min():.3f}"
)

fig6c = genes[[(s, r) in FIG6C_PAIRS for s, r in zip(genes["sender"], genes["receiver"], strict=True)]]
fig6c.to_csv(os.path.join(OUT_DIR, "fig6c_pairs_gene_contributions.csv"), index=False)
for sender, receiver in FIG6C_PAIRS:
    row = pairs[(pairs["sender"] == sender) & (pairs["receiver"] == receiver)].iloc[0]
    print(
        f"{sender} -> {receiver}: {row['original_n_significant']} vs {row['retrained_n_significant']} significant genes, "
        f"gene-level Pearson {row['gene_pearson']:.3f}, Spearman {row['gene_spearman']:.3f}"
    )

# %% Figure: a, interaction strength per pair; b, per-gene contributions for the Fig. 6c pairs
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.8, wspace=0.5)

ax = axes[0]
hi = 1.05 * max(x.max(), y.max())
identity_line(ax, 0, hi)
ax.scatter(x, y, s=9, color=ACCENT, alpha=0.6, linewidths=0, zorder=2)
highlight = pairs[pairs["fig6c_pair"]]
ax.scatter(
    highlight["original_n_significant"],
    highlight["retrained_n_significant"],
    s=22,
    facecolors="none",
    edgecolors=INK,
    linewidths=0.8,
    zorder=3,
    label="Fig. 6c pairs (b)",
)
ax.legend(loc="lower right", fontsize=5.5, frameon=False, handletextpad=0.2, borderaxespad=0.2)
ax.text(
    0.04,
    0.96,
    f"r = {r_pairs:.2f}\nρ = {rho_pairs:.2f}\n{len(pairs)} pairs",
    transform=ax.transAxes,
    va="top",
    fontsize=6,
    color=INK,
)
ax.set_xlabel("Interacting genes, original model")
ax.set_ylabel("Interacting genes, no-ribosomal model")
ax.set_title("Sender → receiver interaction strength", fontsize=6.5)
style(ax)

for i, ((sender, receiver), ax) in enumerate(zip(FIG6C_PAIRS, axes[1:], strict=True)):
    g = fig6c[(fig6c["sender"] == sender) & (fig6c["receiver"] == receiver)]
    both = g["original_significant"] & g["retrained_significant"]
    lo = min(g["original_diff"].min(), g["retrained_diff"].min())
    hi = max(g["original_diff"].max(), g["retrained_diff"].max())
    pad = 0.05 * (hi - lo)
    identity_line(ax, lo - pad, hi + pad)
    ax.scatter(
        g.loc[both, "original_diff"],
        g.loc[both, "retrained_diff"],
        s=5,
        color=ACCENT,
        alpha=0.8,
        linewidths=0,
        zorder=2,
        label="Significant in both",
    )
    ax.scatter(
        g.loc[~both, "original_diff"],
        g.loc[~both, "retrained_diff"],
        s=5,
        color=NEUTRAL,
        linewidths=0,
        zorder=1,
        label="Other genes",
    )
    ax.text(
        0.04,
        0.96,
        f"r = {pearson(g['original_diff'], g['retrained_diff']):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=6,
    )
    ax.set_title(pair_label(sender, receiver), fontsize=6.5)
    ax.set_xlabel("Contribution, original")
    if i == 0:
        ax.set_ylabel("Contribution, no-ribosomal")
    style(ax)
fig.legend(
    *axes[1].get_legend_handles_labels(),
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    fontsize=5.5,
    frameon=False,
    handletextpad=0.2,
    markerscale=1.6,
)

# Panel letters go beside each panel's y-axis label, once the equal aspect has set the axes' final positions.
for ax in axes:
    ax.apply_aspect()
b_left = axes[1].get_position().x0
fig.text(0.005, 0.97, "a", fontsize=8, fontweight="bold", va="top")
fig.text(b_left - 0.06, 0.97, "b", fontsize=8, fontweight="bold", va="top")
fig.text(
    b_left - 0.035,
    0.965,
    f"Per-gene neighbour contribution for the Fig. 6c pairs ({fig6c['gene'].nunique()} shared genes)",
    fontsize=6.5,
    va="top",
)

for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"no_ribo_interaction_correlation.{ext}"), dpi=300)
print(f"saved to {OUT_DIR}")
