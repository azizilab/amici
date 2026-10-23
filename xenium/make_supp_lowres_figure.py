# %% Assemble the full low-resolution comparison supplementary figure (Fig S19).
#
# Panels a-d reuse the interaction matrices computed by xenium_analysis_lowres.py (from its cached ablation
# scores) and panel e reuses the Moran's I scores cached by xenium_within_type_autocorrelation.py, so this
# script only lays out and annotates results rather than recomputing them. Run both of those scripts first.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

from amici.interpretation._ablation_module import AMICIAblationModule

matplotlib_svg_fonttype = "none"
plt.rcParams["svg.fonttype"] = matplotlib_svg_fonttype

# %% Settings
SEED = 18
LOWRES_RUN_ID = "5tfi78lp"
LOWRES_MODEL_DATE = "2026-09-19"
N_PERMUTATIONS = 1000

ABLATION_CACHE_DIR = "./figures/cached_ablation_scores"
HIGHRES_ABLATION_PATH = os.path.join(ABLATION_CACHE_DIR, "highres_model_ablation_scores.pkl")
LOWRES_ABLATION_PATH = os.path.join(
    ABLATION_CACHE_DIR, f"lowres_model_{LOWRES_RUN_ID}_{LOWRES_MODEL_DATE}_ablation_scores.pkl"
)
AUTOCORRELATION_SCORES_PATH = (
    "data/xenium_sample1/within_type_autocorrelation_cache/within_type_autocorrelation_validation_scores.csv"
)
FIGURE_DIR = "./figures"
OUTPUT_STEM = os.path.join(FIGURE_DIR, "SuppFigLoResComp")

# Same grouping map as xenium_preprocess_lowres.py / xenium_analysis_lowres.py
cell_label_map_lowres = {
    "CD8+_T_Cells": "T_Cells",
    "CD4+_T_Cells": "T_Cells",
    "DCIS_1": "DCIS",
    "DCIS_2": "DCIS",
    "IRF7+_DCs": "DCs",
    "LAMP3+_DCs": "DCs",
    "Macrophages_1": "Macrophages",
    "Macrophages_2": "Macrophages",
    "Myoepi_ACTA2+": "Myoepithelial",
    "Myoepi_KRT15+": "Myoepithelial",
    "Perivascular-Like": "Vascular",
    "Endothelial": "Vascular",
}
lowres_order = [
    "T_Cells",
    "DCIS",
    "DCs",
    "Macrophages",
    "Myoepithelial",
    "Invasive_Tumor",
    "B_Cells",
    "Mast_Cells",
    "Vascular",
]
highres_groups = {
    "T_Cells": ["CD8+_T_Cells", "CD4+_T_Cells"],
    "DCIS": ["DCIS_1", "DCIS_2"],
    "DCs": ["IRF7+_DCs", "LAMP3+_DCs"],
    "Macrophages": ["Macrophages_1", "Macrophages_2"],
    "Myoepithelial": ["Myoepi_ACTA2+", "Myoepi_KRT15+"],
    "Invasive_Tumor": ["Invasive_Tumor"],
    "B_Cells": ["B_Cells"],
    "Mast_Cells": ["Mast_Cells"],
    "Vascular": ["Perivascular-Like", "Endothelial"],
}

# %% Load cached interaction matrices
interaction_matrix_old = AMICIAblationModule.load_object(HIGHRES_ABLATION_PATH)._get_interaction_weight_matrix()
interaction_matrix_new = AMICIAblationModule.load_object(LOWRES_ABLATION_PATH)._get_interaction_weight_matrix()

interaction_matrix_old_lowres = interaction_matrix_old.copy()
interaction_matrix_old_lowres.index = [cell_label_map_lowres.get(ct, ct) for ct in interaction_matrix_old.index]
interaction_matrix_old_lowres.columns = [cell_label_map_lowres.get(ct, ct) for ct in interaction_matrix_old.columns]
interaction_matrix_old_lowres = interaction_matrix_old_lowres.groupby(level=0).mean()
interaction_matrix_old_lowres = interaction_matrix_old_lowres.T.groupby(level=0).mean().T

highres_order = [ct for group in lowres_order for ct in highres_groups.get(group, [])]
shared_lowres = [
    ct for ct in lowres_order if ct in interaction_matrix_old_lowres.index and ct in interaction_matrix_new.index
]
shared_highres = [ct for ct in highres_order if ct in interaction_matrix_old.index]

mat_old_hires = interaction_matrix_old.loc[shared_highres, shared_highres]
mat_old_shared = interaction_matrix_old_lowres.loc[shared_lowres, shared_lowres]
mat_new_shared = interaction_matrix_new.loc[shared_lowres, shared_lowres]

# %% Permutation null for the similarity between the aggregated and low-resolution matrices
off_diag_mask = ~np.eye(len(shared_lowres), dtype=bool)
A = mat_old_shared.values[off_diag_mask]
B = mat_new_shared.values[off_diag_mask]

cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
r_matrix = pearsonr(A, B)[0]

rng = np.random.default_rng(SEED)
perm_cosine = np.zeros(N_PERMUTATIONS)
perm_pearson = np.zeros(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    perm = rng.permutation(len(shared_lowres))
    B_perm = mat_new_shared.values[np.ix_(perm, perm)][off_diag_mask]
    perm_cosine[i] = np.dot(A, B_perm) / (np.linalg.norm(A) * np.linalg.norm(B_perm))
    perm_pearson[i] = pearsonr(A, B_perm)[0]


def empirical_pval(observed, null_vals):
    """Fraction of permutations at least as extreme as the observed statistic, or None if none are."""
    p = (null_vals >= observed).sum() / len(null_vals)
    return p if p > 0 else None


def fmt_pval(p, n):
    """Format an empirical p-value, reporting the resolution limit when no permutation is as extreme."""
    return f"< {1 / n:.4f}" if p is None else f"{p:.4f}"


# %% Lay out the figure
fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(3, 6, height_ratios=[1.05, 0.85, 0.95], hspace=0.45, wspace=0.9)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d1 = fig.add_subplot(gs[1, 0:3])
ax_d2 = fig.add_subplot(gs[1, 3:6])
ax_e = fig.add_subplot(gs[2, 1:5])

for ax, matrix, title in (
    (ax_a, mat_old_hires, "High-Resolution Cell-Type Model"),
    (ax_b, mat_old_shared, "High-Resolution Cell-Type Model (aggregated to Low-Resolution)"),
    (ax_c, mat_new_shared, "Low-Resolution Cell-Type Model"),
):
    sns.heatmap(matrix, ax=ax, cmap="Reds", cbar_kws={"label": "Interaction Weight"}, linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Receiver Cell Type")
    ax.set_ylabel("Sender Cell Type")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)

# Blue boxes on panel a mark the high-resolution cell types merged into one low-resolution label
position = {ct: i for i, ct in enumerate(shared_highres)}
for group in lowres_order:
    members = [ct for ct in highres_groups.get(group, []) if ct in position]
    if not members:
        continue
    start = min(position[ct] for ct in members)
    ax_a.add_patch(
        Rectangle((start, start), len(members), len(members), fill=False, edgecolor="blue", linewidth=2.0, zorder=5)
    )

for ax, null_vals, observed, metric in (
    (ax_d1, perm_cosine, cosine_sim, "Cosine Similarity"),
    (ax_d2, perm_pearson, r_matrix, "Pearson r"),
):
    z = (observed - null_vals.mean()) / null_vals.std()
    p_str = fmt_pval(empirical_pval(observed, null_vals), N_PERMUTATIONS)
    ax.hist(null_vals, bins=40, color="steelblue", alpha=0.7, label="Shuffled null")
    ax.axvline(observed, color="crimson", linewidth=2, label=f"Observed ({observed:.3f})")
    ax.axvline(
        null_vals.mean(),
        color="steelblue",
        linewidth=1.5,
        linestyle="--",
        label=f"Null mean ({null_vals.mean():.3f})",
    )
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"{metric}:  z = {z:.2f},  p = {p_str}", fontsize=11)
    ax.legend()

# Panel e: spatial autocorrelation diagnostic, as in xenium_within_type_autocorrelation.plot_score_comparison
scores = pd.read_csv(AUTOCORRELATION_SCORES_PATH)
plot_df = scores[
    ["lowres_label", "deployable_mean_top2_pc_morans_i", "oracle_max_subtype_morans_i", "n_highres_subtypes"]
].sort_values("oracle_max_subtype_morans_i", ascending=False)
melted = plot_df.melt(
    id_vars=["lowres_label", "n_highres_subtypes"],
    value_vars=["deployable_mean_top2_pc_morans_i", "oracle_max_subtype_morans_i"],
    var_name="score_type",
    value_name="morans_i",
)
melted["score_type"] = melted["score_type"].map(
    {
        "deployable_mean_top2_pc_morans_i": "Deployable expression PCs",
        "oracle_max_subtype_morans_i": "High-res subtype oracle",
    }
)
sns.barplot(
    data=melted,
    x="lowres_label",
    y="morans_i",
    hue="score_type",
    order=plot_df["lowres_label"],
    palette=["#4C78A8", "#F58518"],
    ax=ax_e,
)
subtype_counts = plot_df.set_index("lowres_label")["n_highres_subtypes"]
ax_e.set_xticks(range(len(plot_df)))
ax_e.set_xticklabels(
    [f"{label}\n({subtype_counts[label]} subtypes)" for label in plot_df["lowres_label"]], rotation=35, ha="right"
)
ax_e.set_xlabel("Low-resolution cell type")
ax_e.set_ylabel("Moran's I")
ax_e.set_title("Spatial autocorrelation diagnostic by coarse annotation", fontsize=11)
ax_e.legend(frameon=False, title="")
ax_e.grid(axis="y", alpha=0.25)

for ax, letter in ((ax_a, "a."), (ax_b, "b."), (ax_c, "c."), (ax_d1, "d."), (ax_e, "e.")):
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-45, 28),
        textcoords="offset points",
        fontsize=20,
        fontweight="bold",
        va="top",
    )

fig.suptitle("")
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{OUTPUT_STEM}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Cosine similarity: {cosine_sim:.4f} | null {perm_cosine.mean():.4f} | p = {fmt_pval(empirical_pval(cosine_sim, perm_cosine), N_PERMUTATIONS)}")
print(f"Pearson r:         {r_matrix:.4f} | null {perm_pearson.mean():.4f} | p = {fmt_pval(empirical_pval(r_matrix, perm_pearson), N_PERMUTATIONS)}")
print(f"Saved {OUTPUT_STEM}.png/.svg/.pdf")
