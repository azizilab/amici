"""Redraw Fig. S5b from the exported replicate interaction-weight matrices, diagonal dropped.

The published panel reports rho = 0.64 over 210 points, which matches neither the value nor the point
count that xenium_replicate_validation.py produces (rho = 0.582 over 225 points, because it flattens the
full matrix and so plots the 15 structurally zero self-pairs). The code behind the published panel is not
in the repository, so the panel is redrawn here from the committed method with the diagonal excluded:
210 cell-type pairs, which is both the statistically correct set and the published point count.

Reads the matrices exported by the replicate rerun, so it needs no model and no GPU.
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# %% Paths and palette (the palette is the one in xenium_replicate_validation.py)
MATRIX_DIR = "./figures/replicate_validation"
OUT_PREFIX = os.path.join(MATRIX_DIR, "Fig_S5b_redraw")

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

# %% Load the two interaction weight matrices (rows senders, columns receivers)
mat_rep1 = pd.read_csv(os.path.join(MATRIX_DIR, "rep1_interaction_weight_matrix.csv"), index_col=0)
mat_rep2 = pd.read_csv(os.path.join(MATRIX_DIR, "rep2_interaction_weight_matrix.csv"), index_col=0)
mat_rep2 = mat_rep2.loc[mat_rep1.index, mat_rep1.columns]

# %% Drop the diagonal: a cell type is never its own neighbor, so those entries are structurally zero
off_diagonal = ~np.eye(len(mat_rep1), dtype=bool)
pairs = pd.DataFrame(
    {
        "sender": [s for s in mat_rep1.index for _ in mat_rep1.columns],
        "receiver": [r for _ in mat_rep1.index for r in mat_rep1.columns],
        "replicate_1": mat_rep1.values.flatten(),
        "replicate_2": mat_rep2.values.flatten(),
        "self_pair": np.eye(len(mat_rep1), dtype=bool).flatten(),
    }
)
pairs = pairs[~pairs["self_pair"]].drop(columns="self_pair").reset_index(drop=True)

spearman_rho, spearman_p = spearmanr(pairs["replicate_1"], pairs["replicate_2"])
print(f"n cell-type pairs: {len(pairs)}")
print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.3e}")

# %% Redraw the panel
plt.figure(figsize=(6, 6))
ax = sns.scatterplot(
    data=pairs,
    x="replicate_1",
    y="replicate_2",
    hue="sender",
    palette=CELL_TYPE_PALETTE,
    s=45,
    edgecolor="white",
    linewidth=0.4,
)

lims = [
    min(pairs["replicate_1"].min(), pairs["replicate_2"].min()),
    max(pairs["replicate_1"].max(), pairs["replicate_2"].max()),
]
pad = 0.03 * (lims[1] - lims[0])
lims = [lims[0] - pad, lims[1] + pad]
plt.plot(lims, lims, "--", color="grey", linewidth=1, zorder=0)

plt.xlabel("Interaction weight (Replicate 1)")
plt.ylabel("Interaction weight (Replicate 2)")
plt.title(
    f"Spearman ρ = {spearman_rho:.2f} (p = {spearman_p:.1e}), n = {len(pairs)} pairs\nColored by sender"
)
plt.xlim(lims)
plt.ylim(lims)
ax.legend(title="Sender", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
plt.tight_layout()

for ext in ("png", "svg", "pdf"):
    plt.savefig(f"{OUT_PREFIX}.{ext}", dpi=300, bbox_inches="tight")
plt.close()

# %% Save the plotted points and the statistics beside the figure
pairs.to_csv(f"{OUT_PREFIX}_points.csv", index=False)
pd.DataFrame(
    [
        {
            "panel": "Fig. S5b (redrawn, diagonal dropped)",
            "n_pairs": len(pairs),
            "spearman_rho": spearman_rho,
            "spearman_pval": spearman_p,
        }
    ]
).to_csv(f"{OUT_PREFIX}_stats.csv", index=False)
print(f"wrote {OUT_PREFIX}.{{png,svg,pdf}}, _points.csv and _stats.csv")
