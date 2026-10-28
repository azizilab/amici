"""Regenerate the with/without-stroma interaction-score scatter from its cached edge scores.

The published panel prints rho = 0.74 and it is correct: over the 210 non-stromal sender -> receiver
pairs in `with_without_stroma_edge_scores.csv`, rho = 0.7444, p = 2.44e-38, which is what the current
xenium_stromal_bleeding_analysis.py computes (line 281, over the edge-score rows).

The rho = 0.7697, p = 2.37e-45 in the cached `with_without_stroma_interaction_metrics.csv` is NOT a
competing estimate of the same quantity: it was written by an earlier version of the analysis that
correlated the flattened 15 x 15 matrices, so it includes the 15 structurally zero self-pairs. The same
defect that was fixed in xenium_replicate_validation.py. Confirmed two ways: correlating the cached
matrices over all 225 entries reproduces 0.7697 / 2.37e-45 exactly, and the cached mean absolute delta
of 4.68 is the 210-pair value of 5.0143 diluted by 15 zeros (5.0143 * 210 / 225 = 4.680).

So this script does not correct the panel; it regenerates it from cache, needing no model and no GPU,
and reports n in the title. The stale metrics CSV is what needs regenerating.

Construction follows xenium_stromal_bleeding_analysis.py lines 298-316: the same axes, identity line,
viridis colouring by absolute change and colourbar.
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

# %% Paths
CACHE_PATH = "./figures/stromal_bleeding_analysis/cache/with_without_stroma_edge_scores.csv"
OUT_DIR = "./figures/stromal_bleeding_analysis"
OUT_PREFIX = os.path.join(OUT_DIR, "with_without_stroma_interaction_scatter_redraw")

# %% Load the cached per-pair scores (one row per non-stromal sender -> receiver pair)
edge_scores = pd.read_csv(CACHE_PATH)
rho, pval = spearmanr(edge_scores["with_stroma"], edge_scores["without_stroma"])
print(f"n cell-type pairs: {len(edge_scores)}")
print(f"Spearman rho = {rho:.4f}, p = {pval:.3e}")

# %% Redraw the panel
max_score = max(edge_scores["with_stroma"].max(), edge_scores["without_stroma"].max())
fig, ax = plt.subplots(figsize=(6.5, 6))
scatter = ax.scatter(
    edge_scores["with_stroma"],
    edge_scores["without_stroma"],
    c=edge_scores["abs_delta"],
    cmap="viridis",
    s=35,
    alpha=0.75,
    linewidths=0,
)
ax.plot([0, max_score], [0, max_score], color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Interaction score with stromal cells")
ax.set_ylabel("Interaction score after stromal exclusion")
ax.set_title(
    f"Non-Stromal Interaction Score Stability\n"
    f"Spearman ρ = {rho:.2f} (p = {pval:.1e}), n = {len(edge_scores)} pairs"
)
fig.colorbar(scatter, ax=ax, label="Absolute change")
fig.tight_layout()

for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{OUT_PREFIX}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% Save the plotted points and statistics beside the figure
edge_scores.to_csv(f"{OUT_PREFIX}_points.csv", index=False)
pd.DataFrame(
    [
        {
            "panel": "with/without stroma interaction scatter (redrawn)",
            "n_pairs": len(edge_scores),
            "spearman_rho": rho,
            "spearman_pval": pval,
        }
    ]
).to_csv(f"{OUT_PREFIX}_stats.csv", index=False)
print(f"wrote {OUT_PREFIX}.{{png,svg,pdf}}, _points.csv and _stats.csv")
