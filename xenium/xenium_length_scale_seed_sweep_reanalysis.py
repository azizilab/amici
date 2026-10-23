"""Correct reanalysis of the Xenium length-scale seed sweep, using the already-cached models.

The first pass (xenium_length_scale_seed_sweep.py) hardcoded head indices 4 and 5 for every
(sender, receiver) pair, copied from the single trained model (seed 18) behind the published
figure. Head identity in an unordered multi-head attention layer has no reason to be stable
across independent training runs -- confirmed directly: in seeds 101/104/111, every one of a
spot-checked set of interactions specialized on SOME head with base_attention_score far above
the dummy score, just never on head 4 or 5.

v2 fixed that by sweeping every head, keeping only heads whose median length scale is sensible
(0, 50] micrometers (the same filter xenium_analysis.py applies before it hardcodes head_idxs
for the one model it plots), and reporting only the single most-confidently-specialized head
per seed (largest base_attention_score margin over the dummy score). That collapsed cases where
a seed had *two* independently valid heads -- e.g. seed 101's CD8+_T_Cells -> Macrophages_1 had
heads 1/2/7 passing the filter at 5.9/30.4/7.7 um -- down to one number, discarding the fact
that this seed actually specialized on two distinct heads for this interaction.

v3 tried to recover that structure by clustering the pooled length-scale values (across all
seeds and heads) by magnitude, but that imposes a global bimodality assumption and needs an
arbitrary gap-size threshold to decide what counts as "the same" scale.

v4 (this version) drops the clustering entirely. Per (seed, sender, receiver), every head that
passes the (0, 50] filter is its own data point; heads are matched across seeds purely by rank
-- within one seed, the valid heads are sorted ascending by length scale, so the shortest valid
head becomes that seed's "head rank 1" point, the next becomes "head rank 2", etc. Each rank
gets its own violin. Most seeds have one or two valid heads per row (typically a shorter-range
and a longer-range head), so most rows show one or two violins; no magnitude threshold is
involved anywhere.

No retraining is needed -- this only reprocesses the models already saved by the first pass.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../benchmarks"))

from amici import AMICI  # noqa: E402

DATA_DATE = "2025-05-01"
LABELS_KEY = "celltype_train_grouped"
COORD_KEY = "spatial"
DATA_PATH = os.path.join(SCRIPT_DIR, f"data/xenium_sample1/xenium_sample1_filtered_{DATA_DATE}.h5ad")
N_NEIGHBORS = 50

MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models", "xenium_length_scale_seed_sweep")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "xenium_length_scale_seed_sweep_reanalysis_v4")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures", "xenium_length_scale_seed_sweep_reanalysis_v4")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

ATTENTION_THRESHOLD = 0.1
VALID_LENGTH_SCALE_RANGE = (0, 50)  # matches xenium_analysis.py's `heads_positive_length_scale` filter

# (sender, receiver, sample_threshold) -- the unique sender/receiver pairs behind the reference
# figure, with sample_threshold keyed by receiver as in xenium_analysis.py's `interactions` dict.
INTERACTION_PAIRS = [
    ("CD4+_T_Cells", "CD8+_T_Cells", 0.02),
    ("CD4+_T_Cells", "B_Cells", 0.02),
    ("CD8+_T_Cells", "Macrophages_1", 0.02),
    ("B_Cells", "CD4+_T_Cells", 0.02),
    ("CD8+_T_Cells", "Invasive_Tumor", 0.001),
    ("Macrophages_2", "CD4+_T_Cells", 0.02),
    ("Macrophages_2", "CD8+_T_Cells", 0.02),
    ("DCIS_1", "Macrophages_1", 0.02),
    ("Invasive_Tumor", "Macrophages_1", 0.02),
    ("Macrophages_2", "Macrophages_1", 0.02),
]


def row_label(sender, receiver):
    """Format one interaction's row label for the plot."""
    return f"{sender} → {receiver}"


ROW_ORDER = [row_label(s, r) for s, r, _ in INTERACTION_PAIRS]


def discover_seeds():
    """Every seed with a cached, trained model on disk."""
    if not os.path.isdir(MODEL_DIR):
        return []
    seeds = []
    for name in os.listdir(MODEL_DIR):
        if name.startswith("seed_") and os.path.exists(os.path.join(MODEL_DIR, name, "model.pt")):
            seeds.append(int(name.split("_")[1]))
    return sorted(seeds)


def valid_heads_length_scale(counterfactual, sender, n_heads, sample_threshold):
    """Every head for this sender whose median length scale passes the (0, 50] sanity filter.

    Returns a list of (head_idx, median_length_scale, n_samples) tuples, one per valid head.
    """
    length_scale_df = counterfactual._calculate_length_scales(
        head_idxs=list(range(n_heads)),
        sender_types=[sender],
        attention_threshold=ATTENTION_THRESHOLD,
        sample_threshold=sample_threshold,
    )
    per_head_median = length_scale_df.groupby("head_idx")["length_scale"].median()
    per_head_n = length_scale_df.groupby("head_idx").size()
    lo, hi = VALID_LENGTH_SCALE_RANGE
    valid = per_head_median[(per_head_median > lo) & (per_head_median <= hi)]
    return [(int(h), float(valid[h]), int(per_head_n[h])) for h in valid.index]


def compute_seed_rows(seed, adata):
    """Long-form table of every valid head's median for one seed's cached model."""
    run_dir = os.path.join(MODEL_DIR, f"seed_{seed}")
    model = AMICI.load(run_dir, adata=adata)
    AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    n_heads = model.module.n_heads

    records = []
    receivers = sorted({receiver for _, receiver, _ in INTERACTION_PAIRS})
    for receiver in receivers:
        counterfactual = model.get_counterfactual_attention_patterns(cell_type=receiver, adata=adata)
        for sender, _, sample_threshold in [p for p in INTERACTION_PAIRS if p[1] == receiver]:
            for head_idx, length_scale, n_samples in valid_heads_length_scale(
                counterfactual, sender, n_heads, sample_threshold
            ):
                records.append(
                    {
                        "seed": seed,
                        "sender": sender,
                        "receiver": receiver,
                        "row_label": row_label(sender, receiver),
                        "head_idx": head_idx,
                        "length_scale": length_scale,
                        "n_samples": n_samples,
                    }
                )
    return pd.DataFrame(records)


def assign_head_rank(long_df):
    """Rank each seed's valid heads ascending by length scale, within (seed, row_label).

    Rank 0 = that seed's shortest valid head, rank 1 = its next-shortest, etc. This matches
    heads across seeds by relative order rather than by clustering on absolute magnitude, so no
    threshold is involved -- a seed with two valid heads always contributes one point to rank 0
    and one to rank 1, however far apart or close together those two heads' scales are.
    """
    long_df = long_df.sort_values(["row_label", "seed", "length_scale"]).copy()
    long_df["head_rank"] = long_df.groupby(["row_label", "seed"]).cumcount()
    return long_df


RANK_STYLE = [
    {"label": "shortest head", "color": "tab:blue", "offset": -0.18},
    {"label": "2nd shortest head", "color": "tab:orange", "offset": 0.18},
    {"label": "3rd shortest head", "color": "tab:green", "offset": 0.42},
    {"label": "4th shortest head", "color": "tab:red", "offset": -0.42},
    {"label": "5th shortest head", "color": "tab:purple", "offset": 0.66},
]


def plot_seed_violin(long_df, n_seeds):
    """Plot the per-seed length-scale distribution for each interaction, split by head rank."""
    order = ROW_ORDER[::-1]
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.5))
    legend_handles = {}
    for rank, style in enumerate(RANK_STYLE):
        plot_positions, plot_data = [], []
        for pos, label in zip(positions, order, strict=False):
            values = (
                long_df.loc[(long_df["row_label"] == label) & (long_df["head_rank"] == rank), "length_scale"]
                .dropna()
                .to_numpy()
            )
            if len(values) == 0:
                continue
            y = pos + style["offset"]
            ax.scatter(values, np.full(len(values), y), s=8, color="black", alpha=0.4, zorder=3)
            if len(values) >= 2:
                plot_positions.append(y)
                plot_data.append(values)
        if plot_data:
            parts = ax.violinplot(plot_data, positions=plot_positions, vert=False, showmedians=True, widths=0.3)
            for pc in parts["bodies"]:
                pc.set_facecolor(style["color"])
                pc.set_alpha(0.6)
            legend_handles[style["label"]] = plt.Rectangle((0, 0), 1, 1, fc=style["color"], alpha=0.6)

    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("Length scale, per valid head rank within a seed (µm)")
    ax.set_title(f"Xenium: seed-to-seed variability of length scale, by head rank (n={n_seeds} seeds)")
    ax.grid(axis="x", alpha=0.3)
    if legend_handles:
        ax.legend(legend_handles.values(), legend_handles.keys(), loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGURE_DIR, f"xenium_length_scale_seed_variability_reanalysis.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    """Sweep every head of each cached seed model and plot the resulting length-scale variability."""
    seeds = discover_seeds()
    print(f"Found {len(seeds)} cached seed models: {seeds}", flush=True)
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()

    all_long_rows = []
    for seed in seeds:
        seed_csv = os.path.join(RESULTS_DIR, f"seed_{seed}_long.csv")
        if os.path.exists(seed_csv):
            print(f"[skip] seed {seed} already reanalyzed", flush=True)
            all_long_rows.append(pd.read_csv(seed_csv))
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        seed_df = compute_seed_rows(seed, adata)
        seed_df.to_csv(seed_csv, index=False)
        all_long_rows.append(seed_df)
        print(seed_df.to_string(index=False), flush=True)

    long_df = pd.concat(all_long_rows, ignore_index=True)
    long_df = assign_head_rank(long_df)
    long_df.to_csv(os.path.join(RESULTS_DIR, "xenium_length_scale_seed_reanalysis_by_rank.csv"), index=False)
    plot_seed_violin(long_df, len(seeds))

    print("\n=== Summary across seeds, by head rank ===", flush=True)
    summary = long_df.groupby(["row_label", "head_rank"]).agg(
        n_seeds=("seed", "count"),
        mean=("length_scale", "mean"),
        median=("length_scale", "median"),
        std=("length_scale", "std"),
    )
    print(summary.to_string(), flush=True)
    print(f"\nWrote results to {RESULTS_DIR}\nWrote figure to {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    main()
