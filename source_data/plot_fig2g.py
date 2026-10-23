"""Redraw Fig. 2g (AMICI length scales on the realistic breast cancer dataset) from its Source Data sheet.

The realistic run behind the published panel was not kept, so the panel is redrawn from the re-run exported by
fig2_benchmarks.py. Drawing directly from source_data/panels/Fig2/Fig2g.csv keeps the panel and its Source Data
identical. The styling follows the published panel: one KDE per interaction (seaborn defaults, as in
benchmarks/length_scale_task/plot_kde.py) with the ground-truth length scale as a dashed line of the same colour.

Run from the repository root after fig2_benchmarks.py:  python source_data/plot_fig2g.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import _common as C

INTERACTIONS = {
    "Macrophages -> DCIS": ("Macrophages → DCIS", "#E6A532"),
    "T Cells -> Endothelial": ("T Cells → Endothelial", "#1E7B85"),
    "Invasive Tumor -> Myoepithelial": ("Invasive tumor → Myoepi", "#8B3A6B"),
}


def main():
    df = pd.read_csv(C.PANEL_DIR / "Fig2" / "Fig2g.csv")
    missing = set(df["interaction"]) - set(INTERACTIONS)
    assert not missing, f"unexpected interactions: {missing}"
    order = list(INTERACTIONS)
    palette = {k: v[1] for k, v in INTERACTIONS.items()}

    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.kdeplot(data=df, x="length_scale_um", hue="interaction", hue_order=order, palette=palette, ax=ax, legend=False)
    handles = []
    for key in order:
        label, color = INTERACTIONS[key]
        gt = df.loc[df["interaction"] == key, "ground_truth_length_scale_um"].iloc[0]
        ax.axvline(gt, color=color, linestyle="--", linewidth=1.5)
        handles.append(Line2D([0], [0], color=color, lw=2, label=f"Inferred ({label})"))
    for key in order:
        label, color = INTERACTIONS[key]
        handles.append(Line2D([0], [0], color=color, lw=2, linestyle="--", label=f"Ground truth ({label})"))

    ax.set_xlim(-10, 85)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Density")
    ax.set_title("Inferred Interaction Length scales", fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    out = C.OUT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"Fig2g_rerun.{ext}", dpi=300, bbox_inches="tight")
    print(f"  wrote {out.relative_to(C.REPO)}/Fig2g_rerun.{{pdf,svg,png}}")
    for key in order:
        v = df.loc[df["interaction"] == key, "length_scale_um"]
        print(f"  {INTERACTIONS[key][0]}: n = {len(v)}, median {v.median():.1f} um")


if __name__ == "__main__":
    main()
