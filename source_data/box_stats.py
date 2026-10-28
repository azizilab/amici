"""Box and violin statistics for the Fig. 2c-e, 4d and 6d legends.

The checklist asks that every box plot be defined in the legend "in terms of minima, maxima, centre,
bounds of box and whiskers and percentile". All three panels are drawn with library defaults, which is
the Tukey convention, so the numbers a legend needs are the ones ``matplotlib.cbook.boxplot_stats``
returns: median, Q1, Q3 and the 1.5 x IQR whisker limits, plus the data extremes, which only the 4d
violin tails show.

This script reads the exported panel CSVs under ``panels/`` and nothing else, so it needs no model, no
dataset and no GPU. Its output is tabulated in ``legend_numbers.md``.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import cbook  # noqa: E402  (after the Agg backend is set)

PANELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")


def stats(values):
    """Median, quartiles, whisker limits, extremes and outlier count of one box."""
    v = np.asarray(values, dtype=float)
    s = cbook.boxplot_stats(v)[0]
    return {
        "n": len(v),
        "min": v.min(),
        "whisker_low": s["whislo"],
        "q1": s["q1"],
        "median": s["med"],
        "q3": s["q3"],
        "whisker_high": s["whishi"],
        "max": v.max(),
        "n_outliers": int(np.sum((v < s["whislo"]) | (v > s["whishi"]))),
    }


def fig4d():
    df = pd.read_csv(os.path.join(PANELS, "Fig4", "Fig4d_length_scales.csv"))
    rows = []
    for (order, interaction), group in df.groupby(["row_order_top_to_bottom", "interaction"]):
        rows.append({"row": order, "glyph": interaction, **stats(group["length_scale_um"])})
    return pd.DataFrame(rows).sort_values("row")


def fig6d():
    df = pd.read_csv(os.path.join(PANELS, "Fig6", "Fig6d_length_scales.csv"))
    rows = []
    for (head, sender), group in df.groupby(["head_idx", "sender"]):
        rows.append({"row": head, "glyph": f"head {head} | {sender}", **stats(group["length_scale_um"])})
    return pd.DataFrame(rows).sort_values(["row", "glyph"])


def fig2ce():
    rows = []
    for panel in ("Fig2c", "Fig2d", "Fig2e"):
        wide = pd.read_csv(os.path.join(PANELS, "Fig2", f"{panel}.csv"))
        for method in [c for c in wide.columns if c != "dataset_seed"]:
            rows.append({"row": panel, "glyph": f"{panel[-1]} | {method}", **stats(wide[method])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for title, table in (("Fig. 4d (top)", fig4d()), ("Fig. 6d", fig6d()), ("Fig. 2c-e", fig2ce())):
        print(f"\n=== {title} ===")
        print(table.drop(columns="row").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
