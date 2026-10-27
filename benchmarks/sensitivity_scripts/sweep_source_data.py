"""Write the numbers behind the hyperparameter sensitivity panels to disk.

The four sweeps (number of heads, number of neighbors, attention penalty, value penalty) each train
one model per dataset seed per setting, score the three benchmark tasks and plot the result. They
used to call only ``savefig``, so the AUPRCs existed solely as pixels and the panels had no Source
Data. These helpers write the same quantities the panels draw:

``<prefix>_auprc.csv``
    One row per setting x dataset seed x task: the AUPRC behind each point of the box plots.
``<prefix>_pr_curves.csv``
    The precision-recall curves drawn for the example dataset seed, one row per threshold, thinned
    to at most ``MAX_CURVE_POINTS`` so the file stays a sensible size.
"""

import os

import numpy as np
import pandas as pd

MAX_CURVE_POINTS = 501

TASK_LABELS = {
    "gene": "Gene task",
    "neighbor": "Neighbor interaction task",
    "receiver": "Receiver subtype task",
}


def _thin(recall, precision, points=MAX_CURVE_POINTS):
    """Interpolate a PR curve onto a fixed recall grid when it is longer than `points`."""
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if recall.size <= points:
        return recall, precision
    order = np.argsort(recall)
    grid = np.linspace(recall[order][0], recall[order][-1], points)
    return grid, np.interp(grid, recall[order], precision[order])


def write_sweep_tables(base_dir, prefix, setting_name, dataset_seeds, results, curves):
    """Write the AUPRC table and the example-seed PR curves for one sweep.

    Args:
        base_dir: the sensitivity_scripts directory; tables go to its ``results/<prefix>`` folder.
        prefix: file stem, e.g. ``head_analysis``.
        setting_name: what the swept setting is called, e.g. ``n_heads``.
        dataset_seeds: the dataset seeds, in the order the AUPRCs were appended.
        results: ``{task: {setting: [auprc per dataset seed]}}``.
        curves: ``{task: {setting: {"precision": ..., "recall": ..., "auprc": ...}}}``.
    """
    out_dir = os.path.join(base_dir, "results", prefix)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for task, by_setting in results.items():
        for setting, auprcs in by_setting.items():
            for seed, auprc in zip(dataset_seeds, auprcs, strict=False):
                rows.append(
                    {
                        setting_name: setting,
                        "dataset_seed": seed,
                        "task": TASK_LABELS.get(task, task),
                        "auprc": auprc,
                    }
                )
    auprc_path = os.path.join(out_dir, f"{prefix}_auprc.csv")
    pd.DataFrame(rows).to_csv(auprc_path, index=False)

    curve_rows = []
    example_seed = dataset_seeds[0] if len(dataset_seeds) else None
    for task, by_setting in curves.items():
        for setting, curve in by_setting.items():
            recall, precision = _thin(curve["recall"], curve["precision"])
            curve_rows.append(
                pd.DataFrame(
                    {
                        setting_name: setting,
                        "dataset_seed": example_seed,
                        "task": TASK_LABELS.get(task, task),
                        "recall": recall,
                        "precision": precision,
                        "auprc": curve["auprc"],
                    }
                )
            )
    curves_path = os.path.join(out_dir, f"{prefix}_pr_curves.csv")
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(curves_path, index=False)

    print(f"[source data] wrote {auprc_path} and {curves_path}")
    return auprc_path, curves_path
