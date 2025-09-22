import os

import pandas as pd
from benchmark_utils import plot_pr_curves


def main():
    """Plot the PR curves for the receiver subtype task."""
    amici_pr = pd.read_csv(snakemake.input.amici_pr_path)  # noqa: F821
    gitiii_pr = pd.read_csv(snakemake.input.gitiii_pr_path)  # noqa: F821
    cgcom_pr = pd.read_csv(snakemake.input.cgcom_pr_path)  # noqa: F821
    gt_receiver_subtypes = pd.read_csv(snakemake.input.gt_receiver_subtypes_path)  # noqa: F821

    num_positive_classes_interactions = len(gt_receiver_subtypes[gt_receiver_subtypes["class"] == 1.0])

    # Plot the PR curves for all 2 models
    plot_pr_curves(
        [amici_pr, gitiii_pr, cgcom_pr],
        ["AMICI", "GITIII", "CGCom"],
        num_positive_classes=[num_positive_classes_interactions],
        save_dir=os.path.dirname(snakemake.output[0]),  # noqa: F821
        suffix="receiver_subtype_task",
    )

    done_path = snakemake.output[1]  # noqa: F821
    with open(done_path, "w") as _:
        os.utime(done_path, None)


if __name__ == "__main__":
    main()
