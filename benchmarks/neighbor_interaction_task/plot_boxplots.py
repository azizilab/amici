import os

import pandas as pd
from benchmark_utils import plot_boxplots


def main():
    """Plot the boxplots for the neighbor interaction task."""
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    seeds = dataset_config["seeds"]

    amici_auprcs = []
    gitiii_auprcs = []
    cgcom_auprcs = []
    for seed in seeds:
        try:
            amici_pr = pd.read_csv(
                f"results/{snakemake.wildcards.dataset}_{seed}/amici_neighbor_interaction_task_pr.csv"  # noqa: F821
            )
            gitiii_pr = pd.read_csv(
                f"results/{snakemake.wildcards.dataset}_{seed}/gitiii_neighbor_interaction_task_pr.csv"  # noqa: F821
            )
            cgcom_pr = pd.read_csv(
                f"results/{snakemake.wildcards.dataset}_{seed}/cgcom_neighbor_interaction_task_pr.csv"  # noqa: F821
            )
        except FileNotFoundError:
            continue

        amici_auprcs.append(amici_pr["avg_precision_score"].values[0])
        gitiii_auprcs.append(gitiii_pr["avg_precision_score"].values[0])
        cgcom_auprcs.append(cgcom_pr["avg_precision_score"].values[0])

        plot_boxplots(
            [amici_auprcs, gitiii_auprcs, cgcom_auprcs],
            ["AMICI", "GITIII", "CGCom"],
            metric_name="auprc",
            save_dir=os.path.dirname(snakemake.output[0]),  # noqa: F821
            title_task="Neighbor Interaction Task",
            suffix="neighbor_interaction_task",
            save_svg=True,
        )

    with open(snakemake.output[1], "w") as f:  # noqa: F821
        f.write("Done")


if __name__ == "__main__":
    main()
