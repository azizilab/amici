import os

import pandas as pd
from benchmark_utils import plot_boxplots


def main():
    """Plot the boxplots for the gene task."""
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    seeds = dataset_config["seeds"]
    amici_auprcs = []
    gitiii_auprcs = []

    ncem_auprcs = {}
    nichede_auprcs = {}
    for niche_size in dataset_config["ncem_niche_sizes"]:
        ncem_auprcs[niche_size] = []
    for niche_size in dataset_config["nichede_niche_sizes"]:
        nichede_auprcs[niche_size] = []

    for seed in seeds:
        try:
            amici_pr = pd.read_csv(f"results/{snakemake.wildcards.dataset}_{seed}/amici_gene_task_pr.csv")  # noqa: F821
            gitiii_pr = pd.read_csv(f"results/{snakemake.wildcards.dataset}_{seed}/gitiii_gene_task_pr.csv")  # noqa: F821
            ncem_prs = {}
            for niche_size in dataset_config["ncem_niche_sizes"]:
                ncem_prs[niche_size] = (
                    pd.read_csv(f"results/{snakemake.wildcards.dataset}_{seed}/ncem_{niche_size}_gene_task_pr.csv")  # noqa: F821
                )
            nichede_prs = {}
            for niche_size in dataset_config["nichede_niche_sizes"]:
                nichede_prs[niche_size] = (
                    pd.read_csv(f"results/{snakemake.wildcards.dataset}_{seed}/nichede_{niche_size}_gene_task_pr.csv")  # noqa: F821
                )
        except FileNotFoundError:
            continue

        amici_auprcs.append(amici_pr["avg_precision_score"].values[0])
        gitiii_auprcs.append(gitiii_pr["avg_precision_score"].values[0])

        for niche_size in dataset_config["ncem_niche_sizes"]:
            ncem_auprcs[niche_size].append(ncem_prs[niche_size]["avg_precision_score"].values[0])

        for niche_size in dataset_config["nichede_niche_sizes"]:
            nichede_auprcs[niche_size].append(nichede_prs[niche_size]["avg_precision_score"].values[0])

    ncem_model_names = [f"NCEM_{niche_size}" for niche_size in dataset_config["ncem_niche_sizes"]]
    nichede_model_names = [f"NicheDE_{niche_size}" for niche_size in dataset_config["nichede_niche_sizes"]]

    all_ncem_auprcs = [ncem_auprcs[niche_size] for niche_size in dataset_config["ncem_niche_sizes"]]
    all_nichede_auprcs = [nichede_auprcs[niche_size] for niche_size in dataset_config["nichede_niche_sizes"]]

    plot_boxplots(
        [amici_auprcs, gitiii_auprcs] + all_nichede_auprcs + all_ncem_auprcs,
        ["AMICI", "GITIII"] + nichede_model_names + ncem_model_names,
        metric_name="auprc",
        save_dir=os.path.dirname(snakemake.output[0]),  # noqa: F821
        title_task="Gene Task",
        suffix="gene_task",
        save_svg=True,
    )

    with open(snakemake.output[1], "w") as f:  # noqa: F821
        f.write("Done")


if __name__ == "__main__":
    main()
