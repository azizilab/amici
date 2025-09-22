import os

import pandas as pd
from benchmark_utils import plot_pr_curves


def main():
    """Plot the PR curves for the gene task."""
    gt_genes = pd.read_csv(snakemake.input.gt_genes_path)  # noqa: F821

    interactions = snakemake.config["datasets"][snakemake.wildcards.dataset]["gt_interactions"]  # noqa: F821
    num_positive_classes_interactions = []
    for interaction_name in interactions:
        num_positive_classes = len(gt_genes[(gt_genes["interaction"] == interaction_name) & (gt_genes["class"] == 1.0)])
        num_positive_classes_interactions.append(num_positive_classes)

    amici_pr = pd.read_csv(snakemake.input.amici_pr_path)  # noqa: F821
    gitiii_pr = pd.read_csv(snakemake.input.gitiii_pr_path)  # noqa: F821
    ncem_pr_paths = snakemake.input.ncem_pr_paths  # noqa: F821
    nichede_pr_paths = snakemake.input.nichede_pr_paths  # noqa: F821

    # Read the PR curves for NCEM and NicheDE along with the niche sizes
    ncem_prs = []
    ncem_model_names = []
    for ncem_pr_path in ncem_pr_paths:
        ncem_prs.append(pd.read_csv(ncem_pr_path))
        niche_size = ncem_pr_path.split("_")[1]
        ncem_model_names.append(f"NCEM_{niche_size}")

    nichede_prs = []
    nichede_model_names = []
    for nichede_pr_path in nichede_pr_paths:
        nichede_prs.append(pd.read_csv(nichede_pr_path))
        niche_size = nichede_pr_path.split("_")[1]
        nichede_model_names.append(f"NicheDE_{niche_size}")

    # Plot the PR curves for all 4 models
    plot_pr_curves(
        [amici_pr, gitiii_pr] + ncem_prs + nichede_prs,
        ["AMICI", "GITIII"] + ncem_model_names + nichede_model_names,
        num_positive_classes=num_positive_classes_interactions,
        save_dir=os.path.dirname(snakemake.output[0]),  # noqa: F821
        suffix="gene_task",
    )

    done_path = snakemake.output[1]  # noqa: F821
    with open(done_path, "w") as _:
        os.utime(done_path, None)


if __name__ == "__main__":
    main()
