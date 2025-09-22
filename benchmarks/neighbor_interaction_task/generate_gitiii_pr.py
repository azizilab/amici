import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the GITIII model."""
    all_gitiii_scores = pd.read_csv(snakemake.input.gitiii_scores_path)  # noqa: F821
    all_gt_neighbor_interaction_scores = pd.read_csv(snakemake.input.gt_neighbor_interactions_path)  # noqa: F821

    # Only keep the scores for the rows where the cell_idx and neighbor_idx pair is present in gitiii_scores
    gitiii_intersection_scores = all_gitiii_scores.merge(
        all_gt_neighbor_interaction_scores.drop(columns=["class"]),
        on=["cell_idx", "neighbor_idx"],
        suffixes=(None, "_gt"),
        how="inner",
    )
    gt_neighbor_interaction_intersection_scores = all_gt_neighbor_interaction_scores.merge(
        all_gitiii_scores.drop(columns=["gitiii_scores"]),
        on=["cell_idx", "neighbor_idx"],
        suffixes=(None, "_gitiii"),
        how="inner",
    )
    gitiii_intersection_scores.fillna(0, inplace=True)
    gt_neighbor_interaction_intersection_scores.fillna(0, inplace=True)

    gitiii_precision, gitiii_recall, gitiii_avg_precision = get_model_precision_recall_auc(
        gitiii_intersection_scores,
        gt_neighbor_interaction_intersection_scores,
        ["cell_idx", "neighbor_idx"],
        "gitiii_scores",
        "class",
    )

    # Save the precision and recall for the GITIII model
    gitiii_pr = pd.DataFrame(
        {
            "precision": gitiii_precision,
            "recall": gitiii_recall,
            "avg_precision_score": gitiii_avg_precision,
        }
    )
    gitiii_pr.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
