import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the CGCom model for the neighbor interaction task."""
    all_cgcom_scores = pd.read_csv(snakemake.input.cgcom_scores_path)  # noqa: F821
    all_gt_neighbor_interaction_scores = pd.read_csv(snakemake.input.gt_neighbor_interactions_path)  # noqa: F821

    # Only keep the scores for the rows where the cell_idx and neighbor_idx pair is present in cgcom_scores
    cgcom_intersection_scores = all_cgcom_scores.merge(
        all_gt_neighbor_interaction_scores.drop(columns=["class"]),
        on=["cell_idx", "neighbor_idx"],
        suffixes=(None, "_gt"),
        how="inner",
    )

    gt_neighbor_interaction_intersection_scores = all_gt_neighbor_interaction_scores.merge(
        all_cgcom_scores.drop(columns=["cgcom_scores"]),
        on=["cell_idx", "neighbor_idx"],
        suffixes=(None, "_cgcom"),
        how="inner",
    )

    cgcom_intersection_scores.fillna(0, inplace=True)
    gt_neighbor_interaction_intersection_scores.fillna(0, inplace=True)

    cgcom_precision, cgcom_recall, cgcom_avg_precision = get_model_precision_recall_auc(
        cgcom_intersection_scores,
        gt_neighbor_interaction_intersection_scores,
        ["cell_idx", "neighbor_idx"],
        "cgcom_scores",
        "class",
    )

    cgcom_pr = pd.DataFrame(
        {
            "precision": cgcom_precision,
            "recall": cgcom_recall,
            "avg_precision_score": cgcom_avg_precision,
        }
    )
    cgcom_pr.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
