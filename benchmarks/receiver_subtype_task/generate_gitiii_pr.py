import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the GITIII model."""
    all_gitiii_scores = pd.read_csv(snakemake.input.gitiii_scores_path)  # noqa: F821
    all_gt_receiver_subtypes = pd.read_csv(snakemake.input.gt_receiver_subtypes_path)  # noqa: F821

    gitiii_precision, gitiii_recall, gitiii_avg_precision = get_model_precision_recall_auc(
        all_gitiii_scores,
        all_gt_receiver_subtypes,
        ["cell_idx"],
        "gitiii_scores",
        "class",
    )

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
