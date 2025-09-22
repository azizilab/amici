import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the AMICI model."""
    all_amici_scores = pd.read_csv(snakemake.input.amici_scores_path)  # noqa: F821
    all_gt_receiver_subtypes = pd.read_csv(snakemake.input.gt_receiver_subtypes_path)  # noqa: F821

    amici_precision, amici_recall, amici_avg_precision = get_model_precision_recall_auc(
        all_amici_scores,
        all_gt_receiver_subtypes,
        ["cell_idx"],
        "amici_scores",
        "class",
    )

    # Save the precision and recall for the AMICI model
    amici_pr = pd.DataFrame(
        {
            "precision": amici_precision,
            "recall": amici_recall,
            "avg_precision_score": amici_avg_precision,
        }
    )
    amici_pr.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
