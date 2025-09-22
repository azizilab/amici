import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the CGCom model for the receiver subtype task."""
    all_cgcom_scores = pd.read_csv(snakemake.input.cgcom_scores_path)  # noqa: F821
    all_gt_receiver_subtypes = pd.read_csv(snakemake.input.gt_receiver_subtypes_path)  # noqa: F821

    cgcom_precision, cgcom_recall, cgcom_avg_precision = get_model_precision_recall_auc(
        all_cgcom_scores,
        all_gt_receiver_subtypes,
        ["cell_idx"],
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
