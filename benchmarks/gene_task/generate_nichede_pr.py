import numpy as np
import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the NicheDE model."""
    all_nichede_scores = pd.read_csv(snakemake.input.nichede_scores_path)  # noqa: F821
    # Ensure gene, sender and receiver columns are strings
    all_nichede_scores["gene"] = all_nichede_scores["gene"].astype(str)
    all_nichede_scores["sender"] = all_nichede_scores["sender"].astype(str)
    all_nichede_scores["receiver"] = all_nichede_scores["receiver"].astype(str)

    all_gt_gene_scores = pd.read_csv(snakemake.input.gt_gene_scores_path)  # noqa: F821

    # Filter the nichede scores to only include the sender and receiver cell types of interest
    interactions = snakemake.config["datasets"][snakemake.wildcards.dataset]["gt_interactions"]  # noqa: F821
    all_nichede_scores["interaction"] = "none"
    for interaction_name in interactions:
        interaction_config = interactions[interaction_name]

        sender_type = interaction_config["sender"]
        receiver_type = interaction_config["receiver"]
        all_nichede_scores.loc[
            (all_nichede_scores["sender"] == sender_type) & (all_nichede_scores["receiver"] == receiver_type),
            "interaction",
        ] = interaction_name

    nichede_scores = all_nichede_scores[all_nichede_scores["interaction"].isin(list(interactions.keys()))]

    # Reverse the p values for the nichede scores
    nichede_scores["p_value_adj"].fillna(1, inplace=True)
    nichede_scores["p_value_adj"] = -np.log10(nichede_scores["p_value_adj"] + 1e-10)

    # Get the PR metrics for the NicheDE model
    nichede_precision, nichede_recall, nichede_avg_precision = get_model_precision_recall_auc(
        nichede_scores,
        all_gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="p_value_adj",
        gt_class_col="class",
    )

    nichede_pr = pd.DataFrame(
        {
            "precision": nichede_precision,
            "recall": nichede_recall,
            "avg_precision_score": nichede_avg_precision,
        }
    )
    nichede_pr.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
