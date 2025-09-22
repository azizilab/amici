import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the NCEM model."""
    all_ncem_scores = pd.read_csv(snakemake.input.ncem_scores_path)  # noqa: F821

    # Ensure gene, sender and receiver columns are strings
    all_ncem_scores["gene"] = all_ncem_scores["gene"].astype(str)
    all_ncem_scores["sender"] = all_ncem_scores["sender"].astype(str)
    all_ncem_scores["receiver"] = all_ncem_scores["receiver"].astype(str)
    all_ncem_scores["log_fold_change"] = all_ncem_scores["log_fold_change"].astype(float)
    all_ncem_scores["p_value_adj"] = all_ncem_scores["p_value_adj"].astype(float)

    all_gt_gene_scores = pd.read_csv(snakemake.input.gt_gene_scores_path)  # noqa: F821

    # Label the scores in NCEM by interaction and filter by the two interactions
    interactions = snakemake.config["datasets"][snakemake.wildcards.dataset]["gt_interactions"]  # noqa: F821
    all_ncem_scores["interaction"] = "none"
    for interaction_name in interactions:
        interaction_config = interactions[interaction_name]

        sender_type = interaction_config["sender"]
        receiver_type = interaction_config["receiver"]
        all_ncem_scores.loc[
            (all_ncem_scores["sender"] == sender_type) & (all_ncem_scores["receiver"] == receiver_type), "interaction"
        ] = interaction_name

    ncem_scores = all_ncem_scores[all_ncem_scores["interaction"].isin(list(interactions.keys()))]

    ncem_scores.fillna({"p_value_adj": 1}, inplace=True)

    # Calculate positive scores
    ncem_scores["ncem_scores"] = ncem_scores["log_fold_change"].copy()
    ncem_scores.loc[ncem_scores["p_value_adj"] >= 0.05, "ncem_scores"] = 0

    ncem_precision, ncem_recall, ncem_avg_precision = get_model_precision_recall_auc(
        ncem_scores,
        all_gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="ncem_scores",
        gt_class_col="class",
    )

    ncem_pr = pd.DataFrame(
        {
            "precision": ncem_precision,
            "recall": ncem_recall,
            "avg_precision_score": ncem_avg_precision,
        }
    )
    ncem_pr.to_csv(snakemake.output[0], index=False)  # noqa: F821


if __name__ == "__main__":
    main()
