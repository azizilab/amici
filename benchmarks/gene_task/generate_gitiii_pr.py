import pandas as pd
from benchmark_utils import get_model_precision_recall_auc


def main():
    """Generate the precision and recall for the GITIII model."""
    all_gitiii_scores = pd.read_csv(snakemake.input.gitiii_scores_path)  # noqa: F821

    # Ensure gene, sender and receiver columns are strings
    all_gitiii_scores["gene"] = all_gitiii_scores["gene"].astype(str)
    all_gitiii_scores["sender"] = all_gitiii_scores["sender"].astype(str)
    all_gitiii_scores["receiver"] = all_gitiii_scores["receiver"].astype(str)

    all_gitiii_scores["z_score"] = all_gitiii_scores["z_score"].astype(float)
    all_gitiii_scores["p_value_adj"] = all_gitiii_scores["p_value_adj"].astype(float)

    all_gt_gene_scores = pd.read_csv(snakemake.input.gt_gene_scores_path)  # noqa: F821

    # Label the scores in GITIII by interaction and filter by the two interactions
    interactions = snakemake.config["datasets"][snakemake.wildcards.dataset]["gt_interactions"]  # noqa: F821
    all_gitiii_scores["interaction"] = "none"
    for interaction_name in interactions:
        interaction_config = interactions[interaction_name]

        sender_type = interaction_config["sender"]
        receiver_type = interaction_config["receiver"]
        all_gitiii_scores.loc[
            (all_gitiii_scores["sender"] == sender_type) & (all_gitiii_scores["receiver"] == receiver_type),
            "interaction",
        ] = interaction_name

    gitiii_scores = all_gitiii_scores[all_gitiii_scores["interaction"].isin(list(interactions.keys()))]

    gitiii_scores.fillna({"p_value_adj": 1}, inplace=True)

    # Zero out z-scores where p-values are not significant
    gitiii_scores.loc[gitiii_scores["p_value_adj"] >= 0.05, "z_score"] = 0

    # Calculate the PR curve
    gitiii_scores["gitiii_scores"] = gitiii_scores["z_score"].copy()

    gitiii_precision, gitiii_recall, gitiii_avg_precision = get_model_precision_recall_auc(
        gitiii_scores,
        all_gt_gene_scores,
        merge_cols=["gene", "interaction"],
        scores_col="gitiii_scores",
        gt_class_col="class",
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
