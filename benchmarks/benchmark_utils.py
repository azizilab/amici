from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from einops import repeat
from scipy import stats
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve


def get_receiver_gt_ranked_genes(
    adata,
    receiver_type,
    interaction_subtype,
    neutral_subtype,
    subtype_key,
    lfc_threshold=0.2,
    pval_threshold=0.05,
    adjusted_pvals=True,
):
    """
    Get the ground truth ranked genes for an interaction between a sender and receiver cell type.

    Args:
        adata: AnnData object
        sender_type: str, sender cell type
        receiver_type: str, receiver cell type
        receiver_subtype: str, receiver subtype
        subtype_key: str, key in adata.obs that indicates the subtype of the cell
        lfc_threshold: float, log fold change threshold
        pval_threshold: float, p-value threshold
        adjusted_pvals: bool, whether to use adjusted p-values

    Returns
    -------
        gt_ranked_genes_df: pd.DataFrame, ground truth ranked genes
        - gene: gene names
        - log_fold_change: log fold changes
        - p_value: p-values
        - class: 1 if the gene is a ground truth ranked gene, 0 otherwise
    """
    adata_de = adata[adata.obs[subtype_key].isin([neutral_subtype, interaction_subtype])]
    sc.tl.rank_genes_groups(
        adata_de,
        groupby=subtype_key,
        method="t-test",
    )

    gene_names = adata_de.uns["rank_genes_groups"]["names"][interaction_subtype]
    log_fold_changes = adata_de.uns["rank_genes_groups"]["logfoldchanges"][interaction_subtype]
    if adjusted_pvals:
        p_values = adata_de.uns["rank_genes_groups"]["pvals_adj"][interaction_subtype]
    else:
        p_values = adata_de.uns["rank_genes_groups"]["pvals"][interaction_subtype]

    gt_ranked_genes_df = pd.DataFrame(
        {
            "gene": gene_names,
            "log_fold_change": log_fold_changes,
            "p_value": p_values,
        }
    )
    gt_ranked_genes_df["gene"] = gt_ranked_genes_df["gene"].astype(str)

    filtered_genes = gt_ranked_genes_df[
        (gt_ranked_genes_df["log_fold_change"] > lfc_threshold) & (gt_ranked_genes_df["p_value"] <= pval_threshold)
    ]

    gt_ranked_genes_df.loc[gt_ranked_genes_df["gene"].isin(filtered_genes["gene"].values), "class"] = 1
    gt_ranked_genes_df.loc[~gt_ranked_genes_df["gene"].isin(filtered_genes["gene"].values), "class"] = 0

    return gt_ranked_genes_df


def get_interaction_gt_neighbor_classes(
    adata,
    interactions_config,
    labels_key,
):
    """
    Get the ground truth neighbor classes for an interaction between a sender and receiver cell type, including the classes for all neighbors.

    Args:
        adata: AnnData object
        interaction_config: dict, interaction configuration
        labels_key: str, key in adata.obs that indicates the subtype of the cell

    Returns
    -------
        gt_neighbor_classes_df: pd.DataFrame, ground truth neighbor classes
        - cell_idx: cell indices
        - neighbor_idx: neighbor indices
        - class: ground truth neighbor class for the given interaction
    """
    nn_dists = np.array(adata.obsm["_nn_dist"])  # batch x n_neighbors
    nn_idxs = np.array(adata.obsm["_nn_idx"])  # batch x n_neighbors
    obs_idxs = np.arange(len(adata))  # batch
    obs_idxs = repeat(obs_idxs, "b -> b n", n=nn_dists.shape[1])  # batch x n_neighbors

    combined_gt_positive_mask = np.zeros(nn_dists.shape, dtype=int)

    for interaction_name in interactions_config:
        interaction_config = interactions_config[interaction_name]
        sender_type = interaction_config["sender"]
        receiver_type = interaction_config["receiver"]
        length_scale = interaction_config["length_scale"]

        obs_labels = adata.obs[labels_key].values  # batch x 1
        obs_labels = repeat(np.array(obs_labels), "b -> b n", n=nn_dists.shape[1])  # batch x n_neighbors
        nn_labels = adata.obs[labels_key].values[nn_idxs]  # batch x n_neighbors

        # If the sender matches the label and the distance is less than the length scale, the neighbor is a ground truth positive
        gt_positive_mask = (
            (nn_dists <= length_scale)
            & (nn_labels == sender_type).astype(int)
            & (obs_labels == receiver_type).astype(int)
        )

        combined_gt_positive_mask = combined_gt_positive_mask | gt_positive_mask

    assert nn_idxs.shape == obs_idxs.shape == combined_gt_positive_mask.shape

    gt_neighbor_classes_df = pd.DataFrame(
        {
            "cell_idx": adata.obs_names[obs_idxs.flatten()],
            "neighbor_idx": adata.obs_names[nn_idxs.flatten()],
            "class": combined_gt_positive_mask.flatten(),
        }
    )
    return gt_neighbor_classes_df


def get_model_precision_recall_auc(
    model_gene_scores_df,
    gt_ranked_genes_df,
    merge_cols,
    scores_col="amici_scores",
    gt_class_col="class",
):
    """
    Get the PR metrics based on the scores provided for a specific model.

    Args:
        model_gene_scores_df: pd.DataFrame, model scores
        gt_ranked_genes_df: pd.DataFrame, ground truth ranked genes
        merge_cols: list, columns to merge on
        scores_col: str, column name of the scores in the model_gene_scores_df
        gt_class_col: str, column name of the ground truth class in the gt_ranked_genes_df

    Returns
    -------
        precision: float, precision
        recall: float, recall
        avg_precision_score: float, average precision score
    """
    merged_df = pd.merge(
        gt_ranked_genes_df,
        model_gene_scores_df,
        on=merge_cols,
        how="inner",
    )

    gt_gene_classes = merged_df[gt_class_col]
    model_gene_scores = merged_df[scores_col]
    precision, recall, _ = precision_recall_curve(gt_gene_classes, model_gene_scores)
    avg_precision_score = average_precision_score(gt_gene_classes, model_gene_scores)

    return precision, recall, avg_precision_score


def plot_pr_curves(
    pr_score_dfs,
    model_names,
    num_positive_classes=None,
    save_svg=False,
    save_png=True,
    save_dir="./figures",
    suffix=None,
):
    """
    Plot the PR curve for a given set of models.

    Args:
        pr_score_dfs: list, list of dataframes with precision, recall and avg precision score columns
        model_names: list, model names
        num_positive_classes: list, number of positive classes for each interaction
        save_svg: bool, whether to save the svg file
        save_png: bool, whether to save the png file
        save_dir: str, directory to save the figures
        suffix: str, suffix to add to the file name
    """
    _, ax_to_plot = plt.subplots(figsize=(8, 8))
    displays = []
    for pr_score_df in pr_score_dfs:
        display = PrecisionRecallDisplay(
            recall=pr_score_df["recall"].values,
            precision=pr_score_df["precision"].values,
        )
        displays.append(display)

    avg_precision_scores = [pr_score_df["avg_precision_score"].values[0] for pr_score_df in pr_score_dfs]
    # Label the PR curve with the AUC score and add DEG count at the top
    loc = 0.8
    # Add DEG count first
    if num_positive_classes:
        if len(num_positive_classes) > 1:
            for i, num_positive_classes_interaction in enumerate(num_positive_classes):
                plt.text(
                    0.2,
                    loc,
                    f"n={num_positive_classes_interaction} positive classes interaction {i + 1}",
                    fontsize=12,
                    fontweight="bold",
                )
                loc -= 0.05
        else:
            plt.text(
                0.2,
                loc,
                f"n={num_positive_classes[0]} positive classes",
                fontsize=12,
                fontweight="bold",
            )
            loc -= 0.05

    # Then add AURPC scores
    for _, avg_precision_score, model_name in zip(displays, avg_precision_scores, model_names):
        plt.text(0.2, loc, f"{model_name} AUPRC = {avg_precision_score:.2f}", fontsize=12)
        loc -= 0.05

    # Show the plot and save it
    for display, model_name in zip(displays, model_names):
        display.plot(ax=ax_to_plot, name=model_name)

    # Create a wrapped title using textwrap
    title_text = "PR Curve"
    wrapped_title = fill(title_text, width=50)

    plt.title(wrapped_title)
    plt.legend()
    save_path = f"{save_dir}/baseline_pr_curves_{suffix}.png" if suffix else f"{save_dir}/baseline_pr_curves.png"
    if save_png:
        plt.savefig(save_path)
    if save_svg:
        plt.savefig(save_path.replace(".png", ".svg"))
    plt.show()
    plt.close()


def plot_boxplots(
    metrics,
    model_names,
    metric_name,
    save_svg=False,
    save_png=True,
    save_dir="./figures",
    title_task=None,
    suffix=None,
):
    """
    Plot the boxplot for a given set of models containing the metrics for each model on multiple seeds of a given dataset.

    The boxplots are connected across models to show the relationship between different models' performance on the same data points.
    Includes significance test indicators between AMICI and other models using Mann-Whitney U test.

    Args:
        metrics: list[list], list of metrics for each model
        model_names: list, model names
        metric_name: str, metric name
        save_svg: bool, whether to save the svg file
        save_png: bool, whether to save the png file
        save_dir: str, directory to save the figures
        title_task: str, name of the task to add to the title
        suffix: str, suffix to add to the file name
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(len(model_names) * 1.5, 6))

    # Create positions for the boxplots
    positions = range(1, len(model_names) + 1)

    # Plot boxplots
    boxplots = ax.boxplot(metrics, positions=positions, labels=model_names, patch_artist=True)

    # Add connecting lines between corresponding points
    for i in range(len(metrics[0])):
        # Get the y-values for this point across all models
        y_values = [model_metrics[i] for model_metrics in metrics]
        # Plot connecting lines with markers
        ax.plot(
            positions,
            y_values,
            "k-",
            alpha=0.2,
            linewidth=0.5,
            marker="o",
            markersize=3,
            markeredgecolor="black",
            markerfacecolor="black",
        )

    # Customize boxplot colors
    colors = ["lightblue", "lightgreen", "pink", "lightyellow"]
    for patch, color in zip(boxplots["boxes"], colors):
        patch.set_facecolor(color)

    # Perform Mann-Whitney U test between AMICI and other models
    amici_idx = model_names.index("AMICI")
    amici_metrics = metrics[amici_idx]

    # Calculate y-axis limits for star placement
    y_min, y_max = ax.get_ylim()
    star_offset = (y_max - y_min) * 0.05  # 5% of the plot height
    new_y_max = y_max

    for i, (model_metrics, model_name) in enumerate(zip(metrics, model_names)):
        if model_name != "AMICI":
            # Perform Mann-Whitney U test
            statistic, pval = stats.mannwhitneyu(amici_metrics, model_metrics, alternative="two-sided")

            # Adjust p-value for multiple comparisons
            adjusted_pval = pval * (len(model_names) - 1)  # Bonferroni correction

            # Determine significance stars
            if adjusted_pval < 0.001:
                stars = "****"
            elif adjusted_pval < 0.005:
                stars = "***"
            elif adjusted_pval < 0.01:
                stars = "**"
            elif adjusted_pval < 0.05:
                stars = "*"
            else:
                stars = ""

            # Place stars and p-values above the boxplot for this model, only if significant
            if stars:
                box_max = max(model_metrics)
                y_star = box_max + star_offset
                # Format p-value for display
                if adjusted_pval < 0.001:
                    pval_text = f"p={adjusted_pval:.4f}"
                elif adjusted_pval < 0.01:
                    pval_text = f"p={adjusted_pval:.4f}"
                else:
                    pval_text = f"p={adjusted_pval:.4f}"

                # Display stars and p-value
                text_content = f"{stars}\n{pval_text}"
                ax.text(positions[i], y_star, text_content, ha="center", va="bottom", fontsize=12, fontweight="bold")
                if y_star > new_y_max:
                    new_y_max = y_star

    # Create a wrapped title using textwrap
    title_text = f"Task: {title_task if title_task else ''} - Boxplot of {metric_name} for {', '.join(model_names)}"
    wrapped_title = fill(title_text, width=50)

    plt.title(wrapped_title)
    plt.ylabel(metric_name.replace("_", " ").upper())
    plt.ylim(0, new_y_max + star_offset * 1.5)  # Adjust y-axis to accommodate stars
    plt.grid(True, alpha=0.3)

    save_path = (
        f"{save_dir}/baseline_{metric_name}_boxplots_{suffix}.png"
        if suffix
        else f"{save_dir}/baseline_{metric_name}_boxplots.png"
    )
    if save_png:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_svg:
        plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.show()
    plt.close()
