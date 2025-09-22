import numpy as np
import pandas as pd


def get_amici_gene_task_scores(
    model,
    adata,
    sender_type,
    receiver_type,
):
    """
    Get the AMICI scores for predicting significant genes task between a sender and receiver cell type.

    Args:
        model: AMICI model
        adata: AnnData object
        sender_type: str, sender cell type
        receiver_type: str, receiver cell type
        signed: bool, whether to return signed scores

    Returns
    -------
        amici_gene_scores_df: pd.DataFrame, AMICI scores
        - gene: gene names
        - sender: sender cell type
        - receiver: receiver cell type
        - amici_scores: AMICI scores
    """
    # Get the neighbor ablation scores using the head index with the highest explained variance
    neighbor_ablation_scores = model.get_neighbor_ablation_scores(
        cell_type=receiver_type,
        adata=adata,
        ablated_neighbor_ct_sub=[sender_type],
        compute_z_value=True,
    )

    # Get the ablation scores dataframe and keep only ablation columns
    scores_df = neighbor_ablation_scores._ablation_scores_df

    z_value_cols = [col for col in scores_df.columns if col.endswith("_z_value")]
    diff_cols = [col for col in scores_df.columns if col.endswith("_diff")]
    scores_df = scores_df[diff_cols + z_value_cols + ["gene"]]

    # Get sender cell types from column names (removing '_diff' suffix)
    sender_types = np.unique([col.replace("_diff", "").replace("_z_value", "") for col in scores_df.columns])
    sender_types = [sender for sender in sender_types if sender != "gene"]

    # For each sender type, create a row for each gene
    all_ablation_scores = []
    for sender in sender_types:
        amici_scores = scores_df[f"{sender}_z_value"].copy()
        amici_scores[scores_df[f"{sender}_diff"] < 0] = 0
        temp_df = pd.DataFrame(
            {
                "gene": scores_df["gene"],
                "receiver": receiver_type,
                "sender": sender,
                "amici_scores": amici_scores,
            }
        )
        temp_df.fillna({"amici_scores": 0}, inplace=True)

        all_ablation_scores.append(temp_df)

    # Combine all scores into a single dataframe
    amici_gene_scores_df = pd.concat(all_ablation_scores, ignore_index=True)

    return amici_gene_scores_df


def get_amici_neighbor_interaction_scores(
    model,
    adata,
):
    """
    Get the AMICI scores for predicting the interacting neighbor task for all sender-receiver interactions.

    Args:
        model: AMICI model
        adata: AnnData object

    Returns
    -------
        amici_neighbor_interaction_scores_df: pd.DataFrame, AMICI scores
    """
    attention_patterns = model.get_attention_patterns(
        adata=adata,
    )

    attention_patterns_df = attention_patterns._attention_patterns_df.drop(columns=["label", "head"])
    max_attention_scores = attention_patterns_df.groupby("cell_idx").max().reset_index()
    nn_obs_names = attention_patterns._nn_idxs_df
    nn_obs_names["cell_idx"] = adata.obs_names
    melted_attention_scores = pd.melt(
        max_attention_scores,
        id_vars="cell_idx",
        value_vars=[f"neighbor_{i}" for i in range(model.n_neighbors)],
        var_name="neighbor_col",
        value_name="amici_scores",
    )
    melted_nn_obs_names = pd.melt(
        nn_obs_names,
        id_vars="cell_idx",
        value_vars=[f"neighbor_{i}" for i in range(model.n_neighbors)],
        var_name="neighbor_col",
        value_name="neighbor_idx",
    )

    merged_attention_scores = pd.merge(
        melted_attention_scores, melted_nn_obs_names, on=["neighbor_col", "cell_idx"], how="inner"
    ).drop(columns=["neighbor_col"])

    return merged_attention_scores


def get_amici_receiver_subtype_scores(
    model,
    adata,
):
    """
    Get the AMICI scores for predicting the interacting receiver subtype task.

    Args:
        model: AMICI model
        adata: AnnData object

    Returns
    -------
        attention_patterns_df: pd.DataFrame, AMICI scores
    """
    attention_patterns = model.get_attention_patterns(
        adata=adata,
    )

    attention_patterns_df = attention_patterns._attention_patterns_df
    attention_patterns_df = attention_patterns_df.drop(columns=["head"])
    attention_patterns_df = attention_patterns_df.groupby(["label", "cell_idx"]).max().reset_index()
    attention_patterns_df["amici_scores"] = attention_patterns_df[
        [f"neighbor_{i}" for i in range(model.n_neighbors)]
    ].max(axis=1)
    attention_patterns_df = attention_patterns_df.drop(columns=[f"neighbor_{i}" for i in range(model.n_neighbors)])

    return attention_patterns_df
