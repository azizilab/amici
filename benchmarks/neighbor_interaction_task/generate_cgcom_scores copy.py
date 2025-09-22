import scanpy as sc
import torch
from cgcom.scripts import (
    build_dataloaders,
    build_graph_from_spatial_data,
    generate_subgraph_features,
    get_cell_communication_scores,
)
from cgcom.utils import convert_anndata_to_df, get_cell_label_dict, get_exp_params


def main():
    """Main function to generate the CGCom scores for the neighbor interaction task."""
    model_path = snakemake.input.model_path  # noqa: F821
    adata_path = snakemake.input.adata_path  # noqa: F821
    output_path = snakemake.output[0]  # noqa: F821
    labels_key = snakemake.config["datasets"][snakemake.wildcards.dataset]["labels_key"]  # noqa: F821

    # Load the dataset
    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()

    cgcom_model = torch.load(model_path)

    exp_params = get_exp_params()

    # Get the cell-cell communication scores from the loaded CGCom model
    expression_df = convert_anndata_to_df(adata)
    cell_label_dict = get_cell_label_dict(adata, labels_key)
    G, edge_list, node_id_list = build_graph_from_spatial_data(adata, exp_params)
    filtered_features, filtered_edges, filtered_labels, filtered_original_node_ids = generate_subgraph_features(
        G, node_id_list, cell_label_dict, expression_df
    )
    train_loader, validate_loader, test_loader, total_loader, num_classes = build_dataloaders(
        filtered_edges, filtered_features, filtered_labels, exp_params
    )
    attention_scores_df, attention_summary_df = get_cell_communication_scores(
        cgcom_model, total_loader, node_id_list, filtered_original_node_ids, torch.device("cuda")
    )

    # Save the attention scores to the output path
    attention_scores_df = attention_scores_df[["center_cell", "neighbor_cell", "attention_score"]]
    attention_scores_df.rename(
        columns={"center_cell": "cell_idx", "neighbor_cell": "neighbor_idx", "attention_score": "cgcom_scores"},
        inplace=True,
    )
    attention_scores_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
