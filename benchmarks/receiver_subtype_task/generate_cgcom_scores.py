import scanpy as sc
import torch
from cgcom.models import GATGraphClassifier
from cgcom.scripts import (
    build_dataloaders,
    build_graph_from_spatial_data,
    generate_subgraph_features,
    get_cell_communication_scores,
)
from cgcom.utils import convert_anndata_to_df, get_cell_label_dict, get_exp_params, get_model_params


def main():
    """Main function to generate the CGCom scores for the receiver subtype task."""
    model_path = snakemake.input.model_path  # noqa: F821
    adata_path = snakemake.input.adata_path  # noqa: F821
    output_path = snakemake.output[0]  # noqa: F821
    labels_key = snakemake.config["datasets"][snakemake.wildcards.dataset]["labels_key"]  # noqa: F821

    # Load the dataset
    adata = sc.read_h5ad(adata_path)
    adata.obs_names_make_unique()

    model_params = get_model_params(
        fc_hidden_channels_2=500,
        fc_hidden_channels_3=512,
        fc_hidden_channels_4=64,
        num_classes=10,
        device=torch.device("cuda"),
        ligand_channel=500,
        receptor_channel=500,
        TF_channel=500,
        mask_indexes=None,
        disable_lr_masking=True,
    )
    exp_params = get_exp_params()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cgcom_model = GATGraphClassifier(
        FChidden_channels_2=model_params["fc_hidden_channels_2"],
        FChidden_channels_3=model_params["fc_hidden_channels_3"],
        FChidden_channels_4=model_params["fc_hidden_channels_4"],
        num_classes=model_params["num_classes"],
        device=torch.device("cuda"),
        ligand_channel=model_params["ligand_channel"],
        receptor_channel=model_params["receptor_channel"],
        TF_channel=model_params["TF_channel"],
        mask_indexes=model_params["mask_indexes"],
        disable_lr_masking=True,
    ).to(device)

    cgcom_model.load_state_dict(torch.load(model_path))
    cgcom_model.eval()
    cgcom_model.to(torch.device("cuda"))

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
        cgcom_model, total_loader, node_id_list, filtered_original_node_ids, device
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
