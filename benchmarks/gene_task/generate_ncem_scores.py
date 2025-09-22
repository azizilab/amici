import os

import numpy as np
import pandas as pd
import scanpy as sc
from ncem_benchmark_utils import get_model_parameters, load_ncem_from_weights


def main():
    """Generate the NCEM scores for the gene task."""
    try:
        adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821
        adata.obs_names_make_unique()

        if "spatial" not in adata.uns:
            adata.uns["spatial"] = adata.obsm["spatial"].copy()

        niche_size = int(snakemake.wildcards.ncem_niche_size)  # noqa: F821

        dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
        labels_key = dataset_config["labels_key"]

        model_dir = f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/saved_models"  # noqa: F821
        model_path = os.path.join(
            model_dir,
            f"ncem_{niche_size}_checkpoint_{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}",  # noqa: F821
        )
        model_args_path = os.path.join(
            model_dir,
            f"ncem_{niche_size}_{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}.pickle",  # noqa: F821
        )

        exp_params, _, _ = get_model_parameters(niche_size)

        # Load the model from the saved weights
        interpreter = load_ncem_from_weights(
            adata,
            labels_key,
            exp_params,
            model_path,
            model_args_path,
        )

        # Run the interaction analysis
        interpreter.n_eval_nodes_per_graph = 1
        interpreter.get_sender_receiver_effects(params_type="ols")
        interpreter.cv_idx = 0  # TODO: hack since this is not initialized in the model

        save_dir = f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/figures"  # noqa: F821
        os.makedirs(save_dir, exist_ok=True)

        # Run the type coupling analysis
        interpreter.type_coupling_analysis_circular(
            edge_attr="magnitude",
            figsize=(10, 8),
            text_space=1.28,
            de_genes_threshold=0,
            save=save_dir,
            suffix="type_coupling_analysis_circular.png",
        )

        # Extract the fold changes and p-values
        log_fold_changes = interpreter.fold_change
        p_values = interpreter.pvalues
        q_values = interpreter.qvalues

        genes = list(adata.var_names)
        _, _ = np.meshgrid(
            np.arange(len(interpreter.cell_names)), np.arange(len(interpreter.cell_names)), indexing="ij"
        )

        # Create a list to store all rows
        ncem_scores_rows = []
        for receiver_idx in range(len(interpreter.cell_names)):
            for sender_idx in range(len(interpreter.cell_names)):
                for gene_idx, gene in enumerate(genes):
                    ncem_scores_rows.append(
                        {
                            "sender": interpreter.cell_names[sender_idx],
                            "receiver": interpreter.cell_names[receiver_idx],
                            "gene": gene,
                            "log_fold_change": log_fold_changes[receiver_idx, sender_idx, gene_idx],
                            "p_value": p_values[receiver_idx, sender_idx, gene_idx],
                            "p_value_adj": q_values[receiver_idx, sender_idx, gene_idx],
                        }
                    )

        # Create DataFrame from the list of rows
        ncem_scores_df = pd.DataFrame(ncem_scores_rows)

        ncem_scores_path = snakemake.output[0]  # noqa: F821
        with open(ncem_scores_path, "w") as f:
            ncem_scores_df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        print("Scores generated successfully")

    except Exception as e:
        print(f"Error during score generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
