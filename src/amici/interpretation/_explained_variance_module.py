import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from anndata import AnnData
from scipy.stats import false_discovery_control
from scvi import REGISTRY_KEYS

from ._utils import _get_compute_method_kwargs

if TYPE_CHECKING:
    from amici._model import AMICI


@dataclass
class AMICIExplainedVarianceModule:
    _adata: AnnData | None = None
    _explained_variance_df: pd.DataFrame | None = None
    _compute_kwargs: dict | None = None
    _n_permutations: int | None = None

    @classmethod
    def compute(
        cls,
        model: "AMICI",
        adata: AnnData | None = None,
        alpha: float = 0.05,
        run_permutation_test: bool = True,
        n_permutations: int | None = None,
    ) -> "AMICIExplainedVarianceModule":
        """Creates a DataFrame with the explained variance scores.

        DataFrame contains the explained variance scores for each head across cells for all cell types or a subset of cell types.

        Args:
            adata (AnnData, optional): The AnnData object.
            alpha (float, optional): The desired significance level for the explained variance of each head-cell type
                pair via permutation testing. Defaults to 0.05.
            run_permutation_test (bool, optional): Whether to run the permutation test. Defaults to True.
            n_permutations (int, optional): The number of permutations to use for the permutation test.
                If not provided, the minimum number of permutations required to achieve a significance level of alpha will be used.

        Returns
        -------
            AMICIExplainedVarianceModule: The module instance with computed explained variance scores stored
            in _explained_variance_df with columns:
                - head: The index of the head.
                - ct_name: The name of the cell type.
                - gene: The name of the gene.
                - expl_var_head_gene: The explained variance score when all but head is ablated for the gene.
                - expl_var_head: The explained variance score when all but head aggregated across all genes.
            if run_permutation_test is True, the following columns are also added:
                - p_value_head: The p-value of the explained variance score when all but head is ablated.
                - p_value_adj_head: The adjusted p-value of the explained variance score when all but head is ablated.
        """
        _compute_kwargs = _get_compute_method_kwargs(**locals())
        model._check_if_trained(warn=True)

        adata = model._validate_anndata(adata)

        labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key

        n_permutations = (
            cls._min_n_permutations(model.module.n_heads, len(adata.obs[labels_key].unique()), alpha)
            if n_permutations is None
            else n_permutations
        )

        expl_variance_scores = []
        p_values = []
        for head_idx in range(model.module.n_heads):
            for ct in list(adata.obs[labels_key].unique()):
                predictions_head, gene_exp = cls._get_ct_predictions_for_ablated_head(
                    model,
                    adata=adata,
                    head_idx=head_idx,
                    cell_type=ct,
                )
                expl_var_head = cls._compute_expl_variance(predictions_head - gene_exp, gene_exp)
                expl_var_head_agg = cls._compute_expl_variance(
                    predictions_head - gene_exp, gene_exp, aggregate_across_genes=True
                )
                if run_permutation_test:
                    p_value_head = cls._run_permutation_test_expl_variance(predictions_head, gene_exp, n_permutations)
                    p_values.append(p_value_head)

                for i, gene in enumerate(adata.var_names):
                    explained_variance_score_row = {
                        "head": head_idx,
                        "ct_name": ct,
                        "gene": gene,
                        "expl_var_head_gene": expl_var_head[i],
                        "expl_var_head": expl_var_head_agg,
                    }
                    if run_permutation_test:
                        explained_variance_score_row["p_value_head"] = p_value_head
                    expl_variance_scores.append(explained_variance_score_row)
        expl_variance_scores_df = pd.DataFrame(expl_variance_scores)

        # adjust p-values via B-H correction
        if run_permutation_test:
            p_values_adj = false_discovery_control(p_values)
            p_value_adj_map = dict(zip(p_values, p_values_adj))
            expl_variance_scores_df["p_value_adj_head"] = expl_variance_scores_df["p_value_head"].map(p_value_adj_map)

        return cls(
            _adata=adata,
            _explained_variance_df=expl_variance_scores_df,
            _compute_kwargs=_compute_kwargs,
            _n_permutations=n_permutations,
        )

    @staticmethod
    def _min_n_permutations(n_heads: int, n_ct: int, alpha: float) -> int:
        """Compute the minimum number of permutations required to achieve a significance level of alpha for the explained variance scores.

        Args:
            n_heads (int): The number of heads.
            n_ct (int): The number of cell types.
            alpha (float): The desired significance level.

        Returns
        -------
            int: The minimum number of permutations required to achieve a significance level of alpha for the explained variance scores.
        """
        return int(np.ceil((n_heads * n_ct) / alpha))

    @staticmethod
    def _compute_expl_variance(
        residuals: pd.DataFrame,
        gene_exp: pd.DataFrame,
        aggregate_across_genes: bool = False,
    ):
        """Compute the explained variance scores for the given cell type residuals and ground truth gene expression for cell type of interest.

        Args:
            residuals (pd.DataFrame): The residuals for the cell type of interest when ablating all heads except one.
            gene_exp (pd.DataFrame): The ground truth gene expression for the cell type of interest.
            aggregate_across_genes (bool, optional): Whether to aggregate the explained variance across all genes.

        Returns
        -------
            np.ndarray: The explained variance scores for the cell type and head of interest.
        """
        if aggregate_across_genes:
            expl_var = (
                1
                - torch.var(torch.tensor(residuals.values), dim=0).sum()
                / torch.var(torch.tensor(gene_exp.values), dim=0).sum()
            ).item()
        else:
            expl_var = (
                1 - torch.var(torch.tensor(residuals.values), dim=0) / torch.var(torch.tensor(gene_exp.values), dim=0)
            ).numpy()
            expl_var = np.where(torch.var(torch.tensor(gene_exp.values), dim=0) == 0, 0, expl_var)
        return expl_var

    @staticmethod
    def _run_permutation_test_expl_variance(
        predictions: pd.DataFrame,  # n_ct_cells x n_genes
        gene_exp: pd.DataFrame,  # n_ct_cells x n_genes
        n_permutations: int,
        batch_size: int = 32,
    ):
        """Run a permutation test to compute the p-value of the explained variance scores.

        Args:
            predictions (torch.Tensor): The predictions tensor of shape n_ct_cells x n_genes.
            gene_exp (torch.Tensor): The ground truth gene expression tensor of shape n_ct_cells x n_genes.
            n_permutations (int): The number of permutations to run.
            batch_size (int, optional): The batch size to use for the permutation test.

        Returns
        -------
            float: The one-sided p-value computed from the permutation test.
        """
        n_ct_cells = predictions.shape[0]

        predictions = predictions.values
        gene_exp = gene_exp.values

        original_mse = np.mean((predictions - gene_exp) ** 2).item()

        lower_mse_count = 0
        for batch_idx in range(0, n_permutations, batch_size):
            batch_end = min(batch_idx + batch_size, n_permutations)
            batch_size_actual = batch_end - batch_idx

            perm_indices = np.stack(
                [np.random.permutation(n_ct_cells) for _ in range(batch_size_actual)]
            )  # batch_size x n_ct_cells

            permuted_preds = predictions[perm_indices]  # batch_size x n_ct_cells x n_genes
            expanded_gene_exp = gene_exp[None, :]  # 1 x n_ct_cells x n_genes

            batch_mses = np.mean((permuted_preds - expanded_gene_exp) ** 2, axis=(1, 2))
            lower_mse_count += np.sum(batch_mses <= original_mse)

        p_value = lower_mse_count / n_permutations
        return p_value

    @staticmethod
    def _get_ct_predictions_for_ablated_head(
        model: "AMICI",
        adata: AnnData,
        head_idx: int = -1,
        cell_type: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gene expression residuals and ground truth gene expression for head_idx and cell_type.

        Args:
            adata (AnnData, optional): The AnnData object.
            head_idx (int, optional): The index of the head to ablate or to not ablate. Defaults to -1.
            cell_type (str, optional): The cell type to compute the residuals for.

        Returns
        -------
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - predictions: A DataFrame of predictions of shape n_ct_cells x n_genes.
                - gene_exp: A DataFrame of ground truth gene expression of shape n_ct_cells x n_genes.
        """

        def _mt_heads_ablation_hook(
            attn_result,
            hook,
            head_idxs_to_ablate,
            neighbor_mask=None,
        ):
            """Hook function to ablate heads.

            Args:
                attn_result (torch.Tensor): The attention result.
                head_idxs_to_ablate (list): The indices of the heads to zero out.
                neighbor_mask (torch.Tensor, optional): The neighbor mask.

            Returns
            -------
                torch.Tensor: The modified attention result with specified heads ablated.
            """
            for head_idx in head_idxs_to_ablate:
                attn_result[:, head_idx, :, :] = 0.0

            if neighbor_mask is None:
                neighbor_mask = torch.ones_like(attn_result)
            else:
                # append dummy neighbor col
                neighbor_mask = torch.cat(
                    [
                        neighbor_mask,
                        torch.ones((neighbor_mask.shape[0], 1)).to(neighbor_mask.device),
                    ],
                    dim=1,
                )
                neighbor_mask = neighbor_mask.unsqueeze(1).unsqueeze(2).expand_as(attn_result)
            attn_result *= neighbor_mask

            return attn_result

        assert cell_type is not None, "cell_type must be specified"

        labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        ct_indices = np.arange(len(adata))[adata.obs[labels_key] == cell_type]
        scdl = model._make_data_loader(adata=adata, indices=ct_indices)

        predictions = []
        gene_expressions = []
        for tensors in scdl:
            true_X = tensors[REGISTRY_KEYS.X_KEY].cpu()
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model.module.reset_hooks()

            head_idxs_to_ablate = [i for i in range(model.module.n_heads) if i != head_idx]

            head_hook_fn = partial(
                _mt_heads_ablation_hook,
                head_idxs_to_ablate=head_idxs_to_ablate,
            )
            prediction = (
                model.module.run_with_hooks(
                    tensors,
                    fwd_hooks=[("attention_layer.hook_pattern", head_hook_fn)],
                )[1]["prediction"]
                .detach()
                .cpu()
            )
            predictions.append(prediction)
            gene_expressions.append(true_X)
        all_predictions = torch.cat(predictions)
        gene_exp = torch.cat(gene_expressions)

        predictions_df = pd.DataFrame(all_predictions.numpy(), columns=adata.var_names)
        gene_exp_df = pd.DataFrame(gene_exp.numpy(), columns=adata.var_names)

        return predictions_df, gene_exp_df

    def save(self, save_path: str):
        """Save explained variance scores to file"""
        self._explained_variance_df.to_csv(save_path)
        return self

    def compute_max_explained_variance_head(
        self,
        cell_type: str,
    ) -> int:
        """
        Compute the head with the maximum explained variance for the given cell type.

        Args:
            cell_type (str): The cell type to compute the maximum explained variance head for.

        Returns
        -------
            int: The head with the maximum explained variance for the given cell type.
        """
        return (
            self._explained_variance_df[self._explained_variance_df["ct_name"] == cell_type]
            .groupby("head")["expl_var_head_gene"]
            .max()
            .idxmax()
        )

    def plot_explained_variance_barplot(
        self,
        palette=None,
        cell_type_sub=None,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots a barplot of the maximum explained variance score for each head and each cell type.

        Args:
            expl_variance_df (pd.DataFrame): A DataFrame containing the explained variance scores for each head and each cell type
                as returned by `model.get_expl_variance_scores()`.
            palette (str, optional): The color palette to use for the barplot. Defaults to None.
            cell_type_sub (list[str], optional): A list of cell types to include in the barplot. Defaults to None.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases. Defaults to True.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save_png (bool, optional): Whether to save the plot as a PNG file. Defaults to False.
            save_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.
            save_dir (str, optional): The directory to save the plot files. Defaults to "./figures".
        """
        if cell_type_sub is not None:
            expl_variance_sub_df = self._explained_variance_df[
                self._explained_variance_df["ct_name"].isin(cell_type_sub)
            ]
        else:
            expl_variance_sub_df = self._explained_variance_df
        max_var_per_head = expl_variance_sub_df.groupby(["head", "ct_name"])["expl_var_head_gene"].max().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=max_var_per_head,
            x="head",
            y="expl_var_head_gene",
            hue="ct_name",
            palette=palette or "tab10",
        )
        plt.xlabel("Head Index")
        plt.ylabel("Max Variance Score")
        plt.title("Maximum Variance Score Across Genes for Each Head Colored by Cell Type")
        plt.legend(title="Cell Type", bbox_to_anchor=(1, 1))

        if wandb_log:
            wandb.log(
                {
                    "Explained Variance Barplot per Head per Cell Type": wandb.Image(plt),
                }
            )
        if save_png:
            plt.savefig(os.path.join(save_dir, "expl_variance_barplot_per_head.png"))
        if save_svg:
            plt.savefig(os.path.join(save_dir, "expl_variance_barplot_per_head.svg"))
        if show:
            plt.show()
        plt.close()

    def plot_featurewise_explained_variance_heatmap(
        self,
        cell_type_sub=None,
        n_top_genes=20,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots a heatmap of the explained variance scores for each head for n_top_genes genes.

        Args:
            expl_variance_df (pd.DataFrame): A DataFrame containing the explained variance scores for each head and each cell type
                as returned by `model.get_expl_variance_scores()"
            cell_type_sub (list[str], optional): A list of cell types for which to plot the heatmap.
            n_top_genes (int, optional): The number of top genes to include in the heatmap.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases.
            show (bool, optional): Whether to display the plot.
            save_png (bool, optional): Whether to save the plot as a PNG file.
            save_svg (bool, optional): Whether to save the plot as an SVG file.
            save_dir (str, optional): The directory to save the plot files. Defaults to "./figures". Saves the filename as
                "expl_variance_heatmap_{cell_type}.png" or "expl_variance_heatmap_{cell_type}.svg".
        """
        if cell_type_sub is not None:
            expl_variance_sub_df = self._explained_variance_df[
                self._explained_variance_df["ct_name"].isin(cell_type_sub)
            ]
        else:
            expl_variance_sub_df = self._explained_variance_df
        for ct in expl_variance_sub_df["ct_name"].unique():
            expl_variance_ct_df = expl_variance_sub_df[expl_variance_sub_df["ct_name"] == ct]

            top_gene_names = (
                expl_variance_ct_df.groupby("gene")["expl_var_head_gene"].mean().nlargest(n_top_genes).index
            )
            expl_variance_ct_df = expl_variance_ct_df[expl_variance_ct_df["gene"].isin(top_gene_names)]

            pivot_table = expl_variance_ct_df.pivot_table(
                index="gene",
                columns="head",
                values="expl_var_head_gene",
                aggfunc="mean",
                sort=False,
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pivot_table,
                annot=False,
                cmap="RdBu_r",
                cbar_kws={"label": "Explained Variance"},
                center=0,
                xticklabels=True,
                yticklabels=True,
            )
            plt.title(f"Explained Variance Heatmap by Gene and Head for Cell Type {ct}")
            plt.xlabel("Head")
            plt.ylabel("Gene")
            if wandb_log:
                wandb.log(
                    {
                        f"Featurewise Explained Variance for Cell Type {ct}": wandb.Image(plt),
                    }
                )
            if save_png:
                plt.savefig(os.path.join(save_dir, f"expl_variance_heatmap_{ct}.png"))
            if save_svg:
                plt.savefig(os.path.join(save_dir, f"expl_variance_heatmap_{ct}.svg"))
            if show:
                plt.show()
            plt.close()
