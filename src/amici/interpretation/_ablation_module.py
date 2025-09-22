from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openchord as ocd
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import wandb
from anndata import AnnData
from matplotlib.lines import Line2D
from scipy.stats import norm
from scvi import REGISTRY_KEYS
from statsmodels.stats.multitest import multipletests

from amici._constants import NN_REGISTRY_KEYS

from ._utils import _get_compute_method_kwargs

if TYPE_CHECKING:
    from amici._model import AMICI


@dataclass
class AMICIAblationModule:
    _adata: AnnData | None = None
    _ablation_scores_df: pd.DataFrame | None = None
    _compute_kwargs: dict | None = None
    _cell_types: list[str] | None = None
    _z_computed: bool = False
    _labels_key: str | None = None

    @classmethod
    def compute(
        cls,
        model: "AMICI",
        cell_type: str | None,
        head_idx: int | None,
        adata: AnnData | None = None,
        ablated_neighbor_ct_sub: list[str] | None = None,
        ablated_neighbor_indices: list[int] | None = None,
        compute_z_value: bool = False,
    ) -> "AMICIAblationModule":
        """Difference in gene expression prediction error for cell type of interest when ablating neighbor cell types for a specific head of interest.

        Args:
            cell_type (str): The cell type to compute the residuals for. None indicates using all cell types.
            head_idx (int, optional): The index of the head to test. None indicates using all heads.
            adata (AnnData, optional): The AnnData object.
            ablated_neighbor_ct_sub (list[str], optional): The neighbor cell types to ablate.
            ablated_neighbor_indices (list[int], optional): The indices of the neighbor cells to ablate.
            compute_z_value (bool, optional): Whether to save the z-value using the correlation coefficient or not.

        Returns
        -------
            pd.DataFrame: A DataFrame with the difference in gene expression prediction error for cell type of interest when ablating neighbor cell types.
            The DataFrame has the columns:
                - [ablated_neighbor_ct]: The difference in gene expression prediction error when ablating the neighbor cell type ablated_neighbor_ct.
        """
        assert (
            ablated_neighbor_ct_sub is None or ablated_neighbor_indices is None
        ), "Cannot pass in both ablated_neighbor_ct_sub and ablated_neighbor_indices"
        _compute_kwargs = _get_compute_method_kwargs(**locals())
        _z_computed = False

        model._check_if_trained(warn=True)

        adata = model._validate_anndata(adata)

        _labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        _cell_types = adata.obs[_labels_key].unique() if ablated_neighbor_ct_sub is None else ablated_neighbor_ct_sub

        cell_types = [cell_type] if cell_type is not None else adata.obs[_labels_key].unique()

        ablation_dfs = []
        for cell_type in cell_types:
            base_residuals, _ = cls._get_ct_residuals_for_ablated_head(
                model,
                cell_type,
                adata=adata,
                ablate_heads=head_idx is not None,
                ablated_neighbor_ct=None,
                head_idx=head_idx,
            )
            ablation_residuals = {}
            if ablated_neighbor_ct_sub is not None:
                cell_type_sub = ablated_neighbor_ct_sub
            elif ablated_neighbor_indices is not None:
                cell_type_sub = ["ablated_indices"]
            else:
                cell_type_sub = _cell_types
            for ablated_neighbor_ct in cell_type_sub:
                if ablated_neighbor_ct == "ablated_indices":
                    ablated_residuals, gene_exp = cls._get_ct_residuals_for_ablated_head(
                        model,
                        cell_type,
                        adata=adata,
                        ablate_heads=head_idx is not None,
                        ablated_neighbor_indices=ablated_neighbor_indices,
                        head_idx=head_idx,
                    )
                else:
                    ablated_residuals, gene_exp = cls._get_ct_residuals_for_ablated_head(
                        model,
                        cell_type,
                        adata=adata,
                        ablate_heads=head_idx is not None,
                        ablated_neighbor_ct=ablated_neighbor_ct,
                        head_idx=head_idx,
                    )
                ablation_residuals[f"{ablated_neighbor_ct}_ablation"] = (ablated_residuals**2 - base_residuals**2).mean(
                    axis=0
                )
                # neighbor contribution = full prediction - prediction w/o neighbor ct
                diff_residuals = base_residuals - ablated_residuals
                ablation_residuals[f"{ablated_neighbor_ct}_diff"] = diff_residuals.mean(axis=0)

                if compute_z_value:
                    z_scores = []
                    for i in range(len(adata.var_names)):
                        r = torch.corrcoef(
                            torch.stack(
                                (
                                    torch.tensor(diff_residuals.abs().values[:, i]),
                                    torch.tensor(gene_exp.values[:, i]),
                                )
                            )
                        )[0, 1]
                        n = len(adata[adata.obs[_labels_key] == cell_type])
                        z_value = (r * ((n - 2) ** 0.5) / (1 - r**2) ** 0.5).numpy()
                        if np.isnan(z_value) or np.isinf(z_value) or r == 1:
                            z_value = 0
                        z_scores.append(z_value)
                    ablation_residuals[f"{ablated_neighbor_ct}_z_value"] = z_scores
                    p_vals = 1 - norm.cdf(z_scores)

                    # Use statsmodels for Benjamini-Hochberg correction
                    _, adj_p_vals, _, _ = multipletests(p_vals, method="fdr_bh")

                    nl10_pval_adj = -np.log10(adj_p_vals + 1e-8)  # Avoid log(0)
                    nl10_pval = -np.log10(p_vals + 1e-8)  # Avoid log(0)

                    ablation_residuals[f"{ablated_neighbor_ct}_nl10_pval"] = nl10_pval
                    ablation_residuals[f"{ablated_neighbor_ct}_nl10_pval_adj"] = nl10_pval_adj
                    _z_computed = True
            res_df = pd.DataFrame(ablation_residuals)

            res_df["gene_variance"] = gene_exp.values.var(axis=0)
            res_df["gene"] = adata.var_names
            res_df["head_idx"] = head_idx
            res_df["cell_type"] = cell_type
            ablation_dfs.append(res_df)
        ablation_scores_df = pd.concat(ablation_dfs, axis=0).reset_index()

        return cls(
            _adata=adata,
            _ablation_scores_df=ablation_scores_df,
            _compute_kwargs=_compute_kwargs,
            _cell_types=_cell_types,
            _z_computed=_z_computed,
            _labels_key=_labels_key,
        )

    @staticmethod
    def _get_ct_residuals_for_ablated_head(
        model: "AMICI",
        cell_type: str,
        adata: AnnData | None = None,
        ablate_heads: bool = True,
        head_idx: int = -1,
        ablated_neighbor_ct: str | None = None,
        ablated_neighbor_indices: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gene expression residuals and ground truth gene expression for head_idx and cell_type.

        Args:
            cell_type (str): The cell type to compute the residuals for.
            adata (AnnData, optional): The AnnData object.
            ablate_heads (bool, optional): Whether to ablate the heads.
            head_idx (int, optional): The index of the head to use. -1 indicates not using any attention heads.
            ablated_neighbor_ct (str, optional): The neighbor cell type to ablate.
            ablated_neighbor_indices (list[int], optional): The indices of the neighbor cells to ablate.

        Returns
        -------
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - residuals: A DataFrame of gene expression residuals of shape n_ct_cells x n_genes.
                - gene_exp: A DataFrame of ground truth gene expression of shape n_ct_cells x n_genes.
        """
        assert (
            ablated_neighbor_ct is None or ablated_neighbor_indices is None
        ), "Cannot pass in both ablated_neighbor_ct and ablated_neighbor_indices"

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

        labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        ct_indices = np.arange(len(adata))[adata.obs[labels_key] == cell_type]
        scdl = model._make_data_loader(adata=adata, indices=ct_indices)

        residuals = []
        gene_expressions = []
        label_mapping = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        for tensors in scdl:
            true_X = tensors[REGISTRY_KEYS.X_KEY].cpu()
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model.module.reset_hooks()

            head_idxs_to_ablate = [i for i in range(model.module.n_heads) if i != head_idx] if ablate_heads else []

            if ablated_neighbor_ct is not None:
                ablated_neighbor_ct_idx = list(label_mapping).index(ablated_neighbor_ct)
                neighbor_mask = torch.where(
                    tensors[NN_REGISTRY_KEYS.NN_LABELS_KEY] == ablated_neighbor_ct_idx,
                    0,
                    1,
                )[:, :, 0]
            elif ablated_neighbor_indices is not None:
                neighbor_mask = torch.where(
                    torch.isin(
                        tensors[NN_REGISTRY_KEYS.NN_IDX_KEY],
                        torch.tensor(ablated_neighbor_indices).to(tensors[NN_REGISTRY_KEYS.NN_IDX_KEY].device),
                    ),
                    0,
                    1,
                )
            else:
                neighbor_mask = None

            head_hook_fn = partial(
                _mt_heads_ablation_hook,
                head_idxs_to_ablate=head_idxs_to_ablate,
                neighbor_mask=neighbor_mask,
            )
            prediction = (
                model.module.run_with_hooks(
                    tensors,
                    fwd_hooks=[("attention_layer.hook_pattern", head_hook_fn)],
                )[1]["prediction"]
                .detach()
                .cpu()
            )

            residuals.append(prediction - true_X)
            gene_expressions.append(true_X)
        all_residuals = torch.cat(residuals)
        gene_exp = torch.cat(gene_expressions)

        residuals_df = pd.DataFrame(all_residuals.numpy(), columns=adata.var_names)
        gene_exp_df = pd.DataFrame(gene_exp.numpy(), columns=adata.var_names)

        return residuals_df, gene_exp_df

    def save(self, save_path: str):
        """Save ablation scores to file"""
        self._ablation_scores_df.to_csv(save_path)
        return self

    def plot_neighbor_ablation_scores(
        self,
        cell_type=None,
        score_col="ablation",
        palette=None,
        threshold=0.02,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots the ablation scores for each neighbor cell type provided the cell type and head index.

        Args:
            cell_type (str, optional): The cell type to plot the ablation scores for.
            plot_z_value (bool, optional): Whether to plot the z-value rather than ablation scores.
            palette (dict, optional): A dictionary mapping each cell type to a color.
            threshold (float, optional): The threshold for sparsifying the ablation scores.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases.
            show (bool, optional): Whether to display the plot.
            save_png (bool, optional): Whether to save the plot as a PNG file.
            save_svg (bool, optional): Whether to save the plot as an SVG file.
            save_dir (str, optional): The directory to save the plot files.

        Returns
        -------
            None
        """
        if score_col == "z_value":
            assert (
                self._z_computed
            ), "Z-values have not been computed. Please run compute with compute_z_value=True first."
        assert (
            len(self._ablation_scores_df["cell_type"].unique()) == 1
        ) or cell_type is not None, "More than one cell type found. Please pass in cell_type."
        assert len(self._ablation_scores_df["head_idx"].unique()) == 1, "More than one head index found"
        cell_type = self._ablation_scores_df["cell_type"].unique()[0] if cell_type is None else cell_type
        head_idx = self._ablation_scores_df["head_idx"].unique()[0]

        plt.figure(figsize=(10, 6))
        neighbor_ablation_cols = {}

        ablation_cols = [f"{ct_name}_{score_col}" for ct_name in self._cell_types if ct_name != cell_type]
        ct_scores_df = self._ablation_scores_df[self._ablation_scores_df["cell_type"] == cell_type]
        for colname in ablation_cols:
            ct_name = colname.replace(f"_{score_col}", "")
            neighbor_ablation_cols[ct_name] = ct_scores_df[colname].values[:, None]
        neighbor_ablation_residuals = np.concatenate(list(neighbor_ablation_cols.values()), axis=1)
        neighbor_ablation_residuals[np.abs(neighbor_ablation_residuals) < threshold] = 0
        neighbor_ablation_residuals_df = pd.DataFrame(
            neighbor_ablation_residuals,
            index=ct_scores_df["gene"],
            columns=ablation_cols,
        )

        tab10_palette = plt.get_cmap("tab10")
        palette = (
            palette
            if palette is not None
            else {
                ct_color.replace(f"_{score_col}", ""): tab10_palette(i)
                for i, ct_color in enumerate(neighbor_ablation_residuals_df.columns)
            }
        )
        neighbor_ablation_residuals_df.sum().plot(
            kind="bar",
            color=[
                palette.get(ct_color.replace(f"_{score_col}", ""), "#333333")
                for ct_color in neighbor_ablation_residuals_df.columns
            ],
        )
        head_idx_str = f"head {head_idx}" if head_idx is not None else "all heads"
        score_title = "Z-Value" if score_col == "z_value" else "Ablation Score"
        plt.title(f"Neighbor {score_title} for {cell_type} by Ablated Cell Type for {head_idx_str}")
        plt.xlabel("Ablated Cell Type")
        plt.ylabel(f"Neighbor {score_title}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_png:
            plt.savefig(f"{save_dir}/neighbor_{score_col}_residuals_{cell_type}_head_{head_idx}.png")
        if save_svg:
            plt.savefig(f"{save_dir}/neighbor_{score_col}_residuals_{cell_type}_head_{head_idx}.svg")

        if wandb_log:
            wandb.log(f"neighbor_{score_col}_residuals_{cell_type}_head_{head_idx}", plt)

        if show:
            plt.show()

    def plot_featurewise_ablation_heatmap(
        self,
        cell_type=None,
        score_col="ablation",
        n_top_genes=10,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """Plots the heatmap of the top genes by ablation scores for a given cell type and head index.

        Args:
            cell_type (str, optional): The cell type to plot the ablation scores for.
            plot_z_value (bool, optional): Whether to plot the z-value rather than ablation scores.
            n_top_genes (int, optional): The number of top genes to plot.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases.
            show (bool, optional): Whether to display the plot.
            save_png (bool, optional): Whether to save the plot as a PNG file.
            save_svg (bool, optional): Whether to save the plot as an SVG file.
            save_dir (str, optional): The directory to save the plot files.

        Returns
        -------
            None
        """
        if score_col == "z_value":
            assert (
                self._z_computed
            ), "Z-values have not been computed. Please run compute with compute_z_value=True first."
        assert (
            len(self._ablation_scores_df["cell_type"].unique()) == 1
        ) or cell_type is not None, "More than one cell type found. Please pass in cell_type."
        assert len(self._ablation_scores_df["head_idx"].unique()) == 1, "More than one head index found"
        cell_type = self._ablation_scores_df["cell_type"].unique()[0] if cell_type is None else cell_type
        head_idx = self._ablation_scores_df["head_idx"].unique()[0]

        top_genes_per_ct = {}

        ablation_cols = [f"{ct_name}_{score_col}" for ct_name in self._cell_types if ct_name != cell_type]
        ct_scores_df = self._ablation_scores_df[self._ablation_scores_df["cell_type"] == cell_type]
        for colname in ablation_cols:
            ct_name = colname.replace(f"_{score_col}", "")
            # Filter for positive neighbor contribution scores
            diff_filter = ct_scores_df[f"{ct_name}_diff"] > 0
            top_genes_idx = ct_scores_df[diff_filter][colname].astype(float).nlargest(n_top_genes).index
            top_genes_per_ct[ct_name] = top_genes_idx

        # Compile all top genes into a single list
        all_top_genes = []
        for genes in top_genes_per_ct.values():
            for gene in genes:
                if gene not in all_top_genes:
                    all_top_genes.append(gene)

        # Create a heatmap dataframe
        heatmap_data = ct_scores_df.loc[all_top_genes, ablation_cols]
        gene_names = ct_scores_df.loc[all_top_genes, "gene"]
        heatmap_data = heatmap_data.set_index(gene_names)

        # Create a mask for top genes per column
        top_genes_mask = np.zeros_like(heatmap_data, dtype=bool)
        for col_idx, col_name in enumerate(ablation_cols):
            ct_name = col_name.replace(f"_{score_col}", "")
            for gene_idx, gene in enumerate(heatmap_data.index):
                if gene in ct_scores_df.loc[top_genes_per_ct[ct_name], "gene"].values:
                    top_genes_mask[gene_idx, col_idx] = True

        # Plot the heatmap
        head_idx_str = f"head {head_idx}" if head_idx is not None else "all heads"
        score_title = "Z-Value" if score_col == "z_value" else "Ablation Score"
        plt.figure(figsize=(12, len(all_top_genes) * 0.4))

        # Create the main heatmap
        ax = sns.heatmap(
            heatmap_data.astype(float),
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": score_title},
            xticklabels=True,
            yticklabels=True,
        )

        # Add black borders around cells that are in the top genes for that column
        for i in range(top_genes_mask.shape[0]):
            for j in range(top_genes_mask.shape[1]):
                if top_genes_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))

        plt.title(f"Heatmap of Top Genes' {score_title} for {cell_type} for {head_idx_str}")
        plt.xlabel("Ablated Cell Type")
        plt.ylabel("Genes")
        plt.tight_layout()

        if save_png:
            plt.savefig(f"{save_dir}/heatmap_{score_col}_top_genes_{cell_type}_head_{head_idx}.png")
        if save_svg:
            plt.savefig(f"{save_dir}/heatmap_{score_col}_top_genes_{cell_type}_head_{head_idx}.svg")

        if wandb_log:
            wandb.log(f"heatmap_{score_col}_top_genes_{cell_type}_head_{head_idx}", plt)
        if show:
            plt.show()

    def plot_featurewise_contributions_heatmap(
        self,
        cell_type=None,
        sort_by="ablation",
        n_top_genes=10,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """Plots the heatmap of gene contributions for the top genes by ablation scores for a given cell type and head index.

        Args:
            cell_type (str, optional): The cell type to plot the contributions for.
            sort_by (str, optional): The score column to sort the genes by.
            n_top_genes (int, optional): The number of top genes to plot.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases.
            show (bool, optional): Whether to display the plot.
            save_png (bool, optional): Whether to save the plot as a PNG file.
            save_svg (bool, optional): Whether to save the plot as an SVG file.
            save_dir (str, optional): The directory to save the plot files.

        Returns
        -------
            None
        """
        assert (
            len(self._ablation_scores_df["cell_type"].unique()) == 1
        ) or cell_type is not None, "More than one cell type found. Please pass in cell_type."
        assert len(self._ablation_scores_df["head_idx"].unique()) == 1, "More than one head index found"
        cell_type = self._ablation_scores_df["cell_type"].unique()[0] if cell_type is None else cell_type
        head_idx = self._ablation_scores_df["head_idx"].unique()[0]

        top_genes_per_ct = {}

        ablation_cols = [f"{ct_name}_{sort_by}" for ct_name in self._cell_types if ct_name != cell_type]
        diff_cols = [f"{ct_name}_diff" for ct_name in self._cell_types if ct_name != cell_type]
        ct_scores_df = self._ablation_scores_df[self._ablation_scores_df["cell_type"] == cell_type]
        for colname in ablation_cols:
            ct_name = colname.replace(f"_{sort_by}", "")
            # Filter for positive neighbor contribution scores
            diff_filter = ct_scores_df[f"{ct_name}_diff"] > 0
            top_genes_idx = ct_scores_df[diff_filter][colname].astype(float).nlargest(n_top_genes).index
            top_genes_per_ct[ct_name] = top_genes_idx

        # Compile all top genes into a single list
        all_top_genes = []
        for genes in top_genes_per_ct.values():
            for gene in genes:
                if gene not in all_top_genes:
                    all_top_genes.append(gene)

        # Create a heatmap dataframe
        heatmap_data = ct_scores_df.loc[all_top_genes, diff_cols]
        gene_names = ct_scores_df.loc[all_top_genes, "gene"]
        heatmap_data = heatmap_data.set_index(gene_names)

        # Create a mask for top genes per column
        top_genes_mask = np.zeros_like(heatmap_data, dtype=bool)
        for col_idx, col_name in enumerate(diff_cols):
            ct_name = col_name.replace("_diff", "")
            for gene_idx, gene in enumerate(heatmap_data.index):
                if gene in ct_scores_df.loc[top_genes_per_ct[ct_name], "gene"].values:
                    top_genes_mask[gene_idx, col_idx] = True

        # Plot the heatmap
        head_idx_str = f"head {head_idx}" if head_idx is not None else "all heads"
        plt.figure(figsize=(12, len(all_top_genes) * 0.4))

        # Create the main heatmap
        ax = sns.heatmap(
            heatmap_data.astype(float),
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Gene Contribution"},
            xticklabels=True,
            yticklabels=True,
        )

        # Add black borders around cells that are in the top genes for that column
        for i in range(top_genes_mask.shape[0]):
            for j in range(top_genes_mask.shape[1]):
                if top_genes_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))

        plt.title(f"Heatmap of Top Genes' Contributions for {cell_type} for {head_idx_str}")
        plt.xlabel("Ablated Cell Type")
        plt.ylabel("Genes")
        plt.tight_layout()

        if save_png:
            plt.savefig(f"{save_dir}/heatmap_contributions_top_genes_{cell_type}_head_{head_idx}.png")
        if save_svg:
            plt.savefig(f"{save_dir}/heatmap_contributions_top_genes_{cell_type}_head_{head_idx}.svg")

        if wandb_log:
            wandb.log(f"heatmap_contributions_top_genes_{cell_type}_head_{head_idx}", plt)
        if show:
            plt.show()

    def plot_featurewise_contributions_dotplot(
        self,
        cell_type=None,
        color_by="diff",
        size_by="nl10_pval_adj",
        n_top_genes=10,
        dot_max=200,
        dot_min=0,
        step=5,
        min_size_by=0,
        vrange=0.5,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """Plots a dotplot of the top genes by color_by and size_by for a given cell type and head index.

        Args:
            cell_type (str, optional): The cell type to plot the dotplot for.
            color_by (str, optional): The column to color the dots by.
            size_by (str, optional): The column to size the dots by.
            n_top_genes (int, optional): The number of top genes to plot.
            dot_max (int, optional): The maximum size of the dots.
            dot_min (int, optional): The minimum size of the dots.
            step (int, optional): The step size of the dots.
            min_size_by (int, optional): The minimum size_by value to plot.
            min_color_by (int, optional): The minimum color_by value to plot.
            vrange (float, optional): The range for color scaling.
            show (bool, optional): Whether to show the plot.
            save_png (bool, optional): Whether to save the plot as a png.
            save_svg (bool, optional): Whether to save the plot as a svg.
            save_dir (str, optional): The directory to save the plot to.
        """
        assert color_by in [
            "z_value",
            "ablation",
            "diff",
            "pval_adj",
            "nl10_pval_adj",
        ], "color_by must be one of 'z_value', 'ablation', 'diff', 'pval_adj', or 'nl10_pval_adj'"
        assert size_by in [
            "z_value",
            "ablation",
            "diff",
            "pval_adj",
            "nl10_pval_adj",
        ], "size_by must be one of 'z_value', 'ablation', 'diff', 'pval_adj', or 'nl10_pval_adj'"
        assert (
            len(self._ablation_scores_df["cell_type"].unique()) == 1
        ) or cell_type is not None, "More than one cell type found. Please pass in cell_type."
        assert len(self._ablation_scores_df["head_idx"].unique()) == 1, "More than one head index found"
        cell_type = self._ablation_scores_df["cell_type"].unique()[0] if cell_type is None else cell_type
        head_idx = self._ablation_scores_df["head_idx"].unique()[0]

        top_genes_per_ct = {}

        size_by_cols = [f"{ct_name}_{size_by}" for ct_name in self._cell_types if ct_name != cell_type]
        color_by_cols = [f"{ct_name}_{color_by}" for ct_name in self._cell_types if ct_name != cell_type]
        ct_scores_df = self._ablation_scores_df[self._ablation_scores_df["cell_type"] == cell_type]
        for colname in color_by_cols:
            ct_name = colname.replace(f"_{color_by}", "")
            # Filter for positive neighbor contribution scores
            diff_filter = ct_scores_df[f"{ct_name}_diff"] > 0
            pval_filter = ct_scores_df[f"{ct_name}_nl10_pval_adj"] > -np.log10(0.05)
            size_by_filter = ct_scores_df[f"{ct_name}_{size_by}"] > min_size_by
            top_genes_idx = (
                ct_scores_df[diff_filter][colname]
                .astype(float)
                .nlargest(n_top_genes)[size_by_filter][pval_filter]
                .index
            )
            top_genes_per_ct[ct_name] = top_genes_idx

        all_top_genes = []
        for genes in top_genes_per_ct.values():
            for gene in genes:
                if gene not in all_top_genes:
                    all_top_genes.append(gene)

        # Create a dotplot dataframe
        dotplot_data = ct_scores_df.loc[all_top_genes, color_by_cols + size_by_cols]
        gene_names = ct_scores_df.loc[all_top_genes, "gene"]
        dotplot_data = dotplot_data.set_index(gene_names)

        # ------------------------------------------------------------------
        # Handle dot sizes --------------------------------------------------
        # ------------------------------------------------------------------
        # Select the size columns and normalise EACH column to 0-1 range so
        # that dot sizes are comparable across all neighbour cell types,
        # regardless of how many/which are selected in `size_by_cols`.
        dot_size_df = dotplot_data[size_by_cols].copy()

        # For the legend, use the actual data maximum
        actual_data_max = dotplot_data[size_by_cols].max().max()
        legend_max = min(actual_data_max, dot_max)

        # Normalize the data to 0-1 range based on the actual data range
        # This matches what scanpy expects for dot sizes
        dot_size_df = dot_size_df.apply(
            lambda col: np.clip(col, a_min=dot_min, a_max=legend_max) / (legend_max - dot_min)
        )

        for col in dot_size_df.columns:
            dot_size_df.rename(columns={col: col.replace(f"_{size_by}", "")}, inplace=True)
        dot_color_df = dotplot_data[color_by_cols]
        for col in dot_color_df.columns:
            dot_color_df.rename(columns={col: col.replace(f"_{color_by}", "")}, inplace=True)

        neighbor_cell_types = [ct.replace(f"_{size_by}", "") for ct in size_by_cols]

        head_idx_str = f"head {head_idx}" if head_idx is not None else "all heads"
        score_titles = {
            "z_value": "Z-value",
            "ablation": "Ablation",
            "diff": "Neighbor Contribution",
            "pval_adj": "Adjusted P-value",
            "nl10_pval_adj": "Neg Log10 Adjusted P-value",
        }
        fig = sc.pl.dotplot(
            self._adata[self._adata.obs[self._labels_key].isin(neighbor_cell_types)].copy(),
            dot_size_df=dot_size_df.T.astype(float),
            dot_color_df=dot_color_df.T.astype(float),
            var_names=gene_names,
            groupby=self._labels_key,
            vmin=-vrange,
            vmax=vrange,
            vcenter=0,
            cmap="RdBu_r",
            dot_min=0.0,
            dot_max=1.0,
            title=f"Dotplot of {score_titles[size_by]} by {score_titles[color_by]} Scores for {cell_type} for {head_idx_str}",
            size_title=score_titles[size_by],
            colorbar_title=score_titles[color_by],
            return_fig=True,
        )

        def _plot_size_legend(self, size_legend_ax, step, dot_max, dot_min):
            """
            Create a plot for the size legend of the dotplot

            Args:
                size_legend_ax (): The axis object for the size legend.
                step (float): The step size for the size legend.
                dot_max (float): The maximum dot size.
                dot_min (float): The minimum dot size.
            """
            size_legend_ax.clear()

            # ------------------------------------------------------------------
            # Build ticks every <step> units from dot_min to dot_max inclusive
            # ------------------------------------------------------------------
            size_range = np.arange(dot_min, dot_max + 1e-6, step)

            # Normalise to 0–1 for sizing
            denom = (dot_max - dot_min) if dot_max != dot_min else 1.0
            size_values = (size_range - dot_min) / denom

            # Convert into marker areas used by scanpy's dotplot
            size = size_values**self.size_exponent
            size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

            # Increase spacing to prevent overlap
            spacing_factor = 1.3
            x_positions = np.arange(len(size)) * spacing_factor + 0.5

            # plot size bar
            size_legend_ax.scatter(
                x_positions,
                np.repeat(0, len(size)),
                s=size,
                color="gray",
                edgecolor="black",
                linewidth=self.dot_edge_lw,
                zorder=100,
            )
            size_legend_ax.set_xticks(x_positions)
            labels = [str(int(x)) for x in size_range]
            size_legend_ax.set_xticklabels(labels, fontsize="x-small")

            # remove y ticks and labels
            size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

            # remove surrounding lines
            size_legend_ax.spines["right"].set_visible(False)
            size_legend_ax.spines["top"].set_visible(False)
            size_legend_ax.spines["left"].set_visible(False)
            size_legend_ax.spines["bottom"].set_visible(False)
            size_legend_ax.grid(visible=False)

            ymax = size_legend_ax.get_ylim()[1]
            size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
            size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

            xmin, xmax = size_legend_ax.get_xlim()
            size_legend_ax.set_xlim(xmin - 0.5, xmax + 0.5)

        # Draw size legend using the actual data range so that
        # labels correspond to the actual data values present.
        _plot_size_legend(
            fig,
            fig.get_axes()["size_legend_ax"],
            step=step,
            dot_max=legend_max,
            dot_min=dot_min,
        )
        if save_png:
            plt.savefig(
                f"{save_dir}/dotplot_{cell_type}_{head_idx_str}_{size_by}_{color_by}.png",
                dpi=300,
                bbox_inches="tight",
            )
        if save_svg:
            plt.savefig(
                f"{save_dir}/dotplot_{cell_type}_{head_idx_str}_{size_by}_{color_by}.svg",
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()

    def _get_interaction_weight_matrix(
        self,
        significance_threshold=0.05,
    ):
        """
        Returns the interaction weight matrix between the cell types based on significant pvalues of genes.

        Args:
            significance_threshold (float): The significance threshold for the pvalues.

        Returns
        -------
            weight_matrix_df (pd.DataFrame): The interaction weight matrix.
        """
        assert self._compute_kwargs["cell_type"] is None, "Must run compute over all cell types via cell_type=None"

        cell_types = np.sort(self._ablation_scores_df["cell_type"].unique())

        # Create weight matrix using the same logic as the original directed graph
        weight_matrix = np.zeros((len(cell_types), len(cell_types)))
        for i, ct in enumerate(cell_types):
            ct_data = self._ablation_scores_df[self._ablation_scores_df["cell_type"] == ct]
            pval_cols = [f"{ct_name}_nl10_pval_adj" for ct_name in cell_types if ct_name != ct]
            diff_cols = [f"{ct_name}_diff" for ct_name in cell_types if ct_name != ct]
            for col, diff_col in zip(pval_cols, diff_cols):
                neighbor_ct = col.replace("_nl10_pval_adj", "")
                j = np.where(cell_types == neighbor_ct)[0][0]
                weight = ((ct_data[col] > -np.log10(significance_threshold)) & (ct_data[diff_col] > 0)).sum()
                weight_matrix[j, i] = weight

        # Convert the weight matrix to a pandas DataFrame
        weight_df = pd.DataFrame(weight_matrix, index=cell_types, columns=cell_types)

        return weight_df

    def get_interaction_circos_plot(
        self,
        cell_type_sub=None,
        significance_threshold=0.05,
        palette=None,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots a circos plot of the interactions between the cell types based on the ablation scores.

        Args:
            cell_type_sub (list): The cell types to plot.
            significance_threshold (float): The significance threshold for the pvalues.
            palette (dict): The palette to use for the colors.
            show (bool): Whether to show the plot.
            save_png (bool): Whether to save the plot as a png.
            save_svg (bool): Whether to save the plot as a svg.

        Returns
        -------
            fig (): The figure object for the plot.
        """
        assert self._compute_kwargs["cell_type"] is None, "Must run compute over all cell types via cell_type=None"

        weight_matrix_df = self._get_interaction_weight_matrix(significance_threshold=significance_threshold)
        if cell_type_sub is not None:
            cell_types = np.sort(cell_type_sub)
        else:
            cell_types = weight_matrix_df.index.tolist()

        fig = ocd.Chord(weight_matrix_df.values, cell_types)

        fig.colormap = [palette.get(ct, "lightblue") if palette else "lightblue" for ct in cell_types]
        if save_png:
            fig.save_png(f"{save_dir}/interaction_circos_plot.png")
        if save_svg:
            fig.save_svg(f"{save_dir}/interaction_circos_plot.svg")
        return fig

    def plot_interaction_weight_heatmap(
        self,
        significance_threshold=0.05,
        vmin=0,
        vmax=80,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots a heatmap of the interactions between the cell types based on the ablation scores.

        Args:
            cell_type_sub (list): The cell types to plot.
            significance_threshold (float): The significance threshold for the pvalues.
            vmin (float): The minimum value for the colorbar.
            vmax (float): The maximum value for the colorbar.
            save_png (bool): Whether to save the plot as a png.
            save_svg (bool): Whether to save the plot as a svg.
            save_dir (str): The directory to save the plot to.
        """
        assert self._compute_kwargs["cell_type"] is None, "Must run compute over all cell types via cell_type=None"

        weight_matrix_df = self._get_interaction_weight_matrix(significance_threshold=significance_threshold)

        plt.figure(figsize=(10, 8))
        sns.heatmap(data=weight_matrix_df, cbar=True, cmap="Reds", vmin=vmin, vmax=vmax)
        plt.title("Cell Type Interaction Weight Heatmap")
        plt.xlabel("Receiver Cell Type")
        plt.ylabel("Sender Cell Type")
        if save_png:
            plt.savefig(f"{save_dir}/interaction_weight_heatmap.png")
        if save_svg:
            plt.savefig(f"{save_dir}/interaction_weight_heatmap.svg")

    def plot_interaction_directed_graph(
        self,
        cell_type_sub=None,
        significance_threshold=0.05,
        weight_threshold=0,
        node_size=1500,
        palette=None,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plots a directed graph of the interactions between cell types based on the ablation scores.

        Args:
            cell_type_sub (list): The cell types to plot.
            significance_threshold (float): The significance threshold for the pvalues.
            weight_threshold (float): The threshold for the weight of the edges.
            node_size (int): The size of the nodes.
            palette (dict): The palette to use for the colors.
            show (bool): Whether to show the plot.
            save_png (bool): Whether to save the plot as a png.
            save_svg (bool): Whether to save the plot as a svg.
            save_dir (str): The directory to save the plot to.
        """
        assert self._compute_kwargs["cell_type"] is None, "Must run compute over all cell types via cell_type=None"

        # Get weight matrix using the helper function
        weight_matrix_df = self._get_interaction_weight_matrix(significance_threshold=significance_threshold)

        all_cell_types = weight_matrix_df.index.tolist()
        if cell_type_sub is None:
            cell_types = weight_matrix_df.index.tolist()
        else:
            cell_types = cell_type_sub

        G = nx.DiGraph()
        for ct in cell_types:
            G.add_node(ct)

        # Add edges using the weight matrix
        weight_matrix = weight_matrix_df.values
        for i, ct in enumerate(all_cell_types):
            for j, neighbor_ct in enumerate(all_cell_types):
                if i != j and weight_matrix[j, i] >= weight_threshold:  # Don't add self-loops
                    if ct in cell_types and neighbor_ct in cell_types:
                        G.add_edge(
                            neighbor_ct,
                            ct,
                            weight=weight_matrix[j, i],
                        )

        plt.figure(figsize=(8, 8))
        pos = nx.circular_layout(G)

        # Use palette colors for nodes, defaulting to lightblue if no palette color exists
        node_colors = [palette.get(node, "lightblue") if palette else "lightblue" for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)

        max_weight = max(data["weight"] for _, _, data in G.edges(data=True))

        for u, v, data in G.edges(data=True):
            scaled_weight = data["weight"] / max_weight
            width = scaled_weight * 5  # Scale for visibility
            nx.draw_networkx_edges(
                G,
                pos,
                node_size=node_size,
                edgelist=[(u, v)],
                width=width,
                alpha=scaled_weight,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.2",
            )

        legend_weights = np.linspace(0, max_weight, num=5).tolist()

        legend_proxies = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=weight / max_weight * 5,
                alpha=weight / max_weight,
            )
            for weight in legend_weights
        ]
        plt.legend(legend_proxies, legend_weights, title="# Sig. p-values")

        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        plt.title("Inferred Cell Type Interaction Network Based")
        plt.axis("off")
        if save_png:
            plt.savefig(
                f"{save_dir}/interaction_directed_graph_{'all' if cell_type_sub is None else 'subset'}.png",
                dpi=300,
                bbox_inches="tight",
            )
        if save_svg:
            plt.savefig(f"{save_dir}/interaction_directed_graph_{'all' if cell_type_sub is None else 'subset'}.svg")
        if show:
            plt.show()
        return plt.gcf()
