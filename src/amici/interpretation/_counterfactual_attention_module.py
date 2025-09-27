import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from anndata import AnnData
from einops import rearrange, repeat
from scvi import REGISTRY_KEYS

from ._utils import _get_compute_method_kwargs

if TYPE_CHECKING:
    from amici._model import AMICI


@dataclass
class AMICICounterfactualAttentionModule:
    _adata: AnnData | None = None
    _compute_kwargs: dict | None = None
    _counterfactual_attention_df: pd.DataFrame | None = None
    _labels_key: str | None = None

    @classmethod
    def compute(
        cls,
        model: "AMICI",
        cell_type: str,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        head_idxs: list[int] | None = None,
        batch_size: int | None = None,
    ) -> "AMICICounterfactualAttentionModule":
        """
        Compute the counterfactual attention patterns for a given cell type and head indices.

        Args:
            model: The trained AMICI model.
            cell_type: The index cell type to get counterfactual attention scores for.
            adata: Optional AnnData object to use. If None, uses model.adata.
            indices: Optional list of cell indices to get scores for. If None, uses all cells.
            head_idxs: Optional list of attention head indices to get scores for. If None, uses all heads.
            batch_size: Optional batch size to use for the data loader.

        Returns
        -------
            AMICICounterfactualAttentionModule: The module instance with computed counterfactual attention scores stored
            in _counterfactual_attention_df with columns:
                - query_label: Cell type label of the query cell
                - neighbor_idx: Index of the neighbor cell
                - neighbor_label: Cell type label of the neighbor cell
                - head_idx: Attention head index
                - base_attention_score: Base attention score
                - pos_coef: Position coefficient of the counterfactual attention score function
                - dummy_attention_score: Dummy attention score of the model
                - distance_kernel_unit_scale: Distance kernel unit scale
        """
        _compute_kwargs = _get_compute_method_kwargs(**locals())
        _labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        model._check_if_trained(warn=True)

        head_idxs = head_idxs if head_idxs is not None else list(range(model.module.n_heads))

        adata = model._validate_anndata(adata)
        if indices is None:
            indices = list(range(adata.n_obs))
        scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        labels_cat_list = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping.tolist()
        if cell_type not in labels_cat_list:
            raise ValueError(f"Cell type {cell_type} not found in adata")
        labels_cat_val = labels_cat_list.index(cell_type)
        labels_tensor = torch.tensor(labels_cat_val).unsqueeze(0).unsqueeze(0).to(model.device)

        counterfactual_attention_dfs = []
        batch_start_idx = 0

        for neighbor_tensors in scdl:
            batch_neighbor_labels = neighbor_tensors[REGISTRY_KEYS.LABELS_KEY].cpu().detach().numpy().flatten()
            batch_size = len(batch_neighbor_labels)
            nn_X = neighbor_tensors[REGISTRY_KEYS.X_KEY].unsqueeze(0).to(model.device)

            inf_outputs = model.module.inference(
                labels_tensor,
                nn_X,
            )

            gen_outputs = model.module.generative(
                labels_tensor,
                inf_outputs["label_embed"],
                inf_outputs["nn_embed"],
                torch.full((1, nn_X.shape[1]), 0).to(model.device),
                return_attention_scores=True,
            )
            gen_attn_scores = gen_outputs["attention_scores"][0, :, :-1].T  # n_cells x n_heads
            gen_pos_coefs = gen_outputs["pos_coefs"][0, :]  # n_cells x n_heads

            batch_attn_scores = gen_attn_scores.cpu().detach().numpy()
            batch_pos_coefs = gen_pos_coefs.cpu().detach().numpy()
            n_heads = batch_attn_scores.shape[1]

            # Get the indices for this batch
            batch_indices = indices[batch_start_idx : batch_start_idx + batch_size]

            batch_base_attn_df = pd.DataFrame(
                batch_attn_scores,
                columns=range(n_heads),
            ).melt(var_name="head_idx", value_name="base_attention_score")

            batch_pos_coef_df = pd.DataFrame(batch_pos_coefs, columns=range(n_heads)).melt(
                var_name="head_idx", value_name="position_coefficient"
            )

            batch_counterfactual_attention_df = pd.DataFrame(
                {
                    "query_label": cell_type,
                    "neighbor_idx": repeat(np.array(batch_indices), "n -> (h n)", h=n_heads),
                    "neighbor_label": repeat(
                        np.array(labels_cat_list)[batch_neighbor_labels],
                        "n -> (h n)",
                        h=n_heads,
                    ),
                    "head_idx": batch_base_attn_df["head_idx"],
                    "base_attention_score": batch_base_attn_df["base_attention_score"],
                    "position_coefficient": batch_pos_coef_df["position_coefficient"],
                    "dummy_attention_score": model.module.attention_dummy_score,
                    "distance_kernel_unit_scale": model.module.distance_kernel_unit_scale,
                }
            )

            counterfactual_attention_dfs.append(batch_counterfactual_attention_df)
            batch_start_idx += batch_size

        counterfactual_attention_df = pd.concat(counterfactual_attention_dfs, axis=0, ignore_index=True)

        counterfactual_attention_df = counterfactual_attention_df[
            counterfactual_attention_df["neighbor_label"] != cell_type
        ]

        return cls(
            _adata=adata,
            _labels_key=_labels_key,
            _counterfactual_attention_df=counterfactual_attention_df,
            _compute_kwargs=_compute_kwargs,
        )

    def save(self, save_path: str):
        """Save counterfactual attention scores to file"""
        self._counterfactual_attention_df.to_csv(save_path)
        return self

    def calculate_counterfactual_attention_at_distances(
        self,
        head_idx: int,
        distances: list[float],
    ):
        """
        Evaluate counterfactual attention score functions at given distances.

        Args:
            head_idx (int): Head index to evaluate.
            distances (list[float]): List of distances to evaluate.

        Returns
        -------
            pd.DataFrame: DataFrame containing the counterfactual attention scores with columns:
                - query_label: Cell type label of the query cell
                - neighbor_idx: Index of the neighbor cell
                - neighbor_label: Cell type label of the neighbor cell
                - head_{head_idx}: Attention score for that neighbor for head head_idx
                - distance: Counterfactual distance of the neighbor
        """
        counterfactual_attention_eval_dfs = []
        head_counterfactual_attention_df = self._counterfactual_attention_df.loc[
            self._counterfactual_attention_df["head_idx"] == head_idx
        ]
        base_attention_score = head_counterfactual_attention_df["base_attention_score"].to_numpy()
        pos_coef = head_counterfactual_attention_df["position_coefficient"].to_numpy()
        dummy_attention_score = head_counterfactual_attention_df["dummy_attention_score"].to_numpy()
        distance_kernel_unit_scale = head_counterfactual_attention_df["distance_kernel_unit_scale"].to_numpy()

        for distance in distances:
            attention_score = base_attention_score - pos_coef * (distance / distance_kernel_unit_scale)
            attention_pattern = np.exp(attention_score) / (np.exp(attention_score) + np.exp(dummy_attention_score))
            counterfactual_attention_eval_dfs.append(
                pd.DataFrame.from_dict(
                    {
                        "query_label": head_counterfactual_attention_df["query_label"],
                        "neighbor_idx": head_counterfactual_attention_df["neighbor_idx"],
                        "neighbor_label": head_counterfactual_attention_df["neighbor_label"],
                        f"head_{head_idx}": attention_pattern,
                        "distance": distance,
                    }
                )
            )
        return pd.concat(counterfactual_attention_eval_dfs, axis=0)

    def plot_counterfactual_attention_summary(
        self,
        head_idx,
        distances,
        neighbor_ct_sub=None,
        palette=None,
        save_dir="./figures",
        save_svg=False,
        save_png=False,
        show=True,
        wandb_log=False,
    ):
        """
        Plot a summary of counterfactual attention patterns for the query cell type and neighbors in counterfactual_attention_df.

        Args:
            counterfactual_attention_df (pd.DataFrame): DataFrame containing counterfactual attention patterns
                as returned by `model.get_counterfactual_attention_scores()`.
            head_idx (int): Head index to plot.
            distances (list): List of distances to plot.
            neighbor_ct_sub (list, optional): List of neighbor cell types to plot.
            palette (str or list, optional): Color palette for the plot.
            save_dir (str, optional): Directory to save the plot.
            save_svg (bool, optional): Whether to save the plot as an SVG file.
            save_png (bool, optional): Whether to save the plot as a PNG file.
            show (bool, optional): Whether to show the plot.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases.
        """
        counterfactual_attention_eval_df = self.calculate_counterfactual_attention_at_distances(head_idx, distances)

        plt.figure(figsize=(10, 6))
        legend_elements = []

        if neighbor_ct_sub is None:
            neighbor_ct_sub = counterfactual_attention_eval_df["neighbor_label"].unique()

        if palette is None:
            palette = {neighbor_ct: sns.color_palette("tab10")[i] for i, neighbor_ct in enumerate(neighbor_ct_sub)}

        for neighbor_ct in neighbor_ct_sub:
            query_label = counterfactual_attention_eval_df["query_label"].unique()[0]

            neighbor_df = counterfactual_attention_eval_df[
                counterfactual_attention_eval_df["neighbor_label"] == neighbor_ct
            ]

            attention_col = f"head_{head_idx}"

            # Calculate mean and standard deviation
            grouped = neighbor_df.groupby("distance")[attention_col].agg(["mean", "std"]).reset_index()

            # Sample neighbors for background traces
            neighbor_idx_sample = np.random.choice(
                neighbor_df["neighbor_idx"].unique(),
                size=min(100, len(neighbor_df["neighbor_idx"].unique())),
                replace=False,
            )

            # Plot individual traces with the neighbor type color
            for idx in neighbor_idx_sample:
                subset = neighbor_df[neighbor_df["neighbor_idx"] == idx]
                plt.plot(
                    subset["distance"],
                    subset[attention_col],
                    color=palette[neighbor_ct],
                    alpha=0.1,
                    linewidth=0.8,
                )

            # Plot mean and confidence interval with the same color
            plt.plot(
                grouped["distance"],
                grouped["mean"],
                color=palette[neighbor_ct],
                linewidth=2,
                label="Mean",
            )
            plt.fill_between(
                grouped["distance"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                color=palette[neighbor_ct],
                alpha=0.3,
                label="±1 std",
            )

            plt.xlabel("Distance")
            plt.ylabel("Attention Score")
            plt.title(f"Counterfactual Attention Patterns for {query_label} (Head {head_idx})")

            # Create legend with matching colors
            legend_elements.append(
                mpatches.Patch(facecolor=palette[neighbor_ct], label=neighbor_ct),
            )
            legend_elements.append(mpatches.Patch(facecolor=palette[neighbor_ct], alpha=0.3, label="±1 std"))

        plt.legend(
            title="Neighbor Cell Type",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            handles=legend_elements,
        )
        plt.ylim(0, 1)
        plt.tight_layout()

        if wandb_log:
            wandb.log(
                {
                    f"Counterfactual Attention Patterns for {query_label}": wandb.Image(plt),
                }
            )
        if save_svg:
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"counterfactual_attention_patterns_head_{head_idx}_celltype_{query_label}.svg",
                )
            )
        if save_png:
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"counterfactual_attention_patterns_head_{head_idx}_celltype_{query_label}.png",
                )
            )
        if show:
            plt.show()
        plt.close()

    def _correct_length_scale_artifacts(
        self,
        length_scale_df: pd.DataFrame,
        sample_threshold: float,
    ) -> pd.DataFrame:
        """
        Check for length scales that may be artifacts due to a small fraction of receivers with senders within the length scale in the sample data.

        Args:
            length_scale_df (pd.DataFrame): The length scale distribution to correct.
            sample_threshold (float): The threshold for the number of samples to consider for the correction.

        Returns
        -------
            pd.DataFrame: The length scales with the artifacts corrected by setting length scales to 0 if there are not enough samples.
        """
        # Get the median length scales and the query label
        median_length_scale_df = pd.DataFrame(
            length_scale_df.groupby(["head_idx", "sender_type"])["length_scale"].median()
        ).reset_index()
        query_label = self._counterfactual_attention_df["query_label"].unique()[0]

        for sender_type in median_length_scale_df["sender_type"].unique():
            # Look at the sender type of interest only
            median_length_sender_df = median_length_scale_df[median_length_scale_df["sender_type"] == sender_type]
            for length_scale, head_idx in np.array(median_length_sender_df[["length_scale", "head_idx"]]):
                # Get receiver cells, indices and distances to nearest neighbors
                receiver_idxs = np.where(self._adata.obs[self._labels_key] == query_label)[0]
                nn_idxs = self._adata.obsm["_nn_idx"][receiver_idxs]
                nn_dists = self._adata.obsm["_nn_dist"][receiver_idxs]

                # Check how many nearest neighbors are there of the sender type within the length scale
                nn_labels = self._adata.obs[self._labels_key].values[rearrange(nn_idxs, "b n -> (b n)")]
                nn_labels_sender_dist = (
                    rearrange(np.array(nn_labels), "(b n) -> b n", b=nn_idxs.shape[0]) == sender_type
                ) & (nn_dists < length_scale)
                count_lt_d = nn_labels_sender_dist.sum(-1)
                count_receivers = (count_lt_d >= 1).sum()
                all_receivers = (self._adata.obs[self._labels_key] == query_label).sum()

                # Use the threshold to correct the length scale to 0 if the number of receivers is too low
                if count_receivers / all_receivers < sample_threshold:
                    length_scale_df.loc[
                        (length_scale_df["head_idx"] == head_idx) & (length_scale_df["sender_type"] == sender_type),
                        "length_scale",
                    ] = 0

        return length_scale_df

    def _calculate_length_scales(
        self,
        head_idxs: list[int],
        sender_types: list[str],
        attention_threshold: float = 0.1,
        sample_threshold: float = 0.02,
    ):
        """
        Compute the length scales for each neighbor of the query cell type for the given head indices and a set of given sender cell types.

        Args:
            head_idxs (list[int]): The head indices to analyze.
            sender_types (list[str]): The sender cell types to analyze.
            attention_threshold (float, optional): The attention threshold below which we consider the length scale. Defaults to 0.1.
            sample_threshold (float, optional): The threshold for the number of samples to consider for the correction. Defaults to 0.02.

        Returns
        -------
            pd.DataFrame: A DataFrame containing the length scales for each head for the given sender cell types.
        """
        assert (
            self._counterfactual_attention_df["query_label"].unique() not in sender_types
        ), "Sender type cannot be the same as the query label"

        length_scales_per_head = []
        for head_idx in head_idxs:
            for sender_type in sender_types:
                # Filter for head and sender type
                head_counterfactual_attention_df = self._counterfactual_attention_df.loc[
                    (self._counterfactual_attention_df["head_idx"] == head_idx)
                    & (self._counterfactual_attention_df["neighbor_label"] == sender_type)
                ]

                base_attention_score = head_counterfactual_attention_df["base_attention_score"].to_numpy()
                pos_coef = head_counterfactual_attention_df["position_coefficient"].to_numpy()
                distance_kernel_unit_scale = head_counterfactual_attention_df["distance_kernel_unit_scale"].to_numpy()

                # Calculate the length scales with attention threshold
                length_scales = (distance_kernel_unit_scale / pos_coef) * (
                    np.log((1 - attention_threshold) / attention_threshold) + base_attention_score - 3
                )

                length_scale_per_head = pd.DataFrame(
                    {
                        "head_idx": np.repeat(head_idx, len(length_scales)),
                        "sender_type": np.repeat(sender_type, len(length_scales)),
                        "length_scale": length_scales,
                        "neighbor_idx": head_counterfactual_attention_df["neighbor_idx"].to_numpy(),
                    }
                )
                length_scales_per_head.append(length_scale_per_head)

        length_scale_df = pd.concat(length_scales_per_head, axis=0)
        length_scale_df["length_scale"].clip(lower=0, inplace=True)

        # Before returning the length scales, correct for the artifact containing false length scales due to spurious counterfactual attention
        length_scale_df = self._correct_length_scale_artifacts(length_scale_df, sample_threshold)
        return length_scale_df

    def plot_length_scale_distribution(
        self,
        head_idxs: list[int],
        sender_types: list[str],
        attention_threshold: float = 0.1,
        max_length_scale: float = 50,
        sample_threshold: float = 0.02,
        palette: dict | None = None,
        plot_kde: bool = False,
        show: bool = True,
        save_png: bool = False,
        save_svg: bool = False,
        save_dir: str = "./figures",
    ):
        """Plot the distribution of length scales for each head and sender cell types based on counterfactual attention patterns.

        Args:
            head_idxs (list[int]): The head indices to analyze.
            sender_types (list[str]): The sender cell types to analyze.
            attention_threshold (float, optional): The attention threshold below which we consider the length scale. Defaults to 0.1.
            max_length_scale (float, optional): The maximum length scale to consider for the plot. Defaults to 50.
            sample_threshold (float, optional): The threshold for the number of samples to consider for the correction. Defaults to 0.02.
            palette (dict, optional): Dictionary mapping sender cell types to colors. If None, uses default seaborn palette.
            plot_kde (bool, optional): Whether to plot the KDE of the length scales. Defaults to False and plots boxplot.
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_png (bool, optional): Whether to save the plot as a PNG file. Defaults to False.
            save_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.
            save_dir (str, optional): The directory to save the plot files. Defaults to "./figures".
        """
        cell_type = self._counterfactual_attention_df["query_label"].unique()[0]
        length_scale_df = self._calculate_length_scales(head_idxs, sender_types, attention_threshold, sample_threshold)

        # Plot the distributions
        plt.figure(figsize=(12, 6))

        # Find out the order of the sender types based on the median length scale
        median_length_scale = length_scale_df.groupby(["sender_type", "head_idx"])["length_scale"].median()
        max_per_sender = median_length_scale.groupby("sender_type").max()
        sender_types_order = max_per_sender.sort_values(ascending=False).index

        if plot_kde:
            # Calculate number of rows and columns for subplot grid
            n_heads = len(head_idxs)
            n_cols = min(3, n_heads)  # Max 3 columns
            n_rows = (n_heads + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            plt.suptitle(f"Length Scale Distribution by Head and Sender Type for {cell_type}")

            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])  # Make axes iterable for single subplot
            axes = axes.flatten()

            for idx, head_idx in enumerate(head_idxs):
                length_scale_df_head = length_scale_df[length_scale_df["head_idx"] == head_idx]
                if len(length_scale_df_head) == 0:
                    continue

                sns.kdeplot(
                    data=length_scale_df_head,
                    x="length_scale",
                    palette=palette or "tab10",
                    hue="sender_type",
                    ax=axes[idx],
                )
                axes[idx].set_xlabel("Distance")
                axes[idx].set_ylabel("Density")
                axes[idx].set_xlim(0, max_length_scale)
                axes[idx].set_title(f"Head {head_idx}")

            # Remove any empty subplots
            for idx in range(len(head_idxs), len(axes)):
                fig.delaxes(axes[idx])

            # Remove legends from individual subplots to avoid duplicates
            for ax in axes:
                if ax.get_legend() is not None:
                    ax.legend_.remove()

            # Handle KDE plot legend separately
            handles_all = [mpatches.Patch(color=palette[ct], label=ct) for ct in sender_types]
            labels_all = sender_types
            fig.legend(
                handles_all,
                labels_all,
                title="Sender Type",
                bbox_to_anchor=(1.02, 0.98),
                loc="upper left",
                borderaxespad=0.0,
            )

            plt.tight_layout(rect=(0, 0, 0.85, 1))  # leave space on right for legend
        else:
            sns.violinplot(
                data=length_scale_df,
                y="head_idx",
                x="length_scale",
                hue="sender_type",
                hue_order=sender_types_order,
                palette=palette if palette is not None else "tab10",  # Use provided palette or default
            )

            # Set labels based on the provided head_idxs list
            plt.yticks(ticks=range(len(head_idxs)), labels=[f"Head {h}" for h in head_idxs])

            plt.xlabel("Attention Head")
            plt.ylabel("Distance")
            plt.ylim(-0.5, max_length_scale)
            plt.legend(title="Sender Type", bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.title(f"Length Scale Distribution by Head and Sender Type for {cell_type}")

        if save_png:
            plt.savefig(f"{save_dir}/length_scale_distribution_{cell_type}.png")
        if save_svg:
            plt.savefig(f"{save_dir}/length_scale_distribution_{cell_type}.svg")
        if show:
            plt.show()
        plt.close()
