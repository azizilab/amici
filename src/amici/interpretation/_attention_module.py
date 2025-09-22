import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from anndata import AnnData
from einops import einsum, rearrange
from scvi import REGISTRY_KEYS
from tqdm import tqdm

from amici._constants import NN_REGISTRY_KEYS

from ._utils import _get_compute_method_kwargs

if TYPE_CHECKING:
    from amici._model import AMICI


@dataclass
class AMICIAttentionModule:
    _adata: AnnData | None = None
    _labels_key: str | None = None
    _attention_patterns_df: pd.DataFrame | None = None
    _nn_idxs_df: pd.DataFrame | None = None
    _nn_dists_df: pd.DataFrame | None = None
    _compute_kwargs: dict | None = None
    _flavor: (Literal["vanilla", "value-weighted", "info-weighted", "gene-weighted"] | None) = None

    @classmethod
    def compute(
        cls,
        model: "AMICI",
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        flavor: Literal["vanilla", "value-weighted", "info-weighted", "gene-weighted"] = "vanilla",
        prog_bar: bool = True,
    ) -> "AMICIAttentionModule":
        """Different flavors of attention patterns retrieved from cache.

        Returns a DataFrame if return_nn_idxs_and_dists is False, otherwise returns a tuple of DataFrames (attention_patterns, nn_idxs, nn_dists).
        Code adapted from: https://github.com/callummcdougall/CircuitsVis/blob/main/python/circuitsvis/attention.py#L203-L211.
        einsum key:
        - b: n_batch
        - h: n_heads
        - n: n_neighbors
        - g: n_genes
        - m: n_heads * n_head_size (a.k.a. d_model)

        Args:
            adata (AnnData): AnnData object to get attention patterns from.
            indices (list[int], optional): Indices of the cells to get attention patterns from.
            batch_size (int, optional): Batch size to use for the data loader.
            flavor (Literal["vanilla", "value-weighted", "info-weighted", "gene-weighted"], optional): Flavor of attention patterns to retrieve.
            prog_bar (bool, optional): Whether to show a progress bar.

        Stores
        -------
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
                - attention_patterns: DataFrame of attention patterns with the following columns:
                    - "neighbor_{i}": Attention pattern for the i-th neighbor.
                    - "head_idx": Index of the attention head.
                    - "label": Label of the cell.
                    - "cell_idx": Index of the cell.
                - nn_idxs: DataFrame of nearest neighbor indices (only if return_nn_idxs_and_dists is True) with the following columns:
                    - "neighbor_{i}": Index of the i-th neighbor.
                - nn_dists: DataFrame of nearest neighbor distances (only if return_nn_idxs_and_dists is True) with the following columns:
                    - "neighbor_{i}": Distance to the i-th neighbor.
        """
        _compute_kwargs = _get_compute_method_kwargs(**locals())
        model._check_if_trained(warn=True)

        adata = model._validate_anndata(adata)
        if indices is None:
            indices = np.arange(len(adata))
        scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        attention_patterns = []
        nn_idxs = []
        nn_dists = []
        for tensors in tqdm(scdl, disable=not prog_bar):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            _, gen_outputs = model.module(
                tensors,
                generative_kwargs={
                    "return_attention_patterns": True,
                    "return_v": True,
                },
                compute_loss=False,
            )
            batch_attention_patterns = (gen_outputs["attention_patterns"].detach().cpu().numpy())[
                :, :, :-1
            ]  # remove query_len dim and dummy dim
            nn_idxs.append(tensors[NN_REGISTRY_KEYS.NN_IDX_KEY].detach().cpu().numpy())
            nn_dists.append(tensors[NN_REGISTRY_KEYS.NN_DIST_KEY].detach().cpu().numpy())

            if flavor in ("value-weighted", "info-weighted", "gene-weighted"):
                batch_v = gen_outputs["attention_v"].detach().cpu().numpy()  # batch x n_nns x n_heads x n_head_size
                W_O = model.module.attention_layer.W_O.detach().cpu().numpy()  # n_heads x n_head_size x d_model
                info = einsum(batch_v, W_O, "b n h d, m h d -> b h n m")
            if flavor == "gene-weighted":
                proj_linear_W = (
                    model.module.linear_head.weight.detach().cpu().numpy()
                )  # n_genes x (d_model * n_ct_embed)

            if flavor == "value-weighted":
                v_norms = rearrange(np.linalg.norm(batch_v, axis=-1), "b n h -> b h n")
                batch_attention_patterns = batch_attention_patterns * v_norms
            if flavor == "info-weighted":
                info_norms = np.linalg.norm(info, axis=-1)
                batch_attention_patterns = batch_attention_patterns * info_norms
            if flavor == "gene-weighted":
                gene_info = einsum(info, proj_linear_W, "b h n m, g m -> b h n g")
                gene_info_norms = np.linalg.norm(gene_info, axis=-1)
                batch_attention_patterns = batch_attention_patterns * gene_info_norms

            attention_patterns.append(batch_attention_patterns)

        attention_patterns_head = []
        labels_key = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        for head_idx in range(model.module.n_heads):
            attention_patterns_head_idx = np.vstack(attention_patterns)[:, head_idx, :]
            head_idx_labels = np.repeat(head_idx, attention_patterns_head_idx.shape[0]).reshape(-1, 1)
            cell_type_labels = adata[indices].obs[labels_key].values.reshape(-1, 1)
            cell_idxs = adata[indices].obs_names.to_numpy().reshape(-1, 1)
            attention_patterns_head_df = pd.DataFrame(
                np.concatenate(
                    (
                        attention_patterns_head_idx,
                        head_idx_labels,
                        cell_type_labels,
                        cell_idxs,
                    ),
                    axis=1,
                ),
                columns=[f"neighbor_{i}" for i in range(model.n_neighbors)] + ["head", "label", "cell_idx"],
            )
            attention_patterns_head.append(attention_patterns_head_df)
        attention_patterns_df = pd.concat(attention_patterns_head, ignore_index=True)

        nn_idxs_df = pd.DataFrame(
            adata.obs_names.to_numpy()[np.vstack(nn_idxs)],
            columns=[f"neighbor_{i}" for i in range(model.n_neighbors)],
            index=adata.obs_names[indices],
        )
        nn_dists_df = pd.DataFrame(
            np.vstack(nn_dists),
            columns=[f"neighbor_{i}" for i in range(model.n_neighbors)],
            index=adata.obs_names[indices],
        )
        return cls(
            _adata=adata,
            _attention_patterns_df=attention_patterns_df,
            _nn_idxs_df=nn_idxs_df,
            _nn_dists_df=nn_dists_df,
            _labels_key=labels_key,
            _flavor=flavor,
            _compute_kwargs=_compute_kwargs,
        )

    def save(self, save_path: str):
        """Save attention patterns to file"""
        self._attention_patterns_df.to_csv(save_path)
        return self

    def _bin_attention_scores(
        self,
        cell_types=None,
        max_distance=100,
        bin_size=5,
        min_bin_count=50,
        prog_bar=True,
    ):
        """
        Get binned attention scores for each cell type and head.

        Args:
            cell_types (list, optional): List of cell types to consider.
                Defaults to all cell types in `attention_patterns_df`.
            max_distance (int, optional): Maximum distance for binning.
            bin_size (int, optional): Size of each bin.
            min_bin_count (int, optional): Minimum count of bins.
            prog_bar (bool, optional): Whether to show a progress bar.
            binned_attention (bool, optional): If False, returns a pivoted attention dataframe
                with no binning.

        Returns
        -------
            pd.DataFrame: DataFrame containing binned attention scores with the columns:
                - cell_i: index of the cell
                - cell_j: index of the neighbor
                - attention: attention score
                - head: attention head
                - cluster_i: index cell type
                - cluster_j: neighbor cell type
                - distance: distance between cell_i and cell_j
                - distance_bin: distance bin
        """
        cell_types = cell_types or list(set(self._attention_patterns_df["label"].unique()))

        head_idxs = sorted(self._attention_patterns_df["head"].unique())

        filtered_attention_df_list = []
        for ct in tqdm(cell_types, desc="Cell type", disable=not prog_bar):
            attention_df_list = []
            for head_idx in tqdm(head_idxs, desc="Head index", disable=not prog_bar):
                attention_patterns_head_idx = self._attention_patterns_df[
                    (self._attention_patterns_df["head"] == head_idx) & (self._attention_patterns_df["label"] == ct)
                ]

                nn_indices_flat = self._nn_idxs_df.loc[attention_patterns_head_idx["cell_idx"]].to_numpy().flatten()
                distances_flat = self._nn_dists_df.loc[attention_patterns_head_idx["cell_idx"]].to_numpy().flatten()
                attention_values_flat = (
                    attention_patterns_head_idx.drop(["head", "label", "cell_idx"], axis=1).to_numpy().flatten()
                )

                cell_i_repeat = np.repeat(attention_patterns_head_idx["cell_idx"], self._nn_idxs_df.shape[1])

                cell_i_labels = attention_patterns_head_idx["label"].to_numpy()
                cluster_i_repeat = np.repeat(cell_i_labels, self._nn_idxs_df.shape[1])

                index_label_map = self._adata.obs[self._labels_key]
                cluster_j_flat = index_label_map.loc[nn_indices_flat].to_numpy()

                batch_df = pd.DataFrame(
                    {
                        "cell_i": cell_i_repeat,
                        "cell_j": nn_indices_flat,
                        "attention": attention_values_flat,
                        "distance": distances_flat,
                        "cluster_i": cluster_i_repeat,
                        "cluster_j": cluster_j_flat,
                    }
                )

                batch_df = batch_df.dropna(subset=["cell_j"])
                batch_df["head"] = head_idx
                attention_df_list.append(batch_df)

            if not attention_df_list:
                continue
            attention_df = pd.concat(attention_df_list, ignore_index=True)

            xticklabels = np.linspace(0, max_distance - bin_size, num=int(max_distance // bin_size))
            attention_df["distance_bin"] = pd.cut(
                attention_df["distance"],
                bins=np.linspace(0, max_distance, num=int(max_distance // bin_size) + 1),
                right=False,
                include_lowest=True,
                labels=xticklabels,
            ).astype(float)
            # filter out distance bins with low attention values
            group_counts = attention_df.groupby(["distance_bin", "cluster_j"]).size().reset_index(name="count")

            groups_to_remove = group_counts[group_counts["count"] < min_bin_count]
            mask = ~attention_df.set_index(["distance_bin", "cluster_j"]).index.isin(
                groups_to_remove.set_index(["distance_bin", "cluster_j"]).index
            )
            filtered_attention_df = attention_df[mask].reset_index(drop=True)
            filtered_attention_df_list.append(filtered_attention_df)

        return pd.concat(filtered_attention_df_list, ignore_index=True)

    def plot_attention_summary(
        self,
        cell_type_sub=None,
        sel_head=None,
        plot_histogram=False,
        palette=None,
        max_distance=100,
        bin_size=5,
        min_bin_count=50,
        epoch=None,
        wandb_log=False,
        show=True,
        save_png=False,
        save_svg=False,
        save_dir="./figures",
    ):
        """
        Plot a summary of attention patterns for different cell types and heads.

        Args:
            cell_types (list, optional): List of cell types to consider. Defaults to None.
            sel_head (int, optional): Selected head index. Defaults to None.
            plot_histogram (bool, optional): Whether to plot a histogram of attention scores. Defaults to False.
            flavor (str, optional): Flavor of attention to plot. Defaults to "vanilla".
            palette (str or list, optional): Color palette for the plot. Defaults to None.
            max_distance (int, optional): Maximum distance for binning. Defaults to 100.
            bin_size (int, optional): Size of each bin. Defaults to 5.
            min_bin_count (int, optional): Minimum count of bins. Defaults to 50.
            epoch (int, optional): Epoch number for logging. Defaults to None.
            wandb_log (bool, optional): Whether to log the plot to Weights and Biases. Defaults to True.
            show (bool, optional): Whether to show the plot. Defaults to False.
            save_png (bool, optional): Whether to save the plot as a PNG file. Defaults to False.
            save_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.
            save_dir (str, optional): Directory to save the plot. Defaults to "./figures".

        Returns
        -------
            None
        """
        cell_types = cell_type_sub or list(set(self._attention_patterns_df["label"].unique()))
        binned_attention_df = self._bin_attention_scores(
            cell_types=cell_types,
            max_distance=max_distance,
            bin_size=bin_size,
            min_bin_count=min_bin_count,
            prog_bar=True,
        )
        max_attention_value = binned_attention_df["attention"].max()
        for ct in cell_types:
            binned_attention_ct_df = binned_attention_df[binned_attention_df["cluster_i"] == ct]

            xticklabels = np.linspace(0, max_distance - bin_size, num=int(max_distance // bin_size))
            if sel_head is not None:
                # plot single head
                binned_attention_ct_head_df = binned_attention_ct_df[binned_attention_ct_df["head"] == sel_head]
                sns.set_theme(style="whitegrid")
                if plot_histogram:
                    fig, (ax1, ax2) = plt.subplots(
                        2,
                        1,
                        figsize=(10, 6),
                        sharex=True,
                        gridspec_kw={"height_ratios": [3, 1]},
                    )
                else:
                    fig, (ax1) = plt.subplots(
                        1,
                        1,
                        figsize=(10, 6),
                    )

                # Attention plot
                g = sns.pointplot(
                    data=binned_attention_ct_head_df,
                    x="distance_bin",
                    y="attention",
                    hue="cluster_j",
                    errorbar="sd",
                    native_scale=True,
                    palette=palette or "tab10",
                    ax=ax1,
                )
                g.set(xlabel="Distance Bin", ylabel="Attention")
                g.set_xticklabels(
                    [f"{x:.2f}".rstrip("0").rstrip(".") for x in xticklabels],
                    rotation=45,
                    ha="right",
                )
                ax1.set_title(
                    f"{self._flavor.capitalize()} Attention Patterns for Index Cell Type {ct} for Head {sel_head}, Binned by Distance",
                    size=16,
                    pad=10,
                )
                ax1.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
                if self._flavor != "vanilla":
                    ax1.set_ylim(0, max_attention_value)
                else:
                    ax1.set_ylim(0, 1)

                if plot_histogram:
                    # Histogram plot
                    num_bins = int(max_distance // (bin_size / 4))
                    sns.histplot(
                        data=binned_attention_ct_head_df[binned_attention_ct_head_df["distance"] < max_distance],
                        x="distance",
                        bins=num_bins,
                        hue="cluster_j",
                        palette=palette or "tab10",
                        multiple="stack",
                        ax=ax2,
                        legend=False,
                    )
                    ax1.set_xlabel("")
                    ax2.set(xlabel="Distance Bin", ylabel="Number of Edges")
                    ax2.set_xticks(xticklabels)
                    ax2.set_xticklabels(
                        [f"{x:.2f}".rstrip("0").rstrip(".") for x in xticklabels],
                        rotation=45,
                        ha="right",
                    )

                plt.tight_layout()

                if wandb_log:
                    wandb.log(
                        {
                            "epoch": epoch,
                            f"{self._flavor.capitalize()} Attention vs. Distance for Index Cell {ct}": wandb.Image(plt),
                        }
                    )
                if save_svg:
                    plt.savefig(os.path.join(save_dir, f"attn_{ct}_head_{sel_head}.svg"))
                if save_png:
                    plt.savefig(os.path.join(save_dir, f"attn_{ct}_head_{sel_head}.png"))
                if show:
                    plt.show()
                plt.close()
            else:
                # plot facetgrid for all heads
                sns.set_theme(style="whitegrid")
                g = sns.FacetGrid(
                    binned_attention_ct_df,
                    col="head",
                    col_wrap=4,
                    height=4,
                    aspect=1,
                    sharex=True,
                    sharey=True,
                    hue="cluster_j",
                    palette=palette or "tab10",
                )
                g.map_dataframe(
                    sns.pointplot,
                    "distance_bin",
                    "attention",
                    errorbar="sd",
                    native_scale=True,
                )
                g.set_titles("Head {col_name}")
                g.set_axis_labels("Distance Bin", "Attention")
                for ax in g.axes.flat:
                    ax.set_xticks(xticklabels)
                    ax.set_xticklabels(
                        [f"{x:.2f}".rstrip("0").rstrip(".") for x in xticklabels],
                        rotation=45,
                        ha="right",
                    )

                g.figure.suptitle(
                    f"{self._flavor.capitalize()} Attention Patterns for Index Cell Type {ct} per Head, Binned by Distance",
                    size=16,
                )
                g.figure.subplots_adjust(top=0.85)
                g.add_legend(title="Cluster")
                if self._flavor != "vanilla":
                    plt.ylim(0, max_attention_value)
                else:
                    plt.ylim(0, 1)
                plt.tight_layout()

                if wandb_log:
                    wandb.log(
                        {
                            "epoch": epoch,
                            f"{self._flavor.capitalize()} Attention vs. Distance for Index Cell {ct}": wandb.Image(plt),
                        }
                    )
                if save_svg:
                    plt.savefig(os.path.join(save_dir, f"attn_{ct}.svg"))
                if save_png:
                    plt.savefig(os.path.join(save_dir, f"attn_{ct}.png"))
                if show:
                    plt.show()
                plt.close()
