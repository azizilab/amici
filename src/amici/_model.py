import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from einops import einsum, rearrange, repeat
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    ObsmField,
)
from scvi.model.base import BaseModelClass, VAEMixin
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ._anndataloader import SpatialAnnDataLoader
from ._constants import NN_REGISTRY_KEYS
from ._module import AMICIModule
from ._utils import wraps_compute
from ._wandb_training_mixin import WandbUnsupervisedTrainingMixin
from .interpretation import (
    AMICIAblationModule,
    AMICIAttentionModule,
    AMICICounterfactualAttentionModule,
    AMICIExplainedVarianceModule,
)

DEFAULT_N_NEIGHBORS = 30


class AMICI(VAEMixin, WandbUnsupervisedTrainingMixin, BaseModelClass):
    _module_cls = AMICIModule
    _data_loader_cls = SpatialAnnDataLoader

    def __init__(
        self,
        adata: AnnData,
        **model_kwargs,
    ):
        # Hack to ensure the UnsupervisedTrainingMixin uses the SpatialAnnDataLoader
        self._data_splitter_cls.data_loader_cls = self._data_loader_cls

        super().__init__(adata)

        self.n_neighbors = adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY]

        empirical_ct_means = []
        dataset_x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        dataset_labels = self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        for label_idx in range(self.summary_stats.n_labels):
            ct_idxs = np.where(dataset_labels == label_idx)[0]
            ct_means = torch.tensor(dataset_x[ct_idxs].mean(0))
            empirical_ct_means.append(ct_means)
        empirical_ct_means = torch.stack(empirical_ct_means)

        self.module = self._module_cls(
            n_genes=adata.n_vars,
            n_labels=self.summary_stats.n_labels,
            empirical_ct_means=empirical_ct_means,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    @staticmethod
    def _compute_nn(
        adata: AnnData,
        coord_obsm_key: str,
        index_key: str,
        dist_key: str,
        n_neighbors: int,
        labels_obs_key: str | None = None,
        cell_radius_key: str | None = None,
        exclude_self_labels: bool = True,
    ) -> None:
        assert not exclude_self_labels or labels_obs_key is not None
        adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY] = n_neighbors
        coords = adata.obsm[coord_obsm_key]
        if isinstance(coords, pd.DataFrame):
            coords = coords.values

        if not exclude_self_labels:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(
                coords
            )
            nn_dist, nn_idx = nn.kneighbors(coords, return_distance=True)
            # prune self
            nn_dist, nn_idx = nn_dist[:, 1:], nn_idx[:, 1:]

            if cell_radius_key is not None:
                cell_radii = adata.obs[cell_radius_key].values
                nn_radii = cell_radii[nn_idx]
                nn_dist = np.clip(
                    nn_dist - cell_radii.repeat(n_neighbors, axis=1) - nn_radii, 0, None
                )

            adata.obsm[index_key] = nn_idx
            adata.obsm[dist_key] = nn_dist
        else:
            labels = adata.obs[labels_obs_key].values
            adata.obsm[index_key] = np.zeros((adata.n_obs, n_neighbors), dtype=int)
            adata.obsm[dist_key] = np.zeros((adata.n_obs, n_neighbors), dtype=float)
            for label in np.unique(labels):
                label_idxs = np.where(labels == label)[0]
                not_label_idxs = np.where(labels != label)[0]
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(
                    coords[not_label_idxs]
                )
                nn_dist, nn_idx = nn.kneighbors(
                    coords[label_idxs], return_distance=True
                )
                remapped_nn_idx = not_label_idxs[nn_idx]

                if cell_radius_key is not None:
                    cell_radii = adata.obs[cell_radius_key].values[label_idxs]
                    nn_radii = adata.obs[cell_radius_key].values[remapped_nn_idx]
                    cell_radii_repeated = repeat(cell_radii, "b -> b n", n=n_neighbors)
                    nn_dist = np.clip(nn_dist - cell_radii_repeated - nn_radii, 0, None)

                adata.obsm[index_key][label_idxs] = remapped_nn_idx
                adata.obsm[dist_key][label_idxs] = nn_dist

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        labels_key: str | None = None,
        coord_obsm_key: str | None = None,
        nn_dist_key: str = "_nn_dist",
        nn_idx_key: str = "_nn_idx",
        cell_radius_key: str | None = None,
        n_neighbors: int | None = None,
        **kwargs,
    ):
        if n_neighbors is None:
            if NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY in adata.uns:
                n_neighbors = adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY]
            else:
                n_neighbors = DEFAULT_N_NEIGHBORS
        setup_method_args = cls._get_setup_method_args(**locals())

        cls._compute_nn(
            adata,
            coord_obsm_key,
            nn_idx_key,
            nn_dist_key,
            n_neighbors,
            labels_obs_key=labels_key,
            exclude_self_labels=True,
            cell_radius_key=cell_radius_key,
        )

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ObsmField(NN_REGISTRY_KEYS.COORD_KEY, coord_obsm_key),
            ObsmField(NN_REGISTRY_KEYS.NN_IDX_KEY, nn_idx_key),
            ObsmField(NN_REGISTRY_KEYS.NN_DIST_KEY, nn_dist_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @wraps_compute(AMICIAttentionModule)
    def get_attention_patterns(
        self, adata: AnnData | None = None, **kwargs
    ) -> AMICIAttentionModule:
        """See AMICIAttentionModule.compute for more details."""
        return AMICIAttentionModule.compute(self, adata, **kwargs)

    @wraps_compute(AMICICounterfactualAttentionModule)
    def get_counterfactual_attention_patterns(
        self,
        cell_type: str,
        adata: AnnData | None = None,
        **kwargs,
    ) -> AMICICounterfactualAttentionModule:
        """See AMICICounterfactualAttentionModule.compute for more details."""
        return AMICICounterfactualAttentionModule.compute(
            self, cell_type, adata, **kwargs
        )

    @wraps_compute(AMICIExplainedVarianceModule)
    def get_expl_variance_scores(
        self,
        adata: AnnData | None = None,
        **kwargs,
    ) -> AMICIExplainedVarianceModule:
        """See AMICIExplainedVarianceModule.compute for more details."""
        return AMICIExplainedVarianceModule.compute(self, adata, **kwargs)

    @wraps_compute(AMICIAblationModule)
    def get_neighbor_ablation_scores(
        self,
        cell_type: str | None = None,
        head_idx: int | None = None,
        adata: AnnData | None = None,
        **kwargs,
    ) -> AMICIAblationModule:
        """See AMICIAblationModule.compute for more details."""
        return AMICIAblationModule.compute(
            self,
            cell_type,
            head_idx,
            adata=adata,
            **kwargs,
        )

    def get_predictions(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        get_residuals: bool = False,
        prog_bar: bool = True,
    ) -> np.ndarray:
        self._check_if_trained(warn=True)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        out = []
        for tensors in tqdm(scdl, disable=not prog_bar):
            _, outputs = self.module(tensors, compute_loss=False)
            residuals, predictions = (
                outputs["residual"].detach().cpu().numpy(),
                outputs["prediction"].detach().cpu().numpy(),
            )
            if get_residuals:
                out.append(residuals)
            else:
                out.append(predictions)
        return np.vstack(out)

    def get_nn_embed(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        self._check_if_trained(warn=True)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        if indices is None:
            indices = list(range(adata.n_obs))

        dummy_labels_tensor = torch.zeros(1, 1, dtype=int).to(self.device)

        nn_embeds = []
        for neighbor_tensors in scdl:
            nn_X = neighbor_tensors[REGISTRY_KEYS.X_KEY].unsqueeze(0).to(self.device)

            inf_outputs = self.module.inference(
                dummy_labels_tensor,
                nn_X,
            )
            nn_embeds.append(inf_outputs["nn_embed"][0].cpu().detach().numpy())
        return np.concatenate(nn_embeds, axis=0)

    def get_gene_residual_contributions(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        head_idxs: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Get the gene residual contributions for each cell at full attention.

        Compute the gene-wise residual contributions for each cell irrespective of
        the attention score for the head. As the value vectors do not depend on the
        distances or the index cell, we only need to provide the neighbor gene expressions.

        Args:
            adata: The AnnData object.
            indices: The indices of the cells to get gene residual contributions for.
            batch_size: The batch size.
            head_idxs: The indices of the heads to get gene residual contributions for.

        Returns
        -------
            pd.DataFrame: A DataFrame with the gene residual contributions for each cell at full attention.
            The DataFrame has the columns:
                - neighbor: The index of the neighbor cell.
                - head: The index of the head.
                - {gene}: The gene residual contribution for the gene.
        """
        self._check_if_trained(warn=True)

        head_idxs = (
            head_idxs if head_idxs is not None else list(range(self.module.n_heads))
        )

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if indices is None:
            indices = list(range(adata.n_obs))

        dummy_labels_tensor = torch.zeros(1, 1, dtype=int).to(self.device)

        residual_contributions_df_rows = []
        cells_processed = 0
        for neighbor_tensors in scdl:
            dummy_nn_dist = torch.full(
                (1, neighbor_tensors[REGISTRY_KEYS.X_KEY].shape[0]), 1
            ).to(self.device)
            nn_X = neighbor_tensors[REGISTRY_KEYS.X_KEY].unsqueeze(0).to(self.device)

            inf_outputs = self.module.inference(
                dummy_labels_tensor,
                nn_X,
            )

            gen_outputs = self.module.generative(
                dummy_labels_tensor,
                inf_outputs["label_embed"],
                inf_outputs["nn_embed"],
                dummy_nn_dist,
                return_v=True,
            )
            v = gen_outputs["attention_v"][0]  # n_neighbors x head x d_head

            # pass value embeds through final layers
            W_O = self.module.attention_layer.W_O
            attn_outs = einsum(
                v,
                W_O,
                "neighbor head_index d_head, d_model head_index d_head -> neighbor head_index d_model",
            )
            attn_outs = self.module.attention_layer.norm_o(
                attn_outs
            )  # Layer norm w/ residual is same as w/o
            residual_contributions = self.module.linear_head(
                attn_outs
            )  # neighbor x heads x genes
            residual_contributions = residual_contributions[:, head_idxs, :]

            neighbor_indices = np.array(indices)[
                cells_processed : cells_processed + nn_X.shape[1]
            ]
            cells_processed += nn_X.shape[1]

            n_neighbors, n_heads, n_genes = residual_contributions.shape
            df = pd.DataFrame(
                rearrange(
                    residual_contributions.cpu().detach().numpy(),
                    "neighbor head gene -> (neighbor head) gene",
                ),
                columns=adata.var_names,
            )
            df["neighbor"] = np.repeat(neighbor_indices, n_heads)
            df["head"] = np.tile(head_idxs, n_neighbors)
            residual_contributions_df_rows.append(df)
        return pd.concat(residual_contributions_df_rows)
