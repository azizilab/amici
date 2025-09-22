from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset

try:
    from anndata._core.sparse_dataset import SparseDataset
except ImportError:
    # anndata >= 0.10.0
    from anndata._core.sparse_dataset import (
        BaseCompressedSparseDataset as SparseDataset,
    )

from scvi import REGISTRY_KEYS
from scvi.data import AnnTorchDataset

from ._constants import NN_REGISTRY_KEYS

if TYPE_CHECKING:
    from ._manager import AnnDataManager


def _registry_key_to_default_dtype(key: str) -> type:
    """Returns the default dtype for a given registry key."""
    if key in [
        REGISTRY_KEYS.BATCH_KEY,
        REGISTRY_KEYS.LABELS_KEY,
        REGISTRY_KEYS.CAT_COVS_KEY,
        REGISTRY_KEYS.INDICES_KEY,
        NN_REGISTRY_KEYS.NN_IDX_KEY,
        NN_REGISTRY_KEYS.NN_LABELS_KEY,
    ]:
        return np.int64

    return np.float32


def convert_getitem_tensors_to_dict(getitem_tensors):
    if isinstance(getitem_tensors, list):
        return {key: _registry_key_to_default_dtype(key) for key in getitem_tensors}
    else:
        return getitem_tensors


def add_getitem_tensor(getitem_tensors, key):
    getitem_tensors[key] = _registry_key_to_default_dtype(key)


def remove_getitem_tensor(getitem_tensors, key):
    del getitem_tensors[key]


class SpatialAnnTorchDataset(Dataset):
    """
    Wrapper of AnnTorchDataset for Spatial Data.

    Wrapper of AnnTorchDataset that treats REGISTRY_KEYS.NN_IDX_KEY
    differently. Instead of loading the tensor of indices, loads the
    gene expression values and labels at the respective indices.
    """

    def __init__(
        self,
        adata_manager: "AnnDataManager",
        getitem_tensors: Optional[Union[list, dict[str, type]]] = None,
    ):
        super().__init__()

        self.adata_manager = adata_manager

        getitem_tensors = convert_getitem_tensors_to_dict(getitem_tensors)

        self.nn_getitem_tensors = {}
        if getitem_tensors is not None:
            self.orig_getitem_tensors = deepcopy(getitem_tensors)
            if NN_REGISTRY_KEYS.NN_X_KEY in getitem_tensors:
                add_getitem_tensor(self.nn_getitem_tensors, NN_REGISTRY_KEYS.NN_X_KEY)
                remove_getitem_tensor(getitem_tensors, NN_REGISTRY_KEYS.NN_X_KEY)
                add_getitem_tensor(getitem_tensors, NN_REGISTRY_KEYS.NN_IDX_KEY)
                add_getitem_tensor(getitem_tensors, REGISTRY_KEYS.X_KEY)
            if NN_REGISTRY_KEYS.NN_LABELS_KEY in getitem_tensors:
                add_getitem_tensor(self.nn_getitem_tensors, NN_REGISTRY_KEYS.NN_LABELS_KEY)
                remove_getitem_tensor(getitem_tensors, NN_REGISTRY_KEYS.NN_LABELS_KEY)
                add_getitem_tensor(getitem_tensors, NN_REGISTRY_KEYS.NN_IDX_KEY)
                add_getitem_tensor(getitem_tensors, REGISTRY_KEYS.LABELS_KEY)
        else:
            registered_keys = adata_manager.data_registry.keys()
            if NN_REGISTRY_KEYS.NN_IDX_KEY in registered_keys:
                if REGISTRY_KEYS.X_KEY in registered_keys:
                    add_getitem_tensor(self.nn_getitem_tensors, NN_REGISTRY_KEYS.NN_X_KEY)
                if REGISTRY_KEYS.LABELS_KEY in registered_keys:
                    add_getitem_tensor(self.nn_getitem_tensors, NN_REGISTRY_KEYS.NN_LABELS_KEY)
            getitem_tensors = convert_getitem_tensors_to_dict(list(registered_keys))
            self.orig_getitem_tensors = deepcopy(getitem_tensors)
            self.orig_getitem_tensors.update(convert_getitem_tensors_to_dict(list(self.nn_getitem_tensors.keys())))

        self.anntorchdataset = AnnTorchDataset(adata_manager, getitem_tensors)

    def __len__(self):
        return len(self.anntorchdataset)

    def __getitem__(self, indexes: list[int]) -> dict[str, np.ndarray | torch.Tensor]:
        data_map = self.anntorchdataset[indexes]

        if len(self.nn_getitem_tensors) == 0:
            return data_map

        if isinstance(indexes, int):
            indexes = [indexes]  # force batched single observations

        if self.adata_manager.adata.isbacked and isinstance(indexes, (list, np.ndarray)):
            # need to sort indexes for h5py datasets
            indexes = np.sort(indexes)

        assert NN_REGISTRY_KEYS.NN_IDX_KEY in data_map
        nn_idxs = data_map[NN_REGISTRY_KEYS.NN_IDX_KEY]
        for key, dtype in self.nn_getitem_tensors.items():
            if key == NN_REGISTRY_KEYS.NN_X_KEY:
                data_key = REGISTRY_KEYS.X_KEY
            elif key == NN_REGISTRY_KEYS.NN_LABELS_KEY:
                data_key = REGISTRY_KEYS.LABELS_KEY
            else:
                raise AssertionError

            data = self.anntorchdataset.data[data_key]
            if isinstance(data, (np.ndarray, h5py.Dataset)):
                sliced_data = data[nn_idxs].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = np.array(
                    [data.iloc[nn_idx, :].to_numpy().astype(dtype, copy=False) for nn_idx in nn_idxs]
                )
            elif issparse(data) or isinstance(data, SparseDataset):
                # sparse matrices do not support 2d indexing
                sliced_data = np.concatenate(
                    [data[nn_idx].astype(dtype, copy=False).toarray()[None, :] for nn_idx in nn_idxs],
                    axis=0,
                )
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        filtered_data_map = {key: value for key, value in data_map.items() if key in self.orig_getitem_tensors}

        return filtered_data_map

    def get_data(self, scvi_data_key: str):
        return self.anntorchdataset.get_data(scvi_data_key)
