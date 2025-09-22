import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp_sparse

try:
    # anndata >= 0.10
    from anndata.experimental import CSCDataset, CSRDataset

    SparseDataset = (CSRDataset, CSCDataset)
except ImportError:
    from anndata._core.sparse_dataset import SparseDataset


def is_count_data(
    data: pd.DataFrame | npt.NDArray | sp_sparse.spmatrix | h5py.Dataset,
    n_to_check: int = 20,
):
    """
    Source: SCVI data utils (https://github.com/scverse/scvi-tools/blob/main/src/scvi/data/_utils.py#L254-L279)

    Approximately checks if the data to ensure it is count data.

    Args:
        data (pd.DataFrame | npt.NDArray | sp_sparse.spmatrix | h5py.Dataset):
            The data to check if it is count data. It can be a pandas DataFrame,
            numpy array, scipy sparse matrix, or h5py Dataset.
        n_to_check (int, optional):
            The number of samples to check from the data. Defaults to 20.

    Returns
    -------
        bool:
            True if the data is count data, False otherwise.

    Raises
    ------
        TypeError:
            If the data type is not understood.
    """
    if isinstance(data, h5py.Dataset) or isinstance(data, SparseDataset):
        data = data[:100]

    if isinstance(data, np.ndarray):
        data = data
    elif issubclass(type(data), sp_sparse.spmatrix):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError("data type not understood")

    ret = True
    if data.shape[0] != 0:
        inds = np.random.choice(data.shape[0], size=(n_to_check,))
        check = data[inds]
        negative = np.any(check < 0)
        non_integer = np.any(check % 1 != 0)
        ret = not (negative or non_integer)
    return ret
