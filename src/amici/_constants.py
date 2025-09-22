from typing import NamedTuple


class _NN_REGISTRY_KEYS_NT(NamedTuple):
    COORD_KEY: str = "coordinates"
    NN_IDX_KEY: str = "nn_idx"
    NN_DIST_KEY: str = "nn_dist"
    NN_X_KEY: str = "nn_x"
    NN_LABELS_KEY: str = "nn_labels"
    NUM_NEIGHBORS_KEY: str = "n_neighbors"


NN_REGISTRY_KEYS = _NN_REGISTRY_KEYS_NT()
