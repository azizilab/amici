import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from amici import AMICI


@pytest.fixture
def mock_adata():
    # Create mock data
    n_cells = 100
    n_genes = 50
    cell_types = ["Type1", "Type2", "Type3"]
    np.random.seed(42)
    X = np.random.randint(1, 1000, size=(n_cells, n_genes))
    X = csr_matrix(X)
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_cells),
        }
    )
    spatial_coords = np.column_stack((np.random.rand(n_cells), np.random.rand(n_cells)))
    cell_radii = np.random.uniform(1, 4, size=n_cells)
    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial_coords
    adata.obs["cell_radius"] = cell_radii
    return adata


@pytest.mark.parametrize("cell_radius_key", [None, "cell_radius"])
@pytest.mark.parametrize("n_neighbors", [15, 30])
def test_model_setup(mock_adata, cell_radius_key, n_neighbors):
    if cell_radius_key is not None:
        mock_adata_no_radius = mock_adata.copy()

    AMICI.setup_anndata(
        mock_adata,
        labels_key="cell_type",
        coord_obsm_key="spatial",
        cell_radius_key=cell_radius_key,
        n_neighbors=n_neighbors,
    )
    assert mock_adata.obsm["_nn_dist"].shape == (
        mock_adata.n_obs,
        n_neighbors,
    ), "The shape of '_nn_dist' does not match the number of cells and neighbors"

    assert mock_adata.obsm["_nn_idx"].shape == (
        mock_adata.n_obs,
        n_neighbors,
    ), "The shape of '_nn_idx' does not match the number of cells and neighbors"

    if cell_radius_key is not None:
        AMICI.setup_anndata(
            mock_adata_no_radius,
            labels_key="cell_type",
            coord_obsm_key="spatial",
            n_neighbors=n_neighbors,
        )
        assert not np.any(mock_adata.obsm["_nn_dist"] < 0), "The distances to nearest neighbors are less than 0"
        assert not np.any(
            mock_adata_no_radius.obsm["_nn_dist"] < mock_adata.obsm["_nn_dist"]
        ), "The distances to nearest neighbors are greater than those without cell radius adjustment"


@pytest.mark.parametrize("n_heads", [1, 4, 8])
@pytest.mark.parametrize("n_label_embed", [1, 8])
def test_model_dims(
    mock_adata,
    n_heads,
    n_label_embed,
):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    n_obs, n_vars = mock_adata.n_obs, mock_adata.n_vars
    n_nns = mock_adata.obsm["_nn_idx"].shape[1]
    model = AMICI(
        mock_adata,
        n_heads=n_heads,
        n_label_embed=n_label_embed,
    )
    model.train(1)

    # Test for get_predictions method
    predictions = model.get_predictions()
    assert predictions.shape == (
        n_obs,
        n_vars,
    ), "The shape of 'prediction' does not match the number of observations and genes"

    # Test for get_attention_patterns method
    attention_module = model.get_attention_patterns()
    assert (
        attention_module._attention_patterns_df.shape
        == (
            n_obs * model.module.n_heads,
            n_nns + 3,
        )
    ), "The shape of 'attention_patterns' does not match the number of observations, number of heads, and number of nearest neighbors"


@pytest.mark.parametrize("value_l1_penalty_coef", [0.0, 1.0])
@pytest.mark.parametrize("attention_penalty_coef", [0.0, 1.0])
def test_model_bools(
    mock_adata,
    attention_penalty_coef,
    value_l1_penalty_coef,
):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    n_obs, n_vars = mock_adata.n_obs, mock_adata.n_vars
    n_nns = mock_adata.obsm["_nn_idx"].shape[1]
    model = AMICI(
        mock_adata,
        attention_penalty_coef=attention_penalty_coef,
        value_l1_penalty_coef=value_l1_penalty_coef,
    )
    model.train(1)
    model.get_elbo(mock_adata)

    # Test for get_predictions method
    predictions = model.get_predictions()
    assert predictions.shape == (
        n_obs,
        n_vars,
    ), "The shape of 'prediction' does not match the number of observations and genes"

    # Test for get_attention_patterns method
    attention_module = model.get_attention_patterns()
    assert (
        attention_module._attention_patterns_df.shape
        == (
            n_obs * model.module.n_heads,
            n_nns + 3,
        )
    ), "The shape of 'attention_patterns' does not match the number of observations, number of heads, and number of nearest neighbors"


@pytest.mark.parametrize("flavor", ["vanilla", "value-weighted", "info-weighted", "gene-weighted"])
def test_attention_patterns(mock_adata, flavor):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    n_obs = mock_adata.n_obs
    n_nns = mock_adata.obsm["_nn_idx"].shape[1]
    model = AMICI(
        mock_adata,
    )
    model.train(1)

    attention_module = model.get_attention_patterns(flavor=flavor)
    assert attention_module._attention_patterns_df.shape == (
        n_obs * model.module.n_heads,
        n_nns + 3,
    ), f"The shape of 'attention_patterns' with flavor '{flavor}' does not match the expected shape"

    attention_module = model.get_attention_patterns(flavor=flavor, indices=np.arange(4))
    assert attention_module._attention_patterns_df.shape == (
        4 * model.module.n_heads,
        n_nns + 3,
    ), f"The shape of 'attention_patterns' with flavor '{flavor}' does not match the expected shape"
    attention_module.plot_attention_summary(show=False)


@pytest.mark.parametrize("ablated_neighbor_ct_sub", [None, ["Type1", "Type2"], ["Type2", "Type3"]])
@pytest.mark.parametrize("cell_type", ["Type1", "Type2"])
@pytest.mark.parametrize("head_idx", [1, None])
def test_neighbor_ablation_scores(mock_adata, ablated_neighbor_ct_sub, cell_type, head_idx):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    n_vars = mock_adata.n_vars
    model = AMICI(
        mock_adata,
    )
    model.train(1)

    neighbor_ablation_scores = model.get_neighbor_ablation_scores(
        cell_type=cell_type,
        head_idx=head_idx,
        adata=mock_adata,
        ablated_neighbor_ct_sub=ablated_neighbor_ct_sub,
    )
    n_cell_types = (
        len(list(mock_adata.obs["cell_type"].unique()))
        if ablated_neighbor_ct_sub is None
        else len(ablated_neighbor_ct_sub)
    )
    assert neighbor_ablation_scores._ablation_scores_df.shape == (
        n_vars,
        n_cell_types * 2 + 5,
    ), "The shape of 'ct_ablated_neighbor_scores' does not match the expected shape"

    neighbor_ablation_scores = model.get_neighbor_ablation_scores(
        cell_type=cell_type,
        head_idx=head_idx,
        adata=mock_adata,
        ablated_neighbor_ct_sub=ablated_neighbor_ct_sub,
        compute_z_value=True,
    )
    assert neighbor_ablation_scores._ablation_scores_df.shape == (
        n_vars,
        n_cell_types * 5 + 5,
    ), "The shape of 'ct_ablated_neighbor_scores' does not match the expected shape"


def test_expl_variance_scores(mock_adata):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    n_vars = mock_adata.n_vars
    model = AMICI(
        mock_adata,
    )
    model.train(1)

    expl_variance_scores = model.get_expl_variance_scores(
        adata=mock_adata,
        run_permutation_test=False,
    )
    assert expl_variance_scores._explained_variance_df.shape == (
        len(mock_adata.obs["cell_type"].unique()) * model.module.n_heads * n_vars,
        5,
    ), "The shape of 'expl_variance_scores' does not match the expected shape"

    expl_variance_scores = model.get_expl_variance_scores(
        adata=mock_adata,
        run_permutation_test=True,
    )
    assert expl_variance_scores._explained_variance_df.shape == (
        len(mock_adata.obs["cell_type"].unique()) * model.module.n_heads * n_vars,
        7,
    ), "The shape of 'expl_variance_scores' does not match the expected shape"
    expl_variance_scores.plot_explained_variance_barplot(show=False)


def test_counterfactual_attention_patterns(mock_adata):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    model = AMICI(mock_adata)
    model.train(1)
    expected_n_obs = mock_adata[mock_adata.obs["cell_type"] != "Type1"].n_obs
    counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
        "Type1",
    )
    assert counterfactual_attention_patterns._counterfactual_attention_df.shape == (
        model.module.n_heads * expected_n_obs,
        8,
    )

    distances = [0.5, 1.0, 1.5]
    counterfactual_attention_eval_patterns = (
        counterfactual_attention_patterns.calculate_counterfactual_attention_at_distances(
            head_idx=0,
            distances=distances,
        )
    )
    assert counterfactual_attention_eval_patterns.shape == (
        expected_n_obs * len(distances),
        5,
    )
    counterfactual_attention_patterns.plot_counterfactual_attention_summary(0, distances, show=False)

    head_idxs = [0, 1, 2]
    counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
        "Type1",
        head_idxs=head_idxs,
    )
    assert counterfactual_attention_patterns._counterfactual_attention_df.shape == (
        model.module.n_heads * expected_n_obs,
        8,
    )

    nn_indices = [0, 1, 2]
    counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
        "Type1",
        indices=nn_indices,
    )
    assert counterfactual_attention_patterns._counterfactual_attention_df.shape == (
        model.module.n_heads * len(nn_indices),
        8,
    )

    counterfactual_attention_patterns.plot_length_scale_distribution(
        head_idxs=head_idxs,
        sender_types=["Type2", "Type3"],
        attention_threshold=0.1,
        show=False,
    )

    try:
        counterfactual_attention_patterns.plot_length_scale_distribution(
            head_idxs=head_idxs,
            sender_types=["Type1", "Type2"],
            attention_threshold=0.1,
            show=False,
        )
    except AssertionError as e:
        assert str(e) == "Sender type cannot be the same as the query label"


def test_gene_residual_contributions(mock_adata):
    AMICI.setup_anndata(mock_adata, labels_key="cell_type", coord_obsm_key="spatial")
    model = AMICI(mock_adata)
    model.train(1)
    residual_contributions_df = model.get_gene_residual_contributions()
    assert residual_contributions_df.shape == (
        mock_adata.n_obs * model.module.n_heads,
        mock_adata.n_vars + 2,
    )

    head_idx = [0, 1, 2]
    residual_contributions_df = model.get_gene_residual_contributions(head_idxs=head_idx)
    assert residual_contributions_df.shape == (
        mock_adata.n_obs * len(head_idx),
        mock_adata.n_vars + 2,
    )
