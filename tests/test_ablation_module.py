import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from amici.interpretation._ablation_module import AMICIAblationModule

NEAR, FAR = 20.0, 100.0


@pytest.fixture
def contact_zone_adata():
    """Receivers R and senders S on a line: a contact zone, far receivers and far senders.

    Genes, by where they are expressed:
      induced, filler  interacting receivers only (a receiver response; passes every comparison)
      spillover        senders at the contact zone, and less in the receivers they border
                       (passes the two Methods comparisons, fails the near-sender one)
      sender_marker    every sender, and less in interacting receivers (fails near vs far senders)
    """
    rng = np.random.default_rng(0)
    n = 20
    groups = {
        "far_receiver": ("R", -200.0 + np.arange(n)),
        "near_receiver": ("R", np.arange(n, dtype=float)),
        "near_sender": ("S", 0.5 + np.arange(n)),
        "far_sender": ("S", 300.0 + np.arange(n)),
    }
    expression = {
        "induced": {"near_receiver": 3.0},
        "filler": {"near_receiver": 2.0},
        "spillover": {"near_sender": 3.0, "near_receiver": 1.0},
        "sender_marker": {"near_sender": 3.0, "far_sender": 3.0, "near_receiver": 1.0},
        "nonsig": {},
    }
    genes = list(expression)
    labels, coords, X = [], [], []
    for group, (label, x) in groups.items():
        labels += [label] * n
        coords.append(np.column_stack([x, np.zeros(n)]))
        X.append(np.array([[expression[g].get(group, 0.0) for g in genes]] * n) + rng.uniform(0, 0.1, (n, len(genes))))
    adata = AnnData(X=np.vstack(X), obs=pd.DataFrame({"cell_type": labels}), var=pd.DataFrame(index=genes))
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obsm["spatial"] = np.vstack(coords)
    return adata


@pytest.fixture
def ablation_module(contact_zone_adata):
    """Scores for receiver R from sender S, ranked so the flagged genes come first."""
    diff = {"nonsig": 0.6, "spillover": 0.5, "sender_marker": 0.4, "induced": 0.3, "filler": 0.2}
    scores = pd.DataFrame(
        {
            "gene": list(diff),
            "cell_type": "R",
            "head_idx": None,
            "S_diff": list(diff.values()),
            "S_z_value": 10.0,
            "S_nl10_pval_adj": [0.5 if g == "nonsig" else 5.0 for g in diff],
        }
    )
    return AMICIAblationModule(
        _adata=contact_zone_adata,
        _ablation_scores_df=scores,
        _cell_types=["R", "S"],
        _labels_key="cell_type",
    )


@pytest.mark.parametrize(
    ("compare_near_senders", "expected"),
    [(False, {"induced", "filler", "spillover"}), (True, {"induced", "filler"})],
)
def test_segmentation_test_near_sender_comparison(ablation_module, contact_zone_adata, compare_near_senders, expected):
    passing = ablation_module._get_segmentation_passing_gene_pairs(
        genes_by_sender={"S": ["induced", "filler", "spillover", "sender_marker"]},
        receiver_type="R",
        adata=contact_zone_adata,
        near_threshold=NEAR,
        far_threshold=FAR,
        compare_near_senders=compare_near_senders,
    )
    assert passing == {("S", gene) for gene in expected}


@pytest.mark.parametrize(
    ("segmentation_kwargs", "n_top_genes", "expected"),
    [
        # Original selection: top 2 by contribution, then the p-value filter drops nonsig, no backfill.
        (None, 2, ["spillover"]),
        # Flagged genes skipped and backfilled; spillover passes without the near-sender comparison.
        ({"compare_near_senders": False}, 2, ["spillover", "induced"]),
        ({"compare_near_senders": True}, 2, ["induced", "filler"]),
        # One gene per batch, so the test runs down the ranking until a gene passes.
        ({"compare_near_senders": True}, 1, ["induced"]),
    ],
)
def test_dotplot_top_genes(ablation_module, contact_zone_adata, segmentation_kwargs, n_top_genes, expected):
    if segmentation_kwargs is not None:
        segmentation_kwargs = {
            "adata": contact_zone_adata,
            "near_threshold": NEAR,
            "far_threshold": FAR,
            **segmentation_kwargs,
        }
    scores = ablation_module._ablation_scores_df
    top = ablation_module._get_dotplot_top_genes(
        scores, "R", "diff", "z_value", n_top_genes, -10, segmentation_kwargs=segmentation_kwargs
    )
    assert scores.loc[top["S"], "gene"].tolist() == expected


def test_dotplot_excludes_flagged_genes(ablation_module, contact_zone_adata):
    ablation_module.plot_featurewise_contributions_dotplot(
        cell_type="R",
        color_by="diff",
        size_by="z_value",
        n_top_genes=2,
        min_size_by=-10,
        show=False,
        segmentation_adata=contact_zone_adata,
        segmentation_near_threshold=NEAR,
        segmentation_far_threshold=FAR,
        exclude_segmentation_flagged_genes=True,
        segmentation_compare_near_senders=True,
    )
    drawn = [label.get_text() for ax in plt.gcf().get_axes() for label in ax.get_xticklabels()]
    plt.close("all")
    assert [gene for gene in drawn if gene in ablation_module._adata.var_names] == ["induced", "filler"]
