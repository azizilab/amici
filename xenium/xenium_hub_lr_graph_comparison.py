# %% Ligand-receptor weighted spatial graph baseline for Xenium communication hubs
from hub_graph_baseline_utils import (
    auto_radius_from_knn,
    cluster_features,
    compute_attention_hubs,
    compute_lr_weighted_radius_features,
    load_xenium_model,
    save_baseline_outputs,
)


# %%
adata, model, attention_patterns = load_xenium_model()
attention_df, n_clusters = compute_attention_hubs(attention_patterns)

# %%
radius = auto_radius_from_knn(attention_patterns)
print(f"Using LR-weighted radius graph with radius={radius:.3f}")
features_df = compute_lr_weighted_radius_features(adata, radius)
clustered_df = cluster_features(features_df, "lr_graph_cluster", n_clusters)
save_baseline_outputs(
    adata,
    attention_df,
    clustered_df,
    "lr_graph",
    extra_metrics={"radius": radius},
)
