Quick Start Guide
=================

This guide will help you get started with AMICI for analyzing cell-cell interactions in spatial transcriptomics data.

Basic Usage
-----------

1. **Import AMICI and load your data**:

.. code-block:: python

    import anndata
    import scanpy as sc
    from amici import AMICI

    # Load your spatial transcriptomics data
    adata = anndata.read_h5ad("your_data.h5ad")

2. **Setup the data for AMICI**:

.. code-block:: python

    AMICI.setup_anndata(
        adata,
        labels_key="cell_type",  # Column in adata.obs with cell type annotations
        coord_obsm_key="spatial"  # Key in adata.obsm with spatial coordinates
    )

3. **Create and configure the model**:

.. code-block:: python

    model = AMICI(
        adata,
        n_heads=4,  # Number of attention heads
        n_neighbors=30,  # Number of nearest neighbors to consider
        # Add other model parameters as needed
    )

4. **Train the model**:

.. code-block:: python

    model.train(
        max_epochs=100,
        batch_size=128,
        early_stopping=True
    )

5. **Analyze attention patterns**:

.. code-block:: python

    # Get attention patterns
    attention_patterns = model.get_attention_patterns()

    # Plot attention summary
    attention_patterns.plot_attention_summary()

6. **Perform neighbor ablation analysis**:

.. code-block:: python

    # Get ablation scores for a specific cell type
    ablation_scores = model.get_neighbor_ablation_scores(
        cell_type="T_cell",
        head_idx=0
    )

    # Plot ablation results
    ablation_scores.plot_neighbor_ablation_scores()

7. **Analyze explained variance**:

.. code-block:: python

    # Get explained variance scores
    explained_variance = model.get_expl_variance_scores()

    # Plot explained variance
    explained_variance.plot_explained_variance_barplot()

Data Requirements
-----------------

Your AnnData object should contain:

- **Gene expression data**: In ``adata.X`` (can be raw counts or normalized)
- **Cell type annotations**: In ``adata.obs`` (categorical column)
- **Spatial coordinates**: In ``adata.obsm`` (2D coordinates for each cell)

Optional:
- **Cell radius information**: In ``adata.obs`` (for distance calculations)

Example Data Structure
----------------------

.. code-block:: python

    adata.obs:
        cell_type: ['T_cell', 'B_cell', 'Macrophage', ...]
        cell_radius: [2.5, 3.1, 2.8, ...]  # Optional

    adata.obsm:
        spatial: [[x1, y1], [x2, y2], ...]  # 2D coordinates

    adata.X:
        # Gene expression matrix (cells x genes)

Next Steps
----------

- Check out the :doc:`tutorial` for a more detailed walkthrough
- Explore the :doc:`api/index` for comprehensive API documentation
- Read the full paper real data examples
