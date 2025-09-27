AMICI: Attention Mechanism Interpretation of Cell-cell Interactions
====================================================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by-nc/4.0/
   :alt: License

.. image:: https://readthedocs.org/projects/amici/badge/?version=latest
   :target: https://amici.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/amici-st.svg
   :target: https://pypi.org/project/amici-st/
   :alt: PyPI version

.. image:: https://github.com/user-attachments/assets/13b05cb1-d59a-4ef1-8abe-858c8414448e
   :alt: AMICI Framework
   :align: center
   :width: 800px

Cross-attention-based cell-cell interaction inference from spatial transcriptomics data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorial
   api/index

Installation
------------

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_.

Install the latest release of ``amici-st`` from PyPI:

.. code-block:: bash

    pip install amici-st

Or install the latest development version:

.. code-block:: bash

    pip install git+https://github.com/azizilab/amici.git@main

Quick Start
-----------

.. code-block:: python

    import anndata
    from amici import AMICI

    # Load your spatial transcriptomics data
    adata = anndata.read_h5ad("./adata.h5ad")

    # Setup the data for AMICI
    AMICI.setup_anndata(adata, labels_key="cell_type", coord_obsm_key="spatial")

    # Create and train the model
    model = AMICI(adata, **model_params)
    model.train()

    # Get attention patterns
    attention_patterns = model.get_attention_patterns()
    attention_patterns.plot_attention_summary()

Citation
--------

If you find our work useful, please cite our preprint:

.. code-block:: bibtex

    @article{Hong2025.09.22.677860,
        title = {AMICI: Attention Mechanism Interpretation of Cell-cell Interactions},
        author = {Hong, Justin and Desai, Khushi and Nguyen, Tu Duyen and Nazaret, Achille and Levy, Nathan and Ergen, Can and Plitas, George and Azizi, Elham},
        doi = {10.1101/2025.09.22.677860},
        journal = {bioRxiv},
        publisher = {Cold Spring Harbor Laboratory},
        year = {2025},
    }

