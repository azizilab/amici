Installation
============

Requirements
------------

AMICI requires Python 3.10 or newer. The main dependencies include:

- scvi-tools for single-cell analysis
- scanpy for single-cell data processing
- Various scientific computing libraries (numpy, scipy, pandas)

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release:

.. code-block:: bash

    pip install amici-st

From Source (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

    pip install git+https://github.com/azizilab/amici.git@main

For local development:

.. code-block:: bash

    git clone https://github.com/azizilab/amici.git
    cd amici
    pip install -e .

Verification
------------

To verify your installation, try importing AMICI:

.. code-block:: python

    import amici
    print(amici.__version__)

