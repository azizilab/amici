API Reference
=============

.. currentmodule:: amici

This section contains the API reference for AMICI.

Main Classes
------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AMICI
   AMICIModule

Interpretation Modules
----------------------

.. currentmodule:: amici.interpretation

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AMICIAttentionModule
   AMICICounterfactualAttentionModule
   AMICIExplainedVarianceModule
   AMICIAblationModule

Callbacks
---------

.. currentmodule:: amici.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AttentionPenaltyMonitor
   ModelInterpretationLogging

Utilities
---------

.. currentmodule:: amici.tools

.. autosummary::
   :toctree: generated/

   is_count_data
