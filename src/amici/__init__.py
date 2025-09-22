from importlib.metadata import version

from ._constants import NN_REGISTRY_KEYS
from ._model import AMICI
from ._module import AMICIModule

__all__ = ["AMICI", "AMICIModule", "NN_REGISTRY_KEYS"]

__version__ = version("amici")
