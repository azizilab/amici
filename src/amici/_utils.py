import inspect
from functools import update_wrapper


def wraps_compute(cls):
    """Wraps the compute method of a class to copy the signature without model."""

    def wraps_without_model(func):
        update_wrapper(func, cls.compute)
        orig_sig = inspect.signature(cls.compute)
        new_sig = orig_sig.replace(parameters=[p for p in orig_sig.parameters.values() if p.name != "model"])
        func.__signature__ = new_sig
        return func

    return wraps_without_model
