def _get_compute_method_kwargs(**compute_locals) -> dict:
    """Returns a dictionary organizing the arguments used to call ``compute``.

    Must be called with ``**locals()`` at the start of the ``compute`` method
    to avoid the inclusion of any extraneous variables.
    """
    compute_kwargs = {}
    for k, v in compute_locals.items():
        if k not in ["self", "model", "adata"]:
            compute_kwargs[k] = v
    return compute_kwargs
