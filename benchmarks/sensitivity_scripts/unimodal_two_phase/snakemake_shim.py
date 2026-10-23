"""Run a Snakemake ``script:`` file outside of Snakemake.

The benchmark scripts under ``benchmarks/`` are Snakemake script-directive files: they read a
global ``snakemake`` object for their inputs, outputs, wildcards and config. This shim builds
that object from a JSON spec and executes the target script unmodified, so the sweeps in this
folder score GITIII and CGCom with the pipeline's own code rather than a reimplementation.

Only the standard library is used so the shim can run inside any of the model conda envs.

Usage:
    python snakemake_shim.py --script gene_task/generate_gitiii_scores.py --spec spec.json

The spec is ``{"config": {...}, "wildcards": {...}, "input": {...}, "output": {...},
"params": {...}}``. Scripts are run with the working directory set to ``benchmarks/`` because
they resolve paths relative to it (GITIII's scorer chdirs into the model directory and back
out via ``../../..``).
"""

import argparse
import json
import os
import runpy
import sys


class _IndexableNamespace:
    """Attribute and index access, matching Snakemake's input/output objects."""

    def __init__(self, values):
        self._order = list(values)
        for key, value in values.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, int):
            return getattr(self, self._order[key])
        return getattr(self, key)

    def __iter__(self):
        return (getattr(self, key) for key in self._order)

    def __len__(self):
        return len(self._order)

    def __fspath__(self):
        if len(self._order) != 1:
            raise TypeError("os.fspath() only valid for a single-entry namespace")
        return str(getattr(self, self._order[0]))

    def __str__(self):
        if len(self._order) == 1:
            return str(getattr(self, self._order[0]))
        return repr(self)


class _Snakemake:
    def __init__(self, spec):
        self.config = spec.get("config", {})
        self.wildcards = _IndexableNamespace(spec.get("wildcards", {}))
        self.input = _IndexableNamespace(spec.get("input", {}))
        self.output = _IndexableNamespace(spec.get("output", {}))
        self.params = _IndexableNamespace(spec.get("params", {}))
        self.resources = _IndexableNamespace(spec.get("resources", {}))
        self.threads = spec.get("threads", 1)
        self.log = _IndexableNamespace(spec.get("log", {}))


def main():
    """Build a snakemake object from the JSON spec and run the target script with it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="Path to the script, relative to benchmarks/")
    parser.add_argument("--spec", required=True, help="Path to the JSON spec")
    parser.add_argument(
        "--benchmarks-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")),
        help="Directory the pipeline scripts expect as the working directory",
    )
    args = parser.parse_args()

    with open(args.spec) as handle:
        spec = json.load(handle)

    benchmarks_dir = os.path.abspath(args.benchmarks_dir)
    script_path = os.path.join(benchmarks_dir, args.script)
    if not os.path.exists(script_path):
        raise FileNotFoundError(script_path)

    # The scripts import sibling helpers (benchmark_utils, gpu_utils, ...) and resolve data
    # paths relative to benchmarks/, so both sys.path and cwd have to point there.
    sys.path.insert(0, benchmarks_dir)
    sys.path.insert(0, os.path.dirname(script_path))
    os.chdir(benchmarks_dir)

    runpy.run_path(script_path, init_globals={"snakemake": _Snakemake(spec)}, run_name="__main__")


if __name__ == "__main__":
    main()
