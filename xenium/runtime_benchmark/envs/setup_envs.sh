#!/usr/bin/env bash
# Create and configure conda environments for AMICI runtime benchmarks.
# Usage: bash setup_envs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Creating CPU environment ==="
conda env remove -n amici_bench_cpu -y 2>/dev/null || true
conda env create -f "$SCRIPT_DIR/cpu.yaml" -y
conda run -n amici_bench_cpu pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
conda run -n amici_bench_cpu pip install -e "$REPO_ROOT"

echo ""
echo "=== Creating GPU environment ==="
conda env remove -n amici_bench_gpu -y 2>/dev/null || true
conda env create -f "$SCRIPT_DIR/gpu.yaml" -y
conda run -n amici_bench_gpu pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n amici_bench_gpu pip install -e "$REPO_ROOT"

echo ""
echo "=== Done ==="
echo "CPU env: conda activate amici_bench_cpu"
echo "GPU env: conda activate amici_bench_gpu"
