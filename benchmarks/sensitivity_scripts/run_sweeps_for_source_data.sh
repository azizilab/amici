#!/usr/bin/env bash
# Retrain the four hyperparameter sweeps behind Extended Data Fig. 6e-h and write their Source Data.
#
# The published runs wrote only figures and their per-setting models were not kept, so these panels
# have no numbers to export. Each sweep trains one model per dataset seed per setting (10 x 4 = 40
# models, ~160 in all) and now writes the AUPRCs and the example-seed PR curves through
# sweep_source_data.py. Models are cached under saved_models/, so a later rerun only rescores.
#
# Run them one after another: a single GPU, and each script assumes it has it.
#   bash benchmarks/sensitivity_scripts/run_sweeps_for_source_data.sh
set -u

PYTHON=${PYTHON:-/opt/conda/envs/amici/bin/python}
cd "$(dirname "$0")" || exit 1
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for sweep in head_analysis neighbor_analysis value_penalty_analysis attention_penalty_analysis; do
    echo "=== ${sweep} starting $(date -Is)"
    "$PYTHON" "${sweep}.py" > "${LOG_DIR}/${sweep}.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "=== ${sweep} finished $(date -Is)"
    else
        # Keep going: one sweep failing should not cost the other three.
        echo "=== ${sweep} FAILED (exit ${status}) $(date -Is); see ${LOG_DIR}/${sweep}.log"
    fi
done

echo "=== all sweeps done $(date -Is)"
ls -la results/*/[a-z]*_auprc.csv 2>/dev/null || echo "no auprc tables were written"
