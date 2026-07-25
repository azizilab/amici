#!/bin/bash

cd "$(dirname "$0")"

project_name="human_tonsil_sweep_$(date +%Y-%m-%d)"
entity_name="stattention"
SWEEP_ID=$(wandb sweep --project "$project_name" --entity "$entity_name" human_tonsil_sweep.yaml |& tail -n 1 | awk "{print \$NF}")

if [ -z "$SWEEP_ID" ]; then
    echo "No sweep ID found"
    exit 1
fi

echo "Extracted sweep ID: $SWEEP_ID"
wandb agent "$SWEEP_ID"
