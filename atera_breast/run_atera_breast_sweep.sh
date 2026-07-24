#!/bin/bash

cd "$(dirname "$0")"

project_name="atera_breast_sweep_$(date +%Y-%m-%d)"
entity_name="stattention"
SWEEP_ID=$(wandb sweep --project "$project_name" --entity "$entity_name" atera_breast_sweep.yaml |& tail -n 1 | awk "{print \$NF}")

if [ -z "$SWEEP_ID" ]; then
    echo "No sweep ID found"
    exit 1
fi

echo "Extracted sweep ID: $SWEEP_ID"

NUM_AGENTS=1
mkdir -p wandb

for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent "$SWEEP_ID" > "wandb/agent_atera_breast_$i.log" 2>&1 &
done

wait
