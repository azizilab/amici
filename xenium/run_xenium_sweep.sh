#!/bin/bash

# Extract the sweep ID (last word of the output)
project_name="xenium_sweep_$(date +%Y-%m-%d)"
entity_name="stattention"
SWEEP_ID=$(wandb sweep --project $project_name --entity $entity_name xenium_sweep.yaml |& tail -n 1 | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "No sweep ID found"
    exit 1
fi

echo "Extracted sweep ID: $SWEEP_ID"

# Number of agents to run
NUM_AGENTS=4

# Run agents in parallel
for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent $SWEEP_ID > wandb/agent_$i.log 2>&1 &
done

# Wait for all agents to finish (optional)
wait