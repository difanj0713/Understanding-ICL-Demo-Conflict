#!/bin/bash

set -e
cd "$(dirname "$0")/.."

MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-4B"
)

TASKS=(
    "operator_induction_text"
)

SHOTS=(4 6 8)
ROLLOUTS=3
SAMPLES=200

echo "Starting corruption evaluation batch..."
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Shots: ${SHOTS[*]}"
echo "Working directory: $(pwd)"
echo ""

# Simple counter
TOTAL=$((${#MODELS[@]} * ${#TASKS[@]} * ${#SHOTS[@]}))
COUNT=0

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for shots in "${SHOTS[@]}"; do
            COUNT=$((COUNT + 1))

            if [[ "$model" == *"Qwen3"* ]]; then
                model_type="qwen3"
            else
                model_type="llama3"
            fi

            data_dir="./VL-ICL"
            model_short=$(basename "$model")

            echo "[$COUNT/$TOTAL] $model_short | $task | ${shots}-shot"

            timestamp=$(date +%Y%m%d_%H%M%S)
            log_file="results/corruption_run_${model_short}_${task}_${shots}shot.log"

            cmd="python scripts/corruption_evaluation.py \
                --model_name \"$model\" \
                --model_type \"$model_type\" \
                --dataset \"$task\" \
                --data_dir \"$data_dir\" \
                --n_shot $shots \
                --num_samples $SAMPLES \
                --num_rollouts $ROLLOUTS"

            if eval "$cmd" 2>&1 | tee "$log_file"; then
                echo "SUCCESS"
            else
                echo "FAILED"
            fi
        done
    done
done

echo "All runs completed!"
echo "Results saved in: results/corruption_analysis/"
find results/corruption_analysis -name "corruption_*.json" -newer . 2>/dev/null | sort