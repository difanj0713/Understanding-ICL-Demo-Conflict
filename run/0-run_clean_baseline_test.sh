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

SHOTS="0 4 6 8"
SAMPLES=200

echo "Starting clean baseline evaluation batch..."
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Shots: $SHOTS"
echo "Working directory: $(pwd)"
echo ""

TOTAL=$((${#MODELS[@]} * ${#TASKS[@]}))
COUNT=0

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        COUNT=$((COUNT + 1))
        
        if [[ "$model" == *"Qwen3"* ]]; then
            model_type="qwen3"
        elif [[ "$model" == *"Qwen2.5-VL"* ]]; then
            model_type="qwen25"
        elif [[ "$model" == *"InternVL"* ]]; then
            model_type="internvl"
        else
            model_type="llama3"
        fi
        
        data_dir="./VL-ICL"
        
        model_short=$(basename "$model")
        
        echo "[$COUNT/$TOTAL] $model_short | $task | shots: $SHOTS"
        
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="results/run_clean_baseline_${model_short}_${task}.log"
        
        cmd="python scripts/clean_baseline_evaluation.py \
            --model_name \"$model\" \
            --model_type \"$model_type\" \
            --dataset \"$task\" \
            --data_dir \"$data_dir\" \
            --n_shots $SHOTS \
            --num_samples $SAMPLES"
        
        echo "Command: $cmd"
        echo "Log: $log_file"
        
        if eval "$cmd" 2>&1 | tee "$log_file"; then
            echo "SUCCESS"
        else
            echo "FAILED"
        fi
        echo ""
    done
done

echo "Results saved in: results/"
find results -name "clean_baseline_*.json" -newer . 2>/dev/null | sort