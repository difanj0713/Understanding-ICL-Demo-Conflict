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

N_SHOT=4
NUM_SAMPLES=200
SEED=42

echo "=========================================="
echo "Targeted Heads"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "N-shot: $N_SHOT"
echo "Samples: $NUM_SAMPLES"
echo "=========================================="
echo ""

mkdir -p results/heads
mkdir -p logs

TOTAL=$((${#MODELS[@]} * ${#TASKS[@]} * 3))
COUNT=0

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        if [[ "$model" == *"Qwen"* ]]; then
            model_type="qwen3"
        else
            model_type="llama3"
        fi

        model_short=$(basename "$model")

        if [[ "$task" == "fake_word_induction" ]]; then
            task_type="color"
            task_args="--dataset $task --task_type $task_type"
        else
            task_type=""
            task_args="--dataset $task"
        fi

        echo ""
        echo "=========================================="
        echo "Model: $model_short | Task: $task"
        echo "=========================================="

        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Running extract_acs_metrics.py..."
        log_file="logs/acs_${model_short}_${task}.log"

        nohup python interp/heads/extract_acs_metrics.py \
            --model_name "$model" \
            --model_type "$model_type" \
            --data_dir "./VL-ICL" \
            --n_shot $N_SHOT \
            --num_samples $NUM_SAMPLES \
            --seed $SEED \
            $task_args \
            > "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "  -> SUCCESS (see $log_file)"
            tail -5 "$log_file"
        else
            echo "  -> FAILED (see $log_file)"
            tail -10 "$log_file"
            continue
        fi

        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] Running find_susc_heads.py..."
        log_file="logs/susc_${model_short}_${task}.log"

        nohup python interp/heads/find_susc_heads.py \
            --model_name "$model" \
            --model_type "$model_type" \
            --data_dir "./VL-ICL" \
            $task_args \
            > "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "  -> SUCCESS (see $log_file)"
            tail -5 "$log_file"
        else
            echo "  -> FAILED (see $log_file)"
            tail -10 "$log_file"
            continue
        fi

        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] Running comprehensive_head_ablation.py..."
        log_file="logs/ablation_${model_short}_${task}.log"

        nohup python interp/heads/comprehensive_head_ablation.py \
            --model_name "$model" \
            --model_type "$model_type" \
            --data_dir "./VL-ICL" \
            --n_shot $N_SHOT \
            --num_samples $NUM_SAMPLES \
            --seed $SEED \
            $task_args \
            > "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "  -> SUCCESS (see $log_file)"
            tail -10 "$log_file"
        else
            echo "  -> FAILED (see $log_file)"
            tail -10 "$log_file"
        fi

    done
done

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Results saved in: results/heads/"
ls -la results/heads/*.pkl 2>/dev/null || echo "No results found"
