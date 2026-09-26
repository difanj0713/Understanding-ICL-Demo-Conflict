#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import sys
import re
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from tasks.t2t_tasks import (
    OperatorInductionTextTask,
    FakeWordInductionTask
)
from utils.experiment_protocol import evaluate_final_answer, sample_seed
from utils.prompt_utils import build_vl_icl_prompt
from utils.corruption_utils import CorruptionManager

class ComprehensiveEvaluator:
    def __init__(self, model_name: str, model_type: str, data_dir: str, dataset: str = "operator_induction_text", task_type: str = "color"):
        self.model_name = model_name
        self.model_type = model_type
        self.data_dir = data_dir
        self.dataset = dataset
        self.task_type = task_type
        self.corruption_manager = CorruptionManager()

        if self.dataset == "operator_induction_text":
            self.task = OperatorInductionTextTask(self.data_dir)
        elif self.dataset == "operator_induction_interleaved_text":
            self.task = OperatorInductionInterleavedTextTask(self.data_dir)
        elif self.dataset == "fake_word_induction":
            self.task = FakeWordInductionTask(self.data_dir)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.engine = None
        
    def _initialize_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Initializing vLLM engine for {self.model_name}...")
        self.engine = LLM(
            model=self.model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048
        )

        return True
    
    def _analyze_corruption_effect(self, clean_baseline_details: List[Dict], corrupted_details: List[Dict],
                                    demonstrations_list: List[List[Dict]], corruption_position: int) -> Dict:




        if "operator_induction" not in self.dataset:
            return {}

        def extract_numbers(question: str):
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 2:
                return int(numbers[0]), int(numbers[1])
            return None, None

        def compute_answer(num1: int, num2: int, operator: str):
            if operator == '+':
                return num1 + num2
            elif operator == '-':
                return num1 - num2
            elif operator == 'x':
                return num1 * num2
            return None

        def get_corrupted_operator(demonstrations: List[Dict], position: int, gt_operator: str):
            if position >= len(demonstrations):
                return None
            corrupted_demo = demonstrations[position]
            num1, num2 = extract_numbers(corrupted_demo.get('question', ''))
            if num1 is None or num2 is None:
                return None
            corrupted_answer = corrupted_demo.get('answer')
            if '_corrupted_operator' in corrupted_demo:
                return corrupted_demo['_corrupted_operator']
            for op in ['+', '-', 'x']:
                if compute_answer(num1, num2, op) == corrupted_answer and op != gt_operator:
                    return op
            return None

        corrupt_count = 0
        alternative_count = 0
        else_count = 0
        total_degraded = 0

        for i, (clean, corrupted) in enumerate(zip(clean_baseline_details, corrupted_details)):

            question = corrupted['question']
            num1, num2 = extract_numbers(question)


            gt_operator = corrupted.get('operator')
            corrupted_operator = get_corrupted_operator(demonstrations_list[i], corruption_position, gt_operator) if num1 is not None and num2 is not None and gt_operator is not None else None


            all_operators = ['+', '-', 'x']
            operator_results = {}
            if num1 is not None and num2 is not None:
                operator_results = {op: compute_answer(num1, num2, op) for op in all_operators}


            corrupted['corrupted_operator'] = corrupted_operator
            corrupted['operator_results'] = operator_results
            corrupted['corruption_category'] = None


            if clean['is_correct'] and not corrupted['is_correct']:
                total_degraded += 1

                predicted_value = corrupted.get('predicted_value')


                if num1 is None or num2 is None or gt_operator is None or corrupted_operator is None or predicted_value is None:
                    else_count += 1
                    corrupted['corruption_category'] = 'else'
                elif predicted_value == operator_results.get(corrupted_operator):
                    corrupt_count += 1
                    corrupted['corruption_category'] = 'corrupt'
                elif predicted_value in [operator_results[op] for op in all_operators if op != gt_operator and op != corrupted_operator]:
                    alternative_count += 1
                    corrupted['corruption_category'] = 'alternative'
                else:
                    else_count += 1
                    corrupted['corruption_category'] = 'else'

        return {
            'total_degraded': total_degraded,
            'corrupt_count': corrupt_count,
            'alternative_count': alternative_count,
            'else_count': else_count,
            'corrupt_ratio': corrupt_count / total_degraded if total_degraded > 0 else 0,
            'alternative_ratio': alternative_count / total_degraded if total_degraded > 0 else 0,
            'else_ratio': else_count / total_degraded if total_degraded > 0 else 0
        }

    def create_corrupted_demonstrations(self, query: Dict, n_shot: int, corruption_position: int, rollout: int = 1):
        random.seed(sample_seed(query, rollout))

        if self.dataset == "fake_word_induction":
            demonstrations = self.task.select_demonstrations(query, n_shot, task_type=self.task_type)
        else:
            demonstrations = self.task.select_demonstrations(query, n_shot)

        if corruption_position is not None:
            if self.dataset == "fake_word_induction":
                demonstrations = self.corruption_manager.create_corrupted_demonstrations(
                    self.dataset, query, demonstrations, self.task.support_data, corruption_position, task_type=self.task_type
                )
            else:
                demonstrations = self.corruption_manager.create_corrupted_demonstrations(
                    self.dataset, query, demonstrations, self.task.support_data, corruption_position
                )

        return demonstrations
    
    def _evaluate_response(self, query: Dict, response: str, is_baseline: bool = False) -> tuple:
        return evaluate_final_answer(query, response, self.dataset, self.task_type)

    def evaluate_batch(self, queries: List[Dict], prompts: List[str], evaluation_type: str, corruption_position: int = None):
        is_baseline = "BASELINE" in evaluation_type
        sampling_params = self.sampling_params

        chat_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if self.model_type in ("qwen3", "llama3"):
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                chat_prompt = prompt
            chat_prompts.append(chat_prompt)

        try:
            outputs = self.engine.generate(chat_prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]
        except Exception as e:
            raise RuntimeError("Inference failed") from e

        if len(responses) != len(queries):
            raise ValueError('Response count differs from query count')
        correct_count = 0
        eval_details = []

        for query, response in zip(queries, responses):
            is_correct, predicted_value = self._evaluate_response(query, response, is_baseline=is_baseline)
            if is_correct:
                correct_count += 1

            detail = {
                'query_id': query.get('id', 'unknown'),
                'question': query.get('question', ''),
                'correct_answer': query.get('answer'),
                'response': response,
                'is_correct': is_correct,
                'predicted_value': predicted_value
            }

            if "operator_induction" in self.dataset and corruption_position is not None:
                detail['operator'] = query.get('operator')

            eval_details.append(detail)

        accuracy = correct_count / len(queries)
        return accuracy, eval_details
    
    def run_comprehensive_evaluation(self, n_shot: int = 4, num_samples: int = 60, num_rollouts: int = 3):
        if not self._initialize_vllm():
            print("Cannot run evaluation without vLLM")
            return None

        available_samples = len(self.task.query_data)
        actual_samples = min(num_samples, available_samples)

        print(f"\n{'='*80}")
        print(f"Corruption Evaluation (Clean Baseline vs Corrupted)")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Shots: {n_shot}")
        print(f"Samples: {actual_samples} (requested: {num_samples}, available: {available_samples})")
        print(f"Rollouts: {num_rollouts}")
        print("="*80)

        query_samples = self.task.query_data[:actual_samples]

        results = {
            'experiment_info': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'n_shot': n_shot,
                'num_samples': actual_samples,
                'num_rollouts': num_rollouts,
                'timestamp': datetime.now().isoformat()
            },
            'clean_baseline': {},
            'position_results': {}
        }

        try:
            print(f"\n{'='*80}")
            print(f"STEP 1: Clean Baseline Evaluation")
            print("="*80)

            clean_baseline_accuracies = []
            clean_baseline_rollout_data = []

            for rollout in range(num_rollouts):
                print(f"\n--- Clean Baseline Rollout {rollout + 1}/{num_rollouts} ---")

                clean_queries = []
                clean_prompts = []
                clean_demonstrations_per_query = []

                for query in tqdm(query_samples, desc=f"Building clean prompts R{rollout+1}"):
                    demonstrations = self.create_corrupted_demonstrations(query, n_shot, None, rollout=rollout)

                    clean_demonstrations_per_query.append(demonstrations)

                    if self.dataset == "fake_word_induction":
                        clean_prompt = build_vl_icl_prompt(
                            self.task, demonstrations, query,
                            mode="constrained",
                            warned=False,
                            max_images=8,
                            task_type=self.task_type
                        )
                    else:
                        clean_prompt = build_vl_icl_prompt(
                            self.task, demonstrations, query,
                            mode="constrained",
                            warned=False,
                            max_images=8
                        )

                    clean_queries.append(query)
                    clean_prompts.append(clean_prompt)

                clean_result = self.evaluate_batch(clean_queries, clean_prompts, f"CLEAN_BASELINE R{rollout+1}")

                if clean_result is not None:
                    clean_accuracy, clean_details = clean_result
                    clean_baseline_accuracies.append(clean_accuracy)
                    clean_baseline_rollout_data.append({
                        'accuracy': clean_accuracy,
                        'details': clean_details,
                        'demonstrations': clean_demonstrations_per_query
                    })
                    print(f"Clean Baseline R{rollout+1}: {clean_accuracy:.3f}")
                else:
                    print(f"Clean Baseline R{rollout+1}: Failed")

            if clean_baseline_accuracies:
                results['clean_baseline'] = {
                    'rollouts': clean_baseline_accuracies,
                    'mean': float(np.mean(clean_baseline_accuracies)),
                    'std': float(np.std(clean_baseline_accuracies)) if len(clean_baseline_accuracies) > 1 else 0.0,
                    'rollout_data': clean_baseline_rollout_data
                }

                print(f"\n{'='*60}")
                print(f"Clean Baseline Summary:")
                print(f"Mean Accuracy: {results['clean_baseline']['mean']:.3f} ± {results['clean_baseline']['std']:.3f}")
                print("="*60)

            print(f"\n{'='*80}")
            print(f"STEP 2: Position-wise Corruption Tests")
            print("="*80)

            for position in range(n_shot):
                position_data = {
                    'clean_baseline_rollouts': clean_baseline_accuracies,
                    'corrupted_rollouts': [],
                    'clean_baseline_mean': results['clean_baseline']['mean'],
                    'clean_baseline_std': results['clean_baseline']['std'],
                    'corrupted_mean': None,
                    'corrupted_std': None,
                    'degradation_mean': None,
                    'degradation_std': None,
                    'paired_differences': [],
                    'p_value': None
                }

                corrupted_accuracies = []
                degradations = []

                for rollout in range(num_rollouts):
                    print(f"\n--- Position {position} Rollout {rollout + 1}/{num_rollouts} ---")

                    corrupted_queries = []
                    corrupted_prompts = []
                    corrupted_demonstrations_per_query = []

                    for query in tqdm(query_samples, desc=f"Building corrupted prompts P{position} R{rollout+1}"):
                        demonstrations = self.create_corrupted_demonstrations(query, n_shot, position, rollout=rollout)
                        corrupted_demonstrations_per_query.append(demonstrations)

                        if self.dataset == "fake_word_induction":
                            corrupted_prompt = build_vl_icl_prompt(
                                self.task, demonstrations, query,
                                mode="constrained",
                                warned=False,
                                max_images=8,
                                task_type=self.task_type
                            )
                        else:
                            corrupted_prompt = build_vl_icl_prompt(
                                self.task, demonstrations, query,
                                mode="constrained",
                                warned=False,
                                max_images=8
                            )

                        corrupted_queries.append(query)
                        corrupted_prompts.append(corrupted_prompt)

                    corrupted_result = self.evaluate_batch(
                        corrupted_queries, corrupted_prompts,
                        f"CORRUPTED P{position} R{rollout+1}",
                        corruption_position=position
                    )

                    if corrupted_result is not None and rollout < len(clean_baseline_rollout_data):
                        corrupted_accuracy, corrupted_details = corrupted_result
                        clean_baseline_data = clean_baseline_rollout_data[rollout]
                        clean_baseline_accuracy = clean_baseline_data['accuracy']
                        clean_baseline_details = clean_baseline_data['details']

                        corrupted_accuracies.append(corrupted_accuracy)
                        degradation = clean_baseline_accuracy - corrupted_accuracy
                        degradations.append(degradation)


                        corruption_analysis = self._analyze_corruption_effect(
                            clean_baseline_details, corrupted_details,
                            corrupted_demonstrations_per_query, position
                        )

                        position_data[f'rollout_{rollout}'] = {
                            'clean_baseline_details': clean_baseline_details,
                            'corrupted_details': corrupted_details,
                            'clean_baseline_accuracy': clean_baseline_accuracy,
                            'corrupted_accuracy': corrupted_accuracy,
                            'degradation': degradation,
                            'corruption_analysis': corruption_analysis
                        }

                        print(f"Position {position} R{rollout+1}: Clean={clean_baseline_accuracy:.3f}, Corrupted={corrupted_accuracy:.3f}, Degradation={degradation:.3f}")
                    else:
                        print(f"Position {position} R{rollout+1}: Failed")

                if corrupted_accuracies and len(clean_baseline_accuracies) == len(corrupted_accuracies):
                    position_data['corrupted_rollouts'] = corrupted_accuracies
                    position_data['corrupted_mean'] = float(np.mean(corrupted_accuracies))
                    position_data['corrupted_std'] = float(np.std(corrupted_accuracies)) if len(corrupted_accuracies) > 1 else 0.0
                    position_data['paired_differences'] = degradations
                    position_data['degradation_mean'] = float(np.mean(degradations))
                    position_data['degradation_std'] = float(np.std(degradations)) if len(degradations) > 1 else 0.0

                    if len(degradations) > 1:
                        t_stat, p_value = stats.ttest_rel(clean_baseline_accuracies, corrupted_accuracies)
                        position_data['p_value'] = float(p_value)

                    corruption_analyses = [position_data[f'rollout_{r}']['corruption_analysis']
                                          for r in range(num_rollouts)
                                          if f'rollout_{r}' in position_data and 'corruption_analysis' in position_data[f'rollout_{r}']]

                    if corruption_analyses and "operator_induction" in self.dataset:
                        total_degraded = sum(ca['total_degraded'] for ca in corruption_analyses)
                        total_corrupt = sum(ca['corrupt_count'] for ca in corruption_analyses)
                        total_alternative = sum(ca['alternative_count'] for ca in corruption_analyses)
                        total_else = sum(ca['else_count'] for ca in corruption_analyses)

                        position_data['corruption_summary'] = {
                            'total_degraded': total_degraded,
                            'corrupt_count': total_corrupt,
                            'alternative_count': total_alternative,
                            'else_count': total_else,
                            'corrupt_ratio': total_corrupt / total_degraded if total_degraded > 0 else 0,
                            'alternative_ratio': total_alternative / total_degraded if total_degraded > 0 else 0,
                            'else_ratio': total_else / total_degraded if total_degraded > 0 else 0
                        }

                    print(f"\n{'='*60}")
                    print(f"POSITION {position} SUMMARY")
                    print("="*60)
                    print(f"Clean Baseline:        {position_data['clean_baseline_mean']:.3f} ± {position_data['clean_baseline_std']:.3f}")
                    print(f"Corrupted:             {position_data['corrupted_mean']:.3f} ± {position_data['corrupted_std']:.3f}")
                    print(f"Degradation:           {position_data['degradation_mean']:.3f} ± {position_data['degradation_std']:.3f}")
                    if position_data.get('p_value') is not None:
                        print(f"P-value:               {position_data['p_value']:.4f}")

                    if 'corruption_summary' in position_data:
                        cs = position_data['corruption_summary']
                        print(f"\nDegradation Analysis (Correct→Wrong: {cs['total_degraded']} samples):")
                        print(f"  Corrupt (followed corrupted op):  {cs['corrupt_count']:3d} ({cs['corrupt_ratio']:.1%})")
                        print(f"  Alternative (other op):           {cs['alternative_count']:3d} ({cs['alternative_ratio']:.1%})")
                        print(f"  Else (random/other):              {cs['else_count']:3d} ({cs['else_ratio']:.1%})")

                    print("="*60)

                results['position_results'][f'pos_{position}'] = position_data

        finally:
            if hasattr(self, 'engine') and self.engine is not None:
                try:
                    del self.engine
                    self.engine = None
                except:
                    pass

        model_short = self.model_name.split('/')[-1]
        os.makedirs("results/corruption_analysis", exist_ok=True)
        filename = f"results/corruption_analysis/corruption_{self.dataset}_{model_short}_{n_shot}shot_{num_rollouts}rollouts.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filename}")

        return results

def main():
    parser = argparse.ArgumentParser(
        description='Corruption evaluation: clean baseline vs position-wise corruption'
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., Qwen/Qwen3-8B)')
    parser.add_argument('--model_type', type=str, default='qwen3',
                        choices=['qwen3', 'llama3', 'internvl', 'qwen25'],
                        help='Model type for compatibility')
    parser.add_argument('--dataset', type=str, default='operator_induction_text',
                        choices=['operator_induction_text', 'operator_induction_interleaved_text', 'fake_word_induction'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, default='./VL-ICL',
                        help='Data directory path')
    parser.add_argument('--task_type', type=str, default='color',
                        choices=['color', 'object'],
                        help='Task type for fake_word_induction (color or object)')
    parser.add_argument('--n_shot', type=int, default=4,
                        help='Number of shots (demonstrations)')
    parser.add_argument('--num_samples', type=int, default=60,
                        help='Number of query samples to test')
    parser.add_argument('--num_rollouts', type=int, default=3,
                        help='Number of rollouts for evaluation')

    args = parser.parse_args()

    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == 'fake_word_induction':
        print(f"Task Type: {args.task_type}")
    print(f"Shots: {args.n_shot}")
    print(f"Samples: {args.num_samples}")
    print(f"Rollouts: {args.num_rollouts}")

    evaluator = ComprehensiveEvaluator(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        dataset=args.dataset,
        task_type=args.task_type
    )

    results = evaluator.run_comprehensive_evaluation(
        n_shot=args.n_shot,
        num_samples=args.num_samples,
        num_rollouts=args.num_rollouts
    )

    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())