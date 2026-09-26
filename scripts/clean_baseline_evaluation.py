#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import sys
import re
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from tasks.t2t_tasks import (
    OperatorInductionTextTask,
    FakeWordInductionTask
)
from utils.experiment_protocol import evaluate_final_answer, sample_seed
from utils.prompt_utils import build_vl_icl_prompt

class CleanBaselineEvaluator:
    def __init__(self, model_name: str, model_type: str, data_dir: str, dataset: str = "operator_induction_text", task_type: str = "color"):
        self.model_name = model_name
        self.model_type = model_type
        self.data_dir = data_dir
        self.dataset = dataset
        self.task_type = task_type

        if self.dataset == "operator_induction_text":
            self.task = OperatorInductionTextTask(self.data_dir)
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
            gpu_memory_utilization=0.9,
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
    
    def create_clean_demonstrations(self, query: Dict, n_shot: int):
        if self.dataset == "fake_word_induction":
            demonstrations = self.task.select_demonstrations(query, n_shot, task_type=self.task_type, seed=sample_seed(query, 0))
        else:
            demonstrations = self.task.select_demonstrations(query, n_shot, seed=sample_seed(query, 0))

        return demonstrations
    
    def _evaluate_response_harsh(self, query: Dict, response: str) -> bool:
        return evaluate_final_answer(query, response, self.dataset, self.task_type)[0]

    def evaluate_batch(self, queries: List[Dict], prompts: List[str], n_shot: int):
        print(f"Running vLLM batch inference for {n_shot}-shot...")

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
            outputs = self.engine.generate(chat_prompts, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
        except Exception as e:
            print(f"vLLM inference failed: {e}")
            return None

        print(f"Evaluating {len(responses)} responses...")
        if len(responses) != len(queries):
            raise ValueError('Response count differs from query count')
        correct_count = 0
        eval_details = []

        for query, response in zip(queries, responses):
            is_correct = self._evaluate_response_harsh(query, response)
            if is_correct:
                correct_count += 1

            eval_details.append({
                'query_id': query.get('id', 'unknown'),
                'question': query.get('question', ''),
                'correct_answer': query.get('answer'),
                'response': response,
                'is_correct': is_correct
            })

        accuracy = correct_count / len(queries)
        return accuracy, eval_details
    
    def run_clean_baseline_evaluation(self, n_shots: List[int] = [0, 1, 2, 4, 6, 8], num_samples: int = 60):
        if not self._initialize_vllm():
            print("Cannot run evaluation without vLLM")
            return None
        
        available_samples = len(self.task.query_data)
        actual_samples = min(num_samples, available_samples)
        
        print(f"\n{'='*80}")
        print(f"CLEAN BASELINE EVALUATION")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Shots: {n_shots}")
        print(f"Samples: {actual_samples} (requested: {num_samples}, available: {available_samples})")
        print("="*80)
        
        query_samples = self.task.query_data[:actual_samples]
        
        results = {
            'experiment_info': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'n_shots': n_shots,
                'num_samples': actual_samples,
                'timestamp': datetime.now().isoformat()
            },
            'shot_results': {}
        }
        
        try:
            for n_shot in n_shots:
                print(f"\n{'='*60}")
                print(f"EVALUATING {n_shot}-SHOT")
                print("="*60)
                
                queries = []
                prompts = []
                
                for query in tqdm(query_samples, desc=f"Building {n_shot}-shot prompts"):
                    demonstrations = self.create_clean_demonstrations(query, n_shot)

                    prompt = build_vl_icl_prompt(
                        self.task, demonstrations, query,
                        mode="constrained",
                        warned=False,
                        max_images=8,
                        task_type=self.task_type if self.dataset == "fake_word_induction" else None
                    )

                    queries.append(query)
                    prompts.append(prompt)

                result = self.evaluate_batch(queries, prompts, n_shot)

                if result is not None:
                    accuracy, eval_details = result
                    results['shot_results'][f'{n_shot}_shot'] = {
                        'accuracy': float(accuracy),
                        'n_shot': n_shot,
                        'correct_count': int(accuracy * len(queries)),
                        'total_count': len(queries),
                        'details': eval_details
                    }
                    print(f"{n_shot}-shot accuracy: {accuracy:.3f}")
                else:
                    print(f"{n_shot}-shot: Failed")
        
        finally:
            if hasattr(self, 'engine') and self.engine is not None:
                try:
                    del self.engine
                    self.engine = None
                except:
                    pass
        
        model_short = self.model_name.split('/')[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/clean_baseline_{self.dataset}_{model_short}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"CLEAN BASELINE RESULTS SUMMARY")
        print("="*80)
        for shot_key, shot_data in results['shot_results'].items():
            n_shot = shot_data['n_shot']
            accuracy = shot_data['accuracy']
            print(f"{n_shot}-shot: {accuracy:.3f}")
        
        print(f"\nResults saved to: {filename}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Clean baseline evaluation across different shots')
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
    parser.add_argument('--n_shots', type=int, nargs='+', default=[0, 1, 2, 4, 6, 8],
                        help='List of shot numbers to evaluate')
    parser.add_argument('--num_samples', type=int, default=60,
                        help='Number of query samples to test')

    args = parser.parse_args()

    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == 'fake_word_induction':
        print(f"Task Type: {args.task_type}")
    print(f"Shots: {args.n_shots}")
    print(f"Samples: {args.num_samples}")

    evaluator = CleanBaselineEvaluator(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        dataset=args.dataset,
        task_type=args.task_type
    )
    
    results = evaluator.run_clean_baseline_evaluation(
        n_shots=args.n_shots,
        num_samples=args.num_samples
    )
    
    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())