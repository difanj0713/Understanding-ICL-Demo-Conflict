import torch
import numpy as np
import pickle
import os
import sys
import random
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Set

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask, FakeWordInductionTask
from utils.prompt_utils import build_vl_icl_prompt
import json


from interp.heads.head_ops import AttentionHeadMaskingHook, head_geometry
from utils.experiment_protocol import evaluate_final_answer, select_query_split, query_group, sample_seed


def run_batch_inference(model, tokenizer, prompts: List[str], max_new_tokens: int, model_type: str, batch_size: int = 100) -> List[str]:
    all_responses = []
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Batch inference", leave=False):
        batch_prompts = prompts[i:i+batch_size]

        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(model.model.device)

        with torch.no_grad():
            outputs = model.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[len(model_inputs['input_ids'][j]):],
                skip_special_tokens=True
            )
            all_responses.append(response)

    return all_responses


def evaluate_responses(queries: List[Dict], responses: List[str], dataset: str = "operator_induction_text", task_type: str = "color") -> List[bool]:
    if len(queries) != len(responses):
        raise ValueError('Response count differs from query count')
    return [evaluate_final_answer(q, r, dataset, task_type)[0] for q, r in zip(queries, responses)]


def create_corrupted_demonstrations(demonstrations: List[Dict], corruption_value: str, dataset: str, task_type: str = "color") -> List[Dict]:
    corrupted = [demo.copy() for demo in demonstrations]

    demo = corrupted[0]

    if dataset == "operator_induction_text":
        numbers = re.findall(r'\d+', demo['question'])
        if len(numbers) >= 2:
            num1, num2 = int(numbers[0]), int(numbers[1])

            if corruption_value == '+':
                corrupted_answer = num1 + num2
            elif corruption_value == '-':
                corrupted_answer = num1 - num2
            elif corruption_value == 'x':
                corrupted_answer = num1 * num2
            else:
                corrupted_answer = demo['answer']

            demo['answer'] = corrupted_answer
    elif dataset == "fake_word_induction":
        if task_type == "color":
            obj = demo.get('object', '')
            demo['real_phrase'] = f"{corruption_value} {obj}"
        else:
            color = demo.get('color', '')
            demo['real_phrase'] = f"{color} {corruption_value}"

    return corrupted


def validate_discovery_artifact(data, evaluation_queries):
    metadata = data.get('metadata', data)
    groups = metadata.get('query_groups')
    if not groups:
        raise ValueError('Head ranking file lacks query_groups')
    if evaluation_queries is not None and set(groups) & {query_group(q) for q in evaluation_queries}:
        raise ValueError('Head discovery and evaluation query groups overlap')


def load_vulnerability_heads(model_name: str, top_k: int, task_name: str, n_shot=4, evaluation_queries=None):
    model_suffix = model_name.split('/')[-1] if '/' in model_name else model_name
    file_path = f'results/heads/vulnerability_heads_{model_suffix}_{task_name}_{n_shot}shot_pos0.pkl'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    validate_discovery_artifact(data, evaluation_queries)

    cs_results = data['corruption_sensitivity']
    aa_results = data['attention_allocation']

    cs_dict = {}
    for key, metrics in cs_results.items():
        layer_idx, head_idx, position = key
        cs_dict[(layer_idx, head_idx)] = metrics['mean']

    aa_dict = {}
    for key, metrics in aa_results.items():
        layer_idx, head_idx, demo_pos = key
        if demo_pos == 0:
            aa_dict[(layer_idx, head_idx)] = metrics['mean']

    combined_rankings = []
    for (layer_idx, head_idx), cs_score in cs_dict.items():
        aa_score = aa_dict.get((layer_idx, head_idx), 0.0)
        product_score = aa_score * cs_score
        combined_rankings.append((layer_idx, head_idx, product_score))

    combined_rankings.sort(key=lambda x: x[2], reverse=True)

    vulnerability_heads = set()
    for layer, head, _ in combined_rankings[:top_k]:
        vulnerability_heads.add((layer, head))

    return vulnerability_heads


def load_susceptibility_heads(model_name: str, top_k: int, task_name: str, n_shot=4, evaluation_queries=None):
    model_suffix = model_name.split('/')[-1] if '/' in model_name else model_name
    file_path = f'results/heads/susceptibility_heads_{model_suffix}_{task_name}_{n_shot}shot_pos0.pkl'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    validate_discovery_artifact(data, evaluation_queries)

    head_rankings = data['final_head_rankings']

    rankings = []
    for head_name, head_data in head_rankings.items():
        layer_idx = int(head_name[1:head_name.index('H')])
        head_idx = int(head_name[head_name.index('H')+1:])
        rankings.append((layer_idx, head_idx, head_data['anti_resolution_score']))

    rankings.sort(key=lambda x: x[2], reverse=True)

    susceptibility_heads = set()
    for layer, head, _ in rankings[:top_k]:
        susceptibility_heads.add((layer, head))

    return susceptibility_heads


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Head Ablation Study")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model_type", type=str, default="qwen3", choices=['qwen3', 'llama3'])
    parser.add_argument("--data_dir", type=str, default="../../VL-ICL")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="fake_word_induction",
                       choices=['operator_induction_text', 'fake_word_induction'])
    parser.add_argument("--task_type", type=str, default="color", choices=['color', 'object'])
    parser.add_argument('--query_split', choices=['evaluation', 'all'], default='evaluation')
    parser.add_argument('--split_seed', type=int, default=1729)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.n_shot != 4:
        parser.error('ablation supports n_shot=4 only')
    model = create_model(args.model_type, args.model_name)
    tokenizer = model.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "operator_induction_text":
        task = OperatorInductionTextTask(args.data_dir)
    elif args.dataset == "fake_word_induction":
        task = FakeWordInductionTask(args.data_dir)

    query_samples = select_query_split(task.query_data, args.query_split, args.split_seed)[:args.num_samples]

    if args.dataset == "operator_induction_text":
        all_operators = ['+', '-', 'x']
    elif args.dataset == "fake_word_induction":
        corruption_candidates = task.colors if args.task_type == 'color' else task.objects

    task_name = f"{args.dataset}_{args.task_type}" if args.dataset == "fake_word_induction" else args.dataset
    ablation_sizes = [5, 8, 10]
    results = {}
    baseline_cache = {}

    for num_heads in ablation_sizes:
        print(f"\n{'='*80}")
        print(f"Ablating {num_heads} heads")
        print("="*80)

        vulnerability_heads = load_vulnerability_heads(args.model_name, num_heads, task_name, args.n_shot, query_samples)
        susceptibility_heads = load_susceptibility_heads(args.model_name, num_heads, task_name, args.n_shot, query_samples)

        rollout_results = []

        for rollout in range(args.num_rollouts):
            rollout_seed = args.seed + rollout
            random.seed(rollout_seed)
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)

            corrupted_prompts = []
            clean_prompts = []
            queries_batch = []

            for query in tqdm(query_samples, desc=f"Building prompts", leave=False):
                if args.dataset == "fake_word_induction":
                    demonstrations = task.select_demonstrations(query, args.n_shot, seed=sample_seed(query, rollout, args.seed), task_type=args.task_type)
                else:
                    demonstrations = task.select_demonstrations(query, args.n_shot, seed=sample_seed(query, rollout, args.seed))

                if args.dataset == "operator_induction_text":
                    correct_op = query.get('operator')
                    other_operators = [op for op in all_operators if op != correct_op]
                    corruption_value = random.choice(other_operators)
                elif args.dataset == "fake_word_induction":
                    if args.task_type == "color":
                        correct_value = query.get('color')
                    else:
                        correct_value = query.get('query_object', query.get('object'))
                    other_values = [v for v in corruption_candidates if v != correct_value]
                    corruption_value = random.choice(other_values)

                clean_prompts.append(build_vl_icl_prompt(task, demonstrations, query, mode='constrained', task_type=args.task_type))
                corrupted_demos = create_corrupted_demonstrations(demonstrations, corruption_value, args.dataset, args.task_type)

                if args.dataset == "fake_word_induction":
                    corrupted_prompt = build_vl_icl_prompt(task, corrupted_demos, query, mode="constrained", warned=False, task_type=args.task_type)
                else:
                    corrupted_prompt = build_vl_icl_prompt(task, corrupted_demos, query, mode="constrained", warned=False)

                corrupted_prompts.append(corrupted_prompt)
                queries_batch.append(query)

            layers = model.model.model.layers
            universe = [(l, h) for l, layer in enumerate(layers) for h in range(head_geometry(layer.self_attn)[0])]
            random_heads = set(random.Random(rollout_seed + 10000).sample(universe, num_heads))
            conditions = {
                'clean': (clean_prompts, set()),
                'corrupted': (corrupted_prompts, set()),
                'vulnerability': (corrupted_prompts, vulnerability_heads),
                'susceptibility': (corrupted_prompts, susceptibility_heads),
                'random': (corrupted_prompts, random_heads),
            }
            record = {'seed': rollout_seed, 'details': {}, 'heads': {}}
            for condition, (prompts, heads) in conditions.items():
                cache_key = (rollout, condition)
                if condition in ('clean', 'corrupted') and cache_key in baseline_cache:
                    responses, outcomes = baseline_cache[cache_key]
                else:
                    hook_manager = AttentionHeadMaskingHook(heads)
                    hook_manager.register_hooks(model)
                    try:
                        responses = run_batch_inference(model, tokenizer, prompts, 2048, args.model_type, args.batch_size)
                    finally:
                        hook_manager.remove_hooks()
                    outcomes = evaluate_responses(queries_batch, responses, args.dataset, args.task_type)
                    if condition in ('clean', 'corrupted'):
                        baseline_cache[cache_key] = (responses, outcomes)
                record[condition] = float(np.mean(outcomes))
                record['heads'][condition] = sorted(heads)
                record['details'][condition] = [
                    {'query_group': query_group(q), 'query_id': q.get('id'), 'prompt': p, 'response': r, 'correct': c}
                    for q, p, r, c in zip(queries_batch, prompts, responses, outcomes)
                ]
            baseline = record['corrupted']
            record['relative_improvement_percent'] = {
                key: 100 * (record[key] - baseline) / baseline if baseline else None
                for key in ('vulnerability', 'susceptibility', 'random')
            }
            rollout_results.append(record)
            print({k: v for k, v in record.items() if k not in ('details', 'heads')})

        avg_vulnerability = np.mean([r['vulnerability'] for r in rollout_results])
        avg_susceptibility = np.mean([r['susceptibility'] for r in rollout_results])

        results[num_heads] = {
            'avg_vulnerability': avg_vulnerability,
            'avg_susceptibility': avg_susceptibility,
            'rollout_results': rollout_results
        }

        print(f"\nAggregate Results (k={num_heads}):")
        print(f"  Corrupted + mask vuln heads:   {avg_vulnerability:.3f}")
        print(f"  Corrupted + mask susc heads:   {avg_susceptibility:.3f}")

    os.makedirs('results/heads', exist_ok=True)
    model_suffix = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    output_path = f'results/heads/head_ablation_{model_suffix}_{task_name}_{args.n_shot}shot_pos0.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump({
            'model_name': args.model_name,
            'dataset': args.dataset,
            'task_type': args.task_type if args.dataset == "fake_word_induction" else None,
            'task_name': task_name,
            'results': results,
            'ablation_sizes': ablation_sizes,
            'num_rollouts': args.num_rollouts,
            'query_groups': sorted({query_group(q) for q in query_samples}),
            'query_split': args.query_split,
            'split_seed': args.split_seed
        }, f)

    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    main()
