import os
import sys
import torch
import numpy as np
import pickle
import random
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import traceback
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask, FakeWordInductionTask
from utils.model_utils import get_model_tokenizer, get_language_model
from attention_extractor import categorize_tokens_text_fixed
from interp.heads.head_ops import extract_attention_head_data
from utils.experiment_protocol import select_query_split, query_group, sample_seed

def get_embeddings_text(model, tokenizer, prompt, model_type):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.model.device)
    input_ids = inputs.input_ids
    token_texts = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    with torch.no_grad():
        embedding_layer = model.model.model.embed_tokens
        embeddings = embedding_layer(input_ids)

    return embeddings, input_ids[0].tolist(), token_texts

def extract_attention_weights_and_outputs(language_model, embeddings, attention_mask):
    return extract_attention_head_data(language_model, embeddings, attention_mask)


def extract_numbers_from_text_question(question):
    numbers = re.findall(r'\d+', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def compute_answer(num1, num2, operator):
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == 'x':
        return num1 * num2
    return 0


def create_corrupted_demonstration(demo, corruption_value, dataset, task_type="color"):
    corrupted_demo = demo.copy()

    if dataset == "operator_induction_text":
        num1, num2 = extract_numbers_from_text_question(demo['question'])
        if num1 is not None and num2 is not None:
            corrupted_answer = compute_answer(num1, num2, corruption_value)
            corrupted_demo['answer'] = corrupted_answer
    elif dataset == "fake_word_induction":
        if task_type == "color":
            corrupted_demo['real_phrase'] = f"{corruption_value} {demo.get('object', '')}"
        else:
            corrupted_demo['real_phrase'] = f"{demo.get('color', '')} {corruption_value}"

    return corrupted_demo


def create_text_prompt(task, query, n_shot, corrupt_position=None, corruption_value=None, dataset="operator_induction_text", task_type="color", demonstrations=None):
    from utils.prompt_utils import build_vl_icl_prompt

    if demonstrations is None:
        raise ValueError('demonstrations is required')
    demonstrations = [demo.copy() for demo in demonstrations]

    if corrupt_position is not None and corruption_value is not None:
        if 0 <= corrupt_position < len(demonstrations):
            demonstrations[corrupt_position] = create_corrupted_demonstration(
                demonstrations[corrupt_position], corruption_value, dataset, task_type
            )

    if dataset == "fake_word_induction":
        prompt = build_vl_icl_prompt(task, demonstrations, query, mode="constrained", warned=False, task_type=task_type)
    else:
        prompt = build_vl_icl_prompt(task, demonstrations, query, mode="constrained", warned=False)
    return prompt

def compute_attention_allocation(attention_weights, token_categories, n_shot):
    allocation_scores = {}
    qf_pos = token_categories.get('query_forerunner')
    if qf_pos is None:
        return allocation_scores
    
    for layer_idx, layer_weights in enumerate(attention_weights):
        num_heads, seq_len, _ = layer_weights.shape
        
        if qf_pos >= seq_len:
            continue
            
        for head_idx in range(num_heads):
            qf_attention = layer_weights[head_idx, qf_pos, :]
            for demo_pos in range(n_shot):
                demo_attention = 0.0
                demo_forerunner = token_categories.get(f'demo_{demo_pos+1}_forerunner')
                demo_label = token_categories.get(f'demo_{demo_pos+1}_label')

                demo_tokens = token_categories.get(f'demo_{demo_pos+1}_tokens', [])
                if not demo_tokens:
                    raise ValueError(f'Missing full demonstration span for position {demo_pos}')

                for token_pos in demo_tokens:
                    if token_pos < seq_len:
                        demo_attention += float(qf_attention[token_pos].item())
                
                if demo_tokens:
                    demo_attention /= len(demo_tokens)
                
                allocation_scores[(layer_idx, head_idx, demo_pos)] = demo_attention
    
    return allocation_scores


def compute_corruption_sensitivity(model, tokenizer, language_model, task, query, n_shot,
                                 attention_outputs_clean, token_categories, position,
                                 model_type, corruption_values=None, dataset="operator_induction_text", task_type="color", epsilon=1e-8, demonstrations=None):
    qf_pos = token_categories.get('query_forerunner')
    if qf_pos is None:
        return {}

    if dataset == "operator_induction_text":
        all_operators = ['+', '-', 'x']
        correct_operator = query.get('operator', '+')
        corruption_candidates = [op for op in all_operators if op != correct_operator]
    elif dataset == "fake_word_induction":
        if task_type == "color":
            import json
            with open(os.path.join(os.path.dirname(task.data_dir), "data/fake_word_concepts.json"), 'r') as f:
                concepts = json.load(f)
            all_colors = concepts['colors']
            correct_color = query.get('color')
            corruption_candidates = [c for c in all_colors if c != correct_color]
        else:
            import json
            with open(os.path.join(os.path.dirname(task.data_dir), "data/fake_word_concepts.json"), 'r') as f:
                concepts = json.load(f)
            all_objects = concepts['objects']
            correct_object = query.get('query_object', query.get('object'))
            corruption_candidates = [o for o in all_objects if o != correct_object]

    head_outputs_clean = {}

    for layer_idx, layer_outputs in enumerate(attention_outputs_clean):
        num_heads, seq_len, head_dim = layer_outputs.shape
        if qf_pos < seq_len:
            for head_idx in range(num_heads):
                head_output = layer_outputs[head_idx, qf_pos, :]
                head_outputs_clean[(layer_idx, head_idx)] = head_output

    corruption_deviations = defaultdict(list)
    if corruption_values is not None:
        corruption_candidates = corruption_values
    for corruption_value in corruption_candidates[:2]:
        corrupted_prompt = create_text_prompt(task, query, n_shot,
                                            corrupt_position=position,
                                            corruption_value=corruption_value,
                                            dataset=dataset,
                                            task_type=task_type, demonstrations=demonstrations)

        try:
            embeddings, token_ids, token_texts = get_embeddings_text(model, tokenizer, corrupted_prompt, model_type)
            corrupted_categories = categorize_tokens_text_fixed(token_ids, token_texts, n_shot, tokenizer, debug=False, dataset=dataset)
            corrupt_qf_pos = corrupted_categories['query_forerunner']
            if corrupt_qf_pos is None:
                raise ValueError('Missing query forerunner in corrupted prompt')
            attention_mask = torch.ones((1, embeddings.shape[1]), dtype=torch.long, device=embeddings.device)

            _, attention_outputs_corrupted = extract_attention_weights_and_outputs(
                language_model, embeddings, attention_mask
            )
            for layer_idx, layer_outputs in enumerate(attention_outputs_corrupted):
                num_heads, seq_len, head_dim = layer_outputs.shape
                if corrupt_qf_pos < seq_len:
                    for head_idx in range(num_heads):
                        if (layer_idx, head_idx) in head_outputs_clean:
                            clean_output = head_outputs_clean[(layer_idx, head_idx)]
                            corrupted_output = layer_outputs[head_idx, corrupt_qf_pos, :]

                            deviation = corrupted_output - clean_output
                            deviation_norm = torch.norm(deviation, p=2).item()
                            corruption_deviations[(layer_idx, head_idx)].append(deviation_norm)

        except Exception:
            raise

    cs_scores = {}
    for (layer_idx, head_idx) in head_outputs_clean:
        if (layer_idx, head_idx) not in corruption_deviations:
            continue
        if len(corruption_deviations[(layer_idx, head_idx)]) != min(2, len(corruption_candidates)):
            continue
        clean_output = head_outputs_clean[(layer_idx, head_idx)]
        clean_norm = torch.norm(clean_output, p=2).item()

        deviations = corruption_deviations[(layer_idx, head_idx)]
        avg_deviation = np.mean(deviations)
        cs_score = avg_deviation / (clean_norm + epsilon)
        cs_scores[(layer_idx, head_idx, position)] = cs_score

    return cs_scores

def run_acs_extraction(model_name="Qwen/Qwen3-8B", model_type="qwen3", data_dir="../../VL-ICL",
                      n_shot=4, num_samples=60, seed=42, dataset="operator_induction_text", task_type="color", query_split="discovery", split_seed=1729):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = create_model(model_type, model_name)
    tokenizer = get_model_tokenizer(model, model_type)
    language_model = get_language_model(model, model_type)

    if dataset == "operator_induction_text":
        task = OperatorInductionTextTask(data_dir)
    elif dataset == "fake_word_induction":
        task = FakeWordInductionTask(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_queries = select_query_split(task.query_data, query_split, split_seed)[:num_samples]

    attention_allocation_results = {}
    corruption_sensitivity_results = {}

    for sample_idx, query in enumerate(all_queries):
        try:
            kwargs = {'task_type': task_type} if dataset == 'fake_word_induction' else {}
            demonstrations = task.select_demonstrations(query, n_shot, seed=sample_seed(query, 0, seed), **kwargs)
            clean_prompt = create_text_prompt(task, query, n_shot, dataset=dataset, task_type=task_type, demonstrations=demonstrations)
            embeddings, token_ids, token_texts = get_embeddings_text(model, tokenizer, clean_prompt, model_type)

            token_categories = categorize_tokens_text_fixed(token_ids, token_texts, n_shot, tokenizer, debug=False, dataset=dataset)
            attention_mask = torch.ones((1, embeddings.shape[1]), dtype=torch.long, device=embeddings.device)
            attention_weights_clean, attention_outputs_clean = extract_attention_weights_and_outputs(
                language_model, embeddings, attention_mask
            )

            allocation_scores = compute_attention_allocation(attention_weights_clean, token_categories, n_shot)
            for key, score in allocation_scores.items():
                if key not in attention_allocation_results:
                    attention_allocation_results[key] = []
                attention_allocation_results[key].append(score)

            position = 0
            cs_scores = compute_corruption_sensitivity(
                model, tokenizer, language_model, task, query, n_shot,
                attention_outputs_clean, token_categories, position, model_type,
                dataset=dataset, task_type=task_type, demonstrations=demonstrations
            )

            for key, score in cs_scores.items():
                if key not in corruption_sensitivity_results:
                    corruption_sensitivity_results[key] = []
                corruption_sensitivity_results[key].append(score)

        except Exception as e:
            raise

    final_attention_allocation = {}
    for key, scores in attention_allocation_results.items():
        final_attention_allocation[key] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'count': len(scores)
        }

    final_corruption_sensitivity = {}
    for key, scores in corruption_sensitivity_results.items():
        final_corruption_sensitivity[key] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'count': len(scores)
        }

    task_name = f"{dataset}_{task_type}" if dataset == "fake_word_induction" else dataset

    return {
        'attention_allocation': final_attention_allocation,
        'corruption_sensitivity': final_corruption_sensitivity,
        'metadata': {
            'model_name': model_name,
            'model_type': model_type,
            'num_samples': len(all_queries),
            'n_shot': n_shot,
            'corruption_position': 0,
            'seed': seed,
            'dataset': dataset,
            'task_type': task_type if dataset == "fake_word_induction" else None,
            'task_name': task_name,
            'framework': 'chat_template_acs',
            'query_groups': sorted({query_group(q) for q in all_queries}),
            'query_split': query_split,
            'split_seed': split_seed
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Attention Allocation and Corruption Sensitivity Metrics")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name")
    parser.add_argument("--model_type", type=str, default="qwen3",
                       choices=['qwen3', 'llama3'],
                       help="Model type")
    parser.add_argument("--data_dir", type=str, default="../../VL-ICL",
                       help="Data directory (default: ../../VL-ICL)")
    parser.add_argument("--n_shot", type=int, default=4,
                       help="Number of shots")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--dataset", type=str, default="fake_word_induction",
                       choices=['operator_induction_text', 'fake_word_induction'],
                       help="Dataset name")
    parser.add_argument("--task_type", type=str, default="color",
                       choices=['color', 'object'],
                       help="Task type for fake_word_induction")

    parser.add_argument('--query_split', choices=['discovery', 'all'], default='discovery')
    parser.add_argument('--split_seed', type=int, default=1729)
    args = parser.parse_args()

    results = run_acs_extraction(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        n_shot=args.n_shot,
        num_samples=args.num_samples,
        seed=args.seed,
        dataset=args.dataset,
        task_type=args.task_type, query_split=args.query_split, split_seed=args.split_seed
    )

    os.makedirs('results/heads', exist_ok=True)
    model_suffix = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    task_name = results['metadata']['task_name']
    output_path = f'results/heads/vulnerability_heads_{model_suffix}_{task_name}_{args.n_shot}shot_pos0.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved vulnerability heads to: {output_path}")

    cs_results = results['corruption_sensitivity']
    aa_results = results['attention_allocation']

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
        combined_rankings.append((layer_idx, head_idx, product_score, aa_score, cs_score))

    combined_rankings.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP 20 VULNERABILITY HEADS (AA × CS)")
    print("="*100)
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'AA×CS':<15} {'AA':<15} {'CS':<15}")
    print("-"*100)
    for rank, (layer, head, product, aa, cs) in enumerate(combined_rankings[:20], 1):
        print(f"{rank:<6} L{layer:<7} H{head:<7} {product:<15.6f} {aa:<15.6f} {cs:<15.6f}")
    print("="*100)

    return results

if __name__ == "__main__":
    main()