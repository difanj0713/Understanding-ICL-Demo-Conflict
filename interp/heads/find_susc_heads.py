import sys
import os
import torch
from typing import Dict, List
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.dirname(parent_dir))
sys.path.append(script_dir)

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask, FakeWordInductionTask
from attention_extractor import categorize_tokens_text_fixed
import json
from interp.heads.head_ops import projected_head_contributions
from utils.experiment_protocol import select_query_split, query_group, sample_seed

class AntiResolutionHeadFinder:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", model_type: str = "qwen3", dataset: str = "operator_induction_text", task_type: str = "color"):
        self.model_name = model_name
        self.model_type = model_type
        self.dataset = dataset
        self.task_type = task_type
        self.model = None
        self.tokenizer = None
        self.operator_words = {'+': 'plus', '-': 'minus', '*': 'multiplication', 'x': 'multiplication'}
        
    def load_model(self):
        self.model = create_model(self.model_type, self.model_name)
        self.tokenizer = self.model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def create_prompt(self, demonstrations: List[Dict], query: Dict, corruption_pattern: List[int], corrupted_value) -> str:
        prompt_parts = []

        if self.dataset == "operator_induction_text":
            for i, demo in enumerate(demonstrations):
                question = demo['question']
                correct_answer = demo['answer'][0] if isinstance(demo['answer'], list) else demo['answer']

                if corruption_pattern[i] == 1:
                    num1, num2 = self.extract_numbers(question)
                    if num1 is not None and num2 is not None:
                        corrupted_answer = self.compute_answer(num1, num2, corrupted_value)
                        prompt_parts.append(f"{question} Answer: {corrupted_answer}")
                    else:
                        prompt_parts.append(f"{question} Answer: {correct_answer}")
                else:
                    prompt_parts.append(f"{question} Answer: {correct_answer}")

            prompt_parts.extend([
                f"{query['question']}",
                "What mathematical operation does ? represent? Choose from: plus, minus, multiplication",
                "Answer:"
            ])
        elif self.dataset == "fake_word_induction":
            for i, demo in enumerate(demonstrations):
                correct_phrase = demo.get('real_phrase', '')
                fake_phrase = demo.get('fake_phrase', '')

                if corruption_pattern[i] == 1:
                    if self.task_type == "color":
                        obj = demo.get('object', '')
                        corrupted_phrase = f"{corrupted_value} {obj}"
                    else:
                        color = demo.get('color', '')
                        corrupted_phrase = f"{color} {corrupted_value}"
                    prompt_parts.append(f"{fake_phrase} means {corrupted_phrase}")
                else:
                    prompt_parts.append(f"{fake_phrase} means {correct_phrase}")

            if self.task_type == "color":
                prompt_parts.extend([
                    query.get('question_color', ''),
                    "Answer:"
                ])
            else:
                prompt_parts.extend([
                    query.get('question_object', ''),
                    "Answer:"
                ])

        return '\n'.join(prompt_parts)
    
    def extract_numbers(self, question: str):
        import re
        numbers = re.findall(r'\d+', question)
        return (int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else (None, None)
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+': return num1 + num2
        elif operator == '-': return num1 - num2
        elif operator in ('*', 'x'): return num1 * num2
        return 0
    
    def get_target_token_ids(self, query):
        target_tokens = {}

        if self.dataset == "operator_induction_text":
            for word in ['plus', 'minus', 'multiplication']:
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                target_tokens[word] = tokens[0] if tokens else None
        elif self.dataset == "fake_word_induction":
            concepts_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/fake_word_concepts.json")
            with open(concepts_file, 'r') as f:
                concepts = json.load(f)

            if self.task_type == "color":
                for color in concepts['colors']:
                    tokens = self.tokenizer.encode(color, add_special_tokens=False)
                    target_tokens[color] = tokens[0] if tokens else None
            else:
                for obj in concepts['objects']:
                    tokens = self.tokenizer.encode(obj, add_special_tokens=False)
                    target_tokens[obj] = tokens[0] if tokens else None

        return target_tokens
    
    def extract_head_contributions_batch(self, prompts: List[str], queries: List[Dict], layers_to_analyze: List[int]) -> List[Dict]:
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.model.device)
        input_ids = model_inputs['input_ids']
        batch_size = input_ids.shape[0]

        query_positions = []
        valid_indices = []
        for idx in range(batch_size):
            actual_token_texts = [self.tokenizer.decode([tid]) for tid in input_ids[idx].tolist()]
            token_categories = categorize_tokens_text_fixed(
                input_ids[idx].tolist(), actual_token_texts, 4, self.tokenizer, debug=False, dataset=self.dataset
            )
            if token_categories['query_forerunner'] is not None:
                query_positions.append(token_categories['query_forerunner'])
                valid_indices.append(idx)
            else:
                raise ValueError("Query forerunner not found")

        batch_head_contributions = [{} for _ in range(batch_size)]

        def attention_hook(layer_idx, attention):
            def hook(module, inputs):
                for batch_idx in valid_indices:
                    query_pos = query_positions[batch_idx]
                    contributions = projected_head_contributions(attention, inputs[0][batch_idx, query_pos])
                    for head_idx, residual_contrib in enumerate(contributions):
                        batch_head_contributions[batch_idx][f"L{layer_idx}H{head_idx}"] = residual_contrib.detach().cpu()
            return hook

        hooks = []
        for layer_idx in layers_to_analyze:
            attention = self.model.model.model.layers[layer_idx].self_attn
            hooks.append(attention.o_proj.register_forward_pre_hook(attention_hook(layer_idx, attention)))

        try:
            with torch.no_grad():
                outputs = self.model.model(**model_inputs)
                unembedding = self.model.model.lm_head.weight

                results = []
                for batch_idx in range(batch_size):
                    if batch_idx not in valid_indices:
                        results.append(None)
                        continue

                    logits = outputs.logits[batch_idx, query_positions[batch_idx], :]
                    target_tokens = self.get_target_token_ids(queries[batch_idx])
                    head_scores = {}

                    for head_name, residual_contrib in batch_head_contributions[batch_idx].items():
                        logit_attribution = torch.matmul(unembedding, residual_contrib.to(unembedding.device))
                        head_logits = {}
                        for word, token_id in target_tokens.items():
                            if token_id is not None:
                                head_logits[word] = logit_attribution[token_id].item()
                        head_scores[head_name] = head_logits

                    results.append({
                        'head_scores': head_scores,
                        'baseline_logits': {word: logits[token_id].item()
                                          for word, token_id in target_tokens.items() if token_id is not None}
                    })

        finally:
            for hook in hooks:
                hook.remove()

        return results

    def extract_head_contributions(self, prompt: str, query: Dict, layers_to_analyze: List[int]) -> Dict:
        results = self.extract_head_contributions_batch([prompt], [query], layers_to_analyze)
        return results[0] if results else None
    
    def rank_anti_resolution_heads(self, query: Dict, demonstrations: List[Dict],
                                  layers_to_analyze: List[int] = None) -> Dict:
        if layers_to_analyze is None:
            num_layers = self.model.model.config.num_hidden_layers
            layers_to_analyze = list(range(num_layers))

        if self.dataset == "operator_induction_text":
            correct_op = query['operator']
            if correct_op == 'x': correct_op = '*'
            available_ops = [op for op in ['+', '-', '*'] if op != correct_op]
            corrupted_op = available_ops[0]
            correct_word = self.operator_words[query['operator']]
            corrupted_word = self.operator_words[corrupted_op]
            corrupted_value = corrupted_op
        elif self.dataset == "fake_word_induction":
            concepts_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/fake_word_concepts.json")
            with open(concepts_file, 'r') as f:
                concepts = json.load(f)

            if self.task_type == "color":
                correct_word = query.get('color')
                all_colors = concepts['colors']
                corrupted_word = [c for c in all_colors if c != correct_word][0]
            else:
                correct_word = query.get('query_object', query.get('object'))
                all_objects = concepts['objects']
                corrupted_word = [o for o in all_objects if o != correct_word][0]
            corrupted_value = corrupted_word

        clean_pattern = [0, 0, 0, 0]
        minority_corrupt_pattern = [1, 0, 0, 0]

        clean_prompt = self.create_prompt(demonstrations, query, clean_pattern, corrupted_value)
        minority_corrupt_prompt = self.create_prompt(demonstrations, query, minority_corrupt_pattern, corrupted_value)

        results = self.extract_head_contributions_batch([clean_prompt, minority_corrupt_prompt], [query, query], layers_to_analyze)
        clean_result = results[0] if len(results) > 0 else None
        minority_corrupt_result = results[1] if len(results) > 1 else None

        if clean_result is None or minority_corrupt_result is None:
            return {}

        anti_resolution_scores = {}

        for head_name in clean_result['head_scores']:
            if head_name not in minority_corrupt_result['head_scores']:
                continue

            clean_logits = clean_result['head_scores'][head_name]
            minority_corrupt_logits = minority_corrupt_result['head_scores'][head_name]

            clean_correct = clean_logits.get(correct_word, 0)
            clean_corrupted = clean_logits.get(corrupted_word, 0)
            minority_corrupt_correct = minority_corrupt_logits.get(correct_word, 0)
            minority_corrupt_corrupted = minority_corrupt_logits.get(corrupted_word, 0)

            clean_gap = clean_correct - clean_corrupted
            corrupt_gap = minority_corrupt_correct - minority_corrupt_corrupted
            anti_resolution_score = clean_gap - corrupt_gap

            anti_resolution_scores[head_name] = {
                'anti_resolution_score': anti_resolution_score,
                'clean_correct_contrib': clean_correct,
                'clean_corrupted_contrib': clean_corrupted,
                'minority_corrupt_correct_contrib': minority_corrupt_correct,
                'minority_corrupt_corrupted_contrib': minority_corrupt_corrupted,
                'clean_logits': clean_logits,
                'minority_corrupt_logits': minority_corrupt_logits
            }

        return anti_resolution_scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find Susceptibility Heads")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_type", type=str, default="qwen3", choices=['qwen3', 'llama3'])
    parser.add_argument("--data_dir", type=str, default="../../VL-ICL")
    parser.add_argument("--dataset", type=str, default="fake_word_induction",
                       choices=['operator_induction_text', 'fake_word_induction'])
    parser.add_argument("--task_type", type=str, default="color", choices=['color', 'object'])
    parser.add_argument('--query_split', choices=['discovery', 'all'], default='discovery')
    parser.add_argument('--split_seed', type=int, default=1729)
    args = parser.parse_args()

    finder = AntiResolutionHeadFinder(model_name=args.model_name, model_type=args.model_type,
                                     dataset=args.dataset, task_type=args.task_type)
    finder.load_model()

    if args.dataset == "operator_induction_text":
        task = OperatorInductionTextTask(args.data_dir)
    elif args.dataset == "fake_word_induction":
        task = FakeWordInductionTask(args.data_dir)

    all_queries = select_query_split(task.query_data, args.query_split, args.split_seed)
    num_layers = finder.model.model.config.num_hidden_layers
    layers_to_analyze = list(range(num_layers))
    head_score_aggregates = {}

    for query_idx, test_query in enumerate(all_queries):
        if args.dataset == "fake_word_induction":
            demonstrations = task.select_demonstrations(test_query, n_shot=4, seed=sample_seed(test_query, 0), task_type=args.task_type)
        else:
            demonstrations = task.select_demonstrations(test_query, n_shot=4, seed=sample_seed(test_query, 0))
        
        head_rankings = finder.rank_anti_resolution_heads(
            test_query, demonstrations, layers_to_analyze
        )
        
        if head_rankings:
            for head_name, data in head_rankings.items():
                if head_name not in head_score_aggregates:
                    head_score_aggregates[head_name] = {
                        'anti_resolution_scores': [],
                        'clean_gaps': [],
                        'corrupt_gaps': []
                    }

                clean_gap = data['clean_correct_contrib'] - data['clean_corrupted_contrib']
                corrupt_gap = data['minority_corrupt_correct_contrib'] - data['minority_corrupt_corrupted_contrib']
                head_score_aggregates[head_name]['anti_resolution_scores'].append(data['anti_resolution_score'])
                head_score_aggregates[head_name]['clean_gaps'].append(clean_gap)
                head_score_aggregates[head_name]['corrupt_gaps'].append(corrupt_gap)
    
    final_head_rankings = {}
    for head_name, aggregates in head_score_aggregates.items():
        if aggregates['anti_resolution_scores']:
            avg_anti_resolution = sum(aggregates['anti_resolution_scores']) / len(aggregates['anti_resolution_scores'])
            avg_clean_gap = sum(aggregates['clean_gaps']) / len(aggregates['clean_gaps'])
            avg_corrupt_gap = sum(aggregates['corrupt_gaps']) / len(aggregates['corrupt_gaps'])

            final_head_rankings[head_name] = {
                'anti_resolution_score': avg_anti_resolution,
                'clean_gap': avg_clean_gap,
                'corrupt_gap': avg_corrupt_gap,
                'n_samples': len(aggregates['anti_resolution_scores'])
            }
    
    if final_head_rankings:
        task_name = f"{args.dataset}_{args.task_type}" if args.dataset == "fake_word_induction" else args.dataset

        rigorous_results = {
            'final_head_rankings': final_head_rankings,
            'total_queries': len(all_queries),
            'total_layers': len(layers_to_analyze),
            'model_name': finder.model_name,
            'model_type': finder.model_type,
            'n_shot': 4,
            'corruption_position': 0,
            'dataset': args.dataset,
            'task_type': args.task_type if args.dataset == "fake_word_induction" else None,
            'task_name': task_name,
            'methodology': 'DLA averaged across all queries and layers',
            'query_groups': sorted({query_group(q) for q in all_queries}),
            'query_split': args.query_split,
            'split_seed': args.split_seed
        }

        model_suffix = finder.model_name.split('/')[-1] if '/' in finder.model_name else finder.model_name
        os.makedirs('results/heads', exist_ok=True)
        output_path = f'results/heads/susceptibility_heads_{model_suffix}_{task_name}_4shot_pos0.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(rigorous_results, f)
        print(f"Saved susceptibility heads to: {output_path}")

        rankings = []
        for head_name, head_data in final_head_rankings.items():
            layer_idx = int(head_name[1:head_name.index('H')])
            head_idx = int(head_name[head_name.index('H')+1:])
            rankings.append((
                layer_idx,
                head_idx,
                head_data['anti_resolution_score'],
                head_data['clean_gap'],
                head_data['corrupt_gap'],
                head_data['n_samples']
            ))

        rankings.sort(key=lambda x: x[2], reverse=True)

        print(f"\n{'='*100}")
        print(f"TOP 20 SUSCEPTIBILITY HEADS")
        print("="*100)
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Anti-Res':<15} {'Clean Gap':<15} {'Corrupt Gap':<15} {'Samples':<8}")
        print("-"*100)
        for rank, (layer, head, anti_res, clean_contrib, corrupt_contrib, n_samples) in enumerate(rankings[:20], 1):
            print(f"{rank:<6} L{layer:<7} H{head:<7} {anti_res:<15.6f} {clean_contrib:<15.6f} {corrupt_contrib:<15.6f} {n_samples:<8}")
        print("="*100)

if __name__ == "__main__":
    main()