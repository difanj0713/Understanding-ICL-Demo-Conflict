#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask
from utils.prompt_utils import build_vl_icl_prompt
from interp.heads.head_ops import extract_attention_head_data
from utils.model_utils import get_model_tokenizer, get_language_model


def categorize_tokens_text_fixed(actual_token_ids, actual_token_texts, n_shot, tokenizer, debug=True, dataset="operator_induction_text"):
    token_categories = {
        'query_forerunner': None,
        'query_label': None
    }

    for d in range(n_shot):
        token_categories[f'demo_{d+1}_forerunner'] = None
        token_categories[f'demo_{d+1}_label'] = None

    if dataset == "operator_induction_text":
        answer_positions = []
        for i, token in enumerate(actual_token_texts):
            if token.strip().lower().startswith("answer"):
                answer_positions.append(i)

        if len(answer_positions) < n_shot + 1:
            return token_categories

        relevant_answer_positions = answer_positions[-(n_shot + 1):]

        for d in range(n_shot):
            answer_pos = relevant_answer_positions[d]

            for pos in range(answer_pos, min(answer_pos + 3, len(actual_token_texts))):
                if ':' in actual_token_texts[pos]:
                    token_categories[f'demo_{d+1}_forerunner'] = pos

                    label_pos = pos + 1
                    while (label_pos < len(actual_token_texts) and
                           actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                        label_pos += 1
                    if label_pos < len(actual_token_texts):
                        token_categories[f'demo_{d+1}_label'] = label_pos
                    break

        query_answer_pos = relevant_answer_positions[-1]
        for pos in range(query_answer_pos, min(query_answer_pos + 3, len(actual_token_texts))):
            if ':' in actual_token_texts[pos]:
                token_categories['query_forerunner'] = pos

                label_pos = pos + 1
                while (label_pos < len(actual_token_texts) and
                       actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                    label_pos += 1
                token_categories['query_label'] = label_pos if label_pos < len(actual_token_texts) else len(actual_token_texts)
                break

    elif dataset == "fake_word_induction":
        means_positions = []
        for i, token in enumerate(actual_token_texts):
            if 'means' in token.lower():
                means_positions.append(i)

        if len(means_positions) < n_shot:
            return token_categories

        for d in range(n_shot):
            means_pos = means_positions[d]
            token_categories[f'demo_{d+1}_forerunner'] = means_pos

            label_pos = means_pos + 1
            while (label_pos < len(actual_token_texts) and
                   actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                label_pos += 1
            if label_pos < len(actual_token_texts):
                token_categories[f'demo_{d+1}_label'] = label_pos

        answer_positions = []
        for i, token in enumerate(actual_token_texts):
            if token.strip().lower().startswith("answer"):
                answer_positions.append(i)

        if answer_positions:
            query_answer_pos = answer_positions[-1]
            for pos in range(query_answer_pos, min(query_answer_pos + 3, len(actual_token_texts))):
                if ':' in actual_token_texts[pos]:
                    token_categories['query_forerunner'] = pos

                    label_pos = pos + 1
                    while (label_pos < len(actual_token_texts) and
                           actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                        label_pos += 1
                    token_categories['query_label'] = label_pos if label_pos < len(actual_token_texts) else len(actual_token_texts)
                    break

    import re
    text = ''.join(actual_token_texts)
    if dataset == 'operator_induction_text':
        pattern = r'(?m)^[^\n]*\?[^\n]*?(?:\n[ \t]*)?Answer[ \t]*:[ \t]*-?\d+'
    else:
        pattern = r'(?m)^[^\n]+ means [^\n]+'
    matches = list(re.finditer(pattern, text))
    if len(matches) >= n_shot:
        offsets = []
        cursor = 0
        for token in actual_token_texts:
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        for d, match in enumerate(matches[-n_shot:]):
            token_categories[f'demo_{d+1}_tokens'] = [
                i for i, (a, b) in enumerate(offsets) if a < match.end() and b > match.start()
            ]
    return token_categories


class CaptureForAttention:
    def __init__(self):
        self.captured_input_ids = None
        self.captured_inputs_embeds = None

    def generate_hook(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            self.captured_input_ids = input_ids
        return self.orig_generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def first_layer_pre_hook(self):
        def hook_fn(module, inputs):
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
                hidden_states = inputs[0]
                if hidden_states is not None:
                    self.captured_inputs_embeds = hidden_states.detach()
            return None
        return hook_fn

def get_actual_embeddings_and_tokens(model, tokenizer, language_model, prompt, debug=True):
    capture = CaptureForAttention()
    capture.orig_generate = model.model.generate
    model.model.generate = capture.generate_hook
    
    if hasattr(language_model, 'layers'):
        transformer_layers = language_model.layers
    else:
        raise AttributeError(f"Cannot find transformer layers in {type(language_model)}")
    
    hook = transformer_layers[0].register_forward_pre_hook(capture.first_layer_pre_hook())
    
    try:
        gen_cfg = dict(max_new_tokens=1, do_sample=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.model.device)
        model.model.generate(**inputs, **gen_cfg)
        
    finally:
        model.model.generate = capture.orig_generate
        hook.remove()
    
    actual_token_ids = capture.captured_input_ids[0].tolist()
    actual_token_texts = [tokenizer.decode([tid]) for tid in actual_token_ids]
    actual_embeddings = capture.captured_inputs_embeds.detach().clone()
    
    return actual_embeddings, actual_token_ids, actual_token_texts


def extract_attention_with_correct_hooks(language_model, inputs_embeds, attention_mask, debug=True):
    return extract_attention_head_data(language_model, inputs_embeds, attention_mask)


def run_qwen3_attention_extraction(model_name="Qwen/Qwen3-8B",
                                  data_dir="../VL-ICL",
                                  n_shot=4,
                                  num_samples=2,
                                  debug=True):

    model = create_model("qwen3", model_name)
    tokenizer = get_model_tokenizer(model, "qwen3")
    language_model = get_language_model(model, "qwen3")
    task = OperatorInductionTextTask(data_dir)

    config = language_model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // n_heads

    all_samples = task.query_data[:num_samples]
    results = []

    for sample_idx, query in enumerate(all_samples):
        try:
            demonstrations = task.select_demonstrations(query, n_shot=n_shot)
            prompt = build_vl_icl_prompt(task, demonstrations, query, mode="constrained", warned=False)
            
            actual_embeddings, actual_token_ids, actual_token_texts = get_actual_embeddings_and_tokens(
                model, tokenizer, language_model, prompt, debug=debug
            )
            
            token_categories = categorize_tokens_text_fixed(
                actual_token_ids, actual_token_texts, n_shot, tokenizer, debug=debug
            )
            
            attention_mask = torch.ones((1, actual_embeddings.shape[1]), dtype=torch.long, device=actual_embeddings.device)
            
            attention_weights, attention_outputs = extract_attention_with_correct_hooks(
                language_model, actual_embeddings, attention_mask, debug=debug
            )
            print()
            
            success = True
            if len(attention_weights) != n_layers:
                success = False
            
            if attention_weights and debug:
                qf_pos = token_categories.get('query_forerunner')
                demo_positions = []
                for d in range(n_shot):
                    demo_pos = token_categories.get(f'demo_{d+1}_forerunner')
                    if demo_pos is not None:
                        demo_positions.append(demo_pos)
                
                if qf_pos is not None and demo_positions:
                    for layer_idx in range(min(5, len(attention_weights))): 
                        layer_weights = attention_weights[layer_idx]
                        layer_outputs = attention_outputs[layer_idx]
                        
                        for head_idx in range(min(3, n_heads)):
                            if qf_pos < layer_weights.shape[1]:
                                qf_to_demos = layer_weights[head_idx, qf_pos, demo_positions]  
                                qf_output = layer_outputs[head_idx, qf_pos, :]
                                
                                demo_attn_str = ", ".join([f"{a:.4f}" for a in qf_to_demos])
                                output_norm = torch.norm(qf_output).item()
                                
                                print(f"[LAYER {layer_idx:2d}] Head {head_idx:2d}: QF->demos [{demo_attn_str}], output_norm {output_norm:.3f}")
                    
                    for layer_idx in range(min(10, len(attention_weights))):
                        if qf_pos < attention_weights[layer_idx].shape[1] and demo_positions[0] < attention_weights[layer_idx].shape[2]:
                            attn_val = attention_weights[layer_idx][0, qf_pos, demo_positions[0]].item()
                            print(f"[LAYER {layer_idx:2d}] {attn_val:.6f}")
            
            sample_result = {
                'sample_idx': sample_idx,
                'query': query,
                'token_categories': token_categories,
                'actual_token_ids': actual_token_ids,
                'actual_token_texts': actual_token_texts,
                'actual_embeddings': actual_embeddings,
                'attention_weights': attention_weights,
                'attention_outputs': attention_outputs,
                'success': success
            }
            results.append(sample_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    return results


def main():
    try:
        results = run_qwen3_attention_extraction(
            model_name="Qwen/Qwen3-8B",
            data_dir="../VL-ICL", 
            n_shot=4,
            num_samples=60,
            debug=True
        )
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()