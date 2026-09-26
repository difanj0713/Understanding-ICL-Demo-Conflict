import copy
import json
import os
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.heads.head_ops import AttentionHeadMaskingHook, projected_head_contributions, extract_attention_head_data
from interp.heads.comprehensive_head_ablation import validate_discovery_artifact, evaluate_responses
from interp.heads.attention_extractor import categorize_tokens_text_fixed
from interp.heads.extract_acs_metrics import create_text_prompt, compute_attention_allocation, compute_corruption_sensitivity
from utils.experiment_protocol import select_query_split, query_group, evaluate_final_answer
from tasks.t2t_tasks import OperatorInductionTextTask
from scripts.corruption_evaluation import ComprehensiveEvaluator
from interp.probes.data_preprocessing import generate_probe_training_data
from interp.probes.linear_probe_evaluator import generate_evaluation_dataset


def small_model(family='qwen'):
    torch.manual_seed(7)
    config = dict(hidden_size=24, intermediate_size=40, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  vocab_size=64, max_position_embeddings=64)
    cls, cfg = (Qwen3ForCausalLM, Qwen3Config) if family == 'qwen' else (LlamaForCausalLM, LlamaConfig)
    model = cls(cfg(**config)).eval()
    model.config._attn_implementation = 'eager'
    return model


class HeadTests(unittest.TestCase):
    def test_mask_removes_projected_head(self):
        for family in ['qwen', 'llama']:
            with self.subTest(family=family):
                model = small_model(family)
                attn = model.model.layers[0].self_attn
                x = torch.randn(2, 3, 32)
                original = attn.o_proj(x)
                contributions = projected_head_contributions(attn, x)
                torch.testing.assert_close(contributions.sum(-2), original)
                mask = AttentionHeadMaskingHook({(0, 1), (0, 3)})
                mask.register_hooks(model)
                try:
                    actual = attn.o_proj(x)
                    expected = original - contributions[..., 1, :] - contributions[..., 3, :]
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    self.assertFalse(torch.allclose(actual[..., 6:12], torch.zeros_like(actual[..., 6:12])))
                finally:
                    mask.remove_hooks()
                torch.testing.assert_close(attn.o_proj(x), original)

    def test_head_capture_matches_projection_input(self):
        for family in ['qwen', 'llama']:
            with self.subTest(family=family):
                model = small_model(family)
                embeddings = model.model.embed_tokens(torch.tensor([[1, 2, 3, 4]]))
                captured = {}
                hooks = []
                for idx, layer in enumerate(model.model.layers):
                    def capture(module, inputs, idx=idx):
                        captured[idx] = inputs[0].detach().clone()
                    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
                try:
                    weights, values = extract_attention_head_data(model.model, embeddings, torch.ones(1, 4, dtype=torch.long))
                finally:
                    for h in hooks: h.remove()
                self.assertEqual(len(values), 2)
                for idx in range(2):
                    self.assertEqual(values[idx].shape, (4, 4, 8))
                    torch.testing.assert_close(values[idx].permute(1, 0, 2).reshape(1, 4, 32), captured[idx])
                    self.assertEqual(weights[idx].shape, (4, 4, 4))
                mask = AttentionHeadMaskingHook({(0, 1)})
                mask.register_hooks(model)
                try:
                    model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False, pad_token_id=0)
                finally:
                    mask.remove_hooks()
                self.assertFalse(model.model.layers[0].self_attn.o_proj._forward_pre_hooks)

    def test_susceptibility_head_contributions(self):
        from interp.heads.find_susc_heads import AntiResolutionHeadFinder
        model = small_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        captures = {}
        hooks = []
        for idx, layer in enumerate(model.model.layers):
            def capture(module, inputs, idx=idx):
                captures[idx] = inputs[0][0, 2].detach().clone()
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
        with torch.no_grad():
            model(input_ids=ids)
        for hook in hooks: hook.remove()

        class Batch(dict):
            def to(self, device): return self
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs): return 'unused'
            def __call__(self, *args, **kwargs): return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))
            def decode(self, ids): return 'token'
            def encode(self, word, **kwargs): return [{'plus': 5, 'minus': 6, 'multiplication': 7}[word]]
        finder = AntiResolutionHeadFinder()
        finder.model = SimpleNamespace(model=model)
        finder.tokenizer = Tokenizer()
        with patch('interp.heads.find_susc_heads.categorize_tokens_text_fixed', return_value={'query_forerunner': 2}):
            result = finder.extract_head_contributions_batch(['prompt'], [{'operator': '+'}], [0, 1])[0]
        for idx in range(2):
            attn = model.model.layers[idx].self_attn
            for head in range(4):
                z = captures[idx][head*8:(head+1)*8]
                expected = model.lm_head.weight[5] @ (attn.o_proj.weight[:, head*8:(head+1)*8] @ z)
                self.assertAlmostEqual(result['head_scores'][f'L{idx}H{head}']['plus'], expected.item(), places=6)

    def test_reject_invalid_heads_and_overlapping_groups(self):
        with self.assertRaises(ValueError):
            AttentionHeadMaskingHook({(0, 9)}).register_hooks(small_model())
        with self.assertRaisesRegex(ValueError, 'query_groups'):
            validate_discovery_artifact({}, [])
        q = {'question': '2 ? 3', 'operator': '+'}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_discovery_artifact({'query_groups': [query_group(q)]}, [q])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.task = OperatorInductionTextTask(str(ROOT / 'VL-ICL'))

    def test_data_split_groups_shared_arithmetic_inputs(self):
        discovery = select_query_split(self.task.query_data, 'discovery')
        evaluation = select_query_split(self.task.query_data, 'evaluation')
        self.assertFalse({query_group(q) for q in discovery} & {query_group(q) for q in evaluation})
        self.assertEqual(len(discovery) + len(evaluation), len(self.task.query_data))
        self.assertEqual({q['operator'] for q in discovery}, {'+', '-', 'x'})

    def test_behavioral_rollouts_are_paired_and_distinct(self):
        evaluator = ComprehensiveEvaluator('unused', 'qwen3', str(ROOT/'VL-ICL'))
        query = self.task.query_data[0]
        runs = []
        for rollout in range(3):
            clean = evaluator.create_corrupted_demonstrations(query, 4, None, rollout)
            corrupt = evaluator.create_corrupted_demonstrations(query, 4, 2, rollout)
            self.assertEqual(clean[:2] + clean[3:], corrupt[:2] + corrupt[3:])
            self.assertEqual(clean[2]['question'], corrupt[2]['question'])
            self.assertNotEqual(clean[2]['answer'], corrupt[2]['answer'])
            runs.append([d['id'] for d in clean])
        self.assertGreater(len({tuple(r) for r in runs}), 1)

    def test_clean_and_acs_corrupt_prompts_differ_only_at_one_demo(self):
        query = self.task.query_data[0]
        demos = self.task.select_demonstrations(query, 4, seed=42)
        original = copy.deepcopy(demos)
        clean = create_text_prompt(self.task, query, 4, demonstrations=demos)
        corrupt = create_text_prompt(self.task, query, 4, corrupt_position=0, corruption_value='-', demonstrations=demos)
        self.assertEqual(demos, original)
        self.assertEqual(sum(a != b for a, b in zip(clean.splitlines(), corrupt.splitlines())), 1)
        self.assertTrue(clean.endswith('Answer:'))

    def test_fake_word_sampling_requires_matching_candidates(self):
        from tasks.t2t_tasks import FakeWordInductionTask
        task = FakeWordInductionTask.__new__(FakeWordInductionTask)
        task.support_data = [{'color': 'red', 'object': 'hat'}, {'color': 'blue', 'object': 'shoe'}]
        query = {'color': 'red', 'object': 'hat'}
        for mode in ('color', 'object'):
            with self.assertRaises(ValueError):
                task.select_demonstrations(query, 1, seed=42, task_type=mode)

    def test_final_answer_scoring(self):
        query = {'answer_color': 'red'}
        for response in ['red or blue', 'I considered red.\n#### blue', '#### not red', '#### red or blue']:
            self.assertFalse(evaluate_final_answer(query, response, 'fake_word_induction')[0], response)
        for response in ['#### red', 'red.', r'\boxed{red}']:
            self.assertTrue(evaluate_final_answer(query, response, 'fake_word_induction')[0], response)
        self.assertTrue(evaluate_final_answer({'answer': -2}, r'\boxed{-2}', 'operator_induction_text')[0])
        with self.assertRaises(ValueError): evaluate_responses([query], [], 'fake_word_induction')

    def test_full_demo_spans_and_attention_average(self):
        pieces = ['Support Set:\n', '8 ? 6 =\n', 'Answer', ':', ' 14', '\n3 ? 5 =\n', 'Answer', ':', ' 8', '\nQuestion:\n6 ? 3\n', 'Answer', ':']
        cats = categorize_tokens_text_fixed(list(range(len(pieces))), pieces, 2, None, False)
        self.assertEqual(cats['demo_1_tokens'], [1, 2, 3, 4])
        self.assertEqual(cats['demo_2_tokens'], [5, 6, 7, 8])
        weights = [torch.arange(144, dtype=torch.float).reshape(1, 12, 12)]
        result = compute_attention_allocation(weights, cats, 2)
        self.assertEqual(result[(0, 0, 0)], weights[0][0, 11, 1:5].mean().item())

    def test_sensitivity_uses_corrupted_forerunner(self):
        query = self.task.query_data[0]
        demos = self.task.select_demonstrations(query, 4, seed=42)
        clean_outputs = [torch.ones(1, 6, 2)]
        corrupted_outputs = torch.ones(1, 8, 2)
        corrupted_outputs[:, 7, :] = 3
        with patch('interp.heads.extract_acs_metrics.get_embeddings_text', return_value=(torch.zeros(1, 8, 2), [], [])), \
             patch('interp.heads.extract_acs_metrics.categorize_tokens_text_fixed', return_value={'query_forerunner': 7}), \
             patch('interp.heads.extract_acs_metrics.extract_attention_weights_and_outputs', return_value=([], [corrupted_outputs])):
            scores = compute_corruption_sensitivity(None, None, None, self.task, query, 4,
                clean_outputs, {'query_forerunner': 5}, 0, 'qwen3', demonstrations=demos)
        self.assertAlmostEqual(scores[(0, 0, 0)], 2.0, places=6)

    def test_probe_generation_held_out_and_paired(self):
        training = generate_probe_training_data('+', num_samples=40)
        evaluation = generate_evaluation_dataset(10)
        self.assertFalse({query_group(s['query']) for s in training} & {query_group(s['query']) for s in evaluation['all_correct']})
        self.assertEqual(sum(s['label'] for s in training), 20)
        for clean, corrupt in zip(evaluation['all_correct'], evaluation['one_corrupted']):
            self.assertEqual(clean['query'], corrupt['query'])
            self.assertEqual(clean['demonstrations'], corrupt['demonstrations'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
