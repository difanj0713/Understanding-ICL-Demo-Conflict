import random
import re
import copy
import os
import json
from typing import List, Dict, Optional, Tuple
from .base_task import BaseTask
import logging
from utils.experiment_protocol import evaluate_final_answer
from utils.evaluation_utils import extract_number_from_response, extract_text_from_response

logger = logging.getLogger(__name__)

class OperatorInductionTextTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("operator_induction_text", data_dir)
        self.hybrid_evaluator = None
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def _get_evaluator(self):
        if self.hybrid_evaluator is None:
            from evaluation.llm_judge import HybridEvaluator
            self.hybrid_evaluator = HybridEvaluator()
        return self.hybrid_evaluator
    
    def get_task_instruction(self, mode="constrained", warned=False) -> str:
        base_instruction = ("The text contains two digit numbers and a ? representing the mathematical operator. "
                           "Induce the mathematical operator (addition, multiplication, minus) according to the "
                           "results of the in-context examples and calculate the result.")

        if mode == "constrained":
            return base_instruction + " Think step by step, then write your final answer after ####"
        elif mode == "free":
            return base_instruction + " Reason carefully step by step."
        else:
            return base_instruction
    
    def format_demonstration(self, support_item: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "constrained":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        elif mode == "free":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        else:
            return f"{support_item['question']}\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int, seed: Optional[int] = None) -> List[Dict]:
        if n_shot == 0:
            return []
            
        operator_index = {'+': 0, '-': 1, 'x': 2}
        operator = query['operator']
        operator_idx = operator_index[operator]
        
        rng = random.Random(seed) if seed is not None else random
        
        candidates = [s for s in self.support_data if len(set(s['answer'])) == 3]
        if len(candidates) < n_shot:
            raise ValueError('Not enough operator demonstrations')
        selected = rng.sample(candidates, n_shot)
        demonstrations = []
        
        for support in selected:
            demo = copy.deepcopy(support)
            if isinstance(demo['answer'], list):
                demo['answer'] = demo['answer'][operator_idx]
            demonstrations.append(demo)
        
        
        return demonstrations
    
    def format_query(self, query: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "free":
            return f"{query['question']} Think for this question step by step."
        else:
            return f"{query['question']}"
    
    def evaluate_response(self, query: Dict, response: str, mode="constrained") -> bool:
        return evaluate_final_answer(query, response, 'operator_induction_text')[0]


class FakeWordInductionTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("fake_word_induction", data_dir)
        self.hybrid_evaluator = None
        with open(os.path.join(os.path.dirname(data_dir), "data/fake_word_concepts.json"), 'r') as f:
            self.concepts = json.load(f)
        self.colors = self.concepts['colors']
        self.objects = self.concepts['objects']

    def _get_evaluator(self):
        if self.hybrid_evaluator is None:
            from evaluation.llm_judge import HybridEvaluator
            self.hybrid_evaluator = HybridEvaluator()
        return self.hybrid_evaluator

    def get_task_instruction(self, mode="constrained", warned=False, task_type="color") -> str:
        if task_type == "color":
            base_instruction = "You will see examples showing what fake words mean in terms of real colors and objects. Learn the mapping from fake words to real words, then answer the question using real words only."
        else:
            base_instruction = "You will see examples showing what fake words mean in terms of real colors and objects. Learn the mapping from fake words to real words, then answer the question using real words only."

        if mode == "constrained":
            return base_instruction + " Think step by step, then write your final answer after ####"
        elif mode == "free":
            return base_instruction + " Reason carefully step by step."
        else:
            return base_instruction

    def format_demonstration(self, support_item: Dict, include_image_token=True, mode="constrained") -> str:
        return f"{support_item['fake_phrase']} means {support_item['real_phrase']}"

    def select_demonstrations(self, query: Dict, n_shot: int, seed: Optional[int] = None, task_type="color") -> List[Dict]:
        if n_shot == 0:
            return []

        rng = random.Random(seed) if seed is not None else random

        if task_type == "color":
            target_color = query['color']
            query_obj = query.get('query_object', query.get('object'))
            candidates = [s for s in self.support_data
                         if s['color'] == target_color and s['object'] != query_obj]
        else:
            target_object = query.get('query_object', query.get('object'))
            candidates = [s for s in self.support_data
                          if s['object'] == target_object and s['color'] != query['color']]

        if len(candidates) < n_shot:
            raise ValueError('Not enough matching demonstrations')
        selected = rng.sample(candidates, n_shot)


        return selected

    def format_query(self, query: Dict, include_image_token=True, mode="constrained", task_type="color") -> str:
        if task_type == "color":
            return query['question_color']
        else:
            return query['question_object']

    def evaluate_response(self, query: Dict, response: str, mode="constrained", task_type="color") -> bool:
        return evaluate_final_answer(query, response, 'fake_word_induction', task_type)[0]
