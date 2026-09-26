import random
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

class BaseCorruptor(ABC):
    
    @abstractmethod
    def get_corruption_candidates(self, query: Dict, support_data: List[Dict]) -> List[Any]:
        pass
    
    @abstractmethod
    def apply_corruption(self, demonstration: Dict, corruption_value: Any) -> Dict:
        pass
    
    @abstractmethod
    def get_task_name(self) -> str:
        pass


class OperatorInductionCorruptor(BaseCorruptor):
    
    def get_task_name(self) -> str:
        return "operator_induction"
    
    def extract_numbers_from_path(self, image_path: str) -> Tuple[Optional[int], Optional[int]]:
        filename = os.path.basename(image_path)
        numbers = re.findall(r'\d+', filename)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None, None
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == 'x':
            return num1 * num2
        return 0
    
    def get_corruption_candidates(self, query: Dict, support_data: List[Dict]) -> List[str]:
        all_operators = ['+', '-', 'x']
        query_operator = query.get('operator')
        if query_operator:
            return [op for op in all_operators if op != query_operator]
        return all_operators
    
    def apply_corruption(self, demonstration: Dict, corruption_operator: str) -> Dict:
        corrupted_demo = demonstration.copy()
        corrupted_demo['_corrupted_operator'] = corruption_operator
        
        if 'image' in demonstration and len(demonstration['image']) > 0:
            num1, num2 = self.extract_numbers_from_path(demonstration['image'][0])
            if num1 is not None and num2 is not None:
                corrupted_answer = self.compute_answer(num1, num2, corruption_operator)
                if isinstance(demonstration.get('answer'), list):
                    operator_index = {'+': 0, '-': 1, 'x': 2}
                    if corruption_operator in operator_index:
                        corrupted_demo['answer'] = demonstration['answer'].copy()
                        corrupted_demo['answer'][operator_index[corruption_operator]] = corrupted_answer
                else:
                    corrupted_demo['answer'] = corrupted_answer
        
        return corrupted_demo


class OperatorInductionTextCorruptor(BaseCorruptor):
    
    def get_task_name(self) -> str:
        return "operator_induction_text"
    
    def extract_numbers_from_question(self, question: str) -> Tuple[Optional[int], Optional[int]]:
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None, None
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == 'x':
            return num1 * num2
        return 0
    
    def get_corruption_candidates(self, query: Dict, support_data: List[Dict]) -> List[str]:
        all_operators = ['+', '-', 'x']
        query_operator = query.get('operator')
        if query_operator:
            return [op for op in all_operators if op != query_operator]
        return all_operators
    
    def apply_corruption(self, demonstration: Dict, corruption_operator: str) -> Dict:
        corrupted_demo = demonstration.copy()
        corrupted_demo['_corrupted_operator'] = corruption_operator
        
        if 'question' in demonstration:
            num1, num2 = self.extract_numbers_from_question(demonstration['question'])
            if num1 is not None and num2 is not None:
                corrupted_answer = self.compute_answer(num1, num2, corruption_operator)
                if isinstance(demonstration.get('answer'), list):
                    operator_index = {'+': 0, '-': 1, 'x': 2}
                    if corruption_operator in operator_index:
                        corrupted_demo['answer'] = demonstration['answer'].copy()
                        corrupted_demo['answer'][operator_index[corruption_operator]] = corrupted_answer
                else:
                    corrupted_demo['answer'] = corrupted_answer
        
        return corrupted_demo

class CorruptionManager:
    def __init__(self):
        self.corruptors = {
            'operator_induction_text': OperatorInductionTextCorruptor(),
            'operator_induction_interleaved_text': OperatorInductionTextCorruptor(),
        }

    def get_corruptor(self, task_name: str, task_type: str = "color") -> Optional[BaseCorruptor]:
        if task_name == 'fake_word_induction':
            return FakeWordInductionCorruptor(task_type=task_type)
        return self.corruptors.get(task_name)

    def create_corrupted_demonstrations(self, task_name: str, query: Dict, demonstrations: List[Any],
                                     support_data: Any, corruption_position: int, task_type: str = "color") -> List[Any]:
        corruptor = self.get_corruptor(task_name, task_type)
        if not corruptor:
            raise ValueError(f"No corruptor available for task: {task_name}")
        
        corrupted_demos = demonstrations.copy()
        
        if 0 <= corruption_position < len(corrupted_demos):
            candidates = corruptor.get_corruption_candidates(query, support_data)
            
            if candidates:
                corruption_value = random.choice(candidates)
                
                corrupted_demos[corruption_position] = corruptor.apply_corruption(
                    demonstrations[corruption_position], corruption_value
                )
        
        return corrupted_demos
    
    def supports_task(self, task_name: str) -> bool:
        return task_name in self.corruptors

class FakeWordInductionCorruptor(BaseCorruptor):
    def __init__(self, task_type="color"):
        self.task_type = task_type
        import json
        concepts_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/fake_word_concepts.json")
        with open(concepts_file, 'r') as f:
            concepts = json.load(f)
        self.colors = list(concepts['colors'].keys())
        self.objects = list(concepts['objects'].keys())

    def get_task_name(self) -> str:
        return "fake_word_induction"

    def get_corruption_candidates(self, query: Dict, support_data: List[Dict]) -> List[str]:
        if self.task_type == "color":
            target = query['color']
            return [c for c in self.colors if c != target]
        else:
            target = query['object']
            return [o for o in self.objects if o != target]

    def apply_corruption(self, demonstration: Dict, corruption_value: str) -> Dict:
        corrupted_demo = demonstration.copy()
        if self.task_type == "color":
            corrupted_demo['real_phrase'] = f"{corruption_value} {corrupted_demo['object']}"
            corrupted_demo['color'] = corruption_value
        else:
            corrupted_demo['real_phrase'] = f"{corrupted_demo['color']} {corruption_value}"
            corrupted_demo['object'] = corruption_value
        return corrupted_demo
