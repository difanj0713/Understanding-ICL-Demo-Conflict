


import re
from typing import Optional, List

def extract_number_from_response(response: str) -> Optional[int]:
    














    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    

    answer_patterns = [
        r'final\s+answer\s*:\s*(-?\d+)',
        r'answer\s*:\s*(-?\d+)',
        r'result\s*:\s*(-?\d+)',
        r'solution\s*:\s*(-?\d+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:

                return int(matches[-1])
            except ValueError:
                continue
    

    equals_patterns = [
        r'=\s*(-?\d+)',
        r'equals\s+(-?\d+)',
    ]
    
    for pattern in equals_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:

                return int(matches[-1])
            except ValueError:
                continue
    

    context_patterns = [
        r'(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)\s+(-?\d+)',
        r'(?:is|equals?)\s+(-?\d+)',
    ]
    
    for pattern in context_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:

                return int(matches[-1])
            except ValueError:
                continue
    

    all_numbers = re.findall(r'-?\d+', response)
    if all_numbers:
        try:
            return int(all_numbers[-1])
        except ValueError:
            pass
    
    return None


def extract_text_from_response(response: str, query_answer: List[str]) -> Optional[List[str]]:



    if not response or not isinstance(response, str):
        return None
    
    response = response.strip().lower()
    
    if len(query_answer) == 2:
        found_components = []
        for component in query_answer:
            if component.lower() in response:
                found_components.append(component.lower())
        
        if len(found_components) == len(query_answer):
            return found_components
    
    for component in query_answer:
        if component.lower() in response:
            return [component.lower()]
    
    return None