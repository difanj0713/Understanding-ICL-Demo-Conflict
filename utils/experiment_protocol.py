import hashlib
import json
import re


def query_group(query):
    if 'operator' in query:
        numbers = re.findall(r'-?\d+', query['question'])
        return 'operator:' + ','.join(numbers[:2])
    return 'word:' + str(query.get('fake_phrase') or query.get('question_color') or query.get('question') or query.get('id'))


def select_query_split(queries, split, seed=1729):
    if split == 'all':
        return list(queries)
    if split not in ('discovery', 'evaluation'):
        raise ValueError(split)
    groups = sorted({query_group(q) for q in queries}, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    if len(groups) < 2:
        raise ValueError('Need at least two query groups for a held-out split')
    discovery = set(groups[:len(groups)//2])
    return [q for q in queries if (query_group(q) in discovery) == (split == 'discovery')]


def sample_seed(query, rollout, base_seed=42):
    identity = query.get('id') or json.dumps(query, sort_keys=True)
    digest = hashlib.sha256(f'{base_seed}:{rollout}:{identity}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def final_answer_segment(response):
    text = response.strip()
    if '####' in text:
        return text.rsplit('####', 1)[1].strip()
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    if boxed:
        return boxed[-1].strip()
    markers = list(re.finditer(r'\b(?:final answer|answer|result)\s*[:=]', text, re.I))
    if markers:
        return text[markers[-1].end():].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ''


def evaluate_final_answer(query, response, dataset, task_type='color'):
    segment = final_answer_segment(response)
    if dataset == 'fake_word_induction':
        expected = str(query.get('answer_' + task_type, query.get('answer', ''))).strip().lower()
        normalized = segment.strip(' \t\r\n.!,;:\"\'`*{}').lower()
        return bool(expected) and normalized == expected, normalized or None
    numbers = re.findall(r'-?\d+', segment)
    if not numbers:
        numbers = re.findall(r'=\s*(-?\d+)', response) or re.findall(r'-?\d+', response)
    predicted = int(numbers[-1]) if numbers else None
    expected = query.get('answer')
    if isinstance(expected, list):
        expected = expected[{'+': 0, '-': 1, 'x': 2, '*': 2}[query['operator']]]
    return predicted is not None and predicted == expected, predicted
