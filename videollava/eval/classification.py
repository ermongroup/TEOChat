import string
from tqdm import tqdm
from collections import Counter

from videollava.eval.inference import run_inference


def classification_inference(
        dataset,
        model,
        tokenizer,
        processor,
        prompt_strategy,
        chronological_prefix,
        conv_mode,
        temperature,
        max_new_tokens
    ):
    outputs = []
    for example in tqdm(dataset):
        response = run_inference(
            model,
            processor,
            tokenizer,
            example["conversations"][0]['value'],
            example['video'],
            conv_mode=conv_mode,
            timestamps=example['timestamp'],
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        output = {
            'response': response,
            'ground_truth': example["conversations"][1]['value'],
            'task': example['task'],
        }
        outputs.append(output)
    return outputs


def classification_metrics(outputs, ignore_casing=True, ignore_punctuation=True):
    tps = Counter()
    task_counts = Counter()
    for output in outputs:
        response = output['response']
        ground_truth = output['ground_truth']
        task = output['task']
        if ignore_casing:
            response = response.lower()
            ground_truth = ground_truth.lower()
        if ignore_punctuation:
            response = response.translate(str.maketrans('', '', string.punctuation))
            ground_truth = ground_truth.translate(str.maketrans('', '', string.punctuation))
        if response == ground_truth:
            tps[task] += 1
        task_counts[task] += 1
        
    return {f"{task}_accuracy": tp / task_counts[task] for task, tp in tps.items()}
