import string
from collections import Counter


def get_string_cleaner(ignore_casing, ignore_punctuation):
    def clean_string(str):
        if ignore_casing:
            str = str.lower()
        if ignore_punctuation:
            str = str.translate(str.maketrans('', '', string.punctuation))
        return str
    return clean_string


def classification_metrics(outputs, ignore_casing=True, ignore_punctuation=True, keywords=None, **kwargs):
    tps = Counter()
    task_counts = Counter()
    clean_string = get_string_cleaner(ignore_casing, ignore_punctuation)
    for output in outputs:
        response = output['response']
        ground_truth = output['ground_truth']
        task = output['task']
        response = clean_string(response)
        ground_truth = clean_string(ground_truth)
        if keywords is not None:
            matched_keyword = False
            for keyword in keywords:
                if keyword in response and keyword in ground_truth:
                    tps[task] += 1
                    matched_keyword = True
                    break
            if not matched_keyword:
                if response == ground_truth:
                    tps[task] += 1
                else:
                    print(response, ground_truth)
        elif response == ground_truth:
            tps[task] += 1
        task_counts[task] += 1
        
    return {f"{task}_accuracy": tp / task_counts[task] for task, tp in tps.items()}
