"""
Segmentation metric code dapted from code for XView2: A Strong Baseline
Xview2_Strong_Baseline/legacy/xview2_metrics.py
Xview2_Strong_Baseline/legacy/create_masks.py
"""
# add python path
# import sys
# import os
# sys.path.append('/deep/u/emily712/aicc-win24-geo-vlm/videollava/')

import json
import string
import numpy as np
import cv2
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from shapely.geometry import Polygon
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm


def compute_tp_fn_fp(pred: np.ndarray, targ: np.ndarray, c: int):
    """
    Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)

    Args:
        pred (np.ndarray): prediction
        targ (np.ndarray): target
        c (int): positive class
    """
    TP = np.logical_and(pred == c, targ == c).sum()
    FN = np.logical_and(pred != c, targ == c).sum()
    FP = np.logical_and(pred == c, targ != c).sum()
    return [TP, FN, FP]


def accuracy_precision_recall(answer_path, dataset, ignore_punctuation=True, verbose=True):
    # Replace with the path to the answers file
    if type(answer_path) == dict:
        results = answer_path
    else:
        with open(answer_path) as json_data:
            results = json.load(json_data)

    task_total = defaultdict(int)
    task_tp = defaultdict(int)

    binary_classification = defaultdict(bool)
    binary_fp = defaultdict(int)
    binary_fn = defaultdict(int)

    # Dictionary of dictionaries. Key: task. Value: {class: count}
    ground_truths = defaultdict(dict)

    values = defaultdict(list)

    accepted_tasks = [
        "temporal_question_answering",
        "region_based_question_answering",
        "temporal_region_based_question_answering",
        "question_answering",
        "temporal_referring_expression",
        "rural_urban",
        "comp",
        "presence",
        "count",
        "change_to_what",
        "smallest_change",
        "change_or_not",
        "change_ratio",
        "largest_change",
        "change_ratio_types",
        "increase_or_not",
        "decrease_or_not"
    ]

    for result in results.values():
        if "task" in result and not any(result["task"].startswith(task) for task in accepted_tasks):
            continue

        # Clean predicted string if necessary
        result["predicted"] = result["predicted"].lower()
        result["ground_truth"] = result["ground_truth"].lower()
        if ignore_punctuation:
            result["predicted"] = ''.join(ch for ch in result["predicted"] if ch not in string.punctuation)
            result["ground_truth"] = ''.join(ch for ch in result["ground_truth"] if ch not in string.punctuation)
        if verbose:
            values["predicted"].append(result["predicted"])
            values["ground_truth"].append(result["ground_truth"])
            values["correct_incorrect"].append("Correct" if result["predicted"] == result["ground_truth"] else "Incorrect")
        if "task" not in result:
            result["task"] = dataset

        # True positive
        if result["predicted"] == result["ground_truth"]:
            task_tp[result["task"]] += 1
        task_total[result["task"]] += 1

        # If binary classification (yes/no question), calculate precision and recall metrics
        binary_classification[result["task"]] = binary_classification[result["task"]] or (result["ground_truth"] in ["yes", "no"])
        if binary_classification[result["task"]]:
            if result["predicted"] != "no" and result["ground_truth"] == "no":
                binary_fp[result["task"]] += 1
            if result["predicted"] != "yes" and result["ground_truth"] == "yes": 
                binary_fn[result["task"]] += 1

        # Update ground truth counts for the task
        task = result["task"]
        class_label = result["ground_truth"]
        ground_truths[task][class_label] = ground_truths[task].get(class_label, 0) + 1
    
    # Print tab separated values
    if verbose:
        max_len = max(len(v) for v in values["ground_truth"]) + 5
        print("Predicted" + " " * (max_len - 9) + "\tGround Truth" + " " * (max_len - 12) + "\tCorrect/Incorrect")
        for i in range(len(values["predicted"])):
            print(values["predicted"][i] + " " * (max_len - len(values["predicted"][i])) + "\t" + values["ground_truth"][i] + " " * (max_len - len(values["ground_truth"][i])) + "\t" + values["correct_incorrect"][i])

    total_tp = 0
    total_predictions = 0
    for task in task_tp:
        acc_string = "Accuracy"
        if ignore_punctuation:
            acc_string += " (ignoring punctuation)"
        print(f"{acc_string} for {task}: {round((task_tp[task] /  task_total[task]), 4) * 100}%")

        if binary_classification[task]:
            if (task_tp[task] + binary_fp[task]) > 0:
                print(f"Precision (ignoring punctuation) for {task}: {round((task_tp[task] / (task_tp[task] + binary_fp[task])), 3) * 100}%")
            if (task_tp[task] + binary_fn[task]) > 0:
                print(f"Recall (ignoring punctuation) for {task}: {round((task_tp[task] / (task_tp[task] + binary_fn[task])), 3) * 100}%")

        majority_class = max(ground_truths[task], key=ground_truths[task].get)
        majority_class_percentage = (ground_truths[task][majority_class] / task_total[task]) * 100
        print(f"Majority class for {task}: {majority_class}, Percentage: {round(majority_class_percentage, 4)}%")

        total_tp += task_tp[task]
        total_predictions += task_total[task]

    if total_predictions == 0:
        print("No predictions made.")
    else:
        total_accuracy = (total_tp / total_predictions) * 100
        print(f"Overall Accuracy: {round(total_accuracy, 3)}%")

# For testing accuracy/precision/recall on a particular script without running inference
if __name__ == '__main__':
    root_dir = '/deep/u/jirvin16/aicc/aicc-win24-geo-vlm/videollava/scripts/geovlm/eval/QFabric/answers/'
    answer_path =  root_dir + "video-llava-7b-8bit-lora-final-no-metadata-zero-gc-acc8-freq-no-geochat-checkpoint-8000_qfabric_test_aux_data_test_prompt_strategy_interleave_chronological_prefix_True_load_8bit_True_load_4bit_False_delete_system_prompt_False.json"
    accuracy_precision_recall(answer_path, dataset="qfabric", ignore_punctuation=True, verbose=False)
