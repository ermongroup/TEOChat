import json
import numpy as np
from infer_utils import create_mask
from shapely.wkt import loads
from collections import defaultdict
from tqdm import tqdm

def clean_string(s):
    return s.replace(' ', '-').replace('.', '').lower()

def get_class_dict(dataset):
    if dataset == "qfabric":
        class_dict = {
            "temporal_region_based_question_answering: What is the development status in this region [bbox] in image N?":
            {
                "prior-construction": 1, 
                "greenland ": 2, 
                "land-cleared": 3, 
                "excavation": 4, 
                "materials-dumped": 5, 
                "construction-started": 6, 
                "construction-midway": 7, 
                "construction-done": 8, 
                "operational": 9
            },
            "region_based_question_answering: Identify the type of urban development that has occurred in this area [bbox].": 
            {
                "residential": 10,
                "commercial": 11,
                "industrial": 12,
                "road": 13,
                "demolition": 14,
                "mega-projects": 15
            }
        }
    elif dataset == "xbd":
        class_dict = {
            "classification: Classify the level of damage experienced by the building at location [bbox] in the second image. Choose from: No damage, Minor Damage, Major Damage, Destroyed.": 
            {
                "no-damage": 1,
                "minor-damage": 2,
                "major-damage": 3,
                "destroyed": 4,
            }
        }
    else:
        raise ValueError(f"Dataset {dataset} should not be evaluated on segmentation classification.")
    return class_dict



def classification_segmentation(answer_path, dataset, per_class_f1=False, height=256, width=256):
    """
    Given the path to the answer file, this function creates segmentation masks on the original polygon for the predicted and ground truth classes.
    Returns the class-weighted per-pixel F1 between predicted and ground-truth masks.
    """
    with open(answer_path) as f:
        results = json.load(f)

    classes = get_class_dict(dataset)
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})

    for result in tqdm(results.values()):
        if result['task'] not in classes:
            continue
        class_dict = classes[result['task']]
        predicted_class = clean_string(result['predicted'])
        try:
            ground_truth_class = clean_string(result["ground_truth"])
        except:
            ground_truth_class = clean_string(result["original_answer"])
        original_polygon = loads(result['original_input_polygon'])
        
        pred_msk = np.zeros((height, width), dtype='uint8')
        gt_msk = np.zeros((height, width), dtype='uint8')
        _msk = create_mask(original_polygon, im_size=(height, width))

        if predicted_class not in class_dict or ground_truth_class not in class_dict:
            continue
        
        pred_label = class_dict[predicted_class]
        gt_label = class_dict[ground_truth_class]
        pred_msk[_msk > 0] = pred_label
        gt_msk[_msk > 0] = gt_label

        for label in class_dict.values():
            pred_mask = (pred_msk == label)
            gt_mask = (gt_msk == label)
            tp = np.sum(pred_mask & gt_mask)
            fp = np.sum(pred_mask & ~gt_mask)
            fn = np.sum(~pred_mask & gt_mask)
            
            class_stats[label]['tp'] += tp
            class_stats[label]['fp'] += fp
            class_stats[label]['fn'] += fn
            class_stats[label]['count'] += np.sum(gt_mask)

    
    scores_dict = {}

    for task, class_info in classes.items():
        print(f"Task: {task}")
        class_f1_scores = {}
        weighted_f1_score = 0
        total_weight = 0

        tp = 0
        fp = 0
        fn = 0
        for class_name, class_label in class_info.items():
            stats = class_stats[class_label]
            total_samples = sum(stats['count'] for label, stats in class_stats.items() if label in class_info.values())

            if stats['tp'] + stats['fp'] == 0 or stats['tp'] + stats['fn'] == 0:
                f1 = 0.0
            else:
                precision = stats['tp'] / (stats['tp'] + stats['fp'])
                recall = stats['tp'] / (stats['tp'] + stats['fn'])
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
            class_f1_scores[class_name] = f1

            if stats['count'] > 0:
                prevalence_inv = total_samples / stats['count']
                weighted_f1_score += f1 * prevalence_inv
                total_weight += prevalence_inv
            
            tp += stats['tp']
            fp += stats['fp']
            fn += stats['fn']
        
        if tp + fp == 0 or tp + fn == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = tp / (tp + 0.5 * (fp + fn))

        if total_weight > 0:
            weighted_f1_score /= total_weight
        else:
            weighted_f1_score = 0.0

        scores_dict[task] = (class_f1_scores, weighted_f1_score)
        print(f"Per-class F1 scores: {class_f1_scores}")
        if dataset == 'qfabric':
            print(f"Micro average F1 score: ", micro_f1)
        else: 
            print(f"Weighted average F1 score: {weighted_f1_score}")
    
    return scores_dict