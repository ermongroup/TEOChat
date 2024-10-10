

from eval_geochat_referring import get_single_image_results, convert_geochat_string

from collections import defaultdict
import numpy as np
import json
import ast
import re
import cv2
from shapely import wkt, Polygon, box
from infer_utils import create_mask
from matplotlib.path import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
from matplotlib.path import Path



DIMENSIONS = {'FAST': 600,
              'SIOR': 800,
              'SOTA': 1024}

def calc_iou_individual_rotated(pred_box, gt_box, img_size=None):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        gt_box (list of floats): location of ground truth object as
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """

    pred_box = np.array(pred_box)
    gt_box = np.array(gt_box)
    pred_box = pred_box.reshape(4, 2)
    gt_box = gt_box.reshape(4, 2)
    pred_polygon = Polygon(pred_box)
    gt_polygon = Polygon(gt_box)
    intersection = pred_polygon.intersection(gt_polygon).area
    union = pred_polygon.union(gt_polygon).area
    iou = intersection / union

    return iou


def get_single_image_results_rotated(gt_boxes, pred_boxes, iou_thr, img_size=None):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [[x1,y1], [x2,y2], ...]
        pred_boxes (dict): dict of dicts of 'boxes' 
            [[x1,y1], [x2,y2], ...]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual_rotated(pred_box, gt_box, img_size)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def accuracy0_5(answer_path, dataset, aux_dataset="scripts/geochat_bench_dict.json"):
    # Replace with the path to the answers file
    results = None
    if dataset != "geochat_xbd":

        if type(answer_path) == dict:
            results = answer_path
        else:
            results = []
            with open(answer_path) as json_data:
                for line in json_data:
                    results.append(json.loads(line))

    with open(aux_dataset) as json_data:
        aux_results = json.load(json_data)

    img_results = {}
    num_bboxes = 0

    if dataset != "geochat_xbd":
        print("Number of images in Geochat: ", len(aux_results))
        print("Number of images predicted: ", len(results))
    
    i = 0
    # Loop over results and get precision, recall overall
    for id, result in tqdm(aux_results.items()):

        if dataset == "geochat_xbd":
            pred = result['answer']

            img_size = DIMENSIONS[result['dataset']]
            pred = convert_geochat_string(pred, img_size)

            ground_truth = result['ground_truth']
            ground_truth = np.array(ground_truth)
            num_bboxes += len(ground_truth)

            img_results[id] = get_single_image_results_rotated(ground_truth, pred, iou_thr=0.5)

        else:

            geochat_id = id.split(".")[0]

            img_size = DIMENSIONS[aux_results[geochat_id]['dataset']]
            ground_truth = result['ground_truth']
            ground_truth = np.array(ground_truth)
            num_bboxes += len(ground_truth)

            parsed_predicted = results[i]['predicted']
            # Load list of predicted and round truth bounding boxes for a single image
            try:
                predicted_boxes = ast.literal_eval("[" + parsed_predicted + "]")
            except:
                match = re.search(r'\[\[.*\]\]', parsed_predicted)
                if match:
                    predicted_boxes = ast.literal_eval(match.group())
                else:
                    predicted_boxes = []

            predicted_boxes = [[coord * 100 if coord < 1 else coord for coord in box] for box in predicted_boxes]
            
            # scale by img_size
            predicted_boxes = [[coord * img_size / 100 for coord in box] for box in predicted_boxes]

            assert results[i]['ground_truth'] == result['ground_truth']

            # convert the pred bboxes [xmin, ymin, xmax, ymax] to [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            pred_bboxes = []
            for bbox in predicted_boxes:
                x1, y1, x2, y2 = bbox
                pred_bboxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            img_results[id] = get_single_image_results_rotated(ground_truth, pred_bboxes, iou_thr=0.5, img_size=img_size)

        i+=1


    acc = np.sum([res['true_pos'] for res in img_results.values()]) / num_bboxes
    print("Acc@0.5: ", acc)
    return acc



if __name__ == '__main__':
    print("Geochat bench")
    geochat_path =  "scripts/geochat_bench_dict.json"
    answer_path = "scripts/geochat_bench_dict.json"
    acc_geochat = accuracy0_5(answer_path, dataset="geochat_xbd")
    print()


    print("Teochat bench")
    answer_path = "/deep/u/idormoy/aicc-win24-geo-vlm/videollava/scripts/geovlm/eval/QFabric/answers/geochat-referring-checkpoint14000_prompt_strategy_interleave_chronological_prefix_True_load_8bit_True_load_4bit_False_delete_system_prompt_False_tmp_0_end.json"
    acc_teochat = accuracy0_5(answer_path, dataset="geochat")
    print()


    print("Teochat-T bench")
    answer_path = "/deep/u/idormoy/aicc-win24-geo-vlm/videollava/videollava/eval/video/geochat-bench-ckpt8000-FIXED_prompt_strategy_interleave_chronological_prefix_True_load_8bit_False_load_4bit_True_delete_system_prompt_False_tmp_0_end (1).json"
    acc_teochatT = accuracy0_5(answer_path, dataset="geochat")
    print()



    print("VideoLLaVA bench")
    answer_path = "/deep/u/idormoy/aicc-win24-geo-vlm/videollava/videollava/eval/video/geochat-referring-Video-LLaVA-7B_prompt_strategy_interleave_chronological_prefix_True_load_8bit_False_load_4bit_True_delete_system_prompt_False_tmp_0_end (1).json"
    acc_videollava = accuracy0_5(answer_path, dataset="geochat")
    print()



    print("Overall accuracies")
    print("Geochat: ", acc_geochat)
    print("Teochat: ", acc_teochat)
    print("Teochat-T: ", acc_teochatT)
    print("VideoLLaVA: ", acc_videollava)

