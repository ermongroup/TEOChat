"""
Code adapted from calculate_mean_ap.py
author: Timothy C. Arlen
date: 28 Feb 2018
"""
import sys
sys.path.append('/deep/u/joycech/aicc-working/videollava')

from collections import defaultdict
import numpy as np
import json
import ast
import re
import cv2
from shapely import wkt, Polygon, box
from infer_utils import create_mask, create_mask_s2looking


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    try:
        x1_p, y1_p, x2_p, y2_p = pred_box
    except:
        print("Prediction box is malformed? pred box: {}".format(pred_box))
        return 0.0

    if (x1_p > x2_p) or (y1_p > y2_p):
        print("Prediction box is malformed? pred box: {}".format(pred_box))
        return 0.0
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)

    return iou

def get_single_image_bound_results(gt_wkts, pred_boxes, img_size=256, dataset=None, id=None, predicted_mask=None, split=None, question=None):
    """
    Calculates upper bound and lower bound number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_wkts (list of strs): list of wkt strings of input polygons, scaled to raw pixel value
        pred_boxes (list of lists): list of list of boxes, where each box is formatted
            as [x_min, y_min, x_max, y_max] on scale from 0-100
        img_size (int): dimensions of the image. defaults to 256. 
    Returns:
        tuple of dicts: true positives (int), false positives (int), false negatives (int)
    """
    lb_preds = [[num * img_size / 100 for num in box] for box in pred_boxes]
    # add error handling for this type of outputs:  [0, 10, 12, 22], [0, 6, 12, 19], [0, 0], [31, 0]
    try:
        lb_preds = [box(*pred_box) for pred_box in lb_preds]
    except:
        lb_preds = []
        for pred_box in pred_boxes:
            if len(pred_box) == 4:
                lb_preds.append(box(*pred_box))

    if isinstance(gt_wkts, str):
        gt_polygons = [wkt.loads(gt_wkts)]
    elif gt_wkts is None:
        gt_polygons = []
    else: 
        gt_polygons = [wkt.loads(gt_wkt) for gt_wkt in gt_wkts]

    # get mask of all gt_polygons and lb_preds
    if dataset == None:
        gt_mask = create_mask(gt_polygons, (img_size, img_size))
    else:
        gt_mask = create_mask_s2looking(id, split=split, question=question) 
        #gt_mask = create_mask(gt_polygons, (img_size, img_size))

    if dataset != "geochat_s2looking":
        lb_preds_mask = create_mask(lb_preds, (img_size, img_size))
    else:
        lb_preds_mask = predicted_mask
        

    # get lower bound intersection and union masks 
    intersection = np.logical_and(gt_mask, lb_preds_mask)
    union = np.logical_or(gt_mask, lb_preds_mask)

    # compute lb metrics
    lower_bound_iou = np.sum(intersection) / np.sum(union)
    if np.sum(intersection) == 0 and np.sum(union) == 0:
        return None, None
    if np.isnan(lower_bound_iou):
        lower_bound_iou = 0

        
    fp = np.sum(np.logical_and(lb_preds_mask, np.logical_not(gt_mask)))
    tp = np.sum(np.logical_and(lb_preds_mask, gt_mask))
    fn = np.sum(np.logical_and(np.logical_not(lb_preds_mask), gt_mask))
    lb_stats = {'true_pos': tp, 
                'false_pos': fp, 
                'false_neg': fn, 
                'intersection': np.sum(intersection),
                'union': np.sum(union)}

    return lb_stats

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
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
            iou = calc_iou_individual(pred_box, gt_box)
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

def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
        print(true_pos, "true_pos", false_pos, "false_pos", false_neg, "false_neg")
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def extract_bboxes(input_string):
    """
    Takes as an input a string like in the image, there are two buildings that have been changed. the first building is located at [0.0, 0.69, 0.45, 0.9] and the second building is located at [0.46, 0.69, 0.99, 0.91]
    Returns a list of bounding boxes in the format [x_min, y_min, x_max, y_max]
    Input:
        input_string (str): string containing the bounding boxes
    Returns:
        list of lists: list of bounding boxes
    """
    matches = re.findall(r'\[\[.*?\]\]', input_string)
    return [ast.literal_eval(match) for match in matches]


def referring_expression(answer_path, dataset, verbose=False, saving_path_root=None, img_size=256, split=None):
    if type(answer_path) == dict:
        results = answer_path
    else:
        with open(answer_path) as json_data:
            results = json.load(json_data)

    img_results = {}
    lb_results = {}
    # Loop over results and get precision, recall overall
    for id, result in results.items():
        if 'temporal_referring_expression' in result['task']:
            if not "s2looking" in dataset:
                continue  # no bounding box outputs for temporal_referring_expression
        
        # for the geochat s2looking predictions, we work directly with the predicted mask instead of the bounding boxes
        if dataset == 'geochat_s2looking': 
            if 'referring_expression' in result['task'] or 'localization' in result['task']:
                lb_res = get_single_image_bound_results(result['original_input_polygon'], [], dataset=dataset, id=id, predicted_mask=result['predicted_mask'], split=split, question=result["question"])
                if lb_res != None:
                    lb_results[id] = lb_res
                continue
            elif 'question_answering' in result['task']:
                continue

        if 'referring_expression' in result['task'] or 'largest building' in result['task'] or "canonical" in result['task'] or 'localization' in result['task'] \
            or 'geochat_referring' in result['task']:
            # No bounding boxes in predicted string
            if "[" not in result["predicted"]:
                # Ground truth has no bounding boxes
                if result["ground_truth"].startswith("There are no") or "no" in result["ground_truth"] or "No" in result["ground_truth"]:
                    # Discard true negatives
                    continue
                # Ground truth has bounding boxes, not identified by the model --> all false negatives
                else:
                    false_neg = "[" + result["ground_truth"] + "]"
                    false_neg = false_neg.replace(".", "")
                    
                    try:
                        false_neg = len(ast.literal_eval(false_neg))
                    except:
                        # count the number of opening '[' in the string
                        false_neg = false_neg.count('[') - 1
                    if not "s2looking" in dataset:
                        gt_mask = create_mask(wkt.loads(result['original_input_polygon']), (img_size, img_size)) 
                    else:
                        gt_mask = create_mask_s2looking(id, split=split, question=result['question']) 
                        # gt_mask = create_mask(wkt.loads(result['original_input_polygon']), (img_size, img_size)) 
                    img_results[id] = {'true_pos': 0, 'false_pos': 0, 'false_neg': false_neg, 'intersection':0, 'union':false_neg}
                    false_neg = np.sum(gt_mask)
                    lb_results[id] = {'true_pos': 0, 'false_pos': 0, 'false_neg': false_neg, 'intersection':0, 'union':false_neg}

            # Bounding boxes in predicted and output string --> compare bounding boxes
            else:

                # To deal with cases where the model outputs an incomplete bounding box (e.g. "[24, 76,")
                first_open_bracket_ind = result["predicted"].find("[")
                last_close_bracket_ind = result["predicted"].rfind("]")
                if last_close_bracket_ind != -1 and first_open_bracket_ind != -1:
                    parsed_predicted = result["predicted"][first_open_bracket_ind:last_close_bracket_ind+1]
                else:   
                    parsed_predicted = ""      

                # Load list of predicted bounding boxes
                try:
                    predicted_boxes = ast.literal_eval("[" + parsed_predicted + "]")
                except:
                    match = re.search(r'\[\[.*\]\]', result["predicted"])
                    if match:
                        predicted_boxes = ast.literal_eval(match.group())
                    else:
                        predicted_boxes = []

                predicted_boxes = [[coord * 100 if coord < 1 else coord for coord in box] for box in predicted_boxes]

                # Load list of ground truth bounding boxes
                if result["ground_truth"].startswith("There are no") or "no" in result["ground_truth"].lower():
                    # If ground truth contains no boxes
                    ground_truth_boxes = []
                first_open_bracket_ind = result["ground_truth"].find("[")
                last_close_bracket_ind = result["ground_truth"].rfind("]")
                if last_close_bracket_ind != -1 and first_open_bracket_ind != -1:
                    parsed_gt = result["ground_truth"][first_open_bracket_ind:last_close_bracket_ind+1]
                else:   
                    parsed_gt = ""      
                try: 
                    ground_truth_boxes = ast.literal_eval("[" + parsed_gt + "]")
                except: 
                    match = re.search(r'\[\[.*\]\]', result["ground_truth"])
                    if match:
                        ground_truth_boxes = ast.literal_eval(match.group())
                    else:
                        ground_truth_boxes = []

                # Get mask results from the two previous parsings
                gt_wkts = result['original_input_polygon']
                img_results[id] = get_single_image_results(ground_truth_boxes, predicted_boxes, iou_thr=0.5) ######

                if 'referring_expression' in result['task'] or 'largest building' in result['task'] or "canonical" in result['task'] or 'localization' in result['task']:
                    if not "s2looking" in dataset:
                        lb_results[id] = get_single_image_bound_results(gt_wkts, predicted_boxes) 
                    elif dataset=="s2looking":
                        lb_results[id] = get_single_image_bound_results(gt_wkts, predicted_boxes, dataset=dataset, id=id, split=split, question=result["question"])
                    else:
                        lb_results[id] = get_single_image_bound_results(gt_wkts, predicted_boxes, predicted_mask=result['predicted_mask'], split=split, question=result["question"])

    precision, recall = calc_precision_recall(img_results)
    print("Referring expression results (precision, recall): ", precision, recall)
    print("Acc@0.5: ", np.sum([res['true_pos'] for res in img_results.values()]) / len(results.keys()))

    if len(lb_results) != 0:
        lb_intersection = np.sum([res['intersection'] for res in lb_results.values()])
        lb_union = np.sum([res['union'] for res in lb_results.values()])
        print("Lower bound IOU: ", lb_intersection / lb_union if lb_union != 0 else 0)
        lb_precision, lb_recall = calc_precision_recall(lb_results)
        print('Lower bound precision: ', lb_precision)
        print('Lower bound recall: ', lb_recall)
        print("Lower bound F1: ", 2 * (lb_precision * lb_recall) / (lb_precision + lb_recall) if (lb_precision + lb_recall) != 0 else 0)

    if saving_path_root:
        with open(f"{saving_path_root}/referring_expression_scores.json", 'w') as f:
            json.dump(img_results, f)

if __name__ == '__main__':
    answer_path = "scripts/geovlm/eval/xBD/answers/ckpt14000-old-aux-xbd-test-canon-auxiliary_interleave.json"
    referring_expression(answer_path, dataset="xbd")