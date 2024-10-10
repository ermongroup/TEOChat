"""
Code adapted from calculate_mean_ap.py
author: Timothy C. Arlen
date: 28 Feb 2018
"""

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

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

from eval_referring import referring_expression
import matplotlib.pyplot as plt
import time
import math
from matplotlib.path import Path

def convert_geochat_string(build, img_size=256):
    """
    Convert the raw str geochat output {<40><89><56><100>|<57>}, {<0><89><56><100>|<57>}
    to a list of rotated bboxes.
    """
    build = build.strip('{}')
    bbox_segments = build.split("}{")
    # Regular expression to find all numbers inside angle brackets
    pattern = r"<(\d+)>"

    # Extract numbers, convert them to integers, and collect into a list
    bboxes = [
        list(map(int, re.findall(pattern, segment)))
        for segment in bbox_segments
    ]

    rotated_bboxes = []
    for bbox in bboxes:
        try:
            xmin, ymin, xmax, ymax, angle = [float(v) for v in bbox]
        except:
            print("Warning - Malformed bbox: ", bbox)
            print("Original string: ", build)
            print()
            continue

        # Convert percentages to pixel coordinates
        xmin = xmin * img_size / 100
        ymin = ymin * img_size / 100
        xmax = xmax * img_size / 100
        ymax = ymax * img_size / 100
        
        # Calculate rectangle dimensions
        rect_width = xmax - xmin
        rect_height = ymax - ymin
        center_x = xmin + rect_width / 2
        center_y = ymin + rect_height / 2
        
        # Calculate corners before rotation
        corners = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ])
        
        # Rotate corners
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        rotated_corners = []
        for x, y in corners:
            tx = x - center_x
            ty = y - center_y
            rotated_x = tx * cos_angle - ty * sin_angle + center_x
            rotated_y = tx * sin_angle + ty * cos_angle + center_y
            rotated_corners.append([rotated_x, rotated_y])

        rotated_bboxes.append(np.array(rotated_corners))

    return rotated_bboxes

def create_geochat_mask(buildings, img_size=(256, 256)):
    """
    Given a list of buildings in an image, this function 
    - creates an img_size * img_size numpy array for the image
    - returns the mask for all buildings
    Input:
    - buildings: List of geochat strings representing buildings
    - img_size: Tuple indicating the size of the image (height, width)
    """
    mask = np.zeros(img_size, np.uint8)

    # Fill in with ones the pixels that are inside the buildings (rotated bboxes)
    for bbox in buildings:
        path = Path(bbox)
        x, y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        points = np.vstack((x.flatten(), y.flatten())).T
        mask[path.contains_points(points).reshape(img_size)] = 1

    return mask

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
        return 0.0

    if (x1_p > x2_p) or (y1_p > y2_p):
        print("Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        print("Ground Truth box is malformed? true box: {}".format(gt_box))

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

def calc_iou_individual_rotated(pred_box, gt_box):
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
    try:
        pred_box = np.array(pred_box)
        gt_box = np.array(gt_box)
    except:
        return 0.0
    if len(pred_box) == 4:
        pred_box = [[pred_box[0], pred_box[1]], [pred_box[2], pred_box[1]], [pred_box[2], pred_box[3]], [pred_box[0], pred_box[3]]]
    if len(gt_box) == 4:
        gt_box = [[gt_box[0], gt_box[1]], [gt_box[2], gt_box[1]], [gt_box[2], gt_box[3]], [gt_box[0], gt_box[3]]]
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

    # try:
    #     pred_box = np.array(pred_box)
    #     gt_box = np.array(gt_box)
    # except:
    #     return 0.0

    # pred_box = pred_box.reshape(4, 2)
    # gt_box = gt_box.reshape(4, 2)

    # pred_polygon = Polygon(pred_box)
    # gt_polygon = Polygon(gt_box)

    # intersection = pred_polygon.intersection(gt_polygon).area
    # union = pred_polygon.union(gt_polygon).area

    # iou = intersection / union

    # plt.figure()
    # plt.plot(*pred_polygon.exterior.xy, color='r', label='pred')
    # plt.plot(*gt_polygon.exterior.xy, color='b', label='gt')
    # plt.legend()
    # plt.title(f"IoU: {iou}")
    # plt.show()
    # plt.savefig("iou.png")
    # time.sleep(1)
    # plt.close()

    return iou
    

def get_single_image_bound_results(gt_wkts, pred_geochat_string, img_size=256):
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
    if isinstance(gt_wkts, str):
        gt_polygons = [wkt.loads(gt_wkts)]
    else: 
        gt_polygons = [wkt.loads(gt_wkt) for gt_wkt in gt_wkts]

    # # Needs fixing for auxiliary
    # if len(gt_polygons) == 0:
    #     false_neg = np.sum(gt_mask)
    #     ub_stats= {'true_pos': 0, 'false_pos': 0, 'false_neg': false_neg, 'intersection':0, 'union':false_neg}
    #     lb_stats = {'true_pos': 0, 'false_pos': 0, 'false_neg': false_neg, 'intersection':0, 'union':false_neg}
    #     return lb_stats, ub_stats

    lb_preds = convert_geochat_string(pred_geochat_string, img_size)
    # get mask of all gt_polygons and lb_preds
    gt_mask = create_mask(gt_polygons, (img_size, img_size))
    lb_preds_mask = create_geochat_mask(lb_preds, (img_size, img_size))

    # get lower bound intersection and union masks 
    intersection = np.logical_and(gt_mask, lb_preds_mask)
    union = np.logical_or(gt_mask, lb_preds_mask)

    # compute lb metrics
    # lower_bound_iou = np.sum(intersection) / np.sum(union)
    fp = np.sum(np.logical_and(lb_preds_mask, np.logical_not(gt_mask)))
    tp = np.sum(np.logical_and(lb_preds_mask, gt_mask))
    fn = np.sum(np.logical_and(np.logical_not(lb_preds_mask), gt_mask))
    lb_stats = {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'intersection': np.sum(intersection), 'union': np.sum(union)}

    # get upper bound intersection and union masks
    ub_pred_mask = np.logical_and(gt_mask, lb_preds_mask)
    intersection = np.logical_and(ub_pred_mask, gt_mask)
    union = np.logical_or(gt_mask, ub_pred_mask)

    # compute ub metrics
    # upper_bound_iou = np.sum(intersection) / np.sum(union)
    ub_fp = np.sum(np.logical_and(ub_pred_mask, np.logical_not(gt_mask)))
    ub_tp = np.sum(np.logical_and(ub_pred_mask, gt_mask))
    ub_fn = np.sum(np.logical_and(np.logical_not(ub_pred_mask), gt_mask))
    ub_stats = {'true_pos': ub_tp, 'false_pos': ub_fp, 'false_neg': ub_fn, 'intersection': np.sum(intersection), 'union': np.sum(union)}

    return lb_stats, ub_stats

def get_geochat_dataset(image_id):
    if image_id.startswith("P"):
        dataset = "SOTA"
    elif image_id.startswith("train"):
        dataset = "FAST"
    else:
        dataset = "SIOR"
    return dataset

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
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
            iou = calc_iou_individual_rotated(pred_box, gt_box)
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
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)


DIMENSIONS = {'FAST': 600,
              'SIOR': 800,
              'SOTA': 1024}


def referring_expression(answer_path, dataset, verbose=False, saving_path_root=None, img_size=256):
    # Replace with the path to the answers file
    if type(answer_path) == dict:
        results = answer_path
    else:
        with open(answer_path) as json_data:
            results = json.load(json_data)

    img_results = {}
    ub_results = {}
    lb_results = {}
    num_bboxes = 0
    # Loop over results and get precision, recall overall
    for id, result in tqdm(results.items()):

        if dataset == "geochat_xbd":
            pred = result['predicted']

            dataset = get_geochat_dataset(id)
            img_size = (DIMENSIONS[dataset])
            pred = convert_geochat_string(pred, img_size)

            ground_truth = result['ground_truth']
            ground_truth = np.array(ground_truth)
            num_bboxes += len(ground_truth)

            img_results[id] = get_single_image_results(ground_truth, pred, iou_thr=0.5)

            continue

        try:
            if 'referring_expression' not in result['task']:
                continue  # no bounding box outputs for temporal_referring_expression
        except:
            pass
        
        # TODO: LOOP THROUGH IDENTIFY TASKS/QUESTIONS IN THE DATASET
       
        # TODO: HANDLE WHEN THERE ARE NO BOUNDING BOXES IN GROUND TRUTH for auxiliary tasks
        if not result['original_input_polygon']:
            first_open_bracket_ind = result["predicted"].find("{")
            last_close_bracket_ind = result["predicted"].rfind("}")
            if last_close_bracket_ind != -1 and first_open_bracket_ind != -1:
                parsed_predicted = result["predicted"][first_open_bracket_ind:last_close_bracket_ind+1]
            else:   
                parsed_predicted = ""      
            predicted_boxes = convert_geochat_string(parsed_predicted)
            # If ground truth contains no boxes: all predictions are false positives
            false_pos = len(predicted_boxes)
            false_pos_pixels = np.sum(create_geochat_mask(predicted_boxes))
            img_results[id] = {'true_pos': 0, 'false_pos': false_pos, 'false_neg': 0, 'intersection':0, 'union':false_pos_pixels}
            ub_results[id] = {'true_pos': 0, 'false_pos': false_pos_pixels, 'false_neg': 0, 'intersection':0, 'union':false_pos_pixels}
            lb_results[id] = {'true_pos': 0, 'false_pos': false_pos_pixels, 'false_neg': 0, 'intersection':0, 'union':false_pos_pixels}
            continue
        else: # Ground truth contains boxes: find predicted Geochat boxes
            first_open_bracket_ind = result["predicted"].find("{")
            last_close_bracket_ind = result["predicted"].rfind("}")
            if last_close_bracket_ind != -1 and first_open_bracket_ind != -1:
                parsed_predicted = result["predicted"][first_open_bracket_ind:last_close_bracket_ind+1]
            else:   
                parsed_predicted = "" 
            gt_wkts = result['original_input_polygon']  
            lb_results[id], ub_results[id] = get_single_image_bound_results(gt_wkts, parsed_predicted)

    if len(ub_results) != 0:
        ub_intersection = np.sum([res['intersection'] for res in ub_results.values()])
        ub_union = np.sum([res['union'] for res in ub_results.values()])
        lb_intersection = np.sum([res['intersection'] for res in lb_results.values()])
        lb_union = np.sum([res['union'] for res in lb_results.values()])
        print("Upper bound IOU: ", ub_intersection / ub_union if ub_union != 0 else 0)
        print("Lower bound IOU: ", lb_intersection / lb_union if lb_union != 0 else 0)
        ub_precision, ub_recall = calc_precision_recall(ub_results)
        lb_precision, lb_recall = calc_precision_recall(lb_results)
        print('Lower bound precision: ', lb_precision)
        print('Lower bound recall: ', lb_recall)
        print("Upper bound F1: ", 2 * (ub_precision * ub_recall) / (ub_precision + ub_recall) if (ub_precision + ub_recall) != 0 else 0)
        print("Lower bound F1: ", 2 * (lb_precision * lb_recall) / (lb_precision + lb_recall) if (lb_precision + lb_recall) != 0 else 0)

    print("Acc@0.5: ", np.sum([res['true_pos'] for res in img_results.values()]) / num_bboxes)

    if type(answer_path) == dict:
        return 
    
    if saving_path_root:
        with open(f"{saving_path_root}/referring_expression_scores.json", 'w') as f:
            json.dump(img_results, f)

if __name__ == '__main__':
    answer_path = "scripts/geovlm/eval/xBD/answers/ckpt14000-geochat-bench_interleave_test.json"
    referring_expression(answer_path, dataset="geochat_xbd")
    #answer_path = "scripts/geochat/eval/xBD/geochat_xbd_test_auxiliary_dict.json"
    # referring_expression(answer_path, dataset="xbd")

