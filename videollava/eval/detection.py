import re
import numpy as np
from tqdm import tqdm
from shapely import wkt
from shapely.wkt import loads
from PIL import Image, ImageDraw
from collections import defaultdict

from videollava.eval.classification import get_string_cleaner, classification_metrics


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1

    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        assert self.confusion_matrix.shape[0] == 2
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        # Number of observations (total number of classifications)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def Class_Weighted_F1_score(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        # Ignore class 0
        class_weights = 1 / np.sum(self.confusion_matrix, axis=1)[1:]
        class_weights = class_weights / np.sum(class_weights)
        weighted_f1_score = np.sum(class_weights * f1_scores)
        return weighted_f1_score

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def get_classes(dataset, task):
    if dataset == "qfabric":
        class_dict = {
            "temporal_region_based_question_answering: What is the development status in this region [bbox] in image N?":
                ["prior-construction", "greenland ", "land-cleared", "excavation", "materials-dumped", "construction-started",
                 "construction-midway", "construction-done", "operational"],
            "region_based_question_answering: Identify the type of urban development that has occurred in this area [bbox].": 
                ["residential", "commercial", "industrial", "road", "demolition", "mega-projects"]
        }
    elif dataset == "xbd":
        class_dict = {
            "classification: Classify the level of damage experienced by the building at location [bbox] in the second image. Choose from: No damage, Minor Damage, Major Damage, Destroyed.": 
                ["No damage", "Minor damage", "Major damage", "Destroyed"]
        }
    else:
        class_dict = {}
    if task not in class_dict:
        return None
    return class_dict[task]


def create_mask(polygons, im_size):
    # Create a blank image (mask) with the same size as specified
    img = Image.new('L', im_size, 0)
    draw = ImageDraw.Draw(img)

    # Function to draw each polygon
    def draw_polygon(polygon):
        # Draw the exterior of the polygon
        exterior = list(polygon.exterior.coords)
        draw.polygon(exterior, outline=1, fill=1)

    # Check if polygons is iterable
    try:
        iter(polygons)
    except:
        polygons = [polygons]
    for polygon in polygons:
        draw_polygon(polygon)

    # Convert the PIL image to a NumPy array
    mask = np.array(img)
    return mask


def evaluate_masks(results, dataset, height=256, width=256):

    mask_getter = create_mask

    evaluator = Evaluator(num_class=2)
    metrics = ["oa", "mIoU", "kappa", "fwIoU", "precision", "recall", "f1", "IoU"]
    metric2method = {
        "precision": evaluator.Pixel_Precision_Rate,
        "recall": evaluator.Pixel_Recall_Rate,
        "f1": evaluator.Pixel_F1_score,
        "oa": evaluator.Pixel_Accuracy,
        "mIoU": evaluator.Mean_Intersection_over_Union,
        "IoU": evaluator.Intersection_over_Union,
        "kappa": evaluator.Kappa_coefficient,
        "fwIoU": evaluator.Frequency_Weighted_Intersection_over_Union,
        "Damage_F1_score": evaluator.Damage_F1_socore,
        "Class_Weighted_F1_score": evaluator.Class_Weighted_F1_score
    }

    for i, result in tqdm(enumerate(results), total=len(results)):

        if "[" not in result['ground_truth']:
            # No boxes in the ground truth
            gt_mask = np.zeros((height, width), dtype='uint8')
        else:
            gt_mask = mask_getter(wkt.loads(result['polygon']), (height, width))

        if "[" not in result['response']:
            # No boxes predicted
            pred_mask = np.zeros((height, width), dtype='uint8')
        else:
            # Get predicted mask from result
            # result['predicted'] is a string with a comma-separated list of boxes
            # of the form [x1, y1, x2, y2]
            # where each coordinate is normalized to [0, 100], a percentage of the image size
            # We need to obtain a wkt polygon from this
            pred_string = result['response']
            # Parse the string into a list of box strings by separatig based on [ and ]
            box_strings = re.findall(r'\[(.*?)\]', pred_string)
            # Parse each box string into a list of floats
            boxes = []
            for box in box_strings:
                try:
                    boxes.append(list(map(float, box.split(','))))
                except:
                    pass
            # Normalize the boxes to the image size
            boxes = [[box[0] / 100 * width, box[1] / 100 * height, box[2] / 100 * width, box[3] / 100 * height] for box in boxes]
            # Create a polygon from each box
            pred_wkt_string = [f"POLYGON (({box[0]} {box[1]}, {box[0]} {box[3]}, {box[2]} {box[3]}, {box[2]} {box[1]}, {box[0]} {box[1]}))" for box in boxes]
            pred_mask = mask_getter(wkt.loads(pred_wkt_string), (height, width))

        evaluator.add_batch(gt_mask, pred_mask)

    metric_values = {metric: metric2method[metric]() for metric in metrics}

    return metric_values


def change_detection_classification(outputs, classes, skip_classes=[], height=256, width=256, ignore_casing=True, ignore_punctuation=True):
    """
    Given the predicted boxes, this function creates segmentation masks on the original polygon for the predicted and ground truth classes.
    Returns the class-weighted per-pixel F1 between predicted and ground-truth masks.
    """
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})

    clean_string = get_string_cleaner(ignore_casing, ignore_punctuation)

    for output in tqdm(outputs):
        predicted_class = clean_string(output['response'])
        ground_truth_class = clean_string(output["ground_truth"])
        polygon = loads(output['polygon'])

        pred_msk = np.zeros((height, width), dtype='uint8')
        gt_msk = np.zeros((height, width), dtype='uint8')
        _msk = create_mask(polygon, im_size=(height, width))

        if ground_truth_class in skip_classes:
            continue

        if predicted_class not in classes:
            fn = gt_msk.sum()

        else:
            pred_label = classes.index(predicted_class) + 1
            gt_label = classes.index(ground_truth_class) + 1
            pred_msk[_msk > 0] = pred_label
            gt_msk[_msk > 0] = gt_label

            tp = (pred_msk == gt_label).sum()
            fp = (pred_msk == pred_label).sum() - tp
            fn = (gt_msk == gt_label).sum() - tp

            class_stats[predicted_class]['tp'] += tp
            class_stats[predicted_class]['fp'] += fp
        class_stats[ground_truth_class]['fn'] += fn
        class_stats[ground_truth_class]['count'] += np.sum(_msk)

    scores_dict = {}

    total_samples = sum(stats['count'] for stats in class_stats.values())
    prev_weighted_f1_score = 0
    inv_prev_weighted_f1_score = 0
    total_inv_prev_weight = 0
    for class_name in classes:
        tp, fp, fn = class_stats[class_name]['tp'], class_stats[class_name]['fp'], class_stats[class_name]['fn']
        if tp + fp == 0:
            class_precision = 0.0
        else:
            class_precision = tp / (tp + fp)
        if tp + fn == 0:
            class_recall = 0.0
        else:
            class_recall = tp / (tp + fn)
        if class_precision + class_recall == 0:
            class_f1 = 0.0
        else:
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        scores_dict[class_name] = class_f1
        
        # Compute weighted F1
        prevalence = class_stats[class_name]['count'] / total_samples
        prev_weighted_f1_score += class_f1 * prevalence
        if prevalence != 0:
            prevalence_inv = 1 / prevalence
            inv_prev_weighted_f1_score += class_f1 * prevalence_inv
            total_inv_prev_weight += prevalence_inv

    if total_inv_prev_weight > 0:
        inv_prev_weighted_f1_score /= total_inv_prev_weight
    else:
        inv_prev_weighted_f1_score = 0.0

    return {
        "f1": np.mean(list(scores_dict.values())),
        "w_f1": prev_weighted_f1_score,
        "inv_w_f1": inv_prev_weighted_f1_score,
    }


def detection_metrics(outputs, dataset_name, ignore_casing=True, ignore_punctuation=True):

    task2outputs = defaultdict(list)
    for output in outputs:
        task = output['task']
        task2outputs[task].append(output)

    detection_metrics = {}

    for task in task2outputs:
        if 'xbd' in dataset_name:
            if task == 'change_detection_classification':
                assert dataset_name == 'xbd_dmg_cls'
                classes = ["no damage", "minor damage", "major damage", "destroyed"]
                skip_classes = ["unclassified"]
                detection_metrics[f"{task}_f1"] = change_detection_classification(
                    task2outputs[task],
                    classes,
                    skip_classes=skip_classes,
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )["inv_w_f1"]
            elif task == 'change_detection_localization':
                detection_metrics[f"{task}_f1"] = evaluate_masks(task2outputs[task], dataset_name)['f1']
            elif task == 'spatial_referring_expression':
                assert dataset_name == 'xbd_sre_qa_rqa'
                detection_metrics[f"{task}_f1"] = evaluate_masks(task2outputs[task], dataset_name)['f1']
            elif task == 'region_based_question_answering':
                assert dataset_name == 'xbd_sre_qa_rqa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )[f"{task}_accuracy"]
            elif task == 'question_answering':
                assert dataset_name == 'xbd_sre_qa_rqa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation,
                    keywords=["yes", "no", "top left", "top center", "top right", "center left", "center", "center right", "bottom left", "bottom center", "bottom right"],
                )[f"{task}_accuracy"]
            else:
                raise ValueError(f"Unsupported task {task} for dataset {dataset_ame}")

        elif 's2' in dataset_name:
            if task == 'change_detection_detection' and dataset_name == 's2_det':
                detection_metrics[f"{task}_f1"] = evaluate_masks(task2outputs[task], dataset_name)['f1']
            elif task == 'region_based_question_answering':
                assert dataset_name == 's2_rqa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )[f"{task}_accuracy"]
            elif task == 'spatial_referring_expression':
                assert dataset_name == 's2_sre_qa'
                detection_metrics[f"{task}_f1"] = evaluate_masks(task2outputs[task], dataset_name)['f1']
            elif task == 'question_answering':
                assert dataset_name == 's2_sre_qa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )[f"{task}_accuracy"]
            else:
                raise ValueError(f"Unsupported task {task} for dataset {dataset_name}")

        elif 'qfabric' in dataset_name:
            if task == 'region_based_question_answering':
                classes = ["residential", "commercial", "industrial", "road", "demolition", "mega projects"]
                skip_classes = []
                detection_metrics[f"{task}_f1"] = change_detection_classification(
                    task2outputs[task],
                    classes,
                    skip_classes=skip_classes,
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )["w_f1"]
            elif task == 'region_based_temporal_question_answering':
                if dataset_name == 'qfabric_tre_rtqa':
                    detection_metrics[f"{task}_accuracy"] = classification_metrics(
                        task2outputs[task],
                        ignore_casing=ignore_casing,
                        ignore_punctuation=ignore_punctuation
                    )[f"{task}_accuracy"]
                elif dataset_name == 'qfabric_rqa5_rtqa5':
                    classes = ["prior construction", "greenland", "land cleared", "excavation", "materials dumped", "construction started", "construction midway", "construction done", "operational"]
                    skip_classes = []
                    detection_metrics[f"{task}_f1"] = change_detection_classification(
                        task2outputs[task],
                        classes,
                        skip_classes=skip_classes,
                        ignore_casing=ignore_casing,
                        ignore_punctuation=ignore_punctuation
                    )["w_f1"]
                else:
                    raise ValueError(f"Unsupported dataset {dataset_name} for task {task}")
            elif task == 'temporal_referring_expression':
                assert dataset_name == 'qfabric_tre_rtqa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation
                )[f"{task}_accuracy"]
            else:
                raise ValueError(f"Unsupported task: {task} for dataset {dataset_name}")

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    return detection_metrics
