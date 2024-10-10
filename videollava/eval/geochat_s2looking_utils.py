
import json
import numpy as np
import cv2
import re
from eval_referring import referring_expression
import matplotlib.pyplot as plt
from shapely import wkt
import time
import math
from matplotlib.path import Path
from eval_classification import accuracy_precision_recall


def convert_geochat_string(build, img_size=256):
    """
    convert the raw str geochat output {<40><89><56><100>|<57>}, {<0><89><56><100>|<57>}
    to a list of rotated bboxes 
    """
    build = build.strip('{}')
    bbox_segments = build.split("}{")

    # Regular expression to find all numbers inside angle brackets
    pattern = r"<(\d+)>"

    # Extract numbers, convert them to integers, and collect into a list
    bboxes = [
        list(map(int, re.findall(pattern, segment)))
        for segment in bbox_segments]
    
    rotated_bboxes = []
    for bbox in bboxes:
        try:
            xmin, ymin, xmax, ymax, angle = [float(v) for v in bbox]
        except:
            pass

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


def get_changed_buildings(build1, build2, img_size=256, task=None):
    """
    Given a list of predicted buildings in image 1 and image 2, this function 
    - creates two img_size * img_size numpy arrays for both of the images
    - gets the mask differences between the two numpy arrays
    - returns a list of bounding boxes that reflect those differences, as well as the difference mask
    Input:
    - build1: [[x,y],[x,y],[x,y],[x,y]] array of four x,y coordinates of the bounding box of a building
    - task can be either None, constructed or destructed
    Note: those bboxes can be rotated
    """
    image1 = np.zeros((img_size, img_size), np.uint8)
    image2 = np.zeros((img_size, img_size), np.uint8)

    build1 = convert_geochat_string(build1)
    build2 = convert_geochat_string(build2)

     # fill in with ones the pixels that are inside the rotated bboxes
    for b in build1:
        path = Path(b)
        x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
        points = np.vstack((x.flatten(), y.flatten())).T
        image1[path.contains_points(points).reshape(img_size, img_size)] = 1

    for b in build2:
        path = Path(b)
        x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
        points = np.vstack((x.flatten(), y.flatten())).T
        image2[path.contains_points(points).reshape(img_size, img_size)] = 1

    # xor between the two images
    if task == None:
        diff = cv2.bitwise_xor(image1, image2)
    elif task == "constructed":
        # if the task is constructed, we want to find the pixels that are in image2 but not in image1
        diff = cv2.bitwise_and(image2, cv2.bitwise_not(image1))
    elif task == "destructed":
        # if the task is destructed, we want to find the pixels that are in image1 but not in image2
        diff = cv2.bitwise_and(image1, cv2.bitwise_not(image2))

    # get the bounding boxes of the difference pixels
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = y, x, h, w
        bboxes.append([x, y, x+w, y+h])
 
    return bboxes, diff

def get_canonical_answer_dataset(answers):
    """
    This function creates a new dataset with questions and answers for geochat, ready to parse into the evaluation metrics."""

    new_dataset = {}   

    for key, answer in answers.items():
        num, quadrant, geovlmid = key.split("_")
        task = answer['task']
        if geovlmid == "1" in task:
            continue
        
        # find the paired image
        id2 = num + "_" + quadrant + "_" + "1"
        answer1 = answers[key]
        try:
            answer2 = answers[id2]
        except:
            print(f"The associated image to {key} wasn't present in the dataset")
            continue

        # get the pixel diff boxes 
        change_bboxes, mask = get_changed_buildings(answer1['predicted'], answer2['predicted'])

        # create the new dataset adapted for running metrics on it 
        new_line = {}

        new_line['predicted'] = "" 
        if len(change_bboxes)>0:
            for bbox in change_bboxes:
                new_line['predicted'] += str(bbox) + ", "
            new_line['predicted'] = new_line['predicted'][:-2]
        new_line['predicted_mask'] = mask.tolist()         

        new_line['ground_truth'] = answer1['original_answer']
        new_line['question'] = answer1['original_question']
        new_line['task'] = answer1['task']
        new_line['original_input_polygon'] = answer1['original_input_polygon']

        new_key = num + "_" + quadrant  
        new_dataset[new_key] = new_line

    return new_dataset

def postprocess_auxiliary_qa(key, answer, original_answers):
    new_line = {}
    new_line['ground_truth'] = answer['ground_truth']
    new_line['question'] = answer['question']
    new_line['task'] = answer['task']
    new_line['original_input_polygon'] = answer['original_input_polygon']

    # retrieve the original 2 anwers
    answer1 = original_answers[key + '_0']['predicted']
    answer2 = original_answers[key + '_1']['predicted']

    # retrieve the task (construction or destruction)
    setting = None
    if "constructed" or "built" in answer['original_question']:
        setting = "constructed"
    elif "destructed" or "torn down" in answer['original_question']:
        setting = "destructed"
    else:
        print("The task is not recognized")
        print("Original question: ", answer['original_question'])
        print()

    # get the pixel diff boxes
    change_bboxes, mask = get_changed_buildings(answer1, answer2, task=setting) 

    new_line['predicted_mask'] = mask.tolist()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_convex_polygon = False
    for contour in contours:
        # check if the contour is a bounding box (4 vertices, rectangle shape)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            found_convex_polygon = True
            break        

    if found_convex_polygon:
        new_line['predicted'] = "Yes"
    else:
        new_line['predicted'] = "No"

    return new_line


def postprocess_auxiliary_region_qa(key, answer, original_answers, img_size=256):
    """
    There is a bbox in the input polygon, we need to find the changed buildings in the image
    inside that bbox
    """
    new_line = {}
    new_line['ground_truth'] = answer['ground_truth']
    new_line['question'] = answer['question']
    new_line['task'] = answer['task']
    new_line['original_input_polygon'] = answer['original_input_polygon']

    # retrieve the original 2 anwers
    answer1 = original_answers[key + '_0']['predicted']
    answer2 = original_answers[key + '_1']['predicted']

    # get the pixel diff boxes
    change_bboxes, mask = get_changed_buildings(answer1, answer2)

    # get the input bbox
    question = new_line['question']
    # find the positions of '[' and ']'
    start = question.find('[')
    end = question.find(']')
    bbox = question[start+1:end].split(',')
    bbox = [int(b) * img_size // 100 for b in bbox]

    # adapt the mask, put 0s outside the bbox
    mask[:bbox[0], :] = 0
    mask[bbox[2]:, :] = 0
    mask[:, :bbox[1]] = 0
    mask[:, bbox[3]:] = 0

    # predict yes or no if there is a convex polygon in the mask
    found_convex_polygon = False
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # check if the contour is a bounding box (4 vertices, rectangle shape)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            found_convex_polygon = True
            break

    new_line['predicted_mask'] = mask.tolist()
   
    if found_convex_polygon:
        new_line['predicted'] = "Yes"
    else:
        new_line['predicted'] = "No"

    return new_line


def postprocess_auxiliary_referring(key, answer, original_answers):
    new_line = {}
    new_line['ground_truth'] = answer['ground_truth']
    new_line['question'] = answer['question']
    new_line['task'] = answer['task']
    new_line['original_input_polygon'] = answer['original_input_polygon']

    # retrieve the original 2 anwers
    answer1 = original_answers[key + '_0']['predicted']
    answer2 = original_answers[key + '_1']['predicted']

    # retrieve the task (construction or destruction)
    setting = None
    if "constructed" or "built" in answer['original_question']:
        setting = "constructed"
    elif "destructed" or "torn down" in answer['original_question']:
        setting = "destructed"
    else:
        print("The task is not recognized")
        print("Original question: ", answer['original_question'])
        print()

    # get the pixel diff boxes
    change_bboxes, mask = get_changed_buildings(answer1, answer2, task=setting)

    new_line['predicted_mask'] = mask.tolist()
    new_line['predicted'] = ""
    if len(change_bboxes)>0:
        for bbox in change_bboxes:
            new_line['predicted'] += str(bbox) + ", "
        new_line['predicted'] = new_line['predicted'][:-2]
      
    return new_line


def postprocess_auxiliary_geochat_s2looking(canonical_answers, original_answers):
    """
    Postprocess the auxiliary file for geochat_s2looking
    The present questions are
    question1 = 'temporal_question_answering: Are there any buildings in the first image which were {destructed,torn down} in the second?'
    question2 = 'temporal_referring_expression: Identify the buildings in the first image which were {built,constructed,destructed,torn down} as seen in the second image.'
    question3 = 'localization_task: Identify all changed buildings.'
    question4 = 'referring_expression: identify the {constructed, destructed} buildings in the image.'
    question5 = 'question_answering: Have any buildings been task in the area? Please answer with Yes or No'

    The goal is to update the 'predicted' field with the correct bounding boxes of the changed buildings.
    - Localization can be kept as is.
    - For question answering tasks, the 'predicted' field should be updated with 'Yes' or 'No' depending on the answer.
    We output 'Yes' if there is a convex polygon in the 'predicted' field.
    - For referring expression, we first need to identify if the task is 'constructed' or 'destructed' and then update the 'predicted' field with the correct mask of the changed buildings.
    Input:
    - answers: dictionary with the answers paired with the get_canonical_answer_dataset function
    Output:
    - postprocessed_answers: dictionary with 'predicted' and 'predicted_mask' fields updated
    """
    postprocessed_answers = {}

    for key, answer in canonical_answers.items():
        task = answer['task']
       
        if 'localization' in task:
            postprocessed_answers[key] = answer
            continue
        if 'region_based_question_answering' in task:
            answer = postprocess_auxiliary_region_qa(key, answer, original_answers)
            postprocessed_answers[key] = answer
            continue
        if 'question_answering' in task:
            answer = postprocess_auxiliary_qa(key, answer, original_answers)
            postprocessed_answers[key] = answer
            continue
        if 'referring_expression' in task:
            answer = postprocess_auxiliary_referring(key, answer, original_answers)
            postprocessed_answers[key] = answer
            continue
    
    return postprocessed_answers
    

def evaluate_geochat_s2looking(answer_file, dataset_file, split):
    answers = {}
    with open(answer_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            answers[list(line.keys())[0]] = line[list(line.keys())[0]]

    dataset = dataset_file.split("/")[-1]
    if dataset == "dataset_canonical.json":

        # create a new dataset with questions and answers for geochat 
        postprocessed_answers = get_canonical_answer_dataset(answers)

        referring_expression(postprocessed_answers, "geochat_s2looking", False, "s2looking/answers/geochat_canonical_test", split=split)
    
    elif dataset == "dataset_v01_v02_canonical_filtered.json" or dataset == "dataset_RQA.json":
        
        # create a new dataset with questions and answers for geochat
        postprocessed_answers = get_canonical_answer_dataset(answers)
        postprocessed_answers = postprocess_auxiliary_geochat_s2looking(postprocessed_answers, answers)

        print("Referring expression")
        referring_expression(postprocessed_answers, "geochat_s2looking", False, "s2looking/answers/geochat_v01_v02_canonical_filtered_test", split=split)
        print()
        print("Accuracy")
        accuracy_precision_recall(postprocessed_answers, "s2looking", verbose=False)
        print()


        # also run per-question referring expression
        question1 = 'temporal_question_answering: Are there any buildings in the first image which were {destructed,torn down} in the second?'
        question2 = 'temporal_referring_expression: Identify the buildings in the first image which were {built,constructed,destructed,torn down} as seen in the second image.'
        question3 = 'localization_task: Identify all changed buildings.'
        question4 = 'referring_expression: identify the {constructed, destructed} buildings in the image.'
        question5 = 'question_answering: Have any buildings been task in the area? Please answer with Yes or No'


        for question in [question1, question2, question3, question4, question5]:
            dataset_question = {}
            for data in postprocessed_answers:
                if postprocessed_answers[data]['task'] == question:
                    dataset_question[data] = postprocessed_answers[data]

            if len(dataset_question) > 0:
                print('Evaluating for question ', question)
                print('Size of the dataset is ', len(dataset_question))
                referring_expression(dataset_question, "geochat_s2looking", False, "s2looking/answers/geochat_v01_v02_canonical_filtered_test", split=split)
                print()

    else:
        print("Evaluation is not suppored for this dataset. Please provide a valid dataset.")
        print("The supported datasets are: dataset_canonical.json, dataset_v01_v02_canonical_filtered.json")

            

