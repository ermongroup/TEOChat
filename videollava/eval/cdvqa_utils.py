import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from videollava.constants import DEFAULT_VIDEO_TOKEN

from infer_utils import run_inference_single


def run_cdvqa_inference(
        model,
        dataset_path,
        processor,
        tokenizer,
        conv_mode,
        use_video_data=False,
        open_prompt=None,
        repeat_frames=None,
        prompt_strategy="interleave",
        chronological_prefix=True,
        data_frac=1,
        data_size=None,
	    delete_system_prompt=False,
        last_image=False,
        start_ind=None,
        end_ind=None,
        print_prompt=False,
        **kwargs
    ):
    for k, v in kwargs.items():
        print("WARNING: Unused argument:", k, v)

    dataset_path = Path(dataset_path)
    data_dir = dataset_path.parent
    questions_path = data_dir / dataset_path.name.replace(".json", "_questions.json")
    answers_path = data_dir / dataset_path.name.replace(".json", "_answers.json")
    images_path = data_dir / dataset_path.name.replace(".json", "_images.json")

    with open(questions_path) as json_data:
        questionsJSON = json.load(json_data)

    with open(answers_path) as json_data:
        answersJSON = json.load(json_data)

    with open(images_path) as json_data:
        imagesJSON = json.load(json_data)

    if data_size is not None:
        data_size = min(data_size, len(questionsJSON))
        idx = np.random.choice(len(questionsJSON), data_size, replace=False)
        imagesJSON = [imagesJSON[i] for i in idx]
    elif data_frac < 1:
        idx = np.random.choice(len(questionsJSON), int(len(questionsJSON) * data_frac), replace=False)
        imagesJSON = [imagesJSON[i] for i in idx]

    # Get the image IDs of test images
    images_ids = [img['id'] for img in imagesJSON['images'] if img['active']]

    if start_ind is not None and end_ind is not None:
        print("Subsetting data from index", start_ind, "to", end_ind)
        images_ids = images_ids[start_ind:end_ind]
    elif start_ind is not None:
        print("Subsetting data from index", start_ind, "to end")
        images_ids = images_ids[start_ind:]
    elif end_ind is not None:
        print("Subsetting data from start to index", end_ind)
        images_ids = images_ids[:end_ind]

    # Store all predicted answers
    answers = {}
    # Read image corresponding to each ID and get its associated question and answer
    for id in tqdm(images_ids):
        file_name = imagesJSON['images'][id]['file_name']

        image_paths = [
            str(data_dir / "second_dataset" / "im1" / file_name),
            str(data_dir / "second_dataset" / "im2" / file_name),
        ]

        for questionid in imagesJSON['images'][id]['questions_ids']:
            question = questionsJSON['questions'][questionid]
            if not question['active']:
                continue
            inp = "This is a pair of satellite images capturing the same location at different times: "
            inp = inp + DEFAULT_VIDEO_TOKEN + '\n'
            inp = inp + question["question"]
            type_str = question["type"]
            answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']

            if type_str in ["change_or_not", "increase_or_not", "decrease_or_not"]:
                inp = inp + " Answer with yes or no."

            elif type_str == "change_ratio":
                inp = inp + " Choose from one of the following options: 0, 0_to_10, 10_to_20, 20_to_30, 30_to_40, 40_to_50, 50_to_60, 60_to_70, 70_to_80, 80_to_90, 90_to_100."

            elif type_str == "change_ratio_types":
                inp = inp + " Choose from one of the following options: 0, 0_to_10, 10_to_20, 20_to_30, 30_to_40, 40_to_50, 50_to_60, 60_to_70."

            else: # smallest_change, largest_change, change_to_what
                inp = inp + " Choose from one of the following options: buildings, low_vegetation, nonvegetated ground surface, playgrounds, trees, water."
                answer_str = answer_str.replace("NVG_surface", "nonvegetated ground surface")    

            outputs = run_inference_single(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                conv_mode=conv_mode,
                inp=inp,
                image_paths=image_paths,
                metadata=None,
                use_video_data=use_video_data,
                repeat_frames=repeat_frames,
                prompt_strategy=prompt_strategy,
                chronological_prefix=chronological_prefix,
                delete_system_prompt=delete_system_prompt,
                last_image=last_image,
                print_prompt=print_prompt
            )

            answers[f"{id}_{questionid}"] = {
                "predicted": outputs,
                "ground_truth": answer_str,
                "task": type_str
            }

    return answers
