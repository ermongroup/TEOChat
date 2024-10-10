import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from infer_utils import run_inference_single
# For the purposes of an experiment, change the infer_utils to: 
# from infer_utils_mod import run_inference_single

def run_aid_fmow_ucmerced_inference(
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
        print_prompt=False,
        **kwargs
    ):
    for k, v in kwargs.items():
        print("WARNING: Unused argument:", k, v)

    try:
        with open(dataset_path) as f:
            data = json.load(f)
    except:
        data = []
        with open(dataset_path) as f:
            for line in f:
                question = json.loads(line)
                question["id"] = question["question_id"]
                question["conversations"] = [
                    {"value": "This is a satellite image: <video> " + question["text"]},
                    {"value": question["ground_truth"]}
                ]
                question["video"] = [question["image"]]
                data.append(question)

    if data_size is not None:
        data_size = min(data_size, len(data))
        idx = np.random.choice(len(data), data_size, replace=False)
        data = [data[i] for i in idx]
    elif data_frac < 1:
        idx = np.random.choice(len(data), int(len(data) * data_frac), replace=False)
        data = [data[i] for i in idx]

    vision_key = "video" if "video" in data[0] else "image"

    answers = {}
    for question in tqdm(data):
        question_id = question["id"]
        inp = question["conversations"][0]['value']
        if open_prompt == "open":
            # Use an open framing for the question
            inp = inp.split("Which")[0] + "Which class does this image belong to?"
        elif open_prompt == "multi-open":
            inp = inp.split("Which")[0] + "What classes does this image belong to?"
        answer_str = question["conversations"][1]['value']
        if 'metadata' not in question:
            question['metadata'] = None
        metadata = question['metadata']
        image_paths = question[vision_key]

        outputs = run_inference_single(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            conv_mode=conv_mode,
            inp=inp,
            image_paths=image_paths,
            metadata=metadata,
            use_video_data=use_video_data,
            repeat_frames=repeat_frames,
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
	        delete_system_prompt=delete_system_prompt,
            last_image=last_image,
            print_prompt=print_prompt
        )

        answers[question_id] = {
            "predicted": outputs,
            "ground_truth": answer_str
        }

    return answers
