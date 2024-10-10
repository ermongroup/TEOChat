import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from infer_utils import run_inference_single, create_mask


def run_s2looking_inference(
    model,
    dataset_path,
    processor,
    tokenizer,
    conv_mode,
    use_video_data=False,
    open_prompt=None,
    repeat_frames=True,
    prompt_strategy="interleave",
    chronological_prefix=True,
    data_frac=1,
    data_size=None,
    delete_system_prompt=False,
    last_image=False,
    print_prompt=False,
    answer_path=None,
    start_ind=None,
    end_ind=None,
):

    dir = Path(dataset_path)

    with open(dir) as f:
        s2looking_data = json.load(f)
    
    if data_size is not None:
        data_size = min(data_size, len(s2looking_data))
        idx = np.random.choice(len(s2looking_data), data_size, replace=False)
        s2looking_data = [s2looking_data[i] for i in idx]
    elif data_frac < 1:
        idx = np.random.choice(len(s2looking_data), int(len(s2looking_data) * data_frac), replace=False)
        s2looking_data = [s2looking_data[i] for i in idx]

    answers = {}
    for question in tqdm(s2looking_data):
        question_id = question["id"]
        inp = question["conversations"][0]['value']
        answer_str = question["conversations"][1]['value']
        metadata = question['metadata']
        task = question['task']
        image_paths = question['video']
        original_input_polygon = question['original_input_polygon']

        outputs = run_inference_single(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            conv_mode=conv_mode,
            inp=inp,
            image_paths=image_paths,
            metadata=metadata,
            repeat_frames=repeat_frames,
            use_video_data=use_video_data,
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
            delete_system_prompt=delete_system_prompt,
            last_image=last_image,
            print_prompt=print_prompt
        )
        
        answers[question_id] = {
            "predicted": outputs,
            "ground_truth": answer_str,
            "question": inp,
            "task": task,
            "original_input_polygon": original_input_polygon
        }

    return answers
