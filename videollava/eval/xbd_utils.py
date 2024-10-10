import json
from tqdm import tqdm
from pathlib import Path

from infer_utils import run_inference_single
import numpy as np



def run_xbd_inference(
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
    last_image=False,
    delete_system_prompt=False,
    print_prompt=False,
    answer_path=None,
    start_ind=None,
    end_ind=None,
):

    with open(dataset_path) as f:
            xbd_data = json.load(f)

    if data_size is not None:
        data_size = min(data_size, len(xbd_data))
        idx = np.random.choice(len(xbd_data), data_size, replace=False)
        xbd_data = [xbd_data[i] for i in idx]
    elif data_frac < 1:
        idx = np.random.choice(len(xbd_data), int(len(xbd_data) * data_frac), replace=False)
        xbd_data = [xbd_data[i] for i in idx]

    answers = {}
    for question in tqdm(xbd_data):
        question_id = question["id"]
        inp = question["conversations"][0]['value']

        answer_str = question["conversations"][1]['value']
        metadata = question['metadata']
        image_paths = question['video']
        task = question['task']
        original_input_polygon = question['original_input_polygon']

        # TODO: check if you want to add closed framing for yes/no questions
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
            last_image=last_image,
            print_prompt=print_prompt
        )

        answers[question_id] = {
            "question": inp,
            "predicted": outputs,
            "ground_truth": answer_str,
            "task": task,
            "original_input_polygon": original_input_polygon
        }
        # For recording individual answers as inference runs
        entry = {question_id: answers[question_id]}
        with open('/deep/u/joycech/aicc-working/geovlm_xbd_localization.json', 'a') as f:
            f.write(json.dumps(entry) + ',')

    return answers
