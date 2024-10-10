import json
from tqdm import tqdm
from pathlib import Path

from infer_utils import run_inference_single
import numpy as np


def run_geochat_inference(
    model,
    dataset_path,
    processor,
    tokenizer,
    conv_mode,
    answer_path,
    use_video_data=False,
    open_prompt=None,
    repeat_frames=None,
    prompt_strategy="interleave",
    chronological_prefix=True,
    data_frac=1,
    data_size=None,
    delete_system_prompt=False,
    start_ind=0,
    end_ind=None,
    print_prompt=False
):
    
    with open(dataset_path) as f:
        qfabric_data = json.load(f)

    if data_size is not None:
        data_size = min(data_size, len(qfabric_data))
        idx = np.random.choice(len(qfabric_data), data_size, replace=False)
        qfabric_data = [qfabric_data[i] for i in idx]
    elif data_frac < 1:
        idx = np.random.choice(len(qfabric_data), int(len(qfabric_data) * data_frac), replace=False)
        qfabric_data = [qfabric_data[i] for i in idx]

    answers = {}
    answers_tmp = str(answer_path).replace(".json", "_tmp.json")
    if end_ind is not None:
        answers_tmp = str(answers_tmp).replace(".json", f"_{start_ind}_{end_ind}.json")
        qfabric_data = qfabric_data[start_ind:end_ind]
    else:
        answers_tmp = str(answers_tmp).replace(".json", f"_{start_ind}_end.json")
        qfabric_data = qfabric_data[start_ind:]
    
    print("answers_tmp: ", answers_tmp)
    print("start ind: ", start_ind)
    print("end ind: ", end_ind)

    for question in tqdm(qfabric_data):
        question_id = question["id"]
        inp = question["conversations"][0]['value']

        answer_str = question["conversations"][1]['value']
        metadata = question['metadata']
        image_paths = question['video']
        task = question['task']
        original_input_polygon = question['original_input_polygon']
        dataset = question['dataset']

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
            print_prompt=print_prompt
            )

        entry = {
            "id": question_id,
            "question": inp,
            "predicted": outputs,
            "ground_truth": answer_str,
            "task": task,
            "original_input_polygon": original_input_polygon,
            "dataset": dataset,
        }
        answers[question_id] = entry

        with open(answers_tmp, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return answers
