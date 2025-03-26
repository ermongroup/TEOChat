import re
import torch
from tqdm import tqdm
from datetime import datetime

from videollava.conversation import conv_templates, SeparatorStyle
from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


def replace_video_token(prompt, image_paths, prompt_strategy):
    if prompt_strategy is None:
        vid_replace_token = DEFAULT_IMAGE_TOKEN * len(image_paths)
        new_prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
    elif prompt_strategy == 'interleave':
        vid_replace_token = ''.join(f"Image {i+1}: {DEFAULT_IMAGE_TOKEN}" for i in range(len(image_paths)))
        new_prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    return new_prompt


def run_inference_single(
        model,
        processor,
        tokenizer,
        inp,
        image_paths,
        conv_mode="v1",
        timestamps=[],
        prompt_strategy="interleave",
        chronological_prefix=True,
        temperature=0.2,
        max_new_tokens=256,
    ):

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if chronological_prefix:
        prompt = prompt.replace("times:", "times in chronological order:")

    if len(timestamps) > 0:
        # Sort by time
        image_paths, timestamps = zip(*sorted(
            zip(image_paths, timestamps),
            key=lambda t: datetime.strptime(t[1], "%Y-%m-%d")
        ))

    image_tensors = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image_paths]
    tensor = [image_tensor.to(model.device, dtype=torch.float16) for image_tensor in image_tensors]

    prompt = replace_video_token(prompt, image_paths, prompt_strategy)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # removes the end sentence token "</s>" from the output
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace('</s>', '').strip()

    return outputs


def extract_bboxes(bbox_str):
    # Regular expression to find numbers within brackets
    pattern = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
    # Find all matches and convert them to lists of integers
    bboxes = [list(map(int, match.groups())) for match in pattern.finditer(bbox_str)]
    return bboxes


def run_inference(
        dataset,
        model,
        tokenizer,
        processor,
        prompt_strategy,
        chronological_prefix,
        conv_mode,
        temperature,
        max_new_tokens
    ):
    outputs = []
    for example in tqdm(dataset):
        response = run_inference_single(
            model,
            processor,
            tokenizer,
            example["conversations"][0]['value'],
            example['video'],
            conv_mode=conv_mode,
            timestamps=example['timestamp'],
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        output = {
            'response': response,
            'ground_truth': example["conversations"][1]['value'],
            'task': example['task'],
        }
        polygon = example.get('polygon', None)
        if polygon is not None:
            output['polygon'] = polygon
        else:
            if dataset in ["xbd_loc", "xbd_dmg_cls", "s2_det", "qfabric_rqa2",
                           "qfabric_rqa5", "xbd_sre_qa_rqa", "s2_sre_qa", "s2_rqa"]:
                raise ValueError(
                    f"Polygons not found for dataset {dataset}. " +
                    "The TEOChatlas dataset was updated to include these polygons on 25 Mar 2025. " +
                    "Please re-download the json files for these splits."
                )
        input_bboxes = extract_bboxes(example["conversations"][0]['value'])
        output_bboxes = extract_bboxes(example["conversations"][1]['value'])
        if len(input_bboxes) > 0:
            output['input_bboxes'] = input_bboxes
        if len(output_bboxes) > 0:
            output['output_bboxes'] = output_bboxes
        outputs.append(output)
    return outputs
