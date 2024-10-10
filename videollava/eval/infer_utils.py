import cv2
import torch
import warnings
import numpy as np
from datetime import datetime
import cv2
import warnings
import time

from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


def replace_video_token(prompt, image_paths, prompt_strategy):
    if prompt_strategy is None:
        vid_replace_token = DEFAULT_IMAGE_TOKEN * len(image_paths)
    elif prompt_strategy == 'interleave':
        vid_replace_token = ''.join(f"Image {i+1}: {DEFAULT_IMAGE_TOKEN}" for i in range(len(image_paths)))
    else:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    return prompt.replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)


def run_inference_single(
        model,
        processor,
        tokenizer,
        conv_mode,
        inp,
        image_paths,
        metadata=None,
        use_video_data=False,
        repeat_frames=None,
        prompt_strategy=None,
        chronological_prefix=True,
        delete_system_prompt=False,
        print_prompt=False,
        return_prompt=False,
        last_image=False,
        prompt=None
    ):
    conv = conv_templates[conv_mode].copy()
    if prompt is None:
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    if chronological_prefix:
        prompt = prompt.replace("times:", "times in chronological order:")

    if metadata is not None:
         # Sort by time
        image_paths, metadata = zip(*sorted(
            zip(image_paths, metadata),
            key=lambda t: datetime.strptime(t[1]["timestamp"], "%Y-%m-%d")
        ))

    if delete_system_prompt:
        if "This is" in prompt:
            start_index = prompt.find("This is")
        elif "These are" in prompt:
            start_index = prompt.find("These are")
        end_index = prompt.find(":", start_index)
        if start_index != -1 and end_index != -1:
            prompt = prompt[:start_index] + prompt[end_index+1:]
        else:
            warnings.warn("Impossible to remove the system message from the prompt.")

    if use_video_data:
        image_paths = list(image_paths)
        if repeat_frames == "uniform":
            # Repeat up to 8 for now
            num_frames = 8
            if len(image_paths) < num_frames:
                num_repeats = num_frames // len(image_paths)
                index = len(image_paths) - num_frames % len(image_paths)
                image_paths = list(np.repeat(image_paths[:index], num_repeats)) + list(np.repeat(image_paths[index:], num_repeats+1))
        elif repeat_frames == "first":
            # Repeat the first frame
            num_frames = 8
            if len(image_paths) < num_frames:
                repeat_frames = [image_paths[0]] * (num_frames - len(image_paths)) + image_paths
        elif repeat_frames == "last":
            # Repeat the last frame
            num_frames = 8
            if len(image_paths) < num_frames:
                repeat_frames = image_paths + [image_paths[-1]] * (num_frames - len(image_paths))

        video_tensor = processor.preprocess(image_paths, return_tensors='pt')['pixel_values']
        tensor = [video_tensor.to(model.device, dtype=torch.float16)]

    else:
        image_tensors = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image_paths]
        tensor = [image_tensor.to(model.device, dtype=torch.float16) for image_tensor in image_tensors]

        if last_image:
            tensor = [tensor[-1]]
            image_paths = [image_paths[-1]]
            if metadata is not None:
                metadata = [metadata[-1]]

    prompt = replace_video_token(prompt, image_paths, prompt_strategy)

    if print_prompt:
        print(prompt)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=256,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # .replace removes the end sentence token "</s>" from the output
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace('</s>', '').strip()

    if return_prompt:
        return prompt, outputs
    else:
        return outputs


def create_mask(poly, im_size):
    """
    Create mask of given height and width where entries
    inside polygon are 1.
    params:
        - poly (shapely polygon object): polygon to create mask for
        - im_size (tuple): size of image (height, width)
    returns:
        - img_mask (np.array): mask of polygon"""
    img_mask = np.zeros(im_size, np.uint8)
    def int_coords(x): return np.array(x).round().astype(np.int32)
    try:
        exteriors = [int_coords(pol.exterior.coords) for pol in poly]
    except:
        exteriors = [int_coords(poly.exterior.coords)]
    cv2.fillPoly(img_mask, exteriors, 1)
    try:
        interiors = [int_coords(pol.interior.coords) for pol in poly] 
        cv2.fillPoly(img_mask, interiors, 0)
    except:
        pass
    try:
        interiors = [int_coords(poly.interior.coords)]
        cv2.fillPoly(img_mask, interiors, 0)
    except:
        pass
    
    return img_mask


def create_mask_s2looking(img_id, split=None, question=None):
    if split == None:
        raise ValueError("split must be provided for S2Looking evaluation")
    
    if question == None:
        raise ValueError("question must be provided for S2Looking evaluation")

    im1_path = f'/scr/geovlm/S2Looking/{split}/label1' # built
    img2_path = f'/scr/geovlm/S2Looking/{split}/label2' # destroyed
    id, chunk = img_id.split('_')
    # Load image as numpy array
    im1 = cv2.imread(f'{im1_path}/{id}.png', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(f'{img2_path}/{id}.png', cv2.IMREAD_GRAYSCALE)
    # replace any value different from 0 with 1
    im1[im1 != 0] = 1
    im2[im2 != 0] = 1

    # get the corresponding of the 16 chunks
    # 1 is upper left, 16 is lower right
    if chunk == '1':
        mask1 = im1[:256, :256]
        mask2 = im2[:256, :256]
    elif chunk == '2':
        mask1 = im1[:256, 256:2*256]
        mask2 = im2[:256, 256:2*256]
    elif chunk == '3':
        mask1 = im1[:256, 2*256:3*256]
        mask2 = im2[:256, 2*256:3*256]
    elif chunk == '4':
        mask1 = im1[:256, 3*256:]
        mask2 = im2[:256, 3*256:]
    elif chunk == '5':
        mask1 = im1[256:2*256, :256]
        mask2 = im2[256:2*256, :256]
    elif chunk == '6':
        mask1 = im1[256:2*256, 256:2*256]
        mask2 = im2[256:2*256, 256:2*256]
    elif chunk == '7':
        mask1 = im1[256:2*256, 2*256:3*256]
        mask2 = im2[256:2*256, 2*256:3*256]
    elif chunk == '8':
        mask1 = im1[256:2*256, 3*256:]
        mask2 = im2[256:2*256, 3*256:]
    elif chunk == '9':
        mask1 = im1[2*256:3*256, :256]
        mask2 = im2[2*256:3*256, :256]
    elif chunk == '10':
        mask1 = im1[2*256:3*256, 256:2*256]
        mask2 = im2[2*256:3*256, 256:2*256]
    elif chunk == '11':
        mask1 = im1[2*256:3*256, 2*256:3*256]
        mask2 = im2[2*256:3*256, 2*256:3*256]
    elif chunk == '12':
        mask1 = im1[2*256:3*256, 3*256:]
        mask2 = im2[2*256:3*256, 3*256:]
    elif chunk == '13':
        mask1 = im1[3*256:, :256]
        mask2 = im2[3*256:, :256]   
    elif chunk == '14':
        mask1 = im1[3*256:, 256:2*256]
        mask2 = im2[3*256:, 256:2*256]
    elif chunk == '15':
        mask1 = im1[3*256:, 2*256:3*256]
        mask2 = im2[3*256:, 2*256:3*256]
    elif chunk == '16':
        mask1 = im1[3*256:, 3*256:]
        mask2 = im2[3*256:, 3*256:]
    
    task = None
    if 'built' in question or 'constructed' in question:
        task = 'constructing'
    if 'destroyed' in question or 'torn down' in question or 'demolished' in question:
        task = 'destroying'
    if 'changed' in question:
        task = 'changing'
    if (('built' in question) or ('constructed' in question)) and (('destroyed' in question) or ('torn down' in question) or ('demolished' in question)):
        print(question)
        raise ValueError("Question cannot contain both 'built' and 'destroyed'")
    if task is None:
        print(question)
        raise ValueError("Question must contain either 'built', 'destroyed', or 'changed'")
    
    if task == 'constructing':
        mask = mask1
    elif task == 'destroying':
        mask = mask2
    elif task == 'changing':
        mask = np.logical_or(mask1, mask2)

    return mask
