import os
import re
import io
import cv2
import json
import torch
import random
import argparse
import tempfile
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from PIL import Image
from gradio import Brush
from gradio.themes.utils import sizes
from pathlib import Path
from collections import defaultdict

# Add the grandparent directory to the path
# This is necessary to import the videollava package
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from videollava.utils import disable_torch_init
from videollava.model.builder import load_pretrained_model
from videollava.eval.infer_utils import run_inference_single
from videollava.constants import DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, Conversation, conv_templates
from videollava.mm_utils import  get_model_name_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="jirvin16/TEOChat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--quantization", type=str, default="8-bit")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--dont-use-fast-api', action='store_true')
    parser.add_argument('--planet-api-key', type=str, default=None)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--server_name', type=str, default="0.0.0.0")
    args = parser.parse_args()
    return args


def get_bbox_in_polyline_format(x1, y1, x2, y2):
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])


def extract_box_sequences(string):
    # Split the input string into segments where sequences of lists are separated by punctuation other than commas or periods
    segments = re.split(r'[^\[\],\d\s]+', string)
    
    # Pattern to find substrings of the form [a,b,c,d] where a, b, c, d are integers
    pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
    
    result = []
    for segment in segments:
        # Find all matches of the pattern in each segment
        matches = re.findall(pattern, segment)
        if matches:
            # Convert each tuple of strings into a list of integers and collect them into a list
            sublist = [list(map(int, match)) for match in matches]
            result.append(sublist)
    
    return result


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = Image.open(mask)
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        x1, x2 = np.where(cols)[0][[0, -1]]
        y1, y2 = np.where(rows)[0][[0, -1]]

        bbox = '[{}, {}, {}, {}]'.format(x1, y1, x2, y2)
    else:
        bbox = ''

    return bbox


def visualize_all_bbox_together(image_path, generation, bbox_presence):
    # Resize the image to a fixed width and a height that preserves the aspect ratio
    # For visualization in gradio
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    sequence_list = extract_box_sequences(generation)
    if sequence_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for sequence in sequence_list:
            try:
                # TODO: Get object name from the string
                # obj, sequence = sequence.split('</p>')
                obj = 'TODO'
            except ValueError:
                print('wrong string: ', sequence)
                continue
            if "][" in sequence:
                sequence=sequence.replace("][","], [")
            flag = False
            for bbox in sequence:

                if len(bbox) == 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    x1 = x1 / bounding_box_size * image_width
                    y1 = y1 / bounding_box_size * image_height
                    x2 = x2 / bounding_box_size * image_width
                    y2 = y2 / bounding_box_size * image_height

                    entities[obj].append([x1, y1, x2, y2])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        bbox = re.findall(r'-?\d+', generation)
        if len(bbox) == 4:  # it is refer
            mode = 'single'

            entities = list()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1 = x1 / bounding_box_size * image_width
            y1 = y1 / bounding_box_size * image_height
            x2 = x2 / bounding_box_size * image_width
            y2 = y2 / bounding_box_size * image_height
            entities.append([x1, y1, x2, y2])
        else:
            # don't detect any valid bbox to visualize
            return image, ''

    if len(entities) == 0:
        return image, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):

        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invalid image format, {type(image)} for {image}")

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.4
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # used_colors = colors  # random.sample(colors, k=num_bboxes)
    if bbox_presence == 'input':
        color = (255, 0, 0)
        color_string = 'red'
    elif bbox_presence == 'output':
        color = (0, 255, 0)
        color_string = 'green'
    else:
        # Doesn't matter, should never be used
        color = None

    # color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        # color_id += 1
        for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)

            # color = used_colors[entity_idx % len(used_colors)] # tuple(np.random.randint(0, 255, size=3).tolist())
            bbox = get_bbox_in_polyline_format(orig_x1, orig_y1, orig_x2, orig_y2)
            new_image=cv2.polylines(new_image, [bbox.astype(np.int32)], isClosed=True,thickness=2, color=color)

            # TODO: Add this after delimeter
            if False: # mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,
                                                               text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)

                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )

                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    # TODO: Add this after delimeter
    if False: # mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        # For now, just color the bounding box text the same color as the input
        def color_bounding_boxes(text):
            # Regex pattern to find patterns of the form [xmin, xmax, ymin, ymax]
            pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'

            # Function to apply HTML styling
            def replace_with_color(match):
                return f'<span style="color:{color_string};">{match.group()}</span>'

            # Replace all matching patterns with colored version
            colored_text = re.sub(pattern, replace_with_color, text)
            return colored_text

        if bbox_presence is not None:
            # Detect the bounding boxes and replace them with colored versions
            generation_colored = color_bounding_boxes(generation)
        else:
            generation_colored = generation

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[CONV_MODE].copy()
    state_ = conv_templates[CONV_MODE].copy()
    return (
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        True,
        state,
        state_,
        state.to_gradio_chatbot()
    )


def single_example_trigger(image1, textbox):
    return gr.update(value=None, interactive=True), *example_trigger()


def temporal_example_trigger(image1, image_list, textbox):
    return image_list, *example_trigger()


def example_trigger():
    state = conv_templates[CONV_MODE].copy()
    state_ = conv_templates[CONV_MODE].copy()
    return True, state, state_, state.to_gradio_chatbot()


def generate(image1, image_list, textbox_in, first_run, state, state_):
    flag = 1
    if not textbox_in:
        return "Please enter an instruction."

    mask = None
    if image1 is None:
        image1 = []
    elif isinstance(image1, str):
        image1 = [image1]
    elif isinstance(image1, dict):
        mask = image1['layers'][0]
        image1 = [image1['background']]
    if image_list is None:
        image_list = []

    all_image_paths = [path for path in image1 + image_list if os.path.exists(path)]

    if type(state) is not Conversation:
        state = conv_templates[CONV_MODE].copy()
        state_ = conv_templates[CONV_MODE].copy()

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    # Check if user provided bbox in the text input
    integers = re.findall(r'-?\d+', text_en_in)
    bbox_in_input = False
    if len(integers) != 4:
        # No bbox provided in input text. Try to use the bbox from the image editor
        bbox = mask2bbox(mask)
        if bbox:
            bbox_in_input = True
            text_en_in += f" {bbox}"
    else:
        bbox_in_input = True

    text_en_out, state_ = handler.generate(all_image_paths, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]

    # Check if bbox is in the text output
    integers = re.findall(r'-?\d+', text_en_out)
    bbox_in_output = False
    if len(integers) == 4:
        bbox_in_output = True

    show_images = ""
    for idx, image_path in enumerate(all_image_paths, start=1):
        if bbox_in_input and bbox_in_output:
            # If both are present, only display the output bbox in the image
            bbox_presence = "output"
            image, text_en_out = visualize_all_bbox_together(image_path, text_en_out, bbox_presence=bbox_presence)
        elif bbox_in_input and not bbox_in_output:
            bbox_presence = "input"
            image, text_en_in = visualize_all_bbox_together(image_path, text_en_in, bbox_presence=bbox_presence)
        elif bbox_in_output:
            bbox_presence = "output"
            image, text_en_out = visualize_all_bbox_together(image_path, text_en_out, bbox_presence=bbox_presence)
        else:
            # No bboxes, pass in output text
            bbox_presence = None
            image, _ = visualize_all_bbox_together(image_path, text_en_out, bbox_presence=bbox_presence)

        if bbox_presence is not None or first_run:
            new_image_path = os.path.join(os.path.dirname(image_path), next(tempfile._get_candidate_names()) + '.png')
            image.save(new_image_path)
            show_images += f'<div style="margin-bottom: 20px;"><strong>Image {idx}:</strong><br><img src="./file={new_image_path}" style="width: 250px; max-height: 400px;"></div>'

    textbox_out = text_en_out
    textbox_in = text_en_in

    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    return (
        state,
        state_,
        state.to_gradio_chatbot(),
        False,
        gr.update(value=None, interactive=True)
    )


class Chat:
    def __init__(self, model_path, conv_mode, model_base=None, quantization=None, device='cuda', cache_dir=None):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        # Add cache_dir attribute to config.json at model_path
        if cache_dir is not None and cache_dir != "./cache_dir":
            # Model path is a full path
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                # Model path is relative to cache dir
                config_path = os.path.join(cache_dir, model_path, 'config.json')
                if not os.path.exists(config_path):
                    # Model path is a hf repo
                    user, repo_id = model_path.split('/')
                    snapshot_dir = os.path.join(cache_dir, f"models--{user}--{repo_id}", 'snapshots')
                    # Get most recent snapshot
                    snapshots = os.listdir(snapshot_dir)
                    snapshot = max(snapshots, key=lambda x: os.path.getctime(os.path.join(snapshot_dir, x)))
                    snapshot_dir = os.path.join(snapshot_dir, snapshot)
                    config_path = os.path.join(snapshot_dir, 'config.json')
                # Download the model
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_path, cache_dir=cache_dir, use_auth_token=os.getenv('HF_AUTH_TOKEN'))

            with open(config_path, 'r') as f:
                config = json.load(f)
            config['cache_dir'] = cache_dir
            with open(config_path, 'w') as f:
                json.dump(config, f)

        load_8bit = quantization == "8-bit"
        load_4bit = quantization == "4-bit"

        self.tokenizer, self.model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                   load_8bit, load_4bit,
                                                                                   device=device, cache_dir=cache_dir,
                                                                                   use_auth_token=os.getenv('HF_AUTH_TOKEN'))
        self.image_processor = processor['image']
        self.conv_mode = conv_mode
        self.conv = conv_templates[conv_mode].copy()
        self.device = self.model.device

    def get_prompt(self, qs, state):
        state.append_message(state.roles[0], qs)
        state.append_message(state.roles[1], None)
        return state

    @torch.inference_mode()
    def generate(self, image_paths: list, prompt: str, first_run: bool, state):

        if first_run:
            if len(image_paths) == 1:
                prefix = f"This is a satellite image: {DEFAULT_VIDEO_TOKEN}\n"
            else:
                prefix = f"This a sequence of satellite images capturing the same location at different times in chronological order: {DEFAULT_VIDEO_TOKEN}\n"
            prompt = prefix + prompt

        state = self.get_prompt(prompt, state)
        prompt = state.get_prompt()

        prompt, outputs = run_inference_single(
            self.model,
            self.image_processor,
            self.tokenizer,
            self.conv_mode,
            inp=None,
            image_paths=image_paths,
            metadata=None, # Assume no metatdata
            prompt_strategy="interleave",
            chronological_prefix=True,
            prompt=prompt,
            print_prompt=True,
            return_prompt=True,
        )

        print("prompt", prompt)

        outputs = outputs.strip()

        print('response', outputs)
        return outputs, state


def center_map(lat, lon, zoom, basemap):

    fig = go.Figure(go.Scattermapbox())

    basemap2source = {
        "Google Maps": "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}",
        "PlanetScope Q2 2024": "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_2024q2_mosaic/gmap/{z}/{x}/{y}.png?api_key=",
        "PlanetScope Q1 2024": "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_2024q1_mosaic/gmap/{z}/{x}/{y}.png?api_key=",
        "PlanetScope Q4 2023": "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_2023q4_mosaic/gmap/{z}/{x}/{y}.png?api_key=",
        "PlanetScope Q3 2023": "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_2023q3_mosaic/gmap/{z}/{x}/{y}.png?api_key=",
        "United States Geological Survey": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
    }
    source = basemap2source[basemap]
    if "Planet" in basemap and PLANET_API_KEY is None:
        raise ValueError("Please provide a Planet API key using --planet-api-key")
    elif "Planet" in basemap:
        source += PLANET_API_KEY

    # Update the layout to include the map configuration
    fig.update_layout(
        # title="Select Image(s) using Map",
        mapbox={
            "style": "white-bg",
            "layers": [{
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": basemap,
                "source": [source]
            }],
            "center": {"lat": lat, "lon": lon},
            "zoom": zoom  # Adjust zoom level based on your preference
        },
        mapbox_style="white-bg",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700
    )

    return fig


def get_single_map_image(lat, lon, zoom, basemap):
    fig = center_map(lat, lon, zoom, basemap)
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    # Convert to PIL image
    img = Image.open(buf)
    # Center crop to the shortest dimension
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = (width + height) / 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = (height + width) / 2
    img = img.crop((left, top, right, bottom))
    return img


def get_temporal_map_image_paths(lat, lon, zoom):
    first_image = get_single_map_image(lat, lon, zoom, "PlanetScope Q3 2023")
    other_images = []
    for basemap in ["PlanetScope Q2 2024", "PlanetScope Q1 2024", "PlanetScope Q4 2023"]:
        other_images.append(get_single_map_image(lat, lon, zoom, basemap))

    # Save each image to temporary files
    first_image_path = os.path.join(os.getenv('TMPDIR'), next(tempfile._get_candidate_names()) + '.png')
    first_image.save(first_image_path)
    other_image_paths = []
    for image in other_images:
        image_path = os.path.join(os.getenv('TMPDIR'), next(tempfile._get_candidate_names()) + '.png')
        image.save(image_path)
        other_image_paths.append(image_path)

    return first_image_path, other_image_paths


def update_map(lat, lon, zoom, basemap):
    return gr.Plot(center_map(lat, lon, zoom, basemap))


if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Initializing Chat...')
    args = parse_args()

    device = args.device

    bounding_box_size = 100

    dtype = torch.float16

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (210, 210, 0),
        (255, 0, 255),
        (0, 255, 255),
        (114, 128, 250),
        (0, 165, 255),
        (0, 128, 0),
        (144, 238, 144),
        (238, 238, 175),
        (255, 191, 0),
        (0, 128, 0),
        (226, 43, 138),
        (255, 0, 255),
        (0, 215, 255),
    ]

    color_map = {
        f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
        color_id, color in enumerate(colors)
    }

    used_colors = colors

    CONV_MODE = args.conv_mode
    PLANET_API_KEY = args.planet_api_key
    if PLANET_API_KEY is None:
        PLANET_API_KEY = os.getenv('PLANET_API_KEY')

    handler = Chat(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        model_base=args.model_base,
        quantization=args.quantization,
        device=args.device,
        cache_dir=args.cache_dir
    )

    # TODO: Consider adding github stars later
     # <a href='https://github.com/ermongroup/TEOChat/stargazers'><img src='https://img.shields.io/github/stars/ermongroup/TEOChat.svg?style=social'></a>
    title_markdown = ("""
    <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <a href="https://github.com/ermongroup/TEOChat" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        <img src="static/logo.png" alt="TEOChat🛰️" style="max-width: 120px; height: auto;">
    </a>
    <div>
        <h1 >TEOChat: Large Language and Vision Assistant for Temporal Earth Observation Data</h1>
        <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
    </div>
    </div>


    <div align="center">
        <div style="display:flex; gap: 0.25rem;" align="center">
            <a href='https://github.com/ermongroup/TEOChat'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            <a href="http://arxiv.org/abs/2410.06234"><img src="https://img.shields.io/badge/Arxiv-2410.06234-red"></a>
        </div>
    </div>
    """)

    introduction = '''
    **Instructions:**
    <ol>
    <li>Select image(s) to input to TEOChat by doing one of the following:
        <ol>
            <li>(Below) Click the image icon in the First Image widget to upload a single image, then optionally upload additional temporal images by clicking the Optional Additional Image(s) widget.</li>
            <li>(On the right) Enter the latitude, longitude, zoom, and select the basemap to view the map image, then:
                <ol>
                    <li>Upload the map image based on the entered latitude, longitude, zoom, and basemap.</li>
                    <li>Upload a temporal map image (including 4 images from PlanetScope) based on the entered latitude, longitude, and zoom.</li>
                    <li>Pan around and download the current map image by clicking the 📷 icon at the top right, then uploading that image.</li>
                </ol>
            </li>
            <li>(On the bottom) Select prespecified example image(s) (and text input).</li>
        </ol>
    </li>
    <li>Optionally draw a bounding box using the First Image widget by clicking the pen icon on the bottom.</li>
    <li>Enter a text prompt in the text input above.</li>
    <li>Click <b>Send</b> to generate the output.</li>
    </ol>
    '''

    block_css = """
    #buttons button {
        min-width: min(120px,100%);
    }
    """

    tos_markdown = """
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """

    learn_more_markdown = """
    ### License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(cur_dir, 'examples')

    textbox = gr.Textbox(
        show_label=False, placeholder="Upload an image or obtain one using the map viewer, then enter text here and press Send ->", container=False
    )
    with gr.Blocks(title='TEOChat', theme=gr.themes.Default(text_size=sizes.text_lg), css=block_css) as demo:
        gr.Markdown(title_markdown)
        state = gr.State()
        state_ = gr.State()
        first_run = gr.State()

        with gr.Row():
            chatbot = gr.Chatbot(label="TEOChat", bubble_full_width=True)
        with gr.Row():
            with gr.Column(scale=8):
                textbox.render()
            with gr.Column(scale=1, min_width=50):
                submit_btn = gr.Button(
                    value="Send", variant="primary", interactive=True
                )
        with gr.Row(elem_id="buttons") as button_row:
            regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
            clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        with gr.Row():
            with gr.Column(scale=1, elem_id="introduction"):
                gr.Markdown(introduction)
                image1 = gr.ImageEditor(
                    label="First Image",
                    type="filepath",
                    layers=False,
                    transforms=(),
                    sources=('upload', 'clipboard'),
                    brush=Brush(colors=["red"], color_mode="fixed", default_size=3)
                    )
                image_list = gr.File(
                    label="Optional Additional Image(s)",
                    file_count="multiple"
                )

            with gr.Column(scale=1):
                with gr.Row():
                    map_view = gr.Plot(label="Map Image(s)")

                with gr.Row():
                    lat = gr.Number(value=37.43144514632126, label="Latitude")
                    lon = gr.Number(value=-122.16210856357836, label="Longitude")
                    zoom = gr.Number(value=18, label="Zoom")
                    basemap = gr.Dropdown(
                        value="Google Maps",
                        choices=[
                            "Google Maps",
                            "PlanetScope Q2 2024",
                            "PlanetScope Q1 2024",
                            "PlanetScope Q4 2023",
                            "PlanetScope Q3 2023",
                            "United States Geological Survey",
                        ],
                        label="Basemap"
                    )
                with gr.Row():
                    single_map_upload_button = gr.Button("Upload Map based on Lat/Lon/Zoom/Basemap")
                    temporal_map_upload_button = gr.Button("Upload Temporal Map (PlanetScope Q3-Q4 2023, Q1-Q2 2024) based on Lat/Lon/Zoom")

        demo.load(center_map, [lat, lon, zoom, basemap], map_view)

        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        f"{example_dir}/rqa.png",
                        "What is this? [21, 3, 47, 19]",
                    ],
                    [
                        f"{example_dir}/xBD_loc.png",
                        "Identify the location of the building on the right of the image using a bounding box of the form [x_min, y_min, x_max, y_max].",
                    ],
                    [
                        f"{example_dir}/AID_cls.png",
                        "Classify this image as one of: Oil Refinery, Compressor Station, Pipeline, Processing Plant, Well Pad.",
                    ],
                    [
                        f"{example_dir}/HRBEN_qa.png",
                        "Is there a road next to a body of water?",
                    ]
                ],
                inputs=[image1, textbox],
                outputs=[image_list, first_run, state, state_, chatbot],
                label="Single Image Examples",
                fn=single_example_trigger,
                run_on_click=True,
                cache_examples=False
            )
            gr.Examples(
                examples=[
                    [
                        f"{example_dir}/fMoW_cls_1.png",
                        [f"{example_dir}/fMoW_cls_2.png", f"{example_dir}/fMoW_cls_3.png", f"{example_dir}/fMoW_cls_4.png"],
                        "Classify the sequence of images as one of: flooded road, lake or pond, aquaculture, dam, mountain trail.",
                    ],
                    [
                        f"{example_dir}/xBD_dis_1.png",
                        [f"{example_dir}/xBD_dis_2.png"],
                        "What disaster has occurred in the area?",
                    ],
                    [
                        f"{example_dir}/xBD_cls_1.png",
                        [f"{example_dir}/xBD_cls_2.png"],
                        "Classify the level of damage experienced by the building at location [0, 8, 49, 53].",
                    ],
                    [
                        f"{example_dir}/S2Looking_cd_1.png",
                        [f"{example_dir}/S2Looking_cd_2.png"],
                        "Identify all changed buildings using bounding boxes of the form [x_min, y_min, x_max, y_max].",
                    ],
                    [
                        f"{example_dir}/QFabric_rtqa_1.png",
                        [f"{example_dir}/QFabric_rtqa_2.png", f"{example_dir}/QFabric_rtqa_3.png", f"{example_dir}/QFabric_rtqa_4.png", f"{example_dir}/QFabric_rtqa_5.png"],
                        "In which image was construction finished?",
                    ],
                ],
                inputs=[image1, image_list, textbox],
                outputs=[image_list, first_run, state, state_, chatbot],
                label="Temporal Image Examples",
                fn=temporal_example_trigger,
                run_on_click=True,
                cache_examples=False
            )
        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)

        lat.change(fn=update_map, inputs=[lat, lon, zoom, basemap], outputs=[map_view])
        lon.change(fn=update_map, inputs=[lat, lon, zoom, basemap], outputs=[map_view])
        zoom.change(fn=update_map, inputs=[lat, lon, zoom, basemap], outputs=[map_view])
        basemap.change(fn=update_map, inputs=[lat, lon, zoom, basemap], outputs=[map_view])
        single_map_upload_button.click(fn=get_single_map_image, inputs=[lat, lon, zoom, basemap], outputs=[image1])
        temporal_map_upload_button.click(fn=get_temporal_map_image_paths, inputs=[lat, lon, zoom], outputs=[image1, image_list])

        submit_btn.click(
            generate,
            [image1, image_list, textbox, first_run, state, state_],
            [state, state_, chatbot, first_run, textbox]
        )

        regenerate_btn.click(
            regenerate,
            [state, state_], [state, state_, chatbot, first_run]
        ).then(
            generate,
            [image1, image_list, textbox, first_run, state, state_],
            [state, state_, chatbot, first_run, textbox]
        )

        clear_btn.click(
            clear_history,
            [state, state_],
            [image1, image_list, textbox, first_run, state, state_, chatbot]
        )

    demo.queue()

    if args.dont_use_fast_api:
        demo.launch(
            share=False,
            server_name=args.server_name,
            favicon_path='static/logo.svg',
            server_port=args.port,
            allowed_paths=['static/logo.png'],
            )

    else:

        import uvicorn
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        # create a FastAPI app
        app = FastAPI()

        # create a static directory to store the static files
        static_dir = Path('./static')
        static_dir.mkdir(parents=True, exist_ok=True)

        # mount FastAPI StaticFiles server
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        # mount Gradio app to FastAPI app
        app = gr.mount_gradio_app(app, demo, path="/", favicon_path='static/logo.svg')

        uvicorn.run(app, host=args.server_name, port=args.port)
