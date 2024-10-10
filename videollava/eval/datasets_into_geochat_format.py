import pandas as pd 
import re
import json

def qfabric_semiconverted_to_geochat_dataset_format(json_file):
    with open(json_file) as f:
        data = json.load(f)
    for conversation_group in data:
        for item in conversation_group["conversations"]:
            # Remove satellite specifications
            item["value"] = re.sub(r"This is a satellite image :", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image:", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image.*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from.*?:\s*", "", item["value"])
            # Remove strings around <identify> that are redundant
            item["value"] = re.sub(r'What is <identify>|this area {<', lambda x: '[identify]' if 'What is [identify]' in x.group() else '{<', item["value"])
            # Switch out <video> for <image>
            item["value"] = re.sub(r'<video>', '', item["value"])
            # Get rid of "this region" immediately before the bounding box 
            item["value"] = re.sub(r'this region {<', '{<', item["value"])
            # Check for the presence of '<identify>' and modify the string accordingly
            if '[identify]' in item["value"]:
                # Find the position of '<identify>' and the position of the first occurrence of '>}' after '<identify>'
                identify_index = item["value"].find('[identify]')
                identify_word_index = item["value"].find('Identify ', identify_index + 8)  
                # if identify_word_index != -1:
                #     item["value"] = item["value"][:identify_word_index] + item["value"][identify_word_index + 8:]
                closing_brace_index = item["value"].find('>}', identify_index)
    return data

def fmow_to_geochat_dataset_format(json_file):
    with open(json_file) as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        video_count = len(entry.get("video", []))
        if video_count > 1:
            original_videos = entry["video"]
            for idx in range(video_count):
                new_entry = entry.copy()
                new_entry['video'] = [original_videos[idx]]
                new_entry['image'] = original_videos[idx]
                new_entry['linked_id'] = entry['id']
                new_entry['img_idx_from_video_lst_id'] = idx
                data.append(new_entry)
        else: 
            new_entry = entry.copy()
            new_entry['image'] = original_videos[0]
    for conversation_group in data:
        for item in conversation_group["conversations"]:
            item["value"] = re.sub(r"This is a satellite image :", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image:", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image.*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from.*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of low-resolution, optical satellite images capturing the same location at different times: ", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of high-resolution, optical satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of satellite images from .*? the same location at different times:", "", item["value"])
            item["value"] = re.sub(r'This is a high resolution,? optical satellite image .*:\s*<image>\n', '\n', item["value"])
            item["value"] = re.sub(r'^This is a high[- ]resolution,? .*?image:\s*<image>\n', '\n', item["value"], flags=re.IGNORECASE | re.DOTALL)
            
            # Switch out <video> for <image>
            item["value"] = re.sub(r'<video>', '', item["value"])
            # Get rid of "this region" immediately before the bounding box 
            item["value"] = re.sub(r'this region {<', '{<', item["value"])
            # Which class 
            item["value"] = re.sub(r'Which of the following classes does this sequence of images belong to', 'Which of the following classes does this image belong to', item["value"])
            # Please answer using one of the following classes:
            item["value"] = re.sub(r'Please answer using only one of the following classes:', 'Please use one of the following classes:', item["value"])
            # Check for the presence of '<identify>' and modify the string accordingly
            if '[identify]' in item["value"]:
                # Find the position of '<identify>' and the position of the first occurrence of '>}' after '<identify>'
                identify_index = item["value"].find('[identify]')
    for i, entry in enumerate(data):
        video_count = len(entry.get("video", []))
        if video_count > 1:
            data.pop(i)
    return data

def xbd_to_geochat_dataset_format(json_file):

    with open(json_file) as f:
        data = json.load(f)

    new_data = []
    for i, entry in enumerate(data):
        if entry["task"].startswith("localization"):
            new_entry=entry.copy()
            new_entry['image'] = entry['video'][0]
            new_data.append(new_entry)
        if entry["task"].startswith("classification"):
            new_entry=entry.copy()
            new_entry['image'] = entry['video'][1]
            new_data.append(new_entry)
        # Auxiliary tasks all look at the second image
        else: 
            new_entry=entry.copy()
            new_entry['image'] = entry['video'][1]
            new_data.append(new_entry)
            
    for conversation_group in new_data:
        localization=False
        classification=False
        # Add a [refer] token to localization tasks
        if conversation_group["task"].startswith("localization") or "identify" in conversation_group["task"].lower():
            localization=True
        # Add a [identify] token to classification tasks
        if conversation_group["task"].startswith("classification"):
            classification=True
        
        for item in conversation_group["conversations"]:
            item["value"] = re.sub(r"This is a satellite image :", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image:", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image.*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"These are two satellite images from .*? capturing the same location at different times: ", "", item["value"])
            item["value"] = re.sub(r"These are two low-resolution, optical satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"These are two high-resolution, optical satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"These are two high-resolution, optical satellite images from .*? capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"These are two satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"These are two satellite images from .*? capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r'These are two high-resolution,? optical satellite images .*:\s*<image>\n', '<image>\n', item["value"])
            item["value"] = re.sub(r'These are two high resolution,? optical satellite images .*:\s*<image>\n', '<image>\n', item["value"])
            item["value"] = re.sub(r'^This is a high[- ]resolution,? .*?image:\s*<image>\n', '<image>\n', item["value"], flags=re.IGNORECASE | re.DOTALL)

            # Switch out <video> for <image>
            if classification:
                item["value"] = re.sub(r'<video> \n', '<image> \n [identify] ', item["value"])
                item["value"] = re.sub(r' in the second image.', '.', item["value"])
            elif localization:
                item["value"] = re.sub(r'<video> \n', '<image> \n [refer] ', item["value"])
                item["value"] = re.sub(r'Image 1', 'the image', item["value"])
            else:
                item["value"] = re.sub(r'<video> \n', '<image> \n ', item["value"])

            # Replace temporal/multi-image wording for auxiliary tasks
            replacements = {
                'Are there any buildings in the first image which have been damaged in the second image? Answer with one word.': 'Are there any damaged buildings in the image? Answer with one word.',
                'Have any buildings in the first image been damaged in the second image? Answer with one word.': 'Have any buildings been damaged in the area? Answer with one word.',
                'What disaster has occurred between the first and second image?': 'What disaster has occurred here?',
                'Identify the buildings in the first image which were severely damaged or destroyed in the second image. Include a bounding box of the form [x_min, y_min, x_max, y_max] for each identified building in your response. If there are no such buildings, do not output a bounding box.': 'Identify the severely damaged or destroyed buildings in the image. Include a bounding box of the form [x_min, y_min, x_max, y_max] for each identified building in your response. If there are no such buildings, do not output a bounding box.'
            }
            for old, new in replacements.items():
                item['value'] = re.sub(re.escape(old), new, item['value'])


            # Get rid of "this region" immediately before the bounding box 
            item["value"] = re.sub(r'this region {<', '{<', item["value"])
            # Which class 
            item["value"] = re.sub(r'Which of the following classes does this sequence of images belong to', 'Which of the following classes does this image belong to', item["value"])
            # Please answer using one of the following classes:
            item["value"] = re.sub(r'Please answer using only one of the following classes:', 'Please use one of the following classes:', item["value"])
            # Replace bounding box format [79, 27, 85, 81] with {<79><27><85><81>|<0>}
            item["value"] = re.sub(r'\[(\d+), (\d+), (\d+), (\d+)\]', r'{<\1><\2><\3><\4>|<0>}', item["value"])
            # Replace bounding box format [x_min, y_min, x_max, y_max] with {<x_min><y_min><x_max><y_max>|<0>}
            item["value"] = re.sub(r'\[(x_min), (y_min), (x_max), (y_max)\]', r'{<\1><\2><\3><\4>|<0>}', item["value"])
    return new_data

def s2looking_to_geochat_dataset_format(json_file):
    with open(json_file) as f:
        data = json.load(f)

    question = "<image>\n [refer] Identify all buildings in the image."

    new_dataset = []
    for elem in data:
        for i in range(2):
            new_item = {}
            new_item['id'] = elem['id'] + '_' + str(i)
            new_item['metadata'] = elem['metadata'][i]
            new_item['original_input_polygon'] = elem['original_input_polygon']
            new_item['task'] = elem['task']
            new_item['image'] = elem['video'][i]
            new_item['geovlm_id'] = i
            new_item['original_conversation'] = elem['conversations']
            new_item['conversations'] = [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": ""
                }
                ]
            new_dataset.append(new_item)
    
    data = new_dataset

    for conversation_group in data:
        for item in conversation_group["conversations"]:
            # Check if the sentence starts with "This is" or "These are" and contains "<image>"
            if (item["value"].startswith("This is") or item["value"].startswith("These are")) and "<image>" in item["value"]:
                colon_index = item["value"].find(":")
                if colon_index != -1 and item["value"][colon_index+1:].strip().startswith("<image>"):
                    item["value"] = item["value"][colon_index+1:].strip()
            item["value"] = re.sub(r"This is a sequence of high-resolution, optical satellite images from Maxar's GeoEye-1, QuickBird-2, WorldView-2, or WorldView-3 capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of low-resolution, optical satellite images from Sentinel-2 capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image :", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image:", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a high resolution, optical satellite image.*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from .*?:\s*", "", item["value"])
            item["value"] = re.sub(r"This is a satellite image from.*?:\s*", "", item["value"])
            # This one is the one I'm referring to: 
            item["value"] = re.sub(r'^This is a sequence of.*times:$', '', item["value"])
            item["value"] = re.sub(r"This is a sequence of high-resolution, optical satellite images from .*? capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of low-resolution, optical satellite images capturing the same location at different times: ", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of high-resolution, optical satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of satellite images from .*? the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"These are two high-resolution, optical satellite images capturing the same location at different times:", "", item["value"])
            item["value"] = re.sub(r"This is a sequence of images from the satellites GaoFen, SuperView and BeiJing-2, capturing the same location at different times:", "", item["value"])
            
            # Switch out <video> for <image>
            item["value"] = re.sub(r'<video>', '', item["value"])
            # Get rid of "this region" immediately before the bounding box 
            item["value"] = re.sub(r'this region {<', '{<', item["value"])
            # Which class 
            item["value"] = re.sub(r'Which of the following classes does this sequence of images belong to', 'Which of the following classes does this image belong to', item["value"])
            # Please answer using one of the following classes:
            item["value"] = re.sub(r'Please answer using only one of the following classes:', 'Please use one of the following classes:', item["value"])
            # Check for the presence of '<identify>' and modify the string accordingly
            if '[identify]' in item["value"]:
                # Find the position of '<identify>' and the position of the first occurrence of '>}' after '<identify>'
                identify_index = item["value"].find('[identify]')
                closing_brace_index = item["value"].find('>}', identify_index)

            # Fix the bounding box format: 
    for conversation_group in data:
        for item in conversation_group["conversations"]:
            # Replace bounding box format [79, 27, 85, 81] with {<79><27><85><81>|<0>}
            item["value"] = re.sub(r'\[(\d+), (\d+), (\d+), (\d+)\]', r'{<\1><\2><\3><\4>|<0>}', item["value"])
            # Replace bounding box format [x_min, y_min, x_max, y_max] with {<x_min><y_min><x_max><y_max>|<0>}
            item["value"] = re.sub(r'\[(x_min), (y_min), (x_max), (y_max)\]', r'{<\1><\2><\3><\4>|<0>}', item["value"])
    for i, entry in enumerate(data):
        video_count = len(entry.get("video", []))
        if video_count > 1:
            data.pop(i)
    return data

def check_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    for conversation_group in data:
        for item in conversation_group["conversations"]:
            if '<image>' not in item["value"]:
                if item["from"] != 'gpt':
                    print(f"Missing <image> in: {item}")
            if any(sentence.strip().startswith(('This is', 'These are')) for sentence in item["value"].split('.')):
                print(f"Starts with 'This is' or 'These are' in: {item}")
if __name__ == "__main__":

    # Paths to datasets
    fmow_0 = "/scr/geovlm/fmow_low_res_val.json"
    fmow_1 = "/scr/geovlm/fmow_high_res_val.json"

    qfabric_0 = '/scr/geovlm/QFabric/test_geochat_seqlen_5_256.json'
    qfabric_1 = '/scr/geovlm/QFabric/test_geochat_seqlen_2_256.json'

    xbd_0 = '/scr/geovlm/xbd_test_auxiliary_multi_image.json'
    xbd_1 = '/scr/geovlm/xbd_test_canon_classification.json'
    xbd_2 = '/scr/geovlm/xbd_test_canon_localization.json'

    print("Running conversion on all datasets, storing updated datasets in variables")

    from tqdm import tqdm

    dataset_formats = [
        (fmow_to_geochat_dataset_format, fmow_0),
        (fmow_to_geochat_dataset_format, fmow_1),
    ]
    formatted_datasets = []
    for format_func, dataset in tqdm(dataset_formats, desc="Converting datasets"):
        if "xbd_test_auxiliary" in dataset: 
            formatted_datasets.append(format_func(dataset))
    
    fmow_0_formatted, fmow_1_formatted = formatted_datasets

    # Write the formatted data for fmow_0 into a JSON file named geochat_fmow_RECENT_format_low_res.json
    with open('/scr/geovlm/geochat_fmow_RECENT_format_low_res.json', 'w') as file:
        json.dump(fmow_0_formatted, file)

    # Write the formatted data for fmow_1 into a JSON file named geochat_fmow_RECENT_format_low_res_AGG.json
    with open('/scr/geovlm/geochat_fmow_RECENT_format_high_res.json', 'w') as file:
        json.dump(fmow_1_formatted, file)
    
    check_file('/scr/geovlm/geochat_fmow_RECENT_format_low_res.json')
    check_file('/scr/geovlm/geochat_fmow_RECENT_format_high_res.json')
