import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import random

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from eval_classification import *
from datasets_into_geochat_format import s2looking_to_geochat_dataset_format, qfabric_semiconverted_to_geochat_dataset_format, xbd_to_geochat_dataset_format
from geochat_s2looking_utils import evaluate_geochat_s2looking

from PIL import Image
import math
import numpy as np

def aggregate_accuracy(answers_file, output_file):
    """
    Parses geochat inference output and aggregates votes on single images
    across an image sequence into the format needed for geovlm-style evaluation. 

    params: 
        - answers_file: path to the file containing geochat inference output
        - output_file: path to the file where the aggregated output will be saved
    """
    with open(answers_file, 'r') as f:
        answers = [json.loads(line) for line in f]
    
    # dictionary that will contain parsed output
    votes = {}

    # parse answers so that predictions with the same geovlm_id
    # are aggregated into a single item with 'predictions' containing
    # a list of values. All other keys should be the same
    for answer in answers: 
        id = answer['geovlm_id']
        if id not in votes:
            item = {}
            item['predicted'] = [answer['predicted']]
            item['ground_truth'] = answer['ground_truth']
            item['task'] = answer['task']
            item['original_input_polygon'] = answer['original_input_polygon']
            item['question'] = answer['question']
            item['id'] = answer['id']
            votes[id] = item
        else: 
            votes[id]['predicted'].append(answer['predicted'])
    
    # implement voting so that each list in 'predicted' attribute
    # is reduced to the most common value
    for linked_id, predicted_dict in votes.items():
        predicted = predicted_dict['predicted']
        unique, counts = np.unique(predicted, return_counts=True)
        index = np.argmax(counts)
        votes[linked_id]['predicted'] = unique[index]

    with open(output_file, 'w') as f:
        json.dump(votes, f)

    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    print(args)
    print()
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    try: 
        with open(args.question_file, 'r') as f:
            questions = json.load(f)    
    except: 
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    if args.end_ind is not None:
        questions = questions[args.start_ind:args.end_ind]
    else:
        questions = questions[args.start_ind:]
    print("start ind: ", args.start_ind)
    print("end ind: ", args.end_ind)

    # check if the answers file alreay exists
    if not os.path.exists(answers_file) or args.rerun==True:
        print('Running inference...')
        image = Image.open(image_file)

        if args.dataset_size:
            # randomly sample dataset_size number of questions
            questions = random.sample(questions, args.dataset_size)

        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")


        # Model
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, cache_dir=args.cache_dir)
        
        for i in tqdm(range(0,len(questions),args.batch_size)):
            input_batch=[]
            input_image_batch=[]
            count=i
            image_folder=[]     
            batch_end = min(i + args.batch_size, len(questions))

            for j in range(i,batch_end):
                image_file=questions[j]['image']
                qs=questions[j]['conversations'][0]['value']

                # TODO do we keep that?
                
                # if model.config.mm_use_im_start_end:
                #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                #     print("start end token")
                # else:
                #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                print(prompt)

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_batch.append(input_ids)

                image = Image.open(os.path.join(args.image_folder, image_file))

                image_folder.append(image)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            max_length = max(tensor.size(1) for tensor in input_batch)

            final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
            final_input_tensors=torch.cat(final_input_list,dim=0)
            image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values']    

            with torch.inference_mode():
                output_ids = model.generate( final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)

            input_token_len = final_input_tensors.shape[1]
            n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            for k in range(0,len(final_input_list)):
                output = outputs[k].strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()

                ans_id = shortuuid.uuid()
                
                if args.dataset == 'qfabric':
                    ans_file.write(json.dumps({
                                            "id": questions[count]["id"],
                                            "image_id": questions[count]["image"],
                                            "question": questions[count]['conversations'][0]['value'],
                                            "predicted": output,
                                            "ground_truth": questions[count]['conversations'][1]['value'],
                                            "task": questions[count]['task'],
                                            "original_input_polygon": questions[count]['original_input_polygon'],
                                            "geovlm_id": questions[count]['geovlm_id']
                                            }) + "\n")
                elif args.dataset == 's2looking':
                    ans_file.write(json.dumps({
                                            questions[count]["id"] : {
                                            "image_id": questions[count]["image"],
                                            "question": questions[count]['conversations'][0]['value'],
                                            "predicted": output,
                                            "task": questions[count]['task'],
                                            "original_input_polygon": questions[count]['original_input_polygon'],
                                            "geovlm_id": questions[count]['geovlm_id'],
                                            "original_question": questions[count]['conversations'][0]['value'],
                                            "original_answer": questions[count]['conversations'][1]['value']
                                            }}) + "\n")
                elif args.dataset == 'xbd':
                    ans_file.write(json.dumps({
                                            questions[count]["id"] : {
                                            "image_id": questions[count]["image"],
                                            "question": questions[count]['conversations'][0]['value'],
                                            "predicted": output,
                                            "task": questions[count]['task'],
                                            "original_input_polygon": questions[count]['original_input_polygon'],
                                            "original_question": questions[count]['conversations'][0]['value'],
                                            "original_answer": questions[count]['conversations'][1]['value']
                                            }}) + "\n")

                count=count+1
                ans_file.flush()
        ans_file.close()
    
        agg_ans_file = args.answers_file.replace('.json', '_agg.json')
        print("Raw Geochat output saved to ", args.answers_file)

    # determine the split from args.question_file
    if 'test' in args.question_file:
        split = 'test'
    elif 'val' or 'valid' or 'validation' in args.question_file:
        split = 'val'
    elif 'train' in args.question_file:
        split = 'train'
    else:
        raise ValueError("Split not found in question file name")
    
    print("Now parsing and aggregating votes for geovlm evaluation...")
    if args.dataset == 'qfabric':
        aggregate_accuracy(args.answers_file, agg_ans_file)
        print("Aggregated output saved to ", agg_ans_file)

        classification_segmentation(agg_ans_file, 'qfabric')
    elif args.dataset == 's2looking':
        evaluate_geochat_s2looking(args.answers_file, args.question_file, split)
    elif args.dataset == 'xbd':
        classification_segmentation(agg_ans_file, 'xbd')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    parser.add_argument("--start-ind", type=int, default=0)
    parser.add_argument("--end-ind", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--rerun", type=bool, default=False)
    parser.add_argument("--dataset_size", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)
