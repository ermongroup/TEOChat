import os
import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer

from videollava.model import *


def merge_lora(args):
    # Changed as per: https://github.com/artidoro/qlora/issues/29
    device = "cpu"
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_base, cache_dir=args.cache_dir, torch_dtype=torch.float16, device_map={"": device})
    if os.path.exists(os.path.join( args.model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(args.model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    model = PeftModel.from_pretrained(model, args.model_path, torch_dtype=torch.float16, device_map={"": device})
    model = model.merge_and_unload()

    image_tower = model.get_image_tower()
    if not image_tower.is_loaded:
        image_tower.load_model()
    image_tower.to(device=device, dtype=torch.float16)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)

    args = parser.parse_args()

    merge_lora(args)
