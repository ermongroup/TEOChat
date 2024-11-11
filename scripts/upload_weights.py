import sys
import torch
import argparse

# Add the parent directory to the path
# This is necessary to import the videollava package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from videollava.model import *
from transformers import AutoTokenizer


def upload(args):
    device = "cpu"
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, cache_dir=args.cache_dir, device_map={"": device})

    image_tower = model.get_image_tower()
    if not image_tower.is_loaded:
        image_tower.load_model()
    image_tower.to(device=device, dtype=torch.float16)

    model.push_to_hub(args.upload_model_path)

    # Upload tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.push_to_hub(args.upload_model_path)


if __name__ == "__main__":
    # Note: You have to login to Hugging Face using `huggingface-cli login` before running this script
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--upload-model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)

    args = parser.parse_args()

    upload(args)
