## Table of Contents
* [Data preparation](#data-preparation)
* [Training TEOChat](#training-teochat)
* [Validating TEOChat](#validating-teochat)
* [Fine-tuning TEOChat](#fine-tuning-teochat)

## Data preparation

The TEOChatlas dataset and external evaluation datasets are available for download [here](https://huggingface.co/datasets/jirvin16/TEOChatlas).

You can download all of the data using the following code:
```python
from datasets import load_dataset

# Optionally specify a cache directory if you have limited space in your home directory
# Or if you want to place the data somewhere else.
cache_dir = None

# Optionally specify a split if you only want to download a subset of the data
# The splits are defined in the hugingface hub page for the dataset
split = None

dataset = load_dataset("jirvin16/TEOChatlas", split=split, cache_dir=cache_dir, trust_remote_code=True)
```
This will download the data to the machine where the code is run. Running `load_dataset` again will not re-download the data, unless the cache directory is changed. The training code uses `load_dataset` to load the data.

## Training TEOChat

### 1. Download Video-LLaVA projector weights
Navigate to [the Video-LLaVA-Pretrain-7B](https://huggingface.co/LanguageBind/Video-LLaVA-Pretrain-7B/tree/main) model on the Hugging Face model hub and download the `mm_projector.bin` file. This file contains the weights for the Video-LLaVA projector, which will be used to initialize the TEOChat projector.

### 2. Edit the training script
You need to make the following changes in order to train TEOChat:
- Set the `--pretrain_mm_mlp_adapter` to the path of the `mm_projector.bin` file you downloaded in step 1.
- Set the `--output_dir` to the directory where you want to save the model checkpoints and logs. The prefix should be `video-llava-7b-8bit-lora` otherwise there may be issues evaluating the model.
- (Optional) Set the `--cache_dir` to the directory where you want to cache the pretrained models used for initialization (like Video-LLaVA).
- (Optional) Set the `--data_cache_dir` to the directory where you stored the TEOChatlas dataset if you specified a cache directory in the data preparation step.

### 3. Run the training script!

```bash
sh scripts/train_teochat.sh
```

## Validating TEOChat

```bash
sh scripts/eval_teochat.sh <dataset_split> <model_path> <model_base> <cache_dir> <data_cache_dir>
```
See [eval.py](https://github.com/ermongroup/TEOChat/tree/main/videollava/eval/eval.py#L76-L83) for the full list of dataset splits.

For example, to evaluate TEOChat on UC Merced, you can run:
```bash
sh scripts/eval_teochat.sh ucm jirvin16/TEOChat
```
assuming the model and data are stored in the default cache directories.

To evaluate a newly trained model on UC Merced, you can run:
```bash
sh scripts/eval_teochat.sh ucm /path/to/model LanguageBind/Video-LLaVA-7B
```
again assuming the model and data are stored in the default cache directories.

## Fine-tuning TEOChat
Instructions for fine-tuning TEOChat will be provided here.
