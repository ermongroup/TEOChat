import fire
import json
from pathlib import Path

from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import get_model_name_from_path
from videollava.model.multimodal_encoder.languagebind.video.processing_video import LanguageBindVideoProcessor

from eval_classification import accuracy_precision_recall
from eval_referring import referring_expression
from classification_segmentation import classification_segmentation

from ben_utils import run_ben_inference
from aid_fmow_ucmerced_utils import run_aid_fmow_ucmerced_inference
from qfabric_utils import run_qfabric_inference
from geochat_utils import run_geochat_inference
from s2looking_utils import run_s2looking_inference
from xbd_utils import run_xbd_inference
from cdvqa_utils import run_cdvqa_inference


def aggregated(answer_path, dataset=None, verbose=False, split=None):
    """
    Define an aggregated metric for our created instruction-following datasets.
    It includes eval_description and eval_referring metrics.
    """
    saving_path_root = Path(answer_path).parent

    with open(answer_path, 'r') as f:
        answers = json.load(f)
    
    print("Referring expression")
    referring_expression(answer_path, dataset, False, saving_path_root, split=split)
    print()
    print("Accuracy")
    accuracy_precision_recall(answer_path, dataset, verbose=False)
    print()

    # TODO per-task metrics for qfabric and xbd
    
    if dataset == 'qfabric' or dataset == 'xbd':
        classification_segmentation(answer_path, dataset)

    if dataset == "s2looking":
        # also run per-question referring expression
        question1 = 'temporal_question_answering: Are there any buildings in the first image which were {destructed,torn down} in the second?'
        question2 = 'temporal_referring_expression: Identify the buildings in the first image which were {built,constructed,destructed,torn down} as seen in the second image.'
        question3 = 'localization_task: Identify all changed buildings.'
        question4 = 'referring_expression: identify the {constructed, destructed} buildings in the image.'
        question5 = 'question_answering: Have any buildings been task in the area? Please answer with Yes or No'


        for question in [question1, question2, question3, question4, question5]:
            dataset_question = {}
            for data in answers:
                if answers[data]['task'] == question:
                    dataset_question[data] = answers[data]
            if len(dataset_question) > 0:
                print('Evaluating for question ', question)
                print('Size of the dataset is ', len(dataset_question))
                referring_expression(dataset_question, dataset, False, saving_path_root, split=split)
                print()


def load_model(model_path, model_base, cache_dir, device, vision_type=None, load_4bit=False, load_8bit=False):
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        device=device,
        cache_dir=cache_dir,
        vision_type=vision_type,
    )

    if vision_type is None:
        # Automatically determine which to us
        # For now assumes one of the processors is not None and one is None
        vision_types = ['image', 'video']
        if processor['image'] is None and processor['video'] is None:
            raise ValueError("Both image and video processors are None")
        elif processor['image'] is not None and processor['video'] is not None:
            vision_processor = processor['image']
        for vision_type in vision_types:
            vision_processor = processor[vision_type]
            if vision_processor is not None:
                break
    else:
        vision_processor = processor[vision_type]
    use_video_data = vision_type == 'video'
    return tokenizer, model, vision_processor, use_video_data


def infer_eval(
        dataset_path,
        model_path,
        model_base="LanguageBind/Video-LLaVA-7B",
        cache_dir="/deep/group/aicc-bootcamp/geovlm/models/vllava_cache",
        outname=None,
        open_prompt=None, 
        repeat_frames=None,
        prompt_strategy="interleave",
        chronological_prefix=True,
        load_8bit=False, 
        load_4bit=False,
        verbose=False,
        rerun=False,
        vision_type=None,
        data_frac=None,
        data_size=None,
        conv_mode="v1",
        delete_system_prompt=False,
        start_ind=None,
        end_ind=None,
        last_image=None,
        print_prompt=False
    ):
    """
    Args:
        dataset_path: path to dataset
        model_path: path to model
        model_base: model base name
        cache_dir: cache directory
        outname: output file name (uses args if None)
        open_prompt options: None, "open", "multi-open"
        repeat_frames options: None, "uniform", "first", "last"
        prompt_strategy options: None, "interleave"
        chronological_prefix: whether to use chronological prefix "in chronological order"
        load_8bit: whether to load 8-bit model
        load_4bit: whether to load 4-bit model
        verbose: whether to print verbose output
        rerun: whether to rerun inference
        vision_type: "image" or "video"
        data_frac: fraction of data to use
        data_size: number of data samples to use
        conv_mode: conversation mode (should be v1 for our models, geochat, and videollava)
        delete_system_prompt: whether to delete system prompt
        start_ind: start index of data
        end_ind: end index of data
        last_image: whether to use last image in video
        print_prompt: whether to print prompt
    """
    args = locals()
    print(f"Arguments passed to infer_eval:")
    for k, v in args.items():
        print(f"{k} ({type(v).__name__}): {v}")

    # check that data_size and data_frac are not both set
    if data_size is not None and data_frac is not None:
        raise ValueError("data_size and data_frac cannot both be set")
    if data_size is None and data_frac is None:
        data_frac = 1

    dataset2metrics = {
        "lrben": [accuracy_precision_recall],
        "hrben": [accuracy_precision_recall],
        "fmow": [accuracy_precision_recall],
        "s2looking": [aggregated],
        "xbd": [aggregated], 
        "qfabric": [aggregated],
        "aid": [accuracy_precision_recall],
        "ucmerced": [accuracy_precision_recall],
        "cdvqa": [accuracy_precision_recall]
    }

    eval_outdir = Path('scripts/geovlm/eval/')

    # Per dataset configurations
    if "lrben" in dataset_path.lower():
        dataset = "lrben"
        run_inference = run_ben_inference
        outdir = eval_outdir / "RSVQA-LRBEN/answers/"
        if open_prompt is not None:
            raise ValueError("LRBEN dataset does not support open prompt")
    elif "hrben" in dataset_path.lower():
        dataset = "hrben"
        run_inference = run_ben_inference
        outdir = eval_outdir / "RSVQA-HRBEN/answers/"
        if open_prompt is not None:
            raise ValueError("HRBEN dataset does not support open prompt")
    elif "fmow" in dataset_path.lower():
        dataset = "fmow"
        run_inference = run_aid_fmow_ucmerced_inference
        outdir = eval_outdir / "fmow-highres/answers/"
    elif "s2looking" in dataset_path.lower():
        dataset = "s2looking"
        run_inference = run_s2looking_inference
        outdir = eval_outdir / "s2looking/answers/"
    elif "xbd" in dataset_path.lower():
        dataset = "xbd"
        run_inference = run_xbd_inference
        outdir = eval_outdir / "xBD/answers/"
    elif 'qfabric' in dataset_path.lower() or 'geochat' in dataset_path.lower():
        dataset = "qfabric"
        run_inference = run_qfabric_inference
        outdir = eval_outdir / "QFabric/answers/"
    elif 'geochat' in dataset_path.lower():
        dataset = "geochat"
        run_inference = run_geochat_inference
        outdir = eval_outdir / "GeoChat/answers/"
    elif 'aid' in dataset_path.lower():
        dataset = "aid"
        run_inference = run_aid_fmow_ucmerced_inference
        outdir = eval_outdir / "AID/answers/"
    elif 'ucmerced' in dataset_path.lower():
        dataset = "ucmerced"
        run_inference = run_aid_fmow_ucmerced_inference
        outdir = eval_outdir / "UCMerced/answers/"
    elif 'cdvqa' in dataset_path.lower():
        dataset = "cdvqa"
        run_inference = run_cdvqa_inference
        outdir = eval_outdir / "CDVQA/answers/"
    else:
        raise ValueError(f"No supported dataset found in {dataset_path}, supported datasets: fmow, lrben, s2looking, xbd, qfabric, aic, ucmerced")
    
    if (start_ind is not None or end_ind is not None) and dataset not in ['qfabric', 'hrben', 'lrben']:
        raise ValueError("start_ind and end_ind can only be used with qfabric, hrben, or lrben datasets")

    # Determine the split
    if 'test' in dataset_path.lower():
        split = 'test'  
    elif 'val' or 'valid' or 'validation' in dataset_path.lower():
        split = 'val'
    elif 'train' in dataset_path.lower():
        split = 'train'
    else:
        print("Warning: Could not determine split from dataset path")

    args_to_determine_path = [
        'open_prompt',
        'repeat_frames',
        'prompt_strategy',
        'chronological_prefix',
        'load_8bit',
        'load_4bit',
        'data_frac',
        'data_size',
        'delete_system_prompt'
    ]
    
    # Setup answer path
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem

    if 'llava' not in model_name and 'llava' not in model_name.lower() and 'teochat' not in model_name.lower():
        if model_base != None:
            if model_path[-1] == "/":
                model_path = model_path[:-1]
            model_name = model_path.split("/")[-2] + "-" + model_path.split("/")[-1]
            print("Model name used: ", model_name)
        else:
            raise ValueError(f"Model name {model_name} does not contain 'llava'")
    if 'lora' not in model_name:
        print("Warning: Model name does not contain 'lora'")

    if outname is None:
        dataset_path_name = Path(dataset_path).stem
        outname = f"{model_name}_{dataset}_{dataset_path_name}_{split}.json"

    if ".json" not in outname:
        outname = f"{outname}.json"

    args_to_determine_path = [
        'open_prompt',
        'repeat_frames',
        'prompt_strategy',
        'chronological_prefix',
        'load_8bit',
        'load_4bit',
        'data_frac',
        'data_size',
        'delete_system_prompt',
        'start_ind',
        'end_ind',
        'last_image'
    ]
    for arg in args_to_determine_path:
        if args[arg] is not None:
            outname = outname.replace(".json", f"_{arg}_{args[arg]}.json")

    answer_path = outdir / outname

    print(f'answer_path: {answer_path}')

    # Save args to file
    args_path = outdir / outname.replace(".json", "_args.json")

    if len(str(args_path)) < 255:
        with open(args_path, 'w') as f:
            json.dump(args, f)
    else:
        # File name too long. Just use first letter of each arg
        for arg in args_to_determine_path:
            if args[arg] is not None:
                first_letters = ''.join([word[0] for word in arg.split('_')])
                #print("outname before replacing: ", outname)
                outname = outname.replace(f"{arg}", first_letters)
                #print("outname after replacing: ", outname)
        answer_path = outdir / outname
        args_path = outdir / outname.replace(".json", "_args.json")
        with open(args_path, 'w') as f:
            json.dump(args, f)
        print(f'New answer_path: {answer_path}')

    # If answer file exists, compute metrics 
    if answer_path.exists() and not rerun:
        for metric in dataset2metrics[dataset]:
            if dataset == "s2looking":
                metric(answer_path, dataset=dataset, verbose=verbose, split=split)
            else:
                metric(answer_path, dataset=dataset, verbose=verbose)
        return

    # Load model
    disable_torch_init()
    device = 'cuda'
    tokenizer, model, processor, use_video_data = load_model(
        model_path,
        model_base,
        cache_dir,
        device,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        vision_type=vision_type
    )

    if use_video_data:
        if dataset == "lrben":
            raise ValueError("LRBEN dataset does not support video processing")
        # Hack to set backend of video processor
        # NOTE: If we change image size, we might need to change this in the config here too
        # (better solution is to figure out where this config is set when saving the model)
        processor.config.vision_config.video_decode_backend = "image_list"
        processor = LanguageBindVideoProcessor(processor.config, tokenizer) 

    if rerun or not answer_path.exists():
        # Run inference
        answers = run_inference(
            model,
            dataset_path,
            processor,
            tokenizer,
            conv_mode,
            answer_path=answer_path,
            open_prompt=open_prompt,
            repeat_frames=repeat_frames,
            use_video_data = use_video_data,
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
            data_size=data_size,
            data_frac=data_frac,
            delete_system_prompt=delete_system_prompt,
            start_ind=start_ind,
            end_ind=end_ind,
            last_image=last_image,
            print_prompt=print_prompt
        )

        # Save answers
        with open(answer_path, 'w') as f:
            json.dump(answers, f, indent=4)
    else:
        answers = json.load(open(answer_path))


    # Calculate metrics
    for metric in dataset2metrics[dataset]:
        if dataset == "s2looking":
            metric(answer_path, dataset=dataset, verbose=verbose, split=split)
        else:
            metric(answer_path, dataset=dataset, verbose=verbose)


if __name__ == '__main__':
    """Example usage:
    export CUDA_VISIBLE_DEVICES=0; 
    export PYTHONPATH=/path/to/aicc-win24-geo-vlm/videollava/:$PYTHONPATH;
    python videollava/eval/video/infer_eval.py infer_eval\
        --dataset fmow\
        --model_path /path/to/model\
    """
    fire.Fire()
