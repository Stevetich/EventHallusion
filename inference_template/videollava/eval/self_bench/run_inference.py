import math
import argparse
import json
import os

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import random
random.seed(42)

import numpy as np
np.random.seed(42)



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=1024)

    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--suffix", type=str, default="vllava_predictions")

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*16) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.cuda()
    
    # Loading questions
    question_paths = {
        "interleave": "../../../../questions/interleave_questions.json",
        "entire": "../../../../questions/entire_questions.json",
        "misleading": "../../../../questions/misleading_questions.json"
    }

    answer_prompt = "\nPlease answer yes or no:"
    predictions = {}
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pred_file = f"{args.output_path}/{args.suffix}.json"
    
    for split, question_filepath in question_paths.items():
        # split: question category
        
        with open(question_filepath, 'r') as f:
            input_datas = json.load(f)

        predictions[split] = {}
        for video_info in input_datas:
            vid = video_info['id']
            
            if vid not in predictions:
                video_info_with_predictions = video_info.copy()
                video_info_with_predictions["qa"] = []
                
                video_path = os.path.join(args.video_path, split, f"{vid}.mp4")
                
                ### detailed description
                try:
                    desc_inp = "Please describe this video in detail."
                    pred_description = get_model_output(model, processor['video'], tokenizer, video_path, desc_inp, args)
                except Exception as e: 
                    print (f"Inference error: {video_path}, Error Detail: {e}")
                    pred_description = ''
                video_info_with_predictions['desc'] = pred_description

                ### binary classification
                try:
                    a = video_info['questions']
                except:
                    print (f"No questions: {video_path}")
                
                for question in video_info['questions']:
                    inp = question['question'] + answer_prompt
                    try:
                        pred_output = get_model_output(model, processor['video'], tokenizer, video_path, inp, args)
                    except Exception as e:
                        print (f"Inference error: {video_path}, Error Detail: {e}")
                        pred_output = ''
                    video_info_with_predictions["qa"].append({'question': question['question'], 'answer': question['answer'], 'prediction': pred_output})
                
                predictions[split][vid] = video_info_with_predictions
                
                with open(pred_file, 'w') as f:
                    json.dump(predictions, f, indent=4)
            else:
                print (f"{vid} collapse.")
