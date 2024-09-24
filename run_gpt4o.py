import os
import time
import json
from argparse import ArgumentParser

import cv2 
import base64
import requests
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--suffix", type=str, default="gpt4o_predictions")
    args = parser.parse_args()
    return args

def sample_frames(array, num_samples=8):
    length = len(array)
    
    if length <= num_samples:
        return array
    
    indices = np.linspace(0, length - 1, num_samples, dtype=int)
    sampled_array = [array[i] for i in indices]
    
    return sampled_array

def get_chat_gpt_response(prompt, base64Frames, api_key, max_retries=5, retry_delay=2):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system", 
                "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. You are chatting with the user via the ChatGPT iOS app. This means most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. Never use emojis, unless explicitly asked to. Knowledge cutoff: 2023-10 Current date: 2024-08-15. Image input capabilities: Enabled Personality: v2"
            }, 
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    *map(lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                        },
                        "resize": 768
                    }, sample_frames(base64Frames)),
                ]
            }
        ]
    }
    # response = requests.post(url, headers=headers, json=data)
    # return response.json()

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # 如果状态码不是200, 引发HTTPError
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # 在最后一次重试前等待
                time.sleep(retry_delay)
            else:
                return {"error": str(e)}

def process_description(video_key, base64Frames, api_key, prompt): 
    response = get_chat_gpt_response(prompt, base64Frames, api_key)
    
    if 'error' in response:
        print(f"video processing: {video_key} fail.")
        return None
    else:
        pred = response.get('choices', [{}])[0].get('message', {}).get('content', None)
        if pred is not None:
            print(f"video processing: {video_key} succeed.")
            return pred
        else:
            print(f"video processing: {video_key} fail.")
            return None


def load_video_base64(video_path):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64Frames



if __name__ == "__main__":
    args = parse_args()
    
    api_key = ""

    # Loading questions
    question_paths = {
        "entire": "./questions/entire_questions.json",
        "interleave": "./questions/interleave_questions.json",
        "misleading": "./questions/misleading_questions.json"
    }

    answer_prompt = "\nPlease answer yes or no:"
    predictions = {}
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pred_file = f"{args.output_path}/{args.suffix}.json"
    
    for split, question_filepath in question_paths.items():
        ### split: question category
        
        with open(question_filepath, 'r') as f:
            input_datas = json.load(f)

        predictions[split] = {}
        for video_info in input_datas:
            vid = video_info['id']
            
            if vid not in predictions:
                video_info_with_predictions = video_info.copy()
                video_info_with_predictions["qa"] = []

                video_path = os.path.join(args.video_path, split, f"{vid}.mp4")
                base64Frames = load_video_base64(video_path)  
                
                ### detailed description
                try:
                    # video_llm_pred = run_inference(args, tokenizer, model, image_processor, context_len, video_path, inp)
                    desc_inp = "Please describe this video in detail."
                    pred_description = process_description(vid, base64Frames, api_key, desc_inp)
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
                        pred_output = process_description(vid, base64Frames, api_key, inp)
                    except Exception as e:
                        print (f"Inference error: {video_path}, Error Detail: {e}")
                        pred_output = ''
                    video_info_with_predictions["qa"].append({'question': question['question'], 'answer': question['answer'], 'prediction': pred_output})
                
                predictions[split][vid] = video_info_with_predictions
                
                with open(pred_file, 'w') as f:
                    json.dump(predictions, f, indent=4)
            else:
                print (f"{vid} collapse.")