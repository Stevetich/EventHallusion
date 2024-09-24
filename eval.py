import json
import os
import argparse

from constants import *

def extract_pred(video_llm_pred):
    if video_llm_pred is None:
        return None
    
    video_llm_pred = video_llm_pred.lower()
    if video_llm_pred.startswith("yes"):
        return "Yes."
    elif video_llm_pred.startswith("no"):
        return "No."
    else:
        return None

def extract_yes_no_gpt(response_text):
    # Remove "**" for bold in gpt's judgement
    response_text = response_text.replace("*", "").lower()
    
    if response_text.startswith("yes"):
        return "yes"
    elif response_text.startswith("no"):
        return "no"
    else:
        return None

def main(predictions):
    total_ques = 0
    total_ques_correct = 0

    total_videos = 0
    total_desc_correct = 0
        
    ques_not_match = 0
    gpt_not_match = 0
    for split, pred_dict in predictions.items():
        ques_cnt = 0
        ques_correct = 0

        video_cnt = 0
        desc_correct = 0

        for video_key, video_info_with_qa in pred_dict.items():
            video_cnt += 1
            for qa in video_info_with_qa['qa']:
                ques_cnt += 1
                gt_answer = qa['answer']
                pred = extract_pred(qa['prediction'])
                
                if pred is None:
                    ques_not_match += 1
                if gt_answer == pred:
                    ques_correct += 1
            
            gpt_judge = video_info_with_qa['judgement']
            gpt_judge_processed = extract_yes_no_gpt(gpt_judge)

            if gpt_judge_processed is None:
                gpt_not_match += 1
                
            if gpt_judge_processed == "yes":
                desc_correct += 1

        total_ques += ques_cnt
        total_ques_correct += ques_correct
        total_videos += video_cnt
        total_desc_correct += desc_correct


        print (f"{split} ques: {ques_cnt}, qs correct: {ques_correct}, qs acc: {ques_correct / ques_cnt}")
        print (f"{split} videos: {video_cnt}, desc correct: {desc_correct}, desc acc: {desc_correct / video_cnt}")
    print (f"overall: ques: {total_ques}, qs correct: {total_ques_correct}, qs acc: {total_ques_correct / total_ques}")
    print (f"overall: desc: {total_videos}, desc correct: {total_desc_correct}, desc acc: {total_desc_correct / total_videos}")
    print (f"Not matching rate: ques: {ques_not_match / total_ques}, desc: {gpt_not_match / total_videos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    args = parser.parse_args()

    input_file = args.input_file
    with open(input_file, 'r') as f:
        predictions = json.load(f)

    main(predictions)