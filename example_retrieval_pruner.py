# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import re
import os
import sys
import torch
import time
import json
import argparse
from utils import *
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from transformers import BertTokenizer, BertModel
from openicl import DatasetReader, TopkRetriever
from datasets import load_dataset, Dataset, DatasetDict 
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from llama_model.modeling_skim_predictor import SkimPredictor, C2FPromptPruner_PolicyNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument("--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment")
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    parser.add_argument("--max_num_worker", type=int, default=16, help="maximum number of workers for dataloader")
    parser.add_argument("--method", type=str, default="few_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method")
    parser.add_argument("--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought")
    parser.add_argument("--cot_shot_length", type=int, default=8, help="length of shots of cot for few-shot ICL settings")
    parser.add_argument("--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction")
    parser.add_argument("--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction")
    parser.add_argument("--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.")
    parser.add_argument("--api_time_interval", type=float, default=1.0, help="")
    parser.add_argument("--select_from_train", action='store_true', default=False)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--pruner_model', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--topk', action='store_true', default=True)
    parser.add_argument('--add_8shot', action='store_true', default=False)
    parser.add_argument('--add_16shot', action='store_true', default=False)
    parser.add_argument('--candidate_set', type=str)
    parser.add_argument("--continue_index", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')

    load_8bit = args.load_8bit
    tokenizer = AutoTokenizer.from_pretrained(base_model) 
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) 

    return tokenizer, model

def evaluate(tokenizer, model, input=None, temperature=0.8, top_p=0.95, top_k=40, num_beams=1, max_new_tokens=256, **kwargs):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    generation_config = GenerationConfig(
        do_sample=False, # no sampling will give best reasoning performance
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    fix_seed(args.random_seed) 
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    train_data = []
    test_data = []

    f = open(args.candidate_set)
    cot_data = json.load(f) 
    f.close()
    for i in tqdm(range(len(cot_data))):
        question = cot_data[i]['instruction'].replace('\n','')
        output = cot_data[i]['output'].replace('\n','')
        train_data.append({'question':question, 'answer':output})

    if args.dataset == "gsm8k":
        decoder = json.JSONDecoder()
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                question = json_res["question"].strip()
                answer = json_res["answer"].split("#### ")[0]
                test_data.append({'question':question, 'answer':answer})

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                test_data.append({'question':q, 'answer':a})

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                test_data.append({'question':q, 'answer':a})
    
    else:
        raise NotImplementedError("{} dataset is not supported".format(args.dataset))

    tokenizer, model = load_model(args)
    print("Finish Loading Models")
 
    test_subset = Dataset.from_list(test_data) 
    train_subset = Dataset.from_list(train_data) 
    dataset = DatasetDict({"train": train_subset,"test": test_subset})

    data = DatasetReader(dataset, input_columns=['question'], output_column='answer')
    retriever = TopkRetriever(data, ice_num=args.cot_shot_length)
    topk_prompt = retriever.retrieve()

    del retriever, data
    torch.cuda.empty_cache()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_tokenizer.add_tokens(["\n"])
    bert_model = BertModel.from_pretrained('bert-large-cased', output_hidden_states = True)
    bert_model.resize_token_embeddings(len(bert_tokenizer))
    bert_model.eval()

    pruner_model = C2FPromptPruner_PolicyNetwork(alpha1=-1, alpha2=1, target_token=1024, feat_shape=1024)
    pruner_model.load_state_dict(torch.load(args.pruner_model))
    pruner_model.eval()

    if args.add_8shot:
        demo_addshot = create_demo_text(args, True, 8)
    elif args.add_16shot:
        demo_addshot = create_demo_text(args, True, 16)

    total = 0
    correct_list = []        
    for i, data in enumerate(dataloader):
        
        if (i+1) < args.continue_index:
            continue

        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()

        if args.method == "few_shot_cot":
            demo_text = retrieve_demo_text_list(i, dataset, topk_prompt)
            demo = compress_prompt(demo_text, pruner_model, bert_tokenizer, bert_model, 8, 1024)
        else:
            raise NotImplementedError("Retrieval-based ICL only support few-shot-cot")

        demo = demo + "\n\n" + demo_addshot
        #demo = demo_addshot + demo
        x = demo + x

        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z =  evaluate(tokenizer, model, x)
        z = z.split(x)[-1]

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = evaluate(tokenizer, model, z2)
            pred = pred.split(z2)[-1]
            print(z2 + pred)
        else:
            pred = z
            print(x + pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        # Checking answer ...
        pred = pred.replace(',','').replace('\n', '')
        y = y.replace(',','').replace('\n', '')
        if is_number(pred):
            correct = int(float(pred) == float(y))
        else:
            correct = 0
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

def generate_pruned_prompt(full_prompt, shot_selection, token_selection, tokenizer):

    tokenized_text = tokenizer(full_prompt, padding="max_length", truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_text['input_ids']])
    input_ids = torch.squeeze(input_ids, 0)

    pruned_input_ids = input_ids[shot_selection,:]
    pruned_input_ids_flatten = torch.flatten(pruned_input_ids)
    pruned_input_ids_final = pruned_input_ids_flatten[token_selection]
    return pruned_input_ids_final

def compress_prompt(input_prompt, pruner_model, bert_tokenizer, bert_model, target_shot, target_token):

    demo_text = input_prompt
    tokenized_text = bert_tokenizer(demo_text, padding="max_length", truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_text['input_ids']])
    input_ids = torch.squeeze(input_ids, 0)
    with torch.no_grad():
        last_hidden_states = bert_model(input_ids)[0] # Models outputs are now tuples

    _, shot_selection, token_selection, _, _ = pruner_model(last_hidden_states)

    pruned_few_shot_input_id = generate_pruned_prompt(input_prompt, shot_selection, token_selection, bert_tokenizer)
    pruned_prompt_text = bert_tokenizer.decode(pruned_few_shot_input_id, skip_special_tokens=True)

    # format cleanning for results
    pruned_prompt_text = clean_response(pruned_prompt_text)

    return pruned_prompt_text[2:]

  
if __name__ == "__main__":
    main()
