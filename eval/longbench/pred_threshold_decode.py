import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from prism_attn import PrismDecodingQwen3ForCausalLM


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--token_budget", default=2048, type=int)
    parser.add_argument("--start_layer", default=0, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_method", default="prism",
                        choices=["prism", "indexcache", "dense", "fa2", "sdpa"], type=str)
    parser.add_argument("--use_fused_kernel", action="store_true")
    parser.add_argument("--use_batch_exist", action="store_true",
                        help="Use PrismDecoding's batch_exist_generate instead of HF .generate()")
    parser.add_argument("--max_tokens", default=4096, type=int,
                        help="Max new tokens to generate for each prompt")
    parser.add_argument('--output_path', type=str, default="./")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def extract_answer(response):
    if "</think>\n\n" in response:
        response = response.split("</think>\n\n")[1]
    return response

def get_pred(rank, world_size, data, max_gen, prompt_format, dataset, device, model_name, out_path, args, lock):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_name, device, args)
    max_length = model.config.max_position_embeddings - 500

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        prompt = build_chat(tokenizer, prompt, model_name)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # Prism/IndexCache use the custom batch_exist_generate; FA2/SDPA/Dense use HF's built-in .generate()
        if args.attention_method in ["prism", "indexcache"] and args.use_batch_exist:
            output = model.batch_exist_generate(
                input_ids=input.input_ids,
                attention_mask=input.attention_mask,
                max_length=context_length + max_gen,
                do_sample=False,
            )[0]
        elif dataset == "samsum":  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred_all = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred_all = post_process(pred_all, model_name)
        pred = extract_answer(pred_all)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "pred_all": pred_all, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], \
                           "length": json_obj["length"], "input_length": context_length, "output_length": len(output[context_length:]), "all_length": len(output)}, 
                           f, ensure_ascii=False)
                f.write('\n')
    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

    if args.attention_method in ["prism", "indexcache", "dense"]:
        if "qwen3" in path.lower():
            model_class = PrismDecodingQwen3ForCausalLM
        else:
            raise ValueError(f"model: {path} not supported in PrismDecoding")

        model = model_class.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_cache=True,
            token_budget=args.token_budget,
            attention_method=args.attention_method,
            use_flash_rope=args.use_fused_kernel,
            fused_norm=args.use_fused_kernel,
            start_layer=args.start_layer,
        )
    elif args.attention_method == "fa2":
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_cache=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    elif args.attention_method == "sdpa":
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_cache=True,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown attention_method: {args.attention_method}")

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    token_budget = args.token_budget

    if args.e:
        # datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
        #     "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", \
            "trec", "triviaqa", "passage_count", "passage_retrieval_en"]        
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    

    lock = mp.Lock()

    output_dir = os.path.join(args.output_path, "pred") if not args.e else os.path.join(args.output_path, "pred_e")
    model_name_for_path = model_name.split("/")[-1] + "_" + str(token_budget)

    for dataset in datasets:
        print(f"Predicting on {dataset}...")
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"{output_dir}/{model_name_for_path}"):
                os.makedirs(f"{output_dir}/{model_name_for_path}")
            out_path = f"{output_dir}/{model_name_for_path}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"{output_dir}/{model_name_for_path}"):
                os.makedirs(f"{output_dir}/{model_name_for_path}")
            out_path = f"{output_dir}/{model_name_for_path}/{dataset}.jsonl"
        
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping...")
            continue
        
        prompt_format = dataset2prompt[dataset]
        # max_gen = dataset2maxlen[dataset]
        max_gen = args.max_tokens
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        
        # rank = 0
        # get_pred(rank, world_size, data_subsets[rank], \
        #         max_gen, prompt_format, dataset, device, model_name, out_path, args, lock)

        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], \
                        max_gen, prompt_format, dataset, device, model_name, out_path, args, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
