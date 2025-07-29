import json
import os
import time 
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import accelerate
from importlib import import_module
import argparse
import re
import datetime
import tiktoken # type: ignore
from vllm import LLM, SamplingParams # type: ignore
import torch.multiprocessing as mp # type: ignore
import sys

# Multiple-choice answer options
choices = ["A", "B", "C", "D"]

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





# Extract the predicted answer from model output
def extract_answer(prediction: str):
    if (prediction is None) or (prediction == ''):
        return None

    patterns = list(re.finditer(r'answer is\s*(.*)', prediction, re.IGNORECASE))

    if not patterns:
        match = re.findall(r'\(([ABCDabcd])\)', prediction, re.IGNORECASE)
        unique_match = set(match)
        if len(unique_match) == 1:
            return unique_match.pop().upper()
        else:
            return None
    else:
        pattern = patterns[-1]
        unique_answers = re.findall(r'\(\s*[ABCDabcd]\s*\)|[ABCD]\.', pattern.group(1))
        unique_answers = set(unique_answers)

        if len(unique_answers) != 1:
            return None

        answer = unique_answers.pop()
        if '.' in answer:
            answer = re.search(r'[ABCDabcd]', answer).group(0)
        return answer.strip('()').upper()

# Normalize answers for comparison
def normalize(text):
    if text is None:
    	return ""
    return text.replace("(", "").replace(")", "").upper().strip()


# Compare prediction with ground truth
def check_equal(instance) -> bool:
    gt = normalize(instance['answer'])
    pred = normalize(instance['predicted_answers'])

    if gt == pred:
        return True
    else:
        return False

# Compute accuracy and extraction failure statistics
def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    extraction_failures = 0
    failures = []

    task_name, task_results = next(iter(run_results.items()))
    pred_answers = [record['predicted_answers'] for record in task_results]
    gold_answers = [record['answer'] for record in task_results]

    acc = sum(1 for pred, gold in zip(pred_answers, gold_answers) if pred == gold)
    extraction_failures = sum(1 for pred in pred_answers if not pred)
    total_acc += acc
    total_num += len(gold_answers)

    extraction_failure_rate = extraction_failures / total_num

    for record in task_results:
        if not record['predicted_answers']:
            failures.append({
                'index': record['index'],
                'question': record['question'],
                'predicted_explanations': record['predicted_explanations']
            })

    output_failure_filename = f"{output_filename[:-6]}_failures.json"  
    with open(output_failure_filename, 'w') as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    # Print and save metrics to JSON file
    metrics = {
        "accuracy": acc / len(gold_answers),
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failure_rate
    }

    print(f"Accuracy-{task_name}: {metrics['accuracy']:.4f}")
    print(f"Extraction Failures-{task_name}: {metrics['extraction_failures']}")
    print(f"Extraction Failure Rate-{task_name}: {metrics['extraction_failure_rate']:.4f}")

    metrics_output_filename = f"{output_filename[:-6]}_metrics.json"
    with open(metrics_output_filename, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)



# Format one row from MMLU test CSV into prompt form
def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1]) 
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

# Load JSON file describing moral stage characteristics
def load_moral_stages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        moral_stages = json.load(file)
    return moral_stages

# Generate system prompt based on persona type and moral stage
def generate_system_prompt(prompt_type: str, persona: str, moral_stages, morality):
    if prompt_type == "no_persona":
        return ""

    moral_key = morality.get(persona)
    if moral_key == "None":
        module_name = "prompts.neutral_prompt_system_prompt"
    else:
        module_name = "prompts.morality_prompt_system_prompt"

    system_prompt_module = import_module(module_name)

    moral_info = moral_stages[moral_key]

    system_prompt = system_prompt_module.SYSTEM_PROMPT_TEMPLATE.format(
        moral_level=moral_info["moral_level"],
        characteristic_1=moral_info["characteristics"][0],
        characteristic_2=moral_info["characteristics"][1],
        characteristic_3=moral_info["characteristics"][2]
    )

    return system_prompt

# Construct full prompt string (system + user + assistant) per question

def generate_prompt(prompt_type, persona, questions, moral_stages, morality): 
    prompts = []
    for question in questions:
        if prompt_type != "no_persona":
            system_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{generate_system_prompt(prompt_type, persona, moral_stages, morality)}<|eot_id|>"
        else: 
            system_prompt = ""
        
        user_prompt = """<|start_header_id|>user<|end_header_id|>\n\nAnswer the following question. The final answer derived from a series of reasoning processes must always be output in the last sentence as "Therefore, the answer is ...".\n\n""" + f"Q: {question}\nA:<|eot_id|>" 
        assistant_prompt = "<|start_header_id|>assistant<|end_header_id|>"
        prompt = f"{system_prompt}\n\n{user_prompt}\n\n{assistant_prompt}"
        prompts.append(prompt)
    return prompts

# Tokenize inputs for LLM inference
def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

# Load vLLM model and tokenizer
def load(ckpt_dir, model_type, seed):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=True,
        padding_side="left",
        # , use_auth_token="your_token_here" 
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id} ")

    model = LLM(
        model = ckpt_dir,  
        tensor_parallel_size=1,
        trust_remote_code=True)

    return model, tokenizer

# Split prompts into mini-batches
def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

# Perform batched inference with vLLM
def batch_infer(model, tokenizer, prompts, batch_size, max_tokens):
    sampling_params = SamplingParams(temperature=0.1, top_p=1, max_tokens=max_tokens)
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        prompts = [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in encode_inputs['input_ids']]
        outputs = model.generate(prompts, sampling_params)
        
        for output in outputs:
            full_prediction = output.outputs[0].text
            answers.append(full_prediction)
    return answers

# Main execution function
def main(args):
    set_seed(args.seed)

    moral_stages = load_moral_stages(args.moral_file_path)
    model, tokenizer = load(args.ckpt_dir, args.seed)

    safe_persona = args.persona.replace("/", "_").replace(" ", "_")
    output_dir = Path("results") / args.model_type / args.task / args.prompt_type / safe_persona
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running task: {args.task}, prompt type: {args.prompt_type}, persona: {args.persona}, seed: {args.seed}")

    output_filename = output_dir / f"seed{args.seed}_{safe_persona}_predictions.json"
    test_df = pd.read_csv(Path(args.data_dir) / "test" / f"{args.task}_test.csv", header=None)

    questions = [format_example(test_df, i, include_answer=False) for i in range(test_df.shape[0])]
    prompts = generate_prompt(args.prompt_type, args.persona, questions, moral_stages, args.morality)

    if prompts:
        print("First Prompt:\n")
        print(prompts[0])

    pred_explanations = batch_infer(model, tokenizer, prompts, args.batch_size, max_tokens=1024)

    records = []
    for i, (pred, question) in enumerate(zip(pred_explanations, questions)):
        label = test_df.iloc[i, test_df.shape[1]-1]
        predicted_answer = extract_answer(pred)
        records.append({
            'index': i,
            'question': question,
            'answer': label,
            'predicted_explanations': pred,
            'predicted_answers': predicted_answer,
            'is_correct': check_equal({'answer': label, 'predicted_answers': predicted_answer}),
            'prompt': prompts[i],
        })

    with open(output_filename, 'w') as f:
        json.dump({args.task: records}, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU evaluation with moral persona prompts.")
    parser.add_argument('--ckpt_dir', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--task', type=str, default='abstract_algebra')
    parser.add_argument('--prompt_type', type=str, default="morality_prompt")
    parser.add_argument('--persona', type=str, default="Pre-conventional morality level human")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--moral_file_path', type=str, default='./prompts/moral_stages.json')
    parser.add_argument('--morality', type=json.loads, default='{"Pre-conventional morality level human": "Pre-conventional", "Conventional morality level human": "Conventional", "Post-conventional morality level human": "Post-conventional"}')

    args = parser.parse_args()
    main(args)