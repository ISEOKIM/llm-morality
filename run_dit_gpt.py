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
from importlib import import_module
import argparse
import re
import datetime
import sys
from openai import OpenAI




choices = ["A", "B", "C", "D"]


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Extract the final answer from the model’s output
def extract_answer(prediction: str):
    if not prediction:
        return None
    patterns = list(re.finditer(r'answer is\s*(.*)', prediction, re.IGNORECASE))
    if not patterns:
        match = re.findall(r'\(([ABCDabcd])\)', prediction, re.IGNORECASE)
        unique_match = set(match)
        return unique_match.pop().upper() if len(unique_match) == 1 else None
    pattern = patterns[-1]
    unique_answers = re.findall(r'\(\s*[ABCDabcd]\s*\)|[ABCD]\.', pattern.group(1))
    unique_answers = set(unique_answers)
    if len(unique_answers) != 1:
        return None
    answer = unique_answers.pop()
    if '.' in answer:
        answer = re.search(r'[ABCDabcd]', answer).group(0)
    return answer.strip('()').upper()

# Normalize text for evaluation
def normalize(text):
    return "" if text is None else text.replace("(", "").replace(")", "").upper().strip()

# Compare predicted and gold answers
def check_equal(instance) -> bool:
    return normalize(instance['answer']) == normalize(instance['predicted_answers'])

# Compute accuracy and extraction failure stats
def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)

    task_name, task_results = next(iter(run_results.items()))
    pred_answers = [r['predicted_answers'] for r in task_results]
    gold_answers = [r['answer'] for r in task_results]

    acc = sum(p == g for p, g in zip(pred_answers, gold_answers))
    extraction_failures = sum(1 for p in pred_answers if not p)
    total = len(gold_answers)

    failures = [
        {
            'index': r['index'],
            'question': r['question'],
            'predicted_explanations': r['predicted_explanations']
        }
        for r in task_results if not r['predicted_answers']
    ]

    with open(output_filename.replace(".json", "_failures.json"), 'w') as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    metrics = {
        "accuracy": acc / total,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / total
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Extraction Failures: {metrics['extraction_failures']}")
    print(f"Extraction Failure Rate: {metrics['extraction_failure_rate']:.4f}")

    with open(output_filename.replace(".json", "_metrics.json"), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# Convert a row to prompt format
def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

# Load moral stages
def load_moral_stages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        moral_stages = json.load(file)
    return moral_stages


# Generate system prompt template
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

# Generate prompt pairs (system, user)
def generate_prompt(prompt_type, persona, questions, moral_stages, morality):
    prompts = []
    for question in questions:
        if prompt_type != "no_persona":
            system_prompt = generate_system_prompt(prompt_type, persona, moral_stages, morality)
        else:
            system_prompt = ""
        user_prompt = (
            "Answer the following question. The final answer derived from a series of reasoning processes "
            "must always be output in the last sentence as 'Therefore, the answer is ...'.\n\nQ: " + question
        )
        prompts.append((system_prompt, user_prompt))
    return prompts

# Split into batches
def batch_split(prompts, batch_size):
    return [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

# ========= OpenAI inference ==========
def batch_infer_openai(prompts, batch_size, max_tokens, model_name, client):
    answers = []
    for batch in tqdm(batch_split(prompts, batch_size)):
        for system_prompt, user_prompt in batch:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=1
                )
                full_prediction = response.choices[0].message.content
                answers.append(full_prediction)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                answers.append("")
    return answers




# Main execution
def main(args):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    set_seed(args.seed)

    moral_stages = load_moral_stages(args.moral_file_path)
    safe_persona = args.persona.replace("/", "_").replace(" ", "_")
    output_dir = Path("results") / args.model_type / args.task / args.prompt_type / safe_persona
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_dir / f"seed{args.seed}_{safe_persona}_predictions.json"
    test_df = pd.read_csv(Path(args.data_dir) / "test" / f"{args.task}_test.csv", header=None)

    questions = [format_example(test_df, i, include_answer=False) for i in range(test_df.shape[0])]
    prompts = generate_prompt(args.prompt_type, args.persona, questions, moral_stages, args.morality)

    print("Example Prompt:")
    print(prompts[0])

    pred_explanations = batch_infer_openai(prompts, args.batch_size, max_tokens=1024, model_name=args.model_type, client=client)

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

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run moral persona-based evaluation using OpenAI models.")
    parser.add_argument('--model_type', type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument('--task', type=str, default='abstract_algebra')
    parser.add_argument('--prompt_type', type=str, default="morality_prompt")
    parser.add_argument('--persona', type=str, default="Post-conventional morality level human")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--moral_file_path', type=str, default='./prompts/moral_stages.json')
    parser.add_argument('--morality', type=json.loads, default='{"Pre-conventional morality level human": "Pre-conventional", "Conventional morality level human": "Conventional", "Post-conventional morality level human": "Post-conventional"}')

    args = parser.parse_args()
    main(args)