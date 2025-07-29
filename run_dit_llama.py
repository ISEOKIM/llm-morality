import json
import os
import time
import random
import torch
import numpy as np
import argparse
from tqdm import tqdm
from importlib import import_module
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# === Set random seed ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Generate system prompt based on persona and moral level ===
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

    return system_prompt_module.SYSTEM_PROMPT_TEMPLATE.format(
        moral_level=moral_info["moral_level"],
        characteristic_1=moral_info["characteristics"][0],
        characteristic_2=moral_info["characteristics"][1],
        characteristic_3=moral_info["characteristics"][2]
    )

# === Generate user prompt by sampling statements and options ===
def generate_user_prompt(user_prompt_type: str, dilemma: str):
    user_prompt_module = import_module(f"prompts.{user_prompt_type}_user_prompt")
    with open("data/Probing_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    dilemma_info = data[dilemma]

    shuffled_statements = random.sample(dilemma_info['statements'], len(dilemma_info['statements']))
    shuffled_options = random.sample(dilemma_info['options'], len(dilemma_info['options']))

    return user_prompt_module.USER_PROMPT_TEMPLATE.format(
        story=dilemma_info["story"],
        statements_1=shuffled_statements[0], statements_2=shuffled_statements[1],
        statements_3=shuffled_statements[2], statements_4=shuffled_statements[3],
        statements_5=shuffled_statements[4], statements_6=shuffled_statements[5],
        statements_7=shuffled_statements[6], statements_8=shuffled_statements[7],
        statements_9=shuffled_statements[8], statements_10=shuffled_statements[9],
        statements_11=shuffled_statements[10], statements_12=shuffled_statements[11],
        Question1=dilemma_info['Question1'],
        options_1=shuffled_options[0], options_2=shuffled_options[1], options_3=shuffled_options[2]
    )


# === Combine system, user, and assistant prompt segments ===
def generate_prompt(system_prompt_type, user_prompt_type, persona, dilemma, moral_stages, morality):
    system_prompt_text = generate_system_prompt(system_prompt_type, persona, moral_stages, morality)
    user_prompt = generate_user_prompt(user_prompt_type, dilemma)
    assistant_prompt = "<|start_header_id|>assistant<|end_header_id|>"

    if system_prompt_text:
        return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt_text}<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n\n{assistant_prompt}"
    else:
        return f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n\n{assistant_prompt}"

# === Load VLLM model and tokenizer ===
def load_model_vllm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    return model, tokenizer

# === VLLM inference ===
def infer_vllm(model, prompt, max_tokens=1024):
    sampling_params = SamplingParams(temperature=0.6, top_p=1.0, max_tokens=max_tokens)
    outputs = model.generate([prompt], sampling_params)
    return [outputs[0].outputs[0].text]


# === Main experiment loop ===
def main(args):
    with open(args.moral_file_path, 'r', encoding='utf-8') as f:
        moral_stages = json.load(f)

    moral_source = Path(args.moral_file_path).stem.replace("_moral_stages", "")
    morality = args.morality
    model, tokenizer = load_model_vllm(args.model_path)

    for seed in range(1, 9):
        set_seed(seed)
        for dilemma in tqdm(args.dilemmas, desc="Dilemmas"):
            for system_prompt_type in tqdm(args.system_prompt_type_list, desc="Prompt Types", leave=False):
                for persona in tqdm(args.persona_list, desc="Personas", leave=False):
                    for i in range(8):
                        start_time = time.time()
                        user_prompt_type = random.choice(args.user_prompt_type_list)
                        prompt = generate_prompt(system_prompt_type, user_prompt_type, persona, dilemma, moral_stages, morality)
                        answers = infer_vllm(model, prompt)

                        output_dir = Path(f"results_dit/vllm_results/{args.model_type}/{moral_source}/{dilemma}/{system_prompt_type}/")
                        output_dir.mkdir(parents=True, exist_ok=True)

                        run_results = {
                            'dilemma': dilemma,
                            'system_prompt_type': system_prompt_type,
                            'user_prompt_type': user_prompt_type,
                            'persona': persona,
                            'seed': seed,
                            'iteration': i + 1,
                            'answers': answers,
                            'elapsed_time': time.time() - start_time
                        }

                        output_file = output_dir / f"result_{persona}_seed{seed}_iter{i+1}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(run_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--dilemmas', nargs='+', default=["Rajesh", "Monica", "Auroria", "Timmy", "Heinz", "Newspaper", "Student", "Webster", "Prisoner"])
    parser.add_argument('--system_prompt_type_list', nargs='+', default=["morality_prompt"])
    parser.add_argument('--user_prompt_type_list', nargs='+', default=["dit_user_prompt"])
    parser.add_argument('--persona_list', nargs='+', default=["Pre-conventional level human", "Conventional level human", "Post-conventional level human"])
    parser.add_argument('--moral_file_path', type=str, required=True)
    parser.add_argument('--morality', type=json.loads, required=True)

    args = parser.parse_args()
    main(args)
