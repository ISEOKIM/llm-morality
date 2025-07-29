# Moral Conditioning and Reasoning in Language Models

This repository contains the main experimental code used for analyzing how moral conditioning affects reasoning behavior in large language models. We evaluate models using two complementary settings:

1. **Defining Issues Test (DIT)** – to assess moral alignment  
2. **MMLU Benchmark** – to assess general problem-solving performance

The experiment is grounded in **Kohlberg’s theory of moral development**, with three moral stages:
- Pre-conventional
- Conventional
- Post-conventional

A morality-free baseline is also included for comparison.



## ⚙️ Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

```
llm-morality/
├── data/
│   └── test/                        # Public MMLU benchmark files
│                                    # (Probing_data.json is excluded)
├── prompts/
│   ├── dit_user_prompt_user_prompt.py     # User prompt template for DIT
│   ├── morality_prompt_system_prompt.py   # System prompt template for moral personas
│   ├── neutral_prompt_system_prompt.py    # System prompt for morality-free baseline
│   └── moral_stages.json                  # Moral stage definitions for all experiments
├── run_dit_llama.py                # DIT evaluation with LLaMA via vLLM
├── run_dit_gpt.py                  # DIT evaluation with OpenAI GPT models
├── run_mmlu_llama.py              # MMLU evaluation with LLaMA via vLLM
├── run_mmlu_gpt.py                # MMLU evaluation with OpenAI GPT models
├── requirements.txt
├── results_dit/                   # DIT evaluation outputs 
└── results_mmlu/                  # MMLU evaluation outputs 
```

**Note:** 
**Note:**

- **MMLU benchmark files (CSV)** are excluded from this repository due to licensing restrictions.  
  To run MMLU experiments, please download the official test files from the [MMLU GitHub repository](https://github.com/hendrycks/test) and place them under `data/test/`.

- The file `data/Probing_data.json` is also excluded for licensing reasons.  
  To run DIT experiments, you must either create this file manually or request access to the official dilemmas in the expected format.
