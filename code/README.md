# Cloned Repositories

## Repo 1: MMLU Benchmark
- **URL**: https://github.com/hendrycks/test
- **Purpose**: Official MMLU benchmark code and test data
- **Location**: `code/mmlu-benchmark/`
- **Key files**: `evaluate_fewshot.py`, test data CSVs in `data/`
- **Notes**: Contains all 57 subject test/val/dev splits as CSVs. Can be used directly for evaluation.

## Repo 2: HELM (Holistic Evaluation of Language Models)
- **URL**: https://github.com/stanford-crfm/helm
- **Purpose**: Comprehensive LLM evaluation framework with 42 scenarios and 7 metrics
- **Location**: `code/helm/`
- **Key files**: `src/helm/benchmark/`, `src/helm/proxy/`
- **Notes**: Full evaluation pipeline. Install with `pip install crfm-helm`. Supports running standardized evaluations across models.

## Repo 3: FastChat (MT-Bench, Chatbot Arena)
- **URL**: https://github.com/lm-sys/FastChat
- **Purpose**: LLM-as-Judge implementation, MT-Bench questions, Chatbot Arena infrastructure
- **Location**: `code/fastchat/`
- **Key files**: `fastchat/llm_judge/` (MT-Bench), `fastchat/serve/` (Arena)
- **Notes**: Contains 80 MT-bench questions, evaluation scripts, and Bradley-Terry ranking code.

## Repo 4: LLM Decontaminator
- **URL**: https://github.com/lm-sys/llm-decontaminator
- **Purpose**: LLM-based decontamination tool detecting rephrased benchmark contamination
- **Location**: `code/llm-decontaminator/`
- **Key files**: Decontamination scripts and evaluation tools
- **Notes**: Implements embedding similarity + LLM judge pipeline for detecting contamination beyond n-gram overlap.

## Repo 5: Giotto-TDA
- **URL**: https://github.com/giotto-ai/giotto-tda
- **Purpose**: Python library for topological data analysis integrated with scikit-learn
- **Location**: `code/giotto-tda/`
- **Key files**: `gtda/homology/`, `gtda/diagrams/`, `gtda/plotting/`
- **Notes**: Provides persistent homology computation (VietorisRipsPersistence), persistence diagrams, landscapes, and vectorization. Key tool for our topological evaluation framework. Install: `pip install giotto-tda`.

## Repo 6: MIN-K% PROB (Detect Pretrain Code Contamination)
- **URL**: https://github.com/swj0419/detect-pretrain-code-contamination
- **Purpose**: Implementation of MIN-K% PROB method for detecting pretraining data
- **Location**: `code/min-k-prob/`
- **Key files**: Detection scripts for membership inference
- **Notes**: Reference-free MIA method. Can detect copyrighted books, benchmark contamination, and audit machine unlearning.
