# Downloaded Datasets

This directory contains datasets for the research project on mathematically grounded LLM evaluation. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: MMLU (Massive Multitask Language Understanding)

### Overview
- **Source**: HuggingFace `cais/mmlu` (config: "all")
- **Original**: https://github.com/hendrycks/test
- **Size**: ~15,908 questions across 57 subjects
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Multiple-choice knowledge evaluation (4 options)
- **Splits**: test, validation, dev, auxiliary_train
- **License**: MIT

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("cais/mmlu", "all")
dataset.save_to_disk("datasets/mmlu/data")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mmlu/data")
# Access: dataset['test'][0]
# Columns: question, subject, choices, answer
```

### Sample Data
See `mmlu/sample_10.json` for example records. Each record contains:
- `question`: The question text
- `subject`: Subject area (e.g., "abstract_algebra", "professional_law")
- `choices`: List of 4 answer options
- `answer`: Index of correct answer (0-3)

### Notes
- 57 subjects spanning STEM, humanities, social sciences, professional domains
- Key subjects for analysis: abstract_algebra, college_mathematics, machine_learning, professional_law
- Performance varies dramatically across subjects (ideal for topological analysis)

---

## Dataset 2: MT-Bench Human Judgments (Chatbot Arena Alternative)

### Overview
- **Source**: HuggingFace `lmsys/mt_bench_human_judgments`
- **Original requested**: `lmsys/chatbot_arena_conversations` (gated/restricted)
- **Size**: ~3K expert votes + GPT-4 judge evaluations
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Pairwise model comparison with human preferences
- **License**: CC-BY-4.0

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("lmsys/mt_bench_human_judgments")
dataset.save_to_disk("datasets/chatbot_arena/data")
```

**For full Chatbot Arena data (requires approval):**
```python
# Apply for access at: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
dataset = load_dataset("lmsys/chatbot_arena_conversations")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/chatbot_arena/data")
```

### Notes
- Contains both human expert votes and GPT-4 judge votes
- Pairwise comparisons suitable for Bradley-Terry analysis
- The full Chatbot Arena dataset (240K+ votes) requires HF access approval

---

## Dataset 3: HELM Benchmark Results

### Overview
- **Source**: https://crfm.stanford.edu/helm/latest/
- **Status**: Not available as a single downloadable dataset
- **Format**: Interactive leaderboard / API

### Access Methods
1. **Browse online**: https://crfm.stanford.edu/helm/latest/
2. **Run locally**: `pip install crfm-helm && helm-run`
3. **Raw results**: https://storage.googleapis.com/crfm-helm-public/
4. **API**: https://crfm.stanford.edu/helm/latest/api/

### Notes
- HELM evaluates models across 42 scenarios and 7 metric categories
- Results are published as interactive leaderboards
- For experiments, we can either run HELM locally or extract results from the public storage bucket
- See `helm/access_info.json` for details
