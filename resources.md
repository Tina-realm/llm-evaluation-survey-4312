# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Mathematically Grounded Evaluation of Large Language Models: A Survey and New Directions."

---

## Papers
Total papers downloaded: **36** (33 relevant, 3 wrong IDs from specification)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| MMLU | Hendrycks et al. | 2020 | papers/2009.03300.pdf | 57-task knowledge benchmark |
| Codex/HumanEval | Chen et al. | 2021 | papers/2107.03374.pdf | Code evaluation benchmark |
| HELM | Liang et al. | 2022 | papers/2211.09110.pdf | Multi-metric holistic evaluation |
| BIG-bench | Srivastava et al. | 2022 | papers/2206.04615.pdf | 204-task collaborative benchmark |
| GSM8K | Cobbe et al. | 2021 | papers/2110.14168.pdf | Math reasoning + verifiers |
| InstructGPT | Ouyang et al. | 2022 | papers/2203.02155.pdf | RLHF training paradigm |
| DPO | Rafailov et al. | 2023 | papers/2305.18290.pdf | Direct preference optimization |
| Chatbot Arena | Chiang et al. | 2024 | papers/2403.04132.pdf | Crowdsourced pairwise evaluation |
| LLM-as-Judge | Zheng et al. | 2023 | papers/2306.05685.pdf | MT-Bench, judge biases |
| Adversarial Attacks | Zou et al. | 2023 | papers/2307.15043.pdf | Universal adversarial suffixes |
| Jailbreaking | Lapid et al. | 2023 | papers/2309.01446.pdf | Black-box jailbreaking |
| Conformal Prediction | Stutz et al. | 2023 | papers/2307.09302.pdf | CP under ambiguous ground truth |
| Know What They Know | Kadavath et al. | 2022 | papers/2207.05221.pdf | LLM self-evaluation, calibration |
| Hallucination Survey | Huang et al. | 2023 | papers/2311.05232.pdf | Hallucination taxonomy |
| RAGAS | Es et al. | 2023 | papers/2309.15217.pdf | RAG evaluation framework |
| ARES | Saad-Falcon et al. | 2023 | papers/2311.09476.pdf | Automated RAG evaluation |
| Constitutional AI | Bai et al. | 2022 | papers/2212.08073.pdf | AI self-improvement for safety |
| Red Teaming | Casper et al. | 2023 | papers/2306.09442.pdf | Automated red teaming |
| ChatGPT Evaluation | Qin et al. | 2023 | papers/2302.06476.pdf | ChatGPT across NLP tasks |
| MIN-K% PROB | Shi et al. | 2023 | papers/2310.16789.pdf | Pretraining data detection |
| Contamination Rephrasing | Yang et al. | 2023 | papers/2311.04850.pdf | Benchmark contamination |
| Intro to TDA | Chazal & Michel | 2021 | papers/1710.04019.pdf | Persistent homology foundations |
| Intro to Info Geometry | Nielsen | 2020 | papers/1808.08271.pdf | Fisher metric, statistical manifolds |
| KG-to-Text LLMs | Wu et al. | 2023 | papers/2309.11206.pdf | Knowledge graph QA |

### Additional Papers (discovered through search)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Beyond Words: Math Framework for LLMs | - | 2023 | papers/2311.03033.pdf | Hex framework for LLM evaluation |
| Neural Persistence | - | 2018 | papers/1812.09764.pdf | Complexity measure via persistent homology (ICLR 2019) |
| Sheaf Neural Networks | - | 2020 | papers/2012.06333.pdf | Sheaf Laplacian for GNNs |
| Neural Sheaf Diffusion | - | 2022 | papers/2202.04579.pdf | Sheaf theory for GNN heterophily |
| Implicit Self-Regularization via RMT | - | 2018 | papers/1810.01075.pdf | Seminal RMT analysis of DNNs |
| Small Singular Values in Transformers | - | 2024 | papers/2410.17770.pdf | RMT spectral analysis of LLMs |
| Contamination Detection Survey | - | 2024 | papers/2404.00699.pdf | 50+ detection methods reviewed |
| Benchmark Data Contamination Survey | - | 2024 | papers/2406.04244.pdf | Comprehensive BDC survey |
| PH Captures Generalization | - | 2021 | papers/2106.00012.pdf | PH diagrams predict generalization |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: **2** (+ 1 documented)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MMLU | HuggingFace cais/mmlu | ~16K questions | Multi-choice knowledge | datasets/mmlu/ | Primary dataset |
| MT-Bench Judgments | HuggingFace lmsys | ~3K votes | Pairwise preference | datasets/chatbot_arena/ | Alternative to gated Arena data |
| HELM Results | Stanford CRFM | N/A | Multi-metric eval | datasets/helm/ | Access info documented |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: **6**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| MMLU Benchmark | github.com/hendrycks/test | Benchmark code + data | code/mmlu-benchmark/ | Official implementation |
| HELM | github.com/stanford-crfm/helm | Evaluation framework | code/helm/ | Multi-scenario evaluation |
| FastChat | github.com/lm-sys/FastChat | MT-Bench + Arena | code/fastchat/ | LLM-as-judge, BT ranking |
| LLM Decontaminator | github.com/lm-sys/llm-decontaminator | Contamination detection | code/llm-decontaminator/ | LLM-based decontamination |
| Giotto-TDA | github.com/giotto-ai/giotto-tda | Topological data analysis | code/giotto-tda/ | Persistent homology for ML |
| MIN-K% PROB | github.com/swj0419/detect-pretrain-code-contamination | Membership inference | code/min-k-prob/ | Reference-free MIA |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used arXiv API to fetch metadata for all 27 user-specified papers
2. Downloaded all PDFs from arXiv
3. Deep-read mathematical foundations papers (TDA, information geometry) using PDF chunker
4. Read first chunks of all evaluation, contamination, and calibration papers
5. Searched for relevant tools: Giotto-TDA for persistent homology, FastChat for Bradley-Terry ranking

### Selection Criteria
- Papers selected by research specification were prioritized
- Code repos chosen for direct relevance to the four mathematical pillars or baseline evaluation methods
- Datasets chosen per specification: MMLU, Chatbot Arena, HELM

### Challenges Encountered
- Three arXiv IDs in the specification pointed to unrelated papers (physics, robotics, Chinese MRC)
- Chatbot Arena full dataset is gated; used MT-Bench judgments as alternative
- HELM results not available as a single downloadable dataset; documented access methods
- Paper-finder service was not running; manual arXiv search used instead

### Gaps and Workarounds
- No existing implementation combining TDA with LLM evaluation found - this is truly novel research
- No sheaf theory library for ML evaluation; will need custom implementation
- Random matrix theory tools (e.g., RMTool, marchenko_pastur) available in NumPy/SciPy

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset: MMLU
- Use per-subject accuracy vectors as point clouds for persistent homology
- 57 subjects provide rich structure for topological analysis
- Historical results across model generations available for drift detection

### 2. Baseline Methods
- Standard accuracy metrics (MMLU per-subject)
- Bradley-Terry / Elo (from Chatbot Arena / FastChat)
- MIN-K% PROB for contamination detection baseline
- n-gram overlap for decontamination baseline

### 3. Evaluation Approach
- **Topology (persistent homology)**: Apply Giotto-TDA to MMLU performance vectors; compute persistence diagrams across model families; use bottleneck/Wasserstein distance for stability analysis
- **Information geometry (Fisher metric)**: Construct statistical manifold from Bradley-Terry model on Chatbot Arena data; compute Fisher-Rao distances between model versions; detect distributional drift
- **Spectral analysis (RMT)**: Analyze eigenvalue distributions of model comparison matrices; test against Marchenko-Pastur distribution for contamination signatures
- **Sheaf theory**: Formalize HELM's multi-metric evaluation as a sheaf on the category of evaluation scenarios; compute sheaf cohomology to detect inconsistencies

### 4. Code to Adapt/Reuse
- **Giotto-TDA**: Persistent homology computation, persistence diagrams, vectorization
- **FastChat/llm_judge**: Bradley-Terry model fitting, Elo ranking
- **LLM Decontaminator**: Baseline contamination detection
- **HELM**: Standardized evaluation pipeline for running scenarios
- **MIN-K% PROB**: Membership inference baseline
