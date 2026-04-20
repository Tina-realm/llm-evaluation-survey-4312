# Literature Review: Mathematically Grounded Evaluation of Large Language Models

## Research Area Overview

This literature review surveys the intersection of LLM evaluation methodology and mathematical frameworks (topology, information geometry, spectral analysis) that could provide formal guarantees for evaluation. Current LLM evaluation relies on benchmarks with ad-hoc metrics that lack composability, formal stability guarantees, and robustness to contamination. We identify four mathematical pillars that could address these gaps: persistent homology (for multi-scale structural analysis), sheaf theory (for composing local evaluations globally), information geometry/Fisher metric (for detecting distributional drift), and random matrix theory (for contamination detection via spectral signatures).

---

## Key Papers

### A. LLM Evaluation Benchmarks

#### 1. MMLU - Measuring Massive Multitask Language Understanding
- **Authors**: Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt (2020)
- **Source**: ICLR 2021 (arXiv:2009.03300)
- **Key Contribution**: 57-task benchmark covering STEM, humanities, social sciences across elementary to professional difficulty. 15,908 multiple-choice questions.
- **Methodology**: Zero-shot and few-shot evaluation. Questions from practice exams (GRE, USMLE, AP exams).
- **Key Results**: GPT-3 175B achieved 43.9% (expert-level ~89.8%). Performance is lopsided - strong on some subjects, near-random on others. Models poorly calibrated (confidence 24% off from accuracy).
- **Limitations**: Static benchmark susceptible to contamination. Multiple-choice format doesn't capture open-ended abilities.
- **Relevance**: Primary benchmark dataset for our experiments. The lopsided performance pattern is ideal for topological analysis of capability structure.

#### 2. HELM - Holistic Evaluation of Language Models
- **Authors**: Liang, Bommasani, Lee et al. (2022)
- **Source**: arXiv:2211.09110
- **Key Contribution**: Comprehensive evaluation framework covering 42 scenarios, 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) across 30+ models.
- **Methodology**: Standardized evaluation pipeline. Multi-metric approach measuring beyond accuracy.
- **Key Results**: No single model dominates across all scenarios/metrics. Strong models in accuracy may be weak in fairness or robustness.
- **Relevance**: The multi-metric structure of HELM is a natural fit for sheaf-theoretic composition - local evaluations on individual scenarios can be composed into global assessments with formal consistency guarantees.

#### 3. BIG-bench - Beyond the Imitation Game
- **Authors**: Srivastava, Rastogi, Rao et al. (2022)
- **Source**: arXiv:2206.04615
- **Key Contribution**: 204 tasks contributed by 450+ researchers. Tasks designed to probe capabilities beyond current benchmarks.
- **Methodology**: Collaborative benchmark with diverse task types.
- **Key Results**: Models show emergent abilities at scale on some tasks. Performance often follows sharp transitions rather than smooth scaling.
- **Relevance**: Emergent abilities create discontinuities in the capability landscape - persistent homology could detect these topological phase transitions.

#### 4. Codex / HumanEval
- **Authors**: Chen, Tworek, Jun et al. (2021)
- **Source**: arXiv:2107.03374
- **Key Contribution**: HumanEval benchmark (164 programming problems) measuring functional correctness via pass@k metric.
- **Relevance**: Code evaluation provides a well-defined ground truth, making it suitable for formal analysis.

#### 5. GSM8K
- **Authors**: Cobbe, Kosaraju, Bavarian et al. (2021)
- **Source**: arXiv:2110.14168
- **Key Contribution**: 8.5K grade school math word problems. Introduced verifier-based approach (training outcome-based reward models).
- **Relevance**: Math reasoning benchmark with clear correctness criteria.

### B. Human Preference and Alignment Evaluation

#### 6. Chatbot Arena
- **Authors**: Chiang, Zheng, Sheng, Angelopoulos et al. (2024)
- **Source**: arXiv:2403.04132
- **Key Contribution**: Open crowdsourced LLM evaluation platform with 240K+ pairwise votes. Uses Bradley-Terry model and Elo ratings for ranking.
- **Methodology**: Anonymous pairwise comparisons with real users. Adaptive sampling for efficient ranking convergence. E-values from Vovk & Wang (2021) for statistical validity.
- **Key Results**: Crowdsourced votes agree well with expert evaluations. Live platform avoids contamination issues of static benchmarks.
- **Relevance**: The Bradley-Terry model for pairwise comparisons has deep connections to information geometry (it defines an exponential family manifold). The Fisher information metric on this manifold could detect when rankings are unstable or when new models create structural shifts.

#### 7. LLM-as-a-Judge / MT-Bench
- **Authors**: Zheng, Chiang, Sheng et al. (2023)
- **Source**: NeurIPS 2023 (arXiv:2306.05685)
- **Key Contribution**: Systematic study of using GPT-4 as evaluator. MT-Bench: 80 multi-turn questions across 8 categories.
- **Methodology**: Pairwise comparison and single-answer grading. Studied position bias, verbosity bias, self-enhancement bias.
- **Key Results**: GPT-4 achieves 80%+ agreement with humans (same as human-human agreement).
- **Relevance**: LLM-as-judge introduces systematic biases that could be formalized via information geometry. The discrepancy between judge distributions and true preference distributions can be measured via Fisher-Rao distance.

#### 8. InstructGPT
- **Authors**: Ouyang, Wu, Jiang et al. (2022)
- **Source**: arXiv:2203.02155
- **Key Contribution**: RLHF training paradigm. Showed instruction-tuned models preferred over much larger base models.
- **Relevance**: RLHF changes the output distribution geometry; information-geometric tools could track how alignment modifies the manifold structure.

#### 9. DPO - Direct Preference Optimization
- **Authors**: Rafailov, Sharma, Mitchell et al. (2023)
- **Source**: arXiv:2305.18290
- **Key Contribution**: Reformulates RLHF as direct optimization on preferences, eliminating the reward model.
- **Relevance**: DPO has an implicit connection to the Bradley-Terry model and exponential families, making it amenable to information-geometric analysis.

### C. Calibration and Uncertainty

#### 10. Language Models (Mostly) Know What They Know
- **Authors**: Kadavath, Conerly, Askell et al. (Anthropic, 2022)
- **Source**: arXiv:2207.05221
- **Key Contribution**: Studied self-evaluation and calibration in LLMs. Introduced P(True) and P(IK) ("I Know") metrics.
- **Methodology**: Models evaluate their own answers. Calibration improves with scale. "Brainstorming" multiple answers before evaluation helps.
- **Key Results**: Larger models are well-calibrated on multiple-choice. Self-evaluation improves faster than generation quality with scale.
- **Relevance**: Calibration is fundamentally a question of the geometry of probability distributions. The Fisher metric provides the natural distance on the space of calibrated predictions.

#### 11. Conformal Prediction under Ambiguous Ground Truth
- **Authors**: Stutz, Roy, Matejovicova et al. (2023)
- **Source**: arXiv:2307.09302
- **Key Contribution**: Extends conformal prediction to settings with ambiguous labels. Provides formal coverage guarantees.
- **Relevance**: Conformal prediction provides distribution-free coverage guarantees - a formal statistical framework that complements our mathematical approaches.

### D. Contamination Detection

#### 12. Detecting Pretraining Data from Large Language Models (MIN-K% PROB)
- **Authors**: Shi, Ajith, Xia, Huang et al. (2023)
- **Source**: ICLR 2024 (arXiv:2310.16789)
- **Key Contribution**: Reference-free membership inference attack. MIN-K% PROB selects tokens with lowest probabilities and averages their log-likelihood.
- **Methodology**: Hypothesis: unseen text has more outlier low-probability tokens than seen text. WIKIMIA benchmark using Wikipedia temporal splits.
- **Key Results**: 7.4% AUC improvement over baselines. Applied to copyrighted book detection, contamination detection, privacy auditing.
- **Relevance**: The statistical basis of MIN-K% PROB could be strengthened with information-geometric tools. Random matrix theory could provide spectral signatures of contamination in weight matrices.

#### 13. Rethinking Benchmark Contamination with Rephrased Samples
- **Authors**: Yang, Chiang, Zheng et al. (2023)
- **Source**: arXiv:2311.04850
- **Key Contribution**: Showed n-gram decontamination is insufficient. Rephrased test samples bypass detection but still inflate benchmarks.
- **Methodology**: LLM-based rephrasing. A 13B model trained on rephrased MMLU achieves 85.9% (vs GPT-4's 86.4%).
- **Key Results**: 8-18% of HumanEval contaminated in RedPajama/StarCoder. Synthetic data from GPT-3.5/4 also carries contamination risk.
- **Relevance**: Topological approaches could detect contamination at a structural level, beyond surface-level string matching. Persistent homology of embedding spaces could reveal anomalous clustering of contaminated vs. clean data.

### E. Safety and Adversarial Evaluation

#### 14. Universal Adversarial Attacks on Aligned LLMs
- **Authors**: Zou, Wang, Carlini et al. (2023)
- **Source**: arXiv:2307.15043
- **Key Contribution**: GCG attack generating universal adversarial suffixes that transfer across models.
- **Relevance**: Adversarial robustness evaluation requires understanding the topology of the decision boundary.

#### 15. Constitutional AI
- **Authors**: Bai, Kadavath, Kundu et al. (2022)
- **Source**: arXiv:2212.08073
- **Key Contribution**: Self-improvement for harmlessness without human labels. AI provides feedback based on a constitution.
- **Relevance**: The constitutional framework creates implicit constraint manifolds; sheaf-theoretic approaches could formalize consistency of safety properties across contexts.

### F. Hallucination and RAG Evaluation

#### 16. Survey on Hallucination in LLMs
- **Authors**: Huang, Yu, Ma et al. (2023)
- **Source**: arXiv:2311.05232
- **Key Contribution**: Taxonomy of hallucination types. Discusses detection and mitigation strategies.
- **Relevance**: Hallucination detection could benefit from topological analysis of the embedding space.

#### 17. RAGAS and ARES (RAG Evaluation)
- **Sources**: arXiv:2309.15217, arXiv:2311.09476
- **Key Contribution**: Automated evaluation frameworks for retrieval-augmented generation.
- **Relevance**: RAG evaluation requires composing retrieval quality and generation quality - a natural sheaf-theoretic composition problem.

### G. Mathematical Foundations

#### 18. Introduction to Topological Data Analysis
- **Authors**: Chazal, Michel (2017/2021)
- **Source**: arXiv:1710.04019
- **Key Contribution**: Comprehensive TDA introduction covering simplicial complexes, persistent homology, statistical aspects, and machine learning applications.
- **Core Concepts**:
  - **Persistent homology**: Tracks topological features (components, loops, cavities) across scales via filtrations. Produces persistence diagrams/barcodes.
  - **Stability theorem**: Small perturbations in data produce small changes in persistence diagrams (bottleneck distance). This is the key formal guarantee.
  - **Distance-to-measure (DTM)**: Robust alternative to Hausdorff distance, stable under Wasserstein metric perturbations.
  - **Vectorization**: Persistence landscapes, persistence images, Betti curves for integration with ML pipelines.
- **Tools**: GUDHI (C++/Python), Dionysus, Giotto-TDA.
- **Relevance**: Core methodology. Persistent homology can be applied to LLM embedding spaces to detect structural features (clusters, loops, voids) at multiple scales. The stability theorem guarantees that these features are robust to noise.

#### 19. Elementary Introduction to Information Geometry
- **Authors**: Frank Nielsen (2018/2020)
- **Source**: arXiv:1808.08271
- **Key Contribution**: Self-contained introduction to information manifolds, Fisher metric, conjugate connections, dually flat manifolds.
- **Core Concepts**:
  - **Fisher information metric**: The unique invariant metric on statistical manifolds. Measures sensitivity of distributions to parameter changes.
  - **Conjugate connection manifolds (CCM)**: (M, g, nabla, nabla*) - dual connections preserving the metric under dual parallel transport.
  - **Statistical manifolds**: (M, g, C) with Amari-Chentsov cubic tensor. Supports alpha-connections family.
  - **Dually flat manifolds**: From Bregman divergences. Exhibit Pythagorean theorems for projections.
  - **f-divergences**: Unique separable divergences satisfying information monotonicity.
  - **Fisher-Rao distance**: Geodesic distance on statistical manifold.
- **Relevance**: Central to our framework. The Fisher metric on the manifold of LLM output distributions provides a principled way to measure distributional drift (training vs. evaluation, clean vs. contaminated), compare model calibration, and detect when evaluation metrics become unreliable.

---

## Common Methodologies
- **Multiple-choice accuracy**: MMLU, BIG-bench, HELM (simple but limited)
- **Pass@k**: HumanEval code evaluation (functional correctness)
- **Pairwise comparison + Bradley-Terry**: Chatbot Arena, MT-Bench
- **LLM-as-Judge**: MT-Bench, AlpacaEval (scalable but biased)
- **Membership inference**: MIN-K% PROB (contamination detection)
- **Calibration metrics**: ECE, Brier score, P(True)/P(IK)

## Standard Baselines
- GPT-4, Claude, LLaMA family (proprietary and open)
- Random chance (25% for 4-choice), human expert performance
- n-gram overlap for decontamination

## Evaluation Metrics in the Literature
- Accuracy, F1, BLEU/ROUGE (traditional NLP)
- Elo rating, win rate (pairwise comparison)
- AUC for contamination detection
- ECE (Expected Calibration Error) for calibration
- Persistence diagram distances (bottleneck, Wasserstein) for TDA

## Datasets in the Literature
- **MMLU**: 15,908 questions, 57 subjects (we downloaded)
- **Chatbot Arena**: 240K+ pairwise votes (we downloaded)
- **HELM**: 42 scenarios, standardized evaluation results (we documented)
- **GSM8K**: 8.5K math problems
- **HumanEval**: 164 coding problems
- **BIG-bench**: 204 tasks
- **WIKIMIA**: Wikipedia temporal splits for contamination detection

---

## Gaps and Opportunities

### Gap 1: No Formal Stability Guarantees
Current metrics (accuracy, Elo) lack formal guarantees about how they change under perturbations. Persistent homology's stability theorem (bounded change in persistence diagrams under bounded input perturbation) directly addresses this.

### Gap 2: Non-Composable Evaluations
HELM evaluates 7 metrics across 42 scenarios independently. There is no formal framework for composing these into a consistent global assessment. Sheaf theory provides exactly this: local sections (per-scenario evaluations) that must satisfy gluing conditions for global consistency.

### Gap 3: Contamination Detection at Structural Level
Current methods (n-gram, embedding similarity, MIN-K% PROB) operate at the surface level. Random matrix theory applied to weight matrices or embedding covariance matrices could detect contamination via spectral anomalies (deviation from Marchenko-Pastur distribution).

### Gap 4: No Geometric Framework for Distributional Drift
When models are updated or evaluated on new data, there's no principled way to measure how the output distribution has changed. The Fisher-Rao distance on the statistical manifold of output distributions provides this, with information-theoretic interpretations.

---

### Additional Critical Papers (discovered through search)

#### 20. Neural Persistence: A Complexity Measure via Algebraic Topology
- **Source**: ICLR 2019 (arXiv:1812.09764)
- **Key Contribution**: Defines neural persistence as a structural complexity measure for DNNs using persistent homology of weighted graphs derived from network weights.
- **Relevance**: Foundational work directly connecting persistent homology to neural network evaluation.

#### 21. Sheaf Neural Networks / Neural Sheaf Diffusion
- **Sources**: arXiv:2012.06333, arXiv:2202.04579
- **Key Contribution**: Generalizes GNNs via sheaf Laplacian. Shows sheaf-theoretic framework explains heterophily and oversmoothing.
- **Relevance**: Provides precedent for applying sheaf theory to neural network analysis.

#### 22. Implicit Self-Regularization via Random Matrix Theory
- **Source**: arXiv:1810.01075
- **Key Contribution**: Seminal work showing weight matrices of pre-trained DNNs exhibit heavy-tailed spectral distributions deviating from Marchenko-Pastur. Proposes this as indicator of training quality.
- **Relevance**: Core methodology for our spectral analysis pillar. Deviation from RMT predictions indicates learned structure vs. noise.

#### 23. Small Singular Values in Transformers (RMT Analysis)
- **Source**: arXiv:2410.17770
- **Key Contribution**: Analyzes singular-value spectra of weight matrices in pretrained transformers using RMT.
- **Relevance**: Directly applicable to LLM evaluation via spectral signatures.

#### 24. Contamination Detection Surveys
- **Sources**: arXiv:2404.00699, arXiv:2406.04244
- **Key Contribution**: Comprehensive reviews of 50+ detection techniques for benchmark contamination.
- **Relevance**: Establishes the state-of-the-art baseline our mathematical approaches aim to improve upon.

#### 25. Persistent Homology Captures Generalization
- **Source**: arXiv:2106.00012
- **Key Contribution**: Shows persistence diagram distances between consecutive training states correlate with validation accuracy - without needing a validation set.
- **Relevance**: Demonstrates TDA can replace traditional evaluation metrics, supporting our framework's viability.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **MMLU** (primary): Well-structured, 57 subjects, rich performance variation across models - ideal for topological analysis
2. **Chatbot Arena logs**: Pairwise comparisons with Bradley-Terry structure - natural for information-geometric analysis
3. **HELM results**: Multi-metric evaluations - ideal for sheaf-theoretic composition experiments

### Recommended Baselines
1. Standard accuracy metrics (MMLU accuracy per subject)
2. Elo ratings (Chatbot Arena)
3. MIN-K% PROB for contamination detection
4. ECE for calibration assessment

### Recommended Metrics
1. Persistence diagram distances (bottleneck, Wasserstein-p) for topological analysis
2. Fisher-Rao distance for distributional comparison
3. Spectral statistics (Tracy-Widom, Marchenko-Pastur) for contamination
4. Sheaf cohomology dimensions for evaluation consistency

### Methodological Considerations
- Start with MMLU performance vectors across subjects as point clouds for TDA
- Use Chatbot Arena pairwise data to construct Bradley-Terry manifold
- Apply persistent homology to model embedding spaces at multiple scales
- Validate contamination detection using known contaminated/clean splits from WIKIMIA
- Compare topological features before/after known model updates to measure structural drift
