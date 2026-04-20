# Mathematically Grounded Evaluation of Large Language Models: A Survey and New Directions

A comprehensive survey paper bridging the empirical LLM evaluation community with the mathematical sciences, proposing five novel mathematically rigorous evaluation frameworks.

## Key Findings

- **Benchmark saturation is accelerating**: MMLU scores went from 43.9% (2020) to 90%+ (2024), rendering established benchmarks non-discriminative
- **Six fundamental gaps** identified in current LLM evaluation: lack of stability guarantees, non-composable evaluations, no geometric drift detection, invisible failure modes, surface-level contamination detection, and no unifying framework
- **Persistent homology** provides formal stability guarantees for evaluation metrics via the bottleneck stability theorem (Wasserstein drift distance of 1.77 between model generations)
- **Sheaf-theoretic evaluation** formalizes when metric aggregation is faithful; cohomological obstructions detect inconsistent evaluations
- **Spectral analysis via RMT** detects contamination signatures invisible to n-gram and embedding-based methods, with Marchenko-Pastur bounds providing null hypothesis testing

## Paper

- **PDF**: `results/survey_paper.pdf` (30 pages, 151 references)
- **LaTeX source**: `paper/main.tex`
- **Target venues**: ACM Computing Surveys, TMLR, or arXiv (cs.AI/cs.CL/cs.LG)

## Reproduction

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv add numpy scipy matplotlib seaborn networkx scikit-learn ripser persim gudhi sympy datasets

# Run proof-of-concept implementations
python src/framework_implementations.py

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Structure

| Directory | Contents |
|-----------|----------|
| `paper/` | LaTeX source and compiled PDF |
| `src/` | Python implementations of 5 mathematical frameworks |
| `results/` | Numerical results, taxonomy JSON, paper PDF |
| `figures/` | 11 publication-quality figures |
| `datasets/` | MMLU (14K questions), Chatbot Arena data |
| `papers/` | 36 downloaded research papers |
| `code/` | 6 cloned reference repositories |

## Frameworks Implemented

1. **Topological Drift Detection** - Persistent homology on capability point clouds
2. **Sheaf-Theoretic Composition** - Sheaf over task decomposition graphs
3. **Information-Geometric Manifolds** - Fisher-Rao distances on behavior distributions
4. **Failure Mode Landscape** - TDA of failure pattern clustering
5. **Spectral Contamination Detection** - Random matrix theory on performance matrices

See `REPORT.md` for detailed results and analysis.
