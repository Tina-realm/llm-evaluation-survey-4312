"""
Proof-of-concept implementations for Part III of the survey paper:
Five mathematically grounded evaluation frameworks for LLMs.

Framework 1: Topological Drift Detection via Persistent Homology
Framework 2: Sheaf-Theoretic Compositional Evaluation
Framework 3: Information-Geometric Evaluation Manifolds
Framework 4: TDA Failure Mode Landscape Analysis
Framework 5: Spectral Contamination Detection via Random Matrix Theory
"""

import numpy as np
import json
import os
import random
from collections import defaultdict

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_DIR = "/workspaces/llm-evaluation-survey-4312"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


###############################################################################
# Data Loading
###############################################################################

def load_mmlu_data():
    """Load MMLU test set and organize by subject."""
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(BASE_DIR, "datasets/mmlu/data"))
    test = ds["test"]

    subjects = sorted(set(test["subject"]))
    subject_counts = defaultdict(int)
    for s in test["subject"]:
        subject_counts[s] += 1

    return test, subjects, dict(subject_counts)


def load_chatbot_arena_data():
    """Load Chatbot Arena / MT-Bench judgment data."""
    from datasets import load_from_disk
    ds_path = os.path.join(BASE_DIR, "datasets/chatbot_arena/data")
    if os.path.exists(ds_path):
        from datasets import load_from_disk
        ds = load_from_disk(ds_path)
        if isinstance(ds, dict):
            return ds[list(ds.keys())[0]]
        return ds
    # Fallback to sample
    with open(os.path.join(BASE_DIR, "datasets/chatbot_arena/sample_10.json")) as f:
        return json.load(f)


def simulate_model_performance(subjects, n_models=12, noise_level=0.05):
    """
    Simulate realistic model performance profiles across MMLU subjects.
    Models are designed to represent different capability archetypes:
    - Generalist (uniform), STEM-strong, Humanities-strong, etc.
    Based on real MMLU performance patterns from literature.
    """
    n_subjects = len(subjects)

    # Create subject category mapping based on real MMLU categories
    stem_subjects = [s for s in subjects if any(k in s for k in
        ['math', 'physics', 'chemistry', 'biology', 'computer', 'engineering',
         'statistics', 'machine_learning', 'astronomy', 'electrical'])]
    humanities_subjects = [s for s in subjects if any(k in s for k in
        ['philosophy', 'history', 'literature', 'religion', 'law', 'moral',
         'ethics', 'jurisprudence', 'logical_fallacies'])]
    social_subjects = [s for s in subjects if any(k in s for k in
        ['psychology', 'sociology', 'economics', 'geography', 'politics',
         'government', 'security', 'management', 'marketing'])]

    stem_idx = [subjects.index(s) for s in stem_subjects if s in subjects]
    hum_idx = [subjects.index(s) for s in humanities_subjects if s in subjects]
    soc_idx = [subjects.index(s) for s in social_subjects if s in subjects]

    # Model archetypes with realistic accuracy ranges
    model_configs = [
        ("GPT-4 (2023)", 0.82, "generalist"),
        ("GPT-4o (2024)", 0.86, "generalist"),
        ("Claude-3 Opus", 0.83, "humanities_strong"),
        ("Claude-3.5 Sonnet", 0.87, "generalist"),
        ("Gemini 1.5 Pro", 0.81, "stem_strong"),
        ("LLaMA-3 70B", 0.78, "generalist"),
        ("LLaMA-3 8B", 0.62, "weak"),
        ("Mixtral 8x22B", 0.76, "stem_strong"),
        ("Qwen-2 72B", 0.80, "stem_strong"),
        ("GPT-3.5 Turbo", 0.68, "weak"),
        ("PaLM-2", 0.75, "social_strong"),
        ("Mistral 7B", 0.58, "weak"),
    ]

    performances = {}
    for name, base_acc, archetype in model_configs:
        perf = np.full(n_subjects, base_acc)

        if archetype == "stem_strong":
            perf[stem_idx] += 0.05
            perf[hum_idx] -= 0.03
        elif archetype == "humanities_strong":
            perf[hum_idx] += 0.05
            perf[stem_idx] -= 0.02
        elif archetype == "social_strong":
            perf[soc_idx] += 0.04
        elif archetype == "weak":
            # More variance for weaker models
            noise_level_local = noise_level * 2
            perf += np.random.normal(0, noise_level_local, n_subjects)

        # Add subject-specific noise
        perf += np.random.normal(0, noise_level, n_subjects)
        perf = np.clip(perf, 0.25, 0.98)  # 0.25 is random chance for 4-choice
        performances[name] = perf

    return performances


###############################################################################
# Framework 1: Topological Drift Detection via Persistent Homology
###############################################################################

def framework1_topological_drift(performances, subjects):
    """
    Apply persistent homology to model capability profiles.

    Concept: Each model is a point in R^57 (one coordinate per MMLU subject).
    The Vietoris-Rips complex on this point cloud reveals topological structure:
    - Connected components (H0): clusters of similarly-performing models
    - Loops (H1): cyclic capability relationships

    Drift is detected by comparing persistence diagrams across time points.
    """
    import ripser
    from persim import plot_diagrams, wasserstein, bottleneck
    import matplotlib.pyplot as plt

    model_names = list(performances.keys())
    X = np.array([performances[m] for m in model_names])

    # Compute persistence diagram for the full point cloud
    result = ripser.ripser(X, maxdim=1)
    diagrams = result['dgms']

    # Save persistence diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H0 barcode
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf]
    births = h0[:, 0]
    deaths = h0[:, 1].copy()
    deaths[deaths == np.inf] = deaths[deaths != np.inf].max() * 1.2 if len(deaths[deaths != np.inf]) > 0 else 1.0

    ax = axes[0]
    for i in range(len(births)):
        color = 'red' if h0[i, 1] == np.inf else 'steelblue'
        ax.plot([births[i], deaths[i]], [i, i], color=color, linewidth=2)
    ax.set_xlabel('Filtration Value (L2 Distance)', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('$H_0$ Barcode (Connected Components)', fontsize=13)

    # Persistence diagram
    ax = axes[1]
    max_val = 0
    colors = ['steelblue', 'orange']
    labels = ['$H_0$', '$H_1$']
    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        finite = dgm[dgm[:, 1] != np.inf]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], alpha=0.7, s=50,
                      color=colors[dim], label=labels[dim], zorder=3)
            max_val = max(max_val, finite.max())

    ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3, label='Diagonal')
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title('Persistence Diagram', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'persistence_diagrams.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Simulate temporal drift: compare early vs. late model cohorts
    early_models = ["GPT-3.5 Turbo", "LLaMA-3 8B", "Mistral 7B", "PaLM-2"]
    late_models = ["GPT-4o (2024)", "Claude-3.5 Sonnet", "Qwen-2 72B", "Gemini 1.5 Pro"]

    X_early = np.array([performances[m] for m in early_models if m in performances])
    X_late = np.array([performances[m] for m in late_models if m in performances])

    dgm_early = ripser.ripser(X_early, maxdim=1)['dgms']
    dgm_late = ripser.ripser(X_late, maxdim=1)['dgms']

    # Compute Wasserstein and bottleneck distances between diagrams
    drift_metrics = {}
    for dim in range(min(2, len(dgm_early), len(dgm_late))):
        d_early = dgm_early[dim]
        d_late = dgm_late[dim]
        # Filter out infinite values
        d_early_f = d_early[d_early[:, 1] != np.inf]
        d_late_f = d_late[d_late[:, 1] != np.inf]
        if len(d_early_f) > 0 and len(d_late_f) > 0:
            w_dist = wasserstein(d_early_f, d_late_f)
            b_dist = bottleneck(d_early_f, d_late_f)
            drift_metrics[f'H{dim}_wasserstein'] = float(w_dist)
            drift_metrics[f'H{dim}_bottleneck'] = float(b_dist)

    # Plot drift comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (cohort, X_cohort, name) in enumerate(
        [(early_models, X_early, "Early Models (2023)"),
         (late_models, X_late, "Late Models (2024)")]):
        dgm = ripser.ripser(X_cohort, maxdim=1)['dgms']
        ax = axes[idx]
        max_val = 0
        for dim in range(min(2, len(dgm))):
            d = dgm[dim]
            finite = d[d[:, 1] != np.inf]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], alpha=0.7, s=60,
                          color=colors[dim], label=labels[dim], zorder=3)
                max_val = max(max_val, finite.max())
        if max_val > 0:
            ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3)
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(name, fontsize=13)
        ax.legend(fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle('Topological Drift Detection: Persistence Diagram Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'topological_drift.png'), dpi=200, bbox_inches='tight')
    plt.close()

    results = {
        'n_models': len(model_names),
        'n_subjects': len(subjects),
        'h0_features': len(diagrams[0]),
        'h1_features': len(diagrams[1]) if len(diagrams) > 1 else 0,
        'drift_metrics': drift_metrics,
        'early_models': early_models,
        'late_models': late_models,
    }

    print(f"Framework 1 Results:")
    print(f"  H0 features: {results['h0_features']}, H1 features: {results['h1_features']}")
    print(f"  Drift metrics: {drift_metrics}")

    return results


###############################################################################
# Framework 2: Sheaf-Theoretic Compositional Evaluation
###############################################################################

def framework2_sheaf_evaluation(performances, subjects):
    """
    Construct a sheaf over a task decomposition graph.

    Concept: Evaluation dimensions form a category (tasks, metrics, aggregations).
    A presheaf assigns to each evaluation context a set of measurements.
    The sheaf condition requires local consistency: if two overlapping contexts
    agree on shared aspects, they can be glued into a global section.

    Cohomological obstructions (H^1 != 0) indicate inconsistent evaluations.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Define evaluation dimension categories (mapping MMLU subjects)
    categories = {
        'STEM': [s for s in subjects if any(k in s for k in
            ['math', 'physics', 'chemistry', 'biology', 'computer', 'engineering',
             'statistics', 'machine_learning', 'astronomy', 'electrical'])],
        'Humanities': [s for s in subjects if any(k in s for k in
            ['philosophy', 'history', 'literature', 'religion', 'moral', 'ethics',
             'jurisprudence', 'logical_fallacies'])],
        'Social Sciences': [s for s in subjects if any(k in s for k in
            ['psychology', 'sociology', 'economics', 'geography', 'politics',
             'government', 'security', 'management', 'marketing'])],
        'Professional': [s for s in subjects if any(k in s for k in
            ['law', 'medicine', 'clinical', 'nursing', 'anatomy', 'nutrition',
             'professional', 'accounting', 'business'])],
    }
    # Assign uncategorized subjects
    categorized = set()
    for cat_subjects in categories.values():
        categorized.update(cat_subjects)
    categories['Other'] = [s for s in subjects if s not in categorized]

    # Build the task decomposition graph
    G = nx.DiGraph()
    G.add_node('Global', level=0)
    for cat in categories:
        G.add_node(cat, level=1)
        G.add_edge('Global', cat)
        for subj in categories[cat]:
            G.add_node(subj, level=2)
            G.add_edge(cat, subj)

    # Construct presheaf: assign performance vectors to each node
    # For a sheaf, restriction maps must be consistent
    model_names = list(performances.keys())

    sheaf_sections = {}
    cohomology_obstructions = {}

    for model in model_names:
        perf = performances[model]

        # Local sections at subject level
        subject_sections = {s: perf[subjects.index(s)] for s in subjects}

        # Category-level sections (should be consistent average)
        cat_sections = {}
        for cat, cat_subjects in categories.items():
            if cat_subjects:
                cat_sections[cat] = np.mean([subject_sections[s] for s in cat_subjects])

        # Global section
        global_section = np.mean(perf)

        # Check sheaf condition: is the global average consistent with
        # the category-weighted average?
        cat_weighted = sum(cat_sections[c] * len(categories[c]) for c in cat_sections) / len(subjects)

        # Cohomological obstruction: difference between aggregation paths
        obstruction = abs(global_section - cat_weighted)

        # Richer obstruction: variance across categories vs within
        between_var = np.var(list(cat_sections.values()))
        within_vars = []
        for cat, cat_subjects in categories.items():
            if len(cat_subjects) > 1:
                vals = [subject_sections[s] for s in cat_subjects]
                within_vars.append(np.var(vals))
        within_var = np.mean(within_vars) if within_vars else 0

        sheaf_sections[model] = {
            'global': float(global_section),
            'categories': {k: float(v) for k, v in cat_sections.items()},
            'obstruction': float(obstruction),
            'between_category_variance': float(between_var),
            'within_category_variance': float(within_var),
            'consistency_ratio': float(between_var / (within_var + 1e-10)),
        }
        cohomology_obstructions[model] = float(obstruction)

    # Visualize sheaf diagram
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Task decomposition graph
    ax = axes[0]
    pos = {}
    pos['Global'] = (0.5, 1.0)
    cat_names = list(categories.keys())
    for i, cat in enumerate(cat_names):
        pos[cat] = ((i + 0.5) / len(cat_names), 0.6)

    # Only show a few subjects per category for clarity
    shown_subjects = []
    for cat in cat_names:
        for s in categories[cat][:3]:
            shown_subjects.append(s)
            cat_idx = cat_names.index(cat)
            subj_idx = categories[cat].index(s)
            x = (cat_idx + 0.5) / len(cat_names) + (subj_idx - 1) * 0.04
            pos[s] = (x, 0.2)

    sub_G = G.subgraph(['Global'] + cat_names + shown_subjects)
    colors_map = {'Global': '#e74c3c'}
    cat_colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for i, cat in enumerate(cat_names):
        colors_map[cat] = cat_colors[i % len(cat_colors)]
    for s in shown_subjects:
        colors_map[s] = '#95a5a6'

    node_colors = [colors_map.get(n, '#95a5a6') for n in sub_G.nodes()]

    nx.draw(sub_G, pos, ax=ax, with_labels=False, node_color=node_colors,
            node_size=800, arrows=True, arrowsize=15, edge_color='gray',
            font_size=7)

    # Add labels manually for readability
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items() if k in sub_G.nodes()}
    short_labels = {}
    for n in sub_G.nodes():
        if n in cat_names or n == 'Global':
            short_labels[n] = n
        else:
            short_labels[n] = n[:15] + '...' if len(n) > 15 else n
    nx.draw_networkx_labels(sub_G, label_pos, short_labels, ax=ax, font_size=7)

    ax.set_title('Sheaf over Task Decomposition Graph', fontsize=13)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.0, 1.15)

    # Right: Cohomological obstructions by model
    ax = axes[1]
    models_sorted = sorted(cohomology_obstructions.keys(),
                          key=lambda m: sheaf_sections[m]['consistency_ratio'])
    ratios = [sheaf_sections[m]['consistency_ratio'] for m in models_sorted]
    bars = ax.barh(range(len(models_sorted)), ratios, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels(models_sorted, fontsize=9)
    ax.set_xlabel('Consistency Ratio (Between/Within Category Variance)', fontsize=11)
    ax.set_title('Sheaf Cohomological Consistency', fontsize=13)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Equal variance')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sheaf_evaluation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    results = {
        'n_categories': len(categories),
        'category_sizes': {k: len(v) for k, v in categories.items()},
        'sheaf_sections': sheaf_sections,
        'graph_nodes': len(G.nodes()),
        'graph_edges': len(G.edges()),
    }

    print(f"Framework 2 Results:")
    print(f"  Graph: {results['graph_nodes']} nodes, {results['graph_edges']} edges")
    print(f"  Categories: {results['category_sizes']}")

    return results


###############################################################################
# Framework 3: Information-Geometric Evaluation Manifolds
###############################################################################

def framework3_information_geometry(performances, subjects):
    """
    Define a statistical manifold over LLM output distributions
    and compute Fisher-Rao distances for drift detection.

    Concept: Model output distributions on multiple-choice tasks form
    a multinomial manifold. The Fisher information metric provides the
    natural Riemannian metric. Fisher-Rao distances between models
    capture distributional differences that accuracy alone misses.
    """
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    from sklearn.manifold import MDS

    model_names = list(performances.keys())
    n_models = len(model_names)

    # For each model, construct a probability distribution over performance levels
    # Discretize accuracy into bins to form categorical distributions
    n_bins = 10
    bin_edges = np.linspace(0.25, 1.0, n_bins + 1)

    model_distributions = {}
    for name in model_names:
        perf = performances[name]
        hist, _ = np.histogram(perf, bins=bin_edges, density=True)
        # Normalize to probability distribution (add small epsilon for stability)
        p = hist / hist.sum() + 1e-8
        p = p / p.sum()
        model_distributions[name] = p

    # Compute Fisher-Rao distance matrix
    # For categorical distributions, Fisher-Rao distance = 2 * arccos(sum(sqrt(p_i * q_i)))
    # This is the Bhattacharyya angle, which equals Fisher-Rao distance on the probability simplex

    fisher_rao_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            p = model_distributions[model_names[i]]
            q = model_distributions[model_names[j]]
            # Bhattacharyya coefficient
            bc = np.sum(np.sqrt(p * q))
            bc = np.clip(bc, -1, 1)
            fisher_rao_matrix[i, j] = 2 * np.arccos(bc)

    # Also compute KL divergence for comparison
    kl_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            p = model_distributions[model_names[i]]
            q = model_distributions[model_names[j]]
            kl_matrix[i, j] = np.sum(p * np.log(p / q))

    # MDS embedding of Fisher-Rao distances
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED, normalized_stress='auto')
    embedding = mds.fit_transform(fisher_rao_matrix)

    # Compute Fisher information for each model (trace of Fisher information matrix)
    # For multinomial with parameters p_i: F_ii = 1/p_i
    fisher_traces = {}
    for name in model_names:
        p = model_distributions[name]
        fisher_traces[name] = float(np.sum(1.0 / p))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Fisher-Rao distance heatmap
    ax = axes[0]
    im = ax.imshow(fisher_rao_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([n.split('(')[0].strip()[:10] for n in model_names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([n.split('(')[0].strip()[:10] for n in model_names], fontsize=8)
    ax.set_title('Fisher-Rao Distance Matrix', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # MDS embedding
    ax = axes[1]
    # Color by average performance
    avg_perf = [np.mean(performances[m]) for m in model_names]
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=avg_perf,
                        cmap='RdYlGn', s=100, edgecolors='black', linewidth=0.5, zorder=3)
    for i, name in enumerate(model_names):
        short_name = name.split('(')[0].strip()[:12]
        ax.annotate(short_name, (embedding[i, 0], embedding[i, 1]),
                   fontsize=7, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')
    ax.set_xlabel('MDS Dimension 1', fontsize=11)
    ax.set_ylabel('MDS Dimension 2', fontsize=11)
    ax.set_title('Statistical Manifold (MDS of Fisher-Rao)', fontsize=13)
    plt.colorbar(scatter, ax=ax, label='Mean Accuracy', shrink=0.8)

    # Fisher information trace
    ax = axes[2]
    sorted_models = sorted(fisher_traces.keys(), key=lambda m: fisher_traces[m])
    traces = [fisher_traces[m] for m in sorted_models]
    bars = ax.barh(range(len(sorted_models)), traces, color='darkorange', alpha=0.8)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([m.split('(')[0].strip()[:15] for m in sorted_models], fontsize=9)
    ax.set_xlabel('Fisher Information Trace', fontsize=11)
    ax.set_title('Model Sensitivity (Fisher Information)', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'information_geometry.png'), dpi=200, bbox_inches='tight')
    plt.close()

    results = {
        'fisher_rao_matrix': fisher_rao_matrix.tolist(),
        'model_names': model_names,
        'fisher_traces': fisher_traces,
        'mds_embedding': embedding.tolist(),
        'max_fisher_rao': float(fisher_rao_matrix.max()),
        'mean_fisher_rao': float(fisher_rao_matrix[np.triu_indices_from(fisher_rao_matrix, k=1)].mean()),
    }

    print(f"Framework 3 Results:")
    print(f"  Max Fisher-Rao distance: {results['max_fisher_rao']:.4f}")
    print(f"  Mean Fisher-Rao distance: {results['mean_fisher_rao']:.4f}")

    return results


###############################################################################
# Framework 4: TDA Failure Mode Landscape Analysis
###############################################################################

def framework4_failure_modes(performances, subjects, subject_counts):
    """
    Apply persistent homology to failure patterns to identify structural
    clustering of failure modes.

    Concept: For each model, create a binary failure vector across subjects
    (above/below threshold). The Hamming distance between failure patterns
    reveals structural relationships. Persistent homology on this space
    identifies clusters and cycles of correlated failures.
    """
    import ripser
    import matplotlib.pyplot as plt

    model_names = list(performances.keys())
    n_models = len(model_names)

    # Create failure profiles at different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]

    all_failure_results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for t_idx, threshold in enumerate(thresholds):
        # Binary failure matrix: 1 if below threshold
        failure_matrix = np.zeros((n_models, len(subjects)))
        for i, model in enumerate(model_names):
            failure_matrix[i] = (performances[model] < threshold).astype(float)

        # Compute Hamming distances
        from scipy.spatial.distance import pdist, squareform
        hamming_dists = squareform(pdist(failure_matrix, metric='hamming'))

        # Persistent homology on failure space
        result = ripser.ripser(hamming_dists, maxdim=1, distance_matrix=True)
        dgms = result['dgms']

        # Plot persistence diagram
        ax = axes[t_idx // 2][t_idx % 2]
        colors = ['steelblue', 'orange']
        labels_ph = ['$H_0$', '$H_1$']
        max_val = 0
        for dim in range(min(2, len(dgms))):
            d = dgms[dim]
            finite = d[d[:, 1] != np.inf]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], alpha=0.7, s=50,
                          color=colors[dim], label=labels_ph[dim], zorder=3)
                max_val = max(max_val, finite.max())
        if max_val > 0:
            ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3)
        ax.set_xlabel('Birth', fontsize=11)
        ax.set_ylabel('Death', fontsize=11)
        ax.set_title(f'Failure Landscape (threshold={threshold})', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

        # Compute Betti numbers
        betti_0 = sum(1 for b, d in dgms[0] if d == np.inf)
        betti_1 = len(dgms[1]) if len(dgms) > 1 else 0

        # Total persistence (lifetime sum)
        total_persistence_h0 = sum(d - b for b, d in dgms[0] if d != np.inf)
        total_persistence_h1 = sum(d - b for b, d in dgms[1]) if len(dgms) > 1 else 0

        all_failure_results[f'threshold_{threshold}'] = {
            'betti_0': int(betti_0),
            'betti_1': int(betti_1),
            'total_persistence_h0': float(total_persistence_h0),
            'total_persistence_h1': float(total_persistence_h1),
            'failure_rate_mean': float(failure_matrix.mean()),
            'failure_rate_std': float(failure_matrix.mean(axis=1).std()),
        }

    plt.suptitle('Failure Mode Landscape: Persistent Homology at Multiple Thresholds',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'failure_mode_landscape.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Subject-level failure correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    failure_70 = np.zeros((n_models, len(subjects)))
    for i, model in enumerate(model_names):
        failure_70[i] = (performances[model] < 0.7).astype(float)

    # Correlation between subject failure patterns
    if failure_70.shape[0] > 1:
        # Only compute for subjects with variance
        var_mask = failure_70.std(axis=0) > 0
        if var_mask.sum() > 2:
            corr = np.corrcoef(failure_70[:, var_mask].T)
            subj_labels = [subjects[i][:15] for i in range(len(subjects)) if var_mask[i]]
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title('Subject Failure Correlation (threshold=0.7)', fontsize=13)
            plt.colorbar(im, ax=ax, shrink=0.8)
            if len(subj_labels) <= 30:
                ax.set_xticks(range(len(subj_labels)))
                ax.set_xticklabels(subj_labels, rotation=90, fontsize=6)
                ax.set_yticks(range(len(subj_labels)))
                ax.set_yticklabels(subj_labels, fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'failure_correlation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Framework 4 Results:")
    for thresh, res in all_failure_results.items():
        print(f"  {thresh}: Betti_0={res['betti_0']}, Betti_1={res['betti_1']}, "
              f"Mean failure rate={res['failure_rate_mean']:.3f}")

    return all_failure_results


###############################################################################
# Framework 5: Spectral Contamination Detection via Random Matrix Theory
###############################################################################

def framework5_spectral_analysis(performances, subjects):
    """
    Apply random matrix theory to performance matrices for contamination detection.

    Concept: Under the null hypothesis (no contamination), the eigenvalue distribution
    of a performance correlation matrix should follow the Marchenko-Pastur law.
    Deviations (outlier eigenvalues beyond the Tracy-Widom threshold) indicate
    systematic structure—potentially from contamination.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    model_names = list(performances.keys())
    X = np.array([performances[m] for m in model_names])

    n_models, n_subjects = X.shape

    # Construct the correlation matrix (subjects x subjects)
    # Center and scale
    X_centered = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # Sample covariance matrix (Wishart-type)
    # C = (1/n) * X^T X where X is n_models x n_subjects
    C = X_centered.T @ X_centered / n_models

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Marchenko-Pastur parameters
    gamma = n_subjects / n_models  # Aspect ratio
    sigma2 = 1.0  # Variance (data is standardized)

    # Marchenko-Pastur bounds
    lambda_plus = sigma2 * (1 + np.sqrt(1/gamma))**2 if gamma > 0 else np.inf
    lambda_minus = sigma2 * (1 - np.sqrt(1/gamma))**2 if gamma > 0 else 0

    # Marchenko-Pastur density
    def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
        lp = sigma2 * (1 + np.sqrt(1/gamma))**2
        lm = sigma2 * (1 - np.sqrt(1/gamma))**2
        if x < lm or x > lp:
            return 0.0
        return gamma / (2 * np.pi * sigma2 * x) * np.sqrt((lp - x) * (x - lm))

    # Generate MP density for plotting
    x_mp = np.linspace(max(0.01, lambda_minus * 0.9), lambda_plus * 1.1, 500)
    y_mp = np.array([marchenko_pastur_pdf(x, gamma) for x in x_mp])

    # Count outlier eigenvalues (beyond MP bound)
    outlier_eigenvalues = eigenvalues[eigenvalues > lambda_plus]
    n_outliers = len(outlier_eigenvalues)

    # Tracy-Widom scaling for the largest eigenvalue
    # Under MP null, the largest eigenvalue concentrates around lambda_plus
    # with Tracy-Widom fluctuations of order n^{-2/3}
    tw_scale = n_models**(-2/3) * lambda_plus
    tw_statistics = (eigenvalues[0] - lambda_plus) / tw_scale if tw_scale > 0 else 0

    # Simulate: create a "contaminated" performance matrix
    X_contaminated = X.copy()
    # Add systematic boost to 10 subjects (simulating contamination)
    contaminated_subjects = np.random.choice(n_subjects, 10, replace=False)
    for i in range(n_models):
        X_contaminated[i, contaminated_subjects] += np.random.uniform(0.05, 0.15, 10)
    X_contaminated = np.clip(X_contaminated, 0.25, 0.98)

    X_cont_centered = (X_contaminated - X_contaminated.mean(axis=0)) / (X_contaminated.std(axis=0) + 1e-10)
    C_cont = X_cont_centered.T @ X_cont_centered / n_models
    eig_cont = np.sort(np.linalg.eigvalsh(C_cont))[::-1]

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Eigenvalue distribution vs Marchenko-Pastur
    ax = axes[0][0]
    ax.hist(eigenvalues, bins=30, density=True, alpha=0.7, color='steelblue',
            label='Observed (Clean)', edgecolor='white')
    ax.plot(x_mp, y_mp, 'r-', linewidth=2, label='Marchenko-Pastur')
    ax.axvline(x=lambda_plus, color='red', linestyle='--', alpha=0.5, label=f'$\\lambda_+$ = {lambda_plus:.2f}')
    ax.axvline(x=lambda_minus, color='orange', linestyle='--', alpha=0.5, label=f'$\\lambda_-$ = {lambda_minus:.2f}')
    ax.set_xlabel('Eigenvalue', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Eigenvalue Distribution vs Marchenko-Pastur (Clean)', fontsize=12)
    ax.legend(fontsize=9)

    # Contaminated eigenvalue distribution
    ax = axes[0][1]
    ax.hist(eig_cont, bins=30, density=True, alpha=0.7, color='coral',
            label='Observed (Contaminated)', edgecolor='white')
    ax.plot(x_mp, y_mp, 'r-', linewidth=2, label='Marchenko-Pastur')
    ax.axvline(x=lambda_plus, color='red', linestyle='--', alpha=0.5, label=f'$\\lambda_+$ = {lambda_plus:.2f}')
    ax.set_xlabel('Eigenvalue', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Eigenvalue Distribution vs Marchenko-Pastur (Contaminated)', fontsize=12)
    ax.legend(fontsize=9)

    # Top eigenvalue comparison
    ax = axes[1][0]
    k = min(15, len(eigenvalues))
    x_pos = np.arange(k)
    width = 0.35
    ax.bar(x_pos - width/2, eigenvalues[:k], width, label='Clean', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, eig_cont[:k], width, label='Contaminated', color='coral', alpha=0.8)
    ax.axhline(y=lambda_plus, color='red', linestyle='--', alpha=0.5, label=f'MP bound ($\\lambda_+$)')
    ax.set_xlabel('Eigenvalue Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(f'Top {k} Eigenvalues: Clean vs Contaminated', fontsize=12)
    ax.legend(fontsize=9)

    # Eigenvalue spacing distribution (compare to GOE/GUE)
    ax = axes[1][1]
    # Compute normalized spacings for clean data
    sorted_eig = np.sort(eigenvalues)
    spacings = np.diff(sorted_eig)
    mean_spacing = spacings.mean()
    if mean_spacing > 0:
        normalized_spacings = spacings / mean_spacing
        ax.hist(normalized_spacings, bins=20, density=True, alpha=0.7,
                color='steelblue', label='Observed spacings', edgecolor='white')
        # Wigner surmise (GOE)
        s = np.linspace(0, 4, 100)
        wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
        ax.plot(s, wigner, 'r-', linewidth=2, label='Wigner surmise (GOE)')
        # Poisson (uncorrelated)
        poisson = np.exp(-s)
        ax.plot(s, poisson, 'g--', linewidth=2, label='Poisson (uncorrelated)')
    ax.set_xlabel('Normalized Spacing', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Eigenvalue Spacing Distribution', fontsize=12)
    ax.legend(fontsize=9)

    plt.suptitle('Spectral Analysis for Contamination Detection (Random Matrix Theory)',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'spectral_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Detect contaminated subjects via leverage scores
    # Subjects contributing most to outlier eigenvectors
    eigvecs = np.linalg.eigh(C)[1]
    # Top eigenvector (associated with largest eigenvalue)
    top_vec = eigvecs[:, -1]
    leverage_scores = top_vec**2
    top_subjects_idx = np.argsort(leverage_scores)[-10:][::-1]

    results = {
        'n_models': n_models,
        'n_subjects': n_subjects,
        'aspect_ratio_gamma': float(gamma),
        'mp_lambda_plus': float(lambda_plus),
        'mp_lambda_minus': float(lambda_minus),
        'top_5_eigenvalues_clean': eigenvalues[:5].tolist(),
        'top_5_eigenvalues_contaminated': eig_cont[:5].tolist(),
        'n_outliers_clean': int(n_outliers),
        'n_outliers_contaminated': int(sum(eig_cont > lambda_plus)),
        'tracy_widom_statistic': float(tw_statistics),
        'contaminated_subjects_injected': contaminated_subjects.tolist(),
        'top_leverage_subjects': [subjects[i] for i in top_subjects_idx],
    }

    print(f"Framework 5 Results:")
    print(f"  MP bounds: [{results['mp_lambda_minus']:.2f}, {results['mp_lambda_plus']:.2f}]")
    print(f"  Outliers (clean): {results['n_outliers_clean']}, (contaminated): {results['n_outliers_contaminated']}")
    print(f"  Tracy-Widom statistic: {results['tracy_widom_statistic']:.2f}")

    return results


###############################################################################
# Additional Figures
###############################################################################

def generate_taxonomy_figure(subjects):
    """Generate the taxonomy diagram of 11 evaluation dimensions."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dimensions = [
        ("1. Knowledge &\nReasoning", ["MMLU", "ARC", "HellaSwag", "BIG-Bench"]),
        ("2. Code\nGeneration", ["HumanEval", "MBPP", "CodeContests"]),
        ("3. Mathematical\nReasoning", ["GSM8K", "MATH", "Minerva"]),
        ("4. Language\nGeneration", ["BLEU/ROUGE", "BERTScore", "FActScore"]),
        ("5. Instruction\nFollowing", ["MT-Bench", "AlpacaEval", "IFEval"]),
        ("6. Safety &\nAlignment", ["TruthfulQA", "BBQ", "DecodingTrust"]),
        ("7. Calibration &\nUncertainty", ["ECE", "Brier Score", "P(True)"]),
        ("8. Robustness &\nAdversarial", ["GCG", "AutoDAN", "AdvBench"]),
        ("9. Contamination\nDetection", ["MIN-K%", "n-gram", "Rephrasing"]),
        ("10. Human\nPreference", ["Chatbot Arena", "RLHF", "DPO"]),
        ("11. RAG &\nCompound", ["RAGAS", "ARES", "LongBench"]),
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # Title
    ax.text(5.5, 7.5, 'Taxonomy of LLM Evaluation Dimensions', fontsize=18,
            ha='center', fontweight='bold')

    # Central node
    center_x, center_y = 5.5, 4.0
    circle = plt.Circle((center_x, center_y), 0.8, color='#e74c3c', alpha=0.3)
    ax.add_patch(circle)
    ax.text(center_x, center_y, 'LLM\nEvaluation', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Dimension nodes arranged in a circle
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
              '#e67e22', '#e74c3c', '#34495e', '#16a085', '#8e44ad', '#c0392b']

    n_dims = len(dimensions)
    for i, (name, methods) in enumerate(dimensions):
        angle = 2 * np.pi * i / n_dims - np.pi/2
        r = 3.2
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)

        # Draw connection line
        ax.plot([center_x, x], [center_y, y], color='gray', alpha=0.3, linewidth=1)

        # Draw dimension box
        bbox = dict(boxstyle='round,pad=0.4', facecolor=colors[i], alpha=0.3)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=bbox)

        # Add method labels
        method_text = '\n'.join(methods[:3])
        method_angle = angle
        mr = 4.5
        mx = center_x + mr * np.cos(method_angle)
        my = center_y + mr * np.sin(method_angle)
        ax.text(mx, my, method_text, ha='center', va='center', fontsize=6,
                color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'taxonomy_diagram.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated taxonomy diagram")


def generate_benchmark_saturation_timeline():
    """Generate timeline showing benchmark saturation."""
    import matplotlib.pyplot as plt

    # Historical MMLU scores (approximate from literature)
    timeline_data = {
        'MMLU': {
            'years': [2020, 2021, 2022, 2022.5, 2023, 2023.5, 2024, 2024.5],
            'scores': [43.9, 55.0, 67.0, 70.0, 83.0, 86.4, 88.0, 90.0],
            'models': ['GPT-3', 'Chinchilla', 'PaLM', 'GPT-3.5', 'GPT-4', 'GPT-4o', 'Claude-3.5', 'GPT-o1']
        },
        'GSM8K': {
            'years': [2021, 2022, 2023, 2023.5, 2024],
            'scores': [35.0, 58.0, 92.0, 95.0, 97.0],
            'models': ['GPT-3', 'PaLM', 'GPT-4', 'GPT-4o', 'o1-preview']
        },
        'HumanEval': {
            'years': [2021, 2022, 2023, 2023.5, 2024],
            'scores': [28.8, 47.0, 67.0, 84.1, 92.0],
            'models': ['Codex', 'PaLM', 'GPT-4', 'GPT-4o', 'Claude-3.5']
        },
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'MMLU': '#3498db', 'GSM8K': '#e74c3c', 'HumanEval': '#2ecc71'}
    markers = {'MMLU': 'o', 'GSM8K': 's', 'HumanEval': '^'}

    for bench_name, data in timeline_data.items():
        ax.plot(data['years'], data['scores'], '-', color=colors[bench_name],
                marker=markers[bench_name], markersize=8, linewidth=2, label=bench_name)
        # Add model labels for MMLU
        if bench_name == 'MMLU':
            for year, score, model in zip(data['years'], data['scores'], data['models']):
                ax.annotate(model, (year, score), fontsize=7,
                           xytext=(5, 8), textcoords='offset points', color='gray')

    # Saturation zone
    ax.axhspan(90, 100, alpha=0.1, color='red', label='Saturation Zone')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.3)

    # Human performance reference
    ax.axhline(y=89.8, color='green', linestyle=':', alpha=0.5, label='Expert Human (MMLU)')

    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Benchmark Saturation Timeline', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(20, 102)
    ax.set_xlim(2019.5, 2025)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'benchmark_saturation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated benchmark saturation timeline")


def generate_comparison_tables():
    """Generate comparison tables as figures for the paper."""
    import matplotlib.pyplot as plt

    # Survey section comparison table
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    headers = ['Dimension', 'Key Methods', 'Math Basis', 'Strengths', 'Limitations']
    table_data = [
        ['Knowledge', 'MMLU, ARC', 'Accuracy, IRT', 'Standardized', 'Saturating'],
        ['Code', 'HumanEval, MBPP', 'Pass@k', 'Functional correctness', 'Limited scope'],
        ['Math', 'GSM8K, MATH', 'Exact match', 'Clear ground truth', 'Format-dependent'],
        ['Generation', 'BLEU, BERTScore', 'n-gram, embedding', 'Automated', 'Poor correlation'],
        ['Instruction', 'MT-Bench', 'Bradley-Terry', 'Human-aligned', 'Expensive'],
        ['Safety', 'TruthfulQA, BBQ', 'Classification', 'Coverage', 'Adversarial fragility'],
        ['Calibration', 'ECE, Brier', 'Probability theory', 'Principled', 'Metric-dependent'],
        ['Robustness', 'GCG, AutoDAN', 'Optimization', 'Real attacks', 'Non-exhaustive'],
        ['Contamination', 'MIN-K%, n-gram', 'Hypothesis testing', 'Automated', 'Evasion-prone'],
        ['Preference', 'Arena, RLHF', 'Elo, BT model', 'Real users', 'Bias, cost'],
        ['RAG/Compound', 'RAGAS, ARES', 'Component metrics', 'End-to-end', 'Non-composable'],
    ]

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='left', colWidths=[0.12, 0.18, 0.18, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#3498db')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#f0f8ff' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title('Comparison of LLM Evaluation Dimensions', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'comparison_table.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Novel frameworks comparison table
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    headers2 = ['Framework', 'Math Foundation', 'Addresses Gap', 'Key Property', 'Complexity']
    table_data2 = [
        ['Topological Drift', 'Persistent Homology', 'Stability', 'Stability theorem', 'O(n^3)'],
        ['Sheaf Composition', 'Sheaf Cohomology', 'Composability', 'Gluing axiom', 'O(n^2 m)'],
        ['Info-Geometric', 'Fisher Metric', 'Drift detection', 'Riemannian invariance', 'O(n^2)'],
        ['Failure Landscape', 'TDA + Clustering', 'Failure modes', 'Multi-scale analysis', 'O(n^3)'],
        ['Spectral Detection', 'Random Matrix Theory', 'Contamination', 'MP distribution', 'O(n^3)'],
    ]

    table2 = ax.table(cellText=table_data2, colLabels=headers2, loc='center',
                      cellLoc='left', colWidths=[0.18, 0.20, 0.18, 0.22, 0.12])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2.0)

    for j in range(len(headers2)):
        table2[0, j].set_facecolor('#e74c3c')
        table2[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data2) + 1):
        color = '#fff0f0' if i % 2 == 0 else 'white'
        for j in range(len(headers2)):
            table2[i, j].set_facecolor(color)

    ax.set_title('Novel Mathematical Frameworks for LLM Evaluation', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'frameworks_table.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated comparison tables")


###############################################################################
# Main Execution
###############################################################################

def main():
    print("=" * 70)
    print("Proof-of-Concept Implementations: Mathematically Grounded LLM Evaluation")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading MMLU data...")
    test_data, subjects, subject_counts = load_mmlu_data()
    print(f"  Loaded {len(test_data)} questions across {len(subjects)} subjects")

    # Simulate model performance profiles
    print("\n[2/7] Generating model performance profiles...")
    performances = simulate_model_performance(subjects)
    print(f"  Generated profiles for {len(performances)} models")

    # Run all 5 frameworks
    print("\n[3/7] Framework 1: Topological Drift Detection...")
    results_f1 = framework1_topological_drift(performances, subjects)

    print("\n[4/7] Framework 2: Sheaf-Theoretic Compositional Evaluation...")
    results_f2 = framework2_sheaf_evaluation(performances, subjects)

    print("\n[5/7] Framework 3: Information-Geometric Evaluation Manifolds...")
    results_f3 = framework3_information_geometry(performances, subjects)

    print("\n[6/7] Framework 4: Failure Mode Landscape Analysis...")
    results_f4 = framework4_failure_modes(performances, subjects, subject_counts)

    print("\n[7/7] Framework 5: Spectral Contamination Detection...")
    results_f5 = framework5_spectral_analysis(performances, subjects)

    # Generate additional figures
    print("\nGenerating additional figures...")
    generate_taxonomy_figure(subjects)
    generate_benchmark_saturation_timeline()
    generate_comparison_tables()

    # Save all results
    all_results = {
        'framework_1_topological_drift': results_f1,
        'framework_2_sheaf_evaluation': {
            'n_categories': results_f2['n_categories'],
            'category_sizes': results_f2['category_sizes'],
            'graph_nodes': results_f2['graph_nodes'],
            'graph_edges': results_f2['graph_edges'],
            'sheaf_sections': results_f2['sheaf_sections'],
        },
        'framework_3_information_geometry': {
            'max_fisher_rao': results_f3['max_fisher_rao'],
            'mean_fisher_rao': results_f3['mean_fisher_rao'],
            'fisher_traces': results_f3['fisher_traces'],
        },
        'framework_4_failure_modes': results_f4,
        'framework_5_spectral_analysis': results_f5,
        'metadata': {
            'n_subjects': len(subjects),
            'n_models': len(performances),
            'subjects': subjects,
            'model_names': list(performances.keys()),
            'seed': SEED,
        }
    }

    with open(os.path.join(RESULTS_DIR, 'framework_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save performance data
    perf_data = {name: perf.tolist() for name, perf in performances.items()}
    with open(os.path.join(RESULTS_DIR, 'model_performances.json'), 'w') as f:
        json.dump({'performances': perf_data, 'subjects': subjects}, f, indent=2)

    print("\n" + "=" * 70)
    print("All frameworks executed. Results saved to results/")
    print(f"Figures saved to figures/")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
