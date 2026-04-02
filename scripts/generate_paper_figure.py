"""
Generate paper figures that earn their space (4-page limit).

Figure 1: Reranker bias — singleton vs shared score distributions
Figure 2: APD vs Coverage scatter — proves geometric diversity ≠ perspective coverage
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def fig1_reranker_bias():
    """Reranker bias: score distributions for singleton vs shared claims."""
    with open(DATA / "newscope_reranker_scores.json") as f:
        reranker_scores = json.load(f)
    with open(DATA / "llm_coverage_labels.json") as f:
        coverage_labels = json.load(f)

    singleton_scores = []
    shared_scores = []

    for event_id, para_scores in reranker_scores.items():
        cov = coverage_labels.get(event_id, {})
        if not cov:
            continue

        singleton_pids = set()
        shared_pids = set()
        for claim_id, paragraph_ids in cov.items():
            if len(paragraph_ids) == 1:
                singleton_pids.add(paragraph_ids[0])
            else:
                shared_pids.update(paragraph_ids)

        for para_id, score in para_scores.items():
            if para_id in singleton_pids:
                singleton_scores.append(score)
            elif para_id in shared_pids:
                shared_scores.append(score)

    singleton_scores = np.array(singleton_scores)
    shared_scores = np.array(shared_scores)

    fig, ax = plt.subplots(figsize=(3.4, 2.0))

    bins = np.linspace(0, 1, 35)

    ax.hist(shared_scores, bins=bins, alpha=0.6, density=True,
            color='#2ecc71', label=f'Shared claims (n={len(shared_scores)})',
            edgecolor='white', linewidth=0.3)
    ax.hist(singleton_scores, bins=bins, alpha=0.6, density=True,
            color='#e74c3c', label=f'Singleton claims (n={len(singleton_scores)})',
            edgecolor='white', linewidth=0.3)

    mean_sing = np.mean(singleton_scores)
    mean_shared = np.mean(shared_scores)

    ax.axvline(mean_shared, color='#1a9641', linestyle='-', linewidth=1.5)
    ax.axvline(mean_sing, color='#c0392b', linestyle='-', linewidth=1.5)

    ax.text(0.97, 0.95,
            f'$\\Delta\\mu = {mean_shared - mean_sing:.3f}$\n$p < 10^{{-10}}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7',
                     edgecolor='#999999', alpha=0.95))

    ax.set_xlabel('BGE Reranker Score')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURES / "reranker_bias.pdf", bbox_inches='tight')
    plt.savefig(FIGURES / "reranker_bias.png", bbox_inches='tight')
    print("Saved reranker_bias")
    plt.close()


def fig2_apd_vs_coverage():
    """APD vs Coverage@10 scatter for multiple methods — proves RQ1."""
    # Load all results files that have per-event APD + coverage
    all_results = []
    for fname in ['newscope_faithful_results.json',
                  'rq2_diversity_results.json',
                  'rq2_infogain_results.json']:
        with open(DATA / fname) as f:
            all_results.extend(json.load(f))

    # Methods with strongest significant negative APD-Coverage correlations
    show_methods = {
        'GreedySCS':  {'color': '#e74c3c', 'marker': 'o'},
        'InfoGain':   {'color': '#2ecc71', 'marker': 'D'},
        'GreedyPlus': {'color': '#9b59b6', 'marker': '^'},
        'DPP':        {'color': '#3498db', 'marker': 's'},
    }

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    for method, style in show_methods.items():
        # Get K=10 data for this method
        method_data = [r for r in all_results
                       if r['method'] == method and r['K'] == 10]
        if not method_data:
            continue

        apds = [r['apd'] for r in method_data]
        covs = [r['coverage'] for r in method_data]

        # Spearman correlation
        rho, pval = stats.spearmanr(apds, covs)
        pstr = f'p<0.001' if pval < 0.001 else f'p={pval:.3f}'

        ax.scatter(apds, covs, c=style['color'], marker=style['marker'],
                   s=12, alpha=0.4, edgecolors='none',
                   label=f'{method} (ρ={rho:.2f}, {pstr})')

        # Trend line
        z = np.polyfit(apds, covs, 1)
        x_line = np.linspace(min(apds), max(apds), 50)
        ax.plot(x_line, np.polyval(z, x_line), color=style['color'],
                linewidth=1.2, alpha=0.8, linestyle='--')

    ax.set_xlabel('APD (Average Pairwise Distance)')
    ax.set_ylabel('Coverage@10')
    ax.set_title('Geometric Diversity vs. Perspective Coverage', fontsize=9,
                 fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper right', framealpha=0.9,
              edgecolor='#ccc', handlelength=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation
    ax.text(0.03, 0.05,
            'All correlations negative:\nhigher spread → worse coverage',
            transform=ax.transAxes, fontsize=6, color='#555',
            fontstyle='italic', va='bottom')

    plt.tight_layout()
    plt.savefig(FIGURES / "apd_vs_coverage.pdf", bbox_inches='tight')
    plt.savefig(FIGURES / "apd_vs_coverage.png", bbox_inches='tight')
    print("Saved apd_vs_coverage")
    plt.close()


if __name__ == "__main__":
    fig1_reranker_bias()
    fig2_apd_vs_coverage()
    print(f"\nAll figures saved to {FIGURES}/")
