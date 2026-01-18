"""
ANE Visualization Module v4
Publication-quality academic visualizations

Key improvements:
1. Full step display (30 columns, not 7)
2. Numerical annotations in ALL cells
3. No rank cap (shows true rank values)
4. Academic color schemes (blue/green)
5. Fixed label overlap
6. Only flipped cases in evolution
7. New comparison statistics chart
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List
import os

# Academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Academic color schemes
def create_prob_cmap():
    """Blue gradient for probability (higher = darker blue)"""
    colors = ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#084594']
    return LinearSegmentedColormap.from_list('prob_blue', colors, N=256)

def create_rank_cmap():
    """Green gradient for rank (lower rank = darker green = better)"""
    colors = ['#00441b', '#238b45', '#74c476', '#c7e9c0', '#f7fcf5']
    return LinearSegmentedColormap.from_list('rank_green', colors, N=256)


def visualize_prob_heatmap(results: Dict, output_dir: str, exp_name: str, tokens_per_page: int = 60):
    """Probability heatmap with numerical annotations."""

    all_probs = []
    token_labels = []

    for result in results.get('results', []):
        for tok in result.get('tokens', []):
            probs = tok.get('metrics', {}).get('target_prob', [])
            if probs:
                all_probs.append(probs)
                token_labels.append(tok.get('target_token', '?'))

    if not all_probs:
        return

    n_tokens = len(all_probs)
    n_steps = max(len(p) for p in all_probs)
    n_pages = (n_tokens + tokens_per_page - 1) // tokens_per_page

    cmap = create_prob_cmap()

    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(start + tokens_per_page, n_tokens)
        page_probs = all_probs[start:end]
        page_labels = token_labels[start:end]

        # Build matrix
        prob_mat = np.zeros((len(page_probs), n_steps))
        for i, probs in enumerate(page_probs):
            prob_mat[i, :len(probs)] = probs
            if len(probs) < n_steps:
                prob_mat[i, len(probs):] = probs[-1] if probs else 0

        n_rows = len(page_probs)
        fig_height = max(8, n_rows * 0.18)
        fig_width = max(12, n_steps * 0.35)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Heatmap
        im = ax.imshow(prob_mat, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Add numerical annotations (ALWAYS)
        for i in range(prob_mat.shape[0]):
            for j in range(prob_mat.shape[1]):
                val = prob_mat[i, j]
                color = 'white' if val > 0.5 else 'black'
                text = f'{val:.2f}' if val > 0.01 else ''
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=5, color=color, fontweight='medium')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Target Probability', fontsize=9)

        # Labels
        ax.set_xlabel('Optimization Step', fontweight='medium')
        ax.set_ylabel('Token Index', fontweight='medium')
        ax.set_xticks(range(0, n_steps, max(1, n_steps//10)))
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'{i+start}: {l}' for i, l in enumerate(page_labels)], fontsize=6)

        page_suffix = f'_p{page+1}' if n_pages > 1 else ''
        ax.set_title(f'Target Probability Evolution - {exp_name}{page_suffix}', fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.12)
        plt.savefig(f'{output_dir}/{exp_name}_prob_heatmap{page_suffix}.png', facecolor='white')
        plt.close()
        print(f"  Saved: {exp_name}_prob_heatmap{page_suffix}.png")


def visualize_rank_heatmap(results: Dict, output_dir: str, exp_name: str, tokens_per_page: int = 60):
    """Rank heatmap with NO cap (shows true ranks)."""

    all_ranks = []
    token_labels = []

    for result in results.get('results', []):
        for tok in result.get('tokens', []):
            ranks = tok.get('metrics', {}).get('target_rank', [])
            if ranks:
                all_ranks.append(ranks)
                token_labels.append(tok.get('target_token', '?'))

    if not all_ranks:
        return

    n_tokens = len(all_ranks)
    n_steps = max(len(r) for r in all_ranks)
    n_pages = (n_tokens + tokens_per_page - 1) // tokens_per_page

    # NO CAP - use true max rank
    max_rank = max(max(r) for r in all_ranks if r)

    cmap = create_rank_cmap()

    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(start + tokens_per_page, n_tokens)
        page_ranks = all_ranks[start:end]
        page_labels = token_labels[start:end]

        rank_mat = np.zeros((len(page_ranks), n_steps))
        for i, ranks in enumerate(page_ranks):
            rank_mat[i, :len(ranks)] = ranks
            if len(ranks) < n_steps:
                rank_mat[i, len(ranks):] = ranks[-1] if ranks else max_rank

        n_rows = len(page_ranks)
        fig_height = max(8, n_rows * 0.18)
        fig_width = max(12, n_steps * 0.35)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(rank_mat, cmap=cmap, aspect='auto', vmin=1, vmax=max_rank)

        # Numerical annotations
        for i in range(rank_mat.shape[0]):
            for j in range(rank_mat.shape[1]):
                val = int(rank_mat[i, j])
                color = 'white' if val < max_rank * 0.3 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       fontsize=5, color=color, fontweight='medium')

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f'Target Rank (1-{max_rank})', fontsize=9)

        ax.set_xlabel('Optimization Step', fontweight='medium')
        ax.set_ylabel('Token Index', fontweight='medium')
        ax.set_xticks(range(0, n_steps, max(1, n_steps//10)))
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'{i+start}: {l}' for i, l in enumerate(page_labels)], fontsize=6)

        page_suffix = f'_p{page+1}' if n_pages > 1 else ''
        ax.set_title(f'Target Rank Evolution - {exp_name}{page_suffix}', fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.12)
        plt.savefig(f'{output_dir}/{exp_name}_rank_heatmap{page_suffix}.png', facecolor='white')
        plt.close()
        print(f"  Saved: {exp_name}_rank_heatmap{page_suffix}.png")


def visualize_flipped_cases(results: Dict, output_dir: str, exp_name: str, n_cases: int = 9):
    """Visualize ONLY flipped cases (wrong->correct) with rich detail."""

    flipped = results.get('flipped_cases', [])

    # Also find from results
    if not flipped:
        for r in results.get('results', []):
            for tok in r.get('tokens', []):
                if not tok.get('correct_before') and tok.get('correct_after'):
                    flipped.append({
                        **tok,
                        'question': r['question'],
                        'answer': r['answer']
                    })

    if not flipped:
        print("  No flipped cases to visualize")
        return

    n_actual = min(len(flipped), n_cases)
    n_cols = 3
    n_rows = (n_actual + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_actual == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, case in enumerate(flipped[:n_actual]):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        metrics = case.get('metrics', {})
        probs = metrics.get('target_prob', [])
        ranks = metrics.get('target_rank', [])

        if not probs:
            ax.set_visible(False)
            continue

        steps = range(len(probs))

        # Probability (left axis)
        color_prob = '#2171b5'
        ax.plot(steps, probs, color=color_prob, linewidth=2.5, marker='o', markersize=3, label='Probability')
        ax.fill_between(steps, probs, alpha=0.2, color=color_prob)
        ax.set_ylabel('Probability', color=color_prob, fontweight='medium')
        ax.tick_params(axis='y', labelcolor=color_prob)
        ax.set_ylim(0, 1.05)

        # Rank (right axis)
        if ranks:
            ax2 = ax.twinx()
            color_rank = '#238b45'
            ax2.plot(steps, ranks, color=color_rank, linewidth=2, linestyle='--', marker='s', markersize=2, label='Rank')
            ax2.set_ylabel('Rank', color=color_rank, fontweight='medium')
            ax2.tick_params(axis='y', labelcolor=color_rank)
            ax2.invert_yaxis()

        ax.set_xlabel('Step')

        # Title with before/after info
        before = case.get('pred_before', '?')
        after = case.get('pred_after', '?')
        target = case.get('target_token', '?')
        ax.set_title(f"'{before}' \u2192 '{after}' (target: '{target}')",
                    fontweight='bold', color='#2ca25f', fontsize=10)

        # Add prob change annotation
        if len(probs) >= 2:
            delta = probs[-1] - probs[0]
            ax.annotate(f'\u0394prob: {delta:+.3f}', xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=8, ha='left', va='top', color=color_prob)

    # Hide empty
    for idx in range(n_actual, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Flipped Cases: Wrong \u2192 Correct ({exp_name})', fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_flipped_cases.png', facecolor='white')
    plt.close()
    print(f"  Saved: {exp_name}_flipped_cases.png")


def visualize_summary(results: Dict, output_dir: str, exp_name: str):
    """Summary statistics visualization."""

    summary = results.get('summary', {})
    if not summary:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (1) Accuracy comparison
    ax1 = axes[0]
    acc_before = summary.get('accuracy_before', 0) * 100
    acc_after = summary.get('accuracy_after', 0) * 100

    x = ['Before TTA', 'After TTA']
    y = [acc_before, acc_after]
    colors = ['#bdbdbd', '#2171b5']

    bars = ax1.bar(x, y, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, y):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')

    improvement = summary.get('improvement', 0) * 100
    if improvement > 0:
        ax1.annotate(f'+{improvement:.2f}%', xy=(0.5, max(y) * 0.5),
                    fontsize=14, fontweight='bold', color='#2ca25f', ha='center')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, max(y) * 1.3)
    ax1.set_title('Accuracy Comparison', fontweight='bold')

    # (2) Key metrics
    ax2 = axes[1]
    ax2.axis('off')

    config = summary.get('config', {})
    info = [
        f"Model: {summary.get('model', '?')}",
        f"Total Tokens: {summary.get('total_tokens', 0)}",
        f"Flipped Cases: {summary.get('flipped_count', 0)}",
        f"Time: {summary.get('elapsed_time_min', 0):.1f} min",
        f"",
        f"Config:",
        f"  Target: {config.get('update_target', '?')}",
        f"  Layers: {config.get('num_layers', '?')}",
        f"  Steps: {config.get('steps', '?')}",
        f"  LR: {config.get('lr', '?')}"
    ]

    ax2.text(0.1, 0.9, '\n'.join(info), transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'ANE-TTA Results: {exp_name}', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_summary.png', facecolor='white')
    plt.close()
    print(f"  Saved: {exp_name}_summary.png")


def visualize_comparison(results_dict: Dict[str, Dict], output_dir: str):
    """Compare multiple experiments with statistical charts."""

    if len(results_dict) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    names = list(results_dict.keys())
    summaries = [r.get('summary', {}) for r in results_dict.values()]

    # (1) Accuracy comparison
    ax1 = axes[0, 0]
    before = [s.get('accuracy_before', 0) * 100 for s in summaries]
    after = [s.get('accuracy_after', 0) * 100 for s in summaries]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, before, width, label='Before TTA', color='#bdbdbd')
    bars2 = ax1.bar(x + width/2, after, width, label='After TTA', color='#2171b5')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax1.legend()
    ax1.set_title('Accuracy Comparison', fontweight='bold')

    # (2) Improvement
    ax2 = axes[0, 1]
    improvements = [s.get('improvement', 0) * 100 for s in summaries]
    colors = ['#2ca25f' if imp > 0 else '#de2d26' for imp in improvements]

    bars = ax2.barh(names, improvements, color=colors)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('Performance Improvement', fontweight='bold')

    for bar, imp in zip(bars, improvements):
        ax2.annotate(f'{imp:+.2f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', va='center', fontsize=8)

    # (3) Flipped cases
    ax3 = axes[1, 0]
    flipped = [s.get('flipped_count', 0) for s in summaries]
    ax3.bar(names, flipped, color='#756bb1')
    ax3.set_ylabel('Count')
    ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax3.set_title('Flipped Cases (Wrong\u2192Correct)', fontweight='bold')

    # (4) Time
    ax4 = axes[1, 1]
    times = [s.get('elapsed_time_min', 0) for s in summaries]
    ax4.bar(names, times, color='#fd8d3c')
    ax4.set_ylabel('Minutes')
    ax4.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax4.set_title('Runtime', fontweight='bold')

    plt.suptitle('ANE-TTA Experiment Comparison', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_stats.png', facecolor='white')
    plt.close()
    print(f"  Saved: comparison_stats.png")


def visualize_all(results: Dict, output_dir: str, exp_name: str, tokenizer):
    """Generate all visualizations."""

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating visualizations for {exp_name}...")

    # 1. Probability heatmap
    visualize_prob_heatmap(results, output_dir, exp_name)

    # 2. Rank heatmap
    visualize_rank_heatmap(results, output_dir, exp_name)

    # 3. Flipped cases (only wrong->correct)
    visualize_flipped_cases(results, output_dir, exp_name)

    # 4. Summary
    visualize_summary(results, output_dir, exp_name)

    print(f"All visualizations saved to {output_dir}/")
