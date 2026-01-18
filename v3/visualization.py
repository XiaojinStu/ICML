"""
ANE Visualization Module v3
Academic-quality visualizations for publication

Key improvements over v2:
1. Paginated heatmaps to show ALL tokens
2. Numerical annotations in cells with contrast-aware colors
3. Prioritize flipped cases (wrong->correct) for case studies
4. Enhanced color schemes (ColorBrewer-based)
5. Removed accuracy plot (low information density)
6. More cases in subspace evolution (6 instead of 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os

# Academic style configuration
def setup_academic_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
    })

# Color definitions
COLORS = {
    'primary': '#2171b5',      # Deep blue
    'secondary': '#6baed6',    # Medium blue
    'correct': '#2ca25f',      # Green for correct
    'incorrect': '#de2d26',    # Red for incorrect
    'neutral': '#636363',      # Gray
    'target': '#e31a1c',       # Bright red for target
    'background': '#f7f7f7',   # Light gray background
}

# Academic color palettes
def create_prob_cmap():
    """Create probability colormap (blue sequential)."""
    colors = [
        '#f7fbff',  # Almost white (prob ~ 0)
        '#d0e1f2',  # Very light blue
        '#94c4df',  # Light blue
        '#4a98c9',  # Medium blue
        '#1764ab',  # Dark blue
        '#0a4480',  # Very dark blue (prob ~ 1)
    ]
    return LinearSegmentedColormap.from_list('prob_blues', colors, N=256)

def create_rank_cmap():
    """Create rank colormap (green-yellow-red diverging)."""
    colors = [
        '#276419',  # Dark green (rank 1, best)
        '#7fbc41',  # Light green
        '#d9f0a3',  # Yellow-green
        '#fee08b',  # Yellow
        '#f46d43',  # Orange
        '#a50026',  # Dark red (high rank, worst)
    ]
    return LinearSegmentedColormap.from_list('rank_rdylgn', colors, N=256)


def select_visualization_cases(results: List[Dict], n_cases: int = 6) -> List[Dict]:
    """
    Select interesting cases for visualization.

    Priority order:
    1. Flipped cases (wrong -> correct) - most interesting for paper
    2. Stable correct cases (correct -> correct)
    3. Challenging cases (high initial rank that improved)
    """
    all_tokens = []
    for r in results:
        for tok in r.get('tokens', []):
            tok['question'] = r['question']
            tok['answer'] = r['answer']
            all_tokens.append(tok)

    cases = []

    # Priority 1: Wrong -> Correct (flipped) - most valuable
    flipped = [t for t in all_tokens if not t.get('correct_before', True) and t.get('correct_after', False)]
    cases.extend(flipped[:min(3, len(flipped))])

    # Priority 2: Correct -> Correct (stable success)
    if len(cases) < n_cases:
        stable = [t for t in all_tokens if t.get('correct_before', False) and t.get('correct_after', False)]
        remaining = n_cases - len(cases)
        cases.extend(stable[:min(remaining, len(stable))])

    # Priority 3: High initial rank that improved
    if len(cases) < n_cases:
        improving = []
        for t in all_tokens:
            metrics = t.get('metrics', {})
            ranks = metrics.get('target_rank', [])
            if len(ranks) >= 2 and ranks[0] > 5 and ranks[-1] < ranks[0]:
                t['_rank_improvement'] = ranks[0] - ranks[-1]
                improving.append(t)
        improving.sort(key=lambda x: x.get('_rank_improvement', 0), reverse=True)
        remaining = n_cases - len(cases)
        cases.extend(improving[:min(remaining, len(improving))])

    # Fill remaining with any tokens that have metrics
    if len(cases) < n_cases:
        remaining = n_cases - len(cases)
        existing_ids = {id(c) for c in cases}
        other = [t for t in all_tokens if id(t) not in existing_ids and t.get('metrics')]
        cases.extend(other[:remaining])

    return cases[:n_cases]


def get_contrast_color(value: float, is_prob: bool = True) -> str:
    """Get text color that contrasts with background based on value."""
    if is_prob:
        # For probability: dark text on light bg, light text on dark bg
        return '#FFFFFF' if value > 0.5 else '#333333'
    else:
        # For rank: assuming normalized 0-1, lower is better (green)
        return '#FFFFFF' if value > 0.6 else '#333333'


def visualize_prob_heatmap_paginated(results: Dict,
                                     output_dir: str,
                                     exp_name: str,
                                     tokens_per_page: int = 80):
    """
    Generate paginated probability heatmaps to show ALL tokens.

    Creates multiple pages if needed, each showing up to tokens_per_page tokens.
    """
    setup_academic_style()

    # Extract all probability trajectories
    all_probs = []
    token_labels = []
    sample_boundaries = []
    current_idx = 0

    for result in results.get('results', []):
        sample_start = current_idx
        for tok in result.get('tokens', []):
            metrics = tok.get('metrics', {})
            probs = metrics.get('target_prob', [])
            if probs:
                all_probs.append(probs)
                token_labels.append(f"{tok.get('target_token', '?')}")
                current_idx += 1
        if current_idx > sample_start:
            sample_boundaries.append((sample_start, current_idx - 1))

    if not all_probs:
        return

    n_tokens = len(all_probs)
    n_steps = max(len(p) for p in all_probs)
    n_pages = (n_tokens + tokens_per_page - 1) // tokens_per_page

    cmap = create_prob_cmap()

    for page in range(n_pages):
        start_idx = page * tokens_per_page
        end_idx = min(start_idx + tokens_per_page, n_tokens)
        page_probs = all_probs[start_idx:end_idx]
        page_labels = token_labels[start_idx:end_idx]

        # Pad to same length
        prob_mat = np.zeros((len(page_probs), n_steps))
        for i, probs in enumerate(page_probs):
            prob_mat[i, :len(probs)] = probs
            if len(probs) < n_steps:
                prob_mat[i, len(probs):] = probs[-1] if probs else 0

        # Figure sizing
        n_page_tokens = len(page_probs)
        fig_height = max(6, n_page_tokens * 0.12)
        fig_width = max(10, n_steps * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Determine annotation settings
        do_annotate = n_page_tokens <= 50 and n_steps <= 35
        annot_fontsize = max(5, min(7, 350 / (n_page_tokens * n_steps)))

        # Create heatmap
        if do_annotate:
            # Create annotation matrix with smart formatting
            annot_matrix = np.empty_like(prob_mat, dtype=object)
            for i in range(prob_mat.shape[0]):
                for j in range(prob_mat.shape[1]):
                    val = prob_mat[i, j]
                    if val > 0.005:
                        annot_matrix[i, j] = f'{val:.2f}'
                    else:
                        annot_matrix[i, j] = ''

            sns.heatmap(prob_mat, cmap=cmap, vmin=0, vmax=1,
                       annot=annot_matrix, fmt='',
                       cbar_kws={'label': 'Target Probability', 'shrink': 0.6},
                       linewidths=0.3, linecolor='white',
                       ax=ax, annot_kws={'fontsize': annot_fontsize})
        else:
            sns.heatmap(prob_mat, cmap=cmap, vmin=0, vmax=1,
                       cbar_kws={'label': 'Target Probability', 'shrink': 0.6},
                       linewidths=0.1, linecolor='white',
                       ax=ax)

        # Draw sample boundaries
        for s_start, s_end in sample_boundaries:
            if start_idx <= s_start < end_idx:
                ax.axhline(y=s_start - start_idx, color='#333333', linewidth=1, linestyle='-')
            if start_idx < s_end < end_idx:
                ax.axhline(y=s_end - start_idx + 1, color='#333333', linewidth=1, linestyle='-')

        # Labels
        ax.set_xlabel('Optimization Step', fontweight='medium')
        ax.set_ylabel(f'Token Index ({start_idx+1}-{end_idx})', fontweight='medium')
        ax.set_yticks(np.arange(n_page_tokens) + 0.5)
        ax.set_yticklabels([f'{i+start_idx}: {l}' for i, l in enumerate(page_labels)], fontsize=7)

        page_suffix = f'_page{page+1}' if n_pages > 1 else ''
        title = f'Target Probability Evolution ({exp_name})'
        if n_pages > 1:
            title += f' - Page {page+1}/{n_pages}'
        ax.set_title(title, fontweight='bold', pad=10)

        plt.tight_layout()
        output_path = f'{output_dir}/{exp_name}_prob_heatmap{page_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_path}")


def visualize_rank_heatmap_paginated(results: Dict,
                                     output_dir: str,
                                     exp_name: str,
                                     tokens_per_page: int = 80):
    """
    Generate paginated rank heatmaps to show ALL tokens.
    """
    setup_academic_style()

    # Extract all rank trajectories
    all_ranks = []
    token_labels = []
    sample_boundaries = []
    current_idx = 0

    for result in results.get('results', []):
        sample_start = current_idx
        for tok in result.get('tokens', []):
            metrics = tok.get('metrics', {})
            ranks = metrics.get('target_rank', [])
            if ranks:
                all_ranks.append(ranks)
                token_labels.append(f"{tok.get('target_token', '?')}")
                current_idx += 1
        if current_idx > sample_start:
            sample_boundaries.append((sample_start, current_idx - 1))

    if not all_ranks:
        return

    n_tokens = len(all_ranks)
    n_steps = max(len(r) for r in all_ranks)
    n_pages = (n_tokens + tokens_per_page - 1) // tokens_per_page

    # Determine max rank for normalization
    max_rank = max(max(r) for r in all_ranks if r)
    max_rank = min(100, max_rank)  # Cap at 100 for visualization

    cmap = create_rank_cmap()

    for page in range(n_pages):
        start_idx = page * tokens_per_page
        end_idx = min(start_idx + tokens_per_page, n_tokens)
        page_ranks = all_ranks[start_idx:end_idx]
        page_labels = token_labels[start_idx:end_idx]

        # Pad to same length
        rank_mat = np.zeros((len(page_ranks), n_steps))
        for i, ranks in enumerate(page_ranks):
            rank_mat[i, :len(ranks)] = ranks
            if len(ranks) < n_steps:
                rank_mat[i, len(ranks):] = ranks[-1] if ranks else max_rank

        # Cap values
        rank_mat = np.clip(rank_mat, 1, max_rank)

        # Figure sizing
        n_page_tokens = len(page_ranks)
        fig_height = max(6, n_page_tokens * 0.12)
        fig_width = max(10, n_steps * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Determine annotation settings
        do_annotate = n_page_tokens <= 50 and n_steps <= 35
        annot_fontsize = max(5, min(7, 350 / (n_page_tokens * n_steps)))

        if do_annotate:
            sns.heatmap(rank_mat, cmap=cmap, vmin=1, vmax=max_rank,
                       annot=True, fmt='.0f',
                       cbar_kws={'label': 'Target Rank', 'shrink': 0.6},
                       linewidths=0.3, linecolor='white',
                       ax=ax, annot_kws={'fontsize': annot_fontsize, 'color': '#333333'})
        else:
            sns.heatmap(rank_mat, cmap=cmap, vmin=1, vmax=max_rank,
                       cbar_kws={'label': 'Target Rank', 'shrink': 0.6},
                       linewidths=0.1, linecolor='white',
                       ax=ax)

        # Highlight cells where rank improved significantly (from previous step)
        for i in range(rank_mat.shape[0]):
            for j in range(1, rank_mat.shape[1]):
                if rank_mat[i, j] < rank_mat[i, j-1] * 0.7:  # 30%+ improvement
                    rect = mpatches.Rectangle((j-0.05, i-0.05), 1.1, 1.1,
                                             linewidth=1.5,
                                             edgecolor=COLORS['correct'],
                                             facecolor='none', alpha=0.6)
                    ax.add_patch(rect)

        # Draw sample boundaries
        for s_start, s_end in sample_boundaries:
            if start_idx <= s_start < end_idx:
                ax.axhline(y=s_start - start_idx, color='#333333', linewidth=1)
            if start_idx < s_end < end_idx:
                ax.axhline(y=s_end - start_idx + 1, color='#333333', linewidth=1)

        # Labels
        ax.set_xlabel('Optimization Step', fontweight='medium')
        ax.set_ylabel(f'Token Index ({start_idx+1}-{end_idx})', fontweight='medium')
        ax.set_yticks(np.arange(n_page_tokens) + 0.5)
        ax.set_yticklabels([f'{i+start_idx}: {l}' for i, l in enumerate(page_labels)], fontsize=7)

        page_suffix = f'_page{page+1}' if n_pages > 1 else ''
        title = f'Target Rank Evolution ({exp_name})'
        if n_pages > 1:
            title += f' - Page {page+1}/{n_pages}'
        ax.set_title(title, fontweight='bold', pad=10)

        plt.tight_layout()
        output_path = f'{output_dir}/{exp_name}_rank_heatmap{page_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_path}")


def visualize_subspace_evolution(results: Dict,
                                 output_dir: str,
                                 exp_name: str,
                                 n_cases: int = 6):
    """
    Visualize numerical subspace probability evolution.

    Shows how probability mass concentrates during optimization.
    Prioritizes flipped cases (wrong->correct).
    """
    setup_academic_style()

    cases = select_visualization_cases(results.get('results', []), n_cases)
    if not cases:
        return

    n_actual = len(cases)
    n_cols = min(3, n_actual)
    n_rows = (n_actual + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_actual == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, case in enumerate(cases):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        metrics = case.get('metrics', {})
        probs = metrics.get('target_prob', [])
        ranks = metrics.get('target_rank', [])

        if not probs:
            ax.set_visible(False)
            continue

        steps = range(len(probs))

        # Plot probability and rank evolution
        ax2 = ax.twinx()

        # Probability line
        line1, = ax.plot(steps, probs, color=COLORS['primary'], linewidth=2,
                        marker='o', markersize=4, label='Probability')
        ax.fill_between(steps, probs, alpha=0.2, color=COLORS['primary'])

        # Rank line (inverted scale for intuition: lower rank = higher on plot)
        if ranks:
            line2, = ax2.plot(steps, ranks, color=COLORS['incorrect'], linewidth=2,
                             linestyle='--', marker='s', markersize=3, label='Rank')
            ax2.set_ylabel('Rank', color=COLORS['incorrect'])
            ax2.tick_params(axis='y', labelcolor=COLORS['incorrect'])
            ax2.invert_yaxis()  # Lower rank at top

        ax.set_xlabel('Step')
        ax.set_ylabel('Probability', color=COLORS['primary'])
        ax.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax.set_ylim(0, 1.05)

        # Title with case info
        is_flipped = not case.get('correct_before', True) and case.get('correct_after', False)
        status = "FLIPPED" if is_flipped else ("Correct" if case.get('correct_after') else "Wrong")
        status_color = COLORS['correct'] if is_flipped or case.get('correct_after') else COLORS['incorrect']

        title = f"Token: '{case.get('target_token', '?')}' [{status}]"
        ax.set_title(title, fontweight='bold', color=status_color)

        # Legend
        lines = [line1]
        labels = ['Probability']
        if ranks:
            lines.append(line2)
            labels.append('Rank')
        ax.legend(lines, labels, loc='center right', fontsize=8)

    # Hide empty subplots
    for idx in range(n_actual, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Numerical Subspace Evolution ({exp_name})', fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = f'{output_dir}/{exp_name}_subspace_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_top10_evolution(results: Dict,
                              output_dir: str,
                              exp_name: str,
                              tokenizer,
                              n_cases: int = 6):
    """
    Visualize top-10 token probability evolution.

    Shows how the ranking of top tokens changes during optimization.
    Prioritizes flipped cases.
    """
    setup_academic_style()

    cases = select_visualization_cases(results.get('results', []), n_cases)
    if not cases:
        return

    n_actual = min(len(cases), n_cases)
    n_cols = min(3, n_actual)
    n_rows = (n_actual + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_actual == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, case in enumerate(cases):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        metrics = case.get('metrics', {})
        top10_probs = metrics.get('top10_probs', [])
        snapshots = metrics.get('top10_tokens_snapshots', [])

        if not top10_probs or len(top10_probs) < 2:
            ax.set_visible(False)
            continue

        # Get token labels from first snapshot
        token_labels = []
        if snapshots:
            for tok_info in snapshots[0].get('tokens', [])[:10]:
                token_labels.append(tok_info.get('token', '?'))
        else:
            token_labels = [f'T{i}' for i in range(10)]

        # Plot each token's probability trajectory
        probs_array = np.array(top10_probs)  # (n_steps, 10)
        steps = range(len(probs_array))

        target_token = case.get('target_token', '')

        for i in range(min(10, probs_array.shape[1])):
            is_target = token_labels[i].strip() == target_token.strip() if i < len(token_labels) else False

            if is_target:
                ax.plot(steps, probs_array[:, i], color=COLORS['target'],
                       linewidth=2.5, marker='*', markersize=8,
                       label=f"'{token_labels[i]}' (TARGET)", zorder=10)
            else:
                ax.plot(steps, probs_array[:, i], color=colors[i],
                       linewidth=1.2, alpha=0.7,
                       label=f"'{token_labels[i]}'")

        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.05)

        # Title
        is_flipped = not case.get('correct_before', True) and case.get('correct_after', False)
        status = "FLIPPED" if is_flipped else ("Correct" if case.get('correct_after') else "Wrong")

        title = f"Top-10 Evolution: '{target_token}' [{status}]"
        ax.set_title(title, fontweight='bold')

        # Legend (outside plot)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7,
                 frameon=True, fancybox=True)

    # Hide empty subplots
    for idx in range(n_actual, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Top-10 Token Probability Evolution ({exp_name})', fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = f'{output_dir}/{exp_name}_top10_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_flipped_cases(results: Dict,
                            output_dir: str,
                            exp_name: str):
    """
    Create dedicated visualization for flipped cases (wrong -> correct).

    This is the most valuable visualization for the paper.
    """
    setup_academic_style()

    flipped = results.get('flipped_cases', [])
    if not flipped:
        # Try to find flipped cases from results
        for result in results.get('results', []):
            for tok in result.get('tokens', []):
                if not tok.get('correct_before', True) and tok.get('correct_after', False):
                    flipped.append({
                        **tok,
                        'question': result['question'],
                        'answer': result['answer']
                    })

    if not flipped:
        print("  No flipped cases found")
        return

    n_cases = min(len(flipped), 6)
    n_cols = min(3, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cases == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, case in enumerate(flipped[:n_cases]):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        metrics = case.get('metrics', {})
        probs = metrics.get('target_prob', [])
        ranks = metrics.get('target_rank', [])

        if not probs:
            ax.set_visible(False)
            continue

        steps = range(len(probs))

        # Probability trajectory
        ax.plot(steps, probs, color=COLORS['correct'], linewidth=2.5,
               marker='o', markersize=5, label='Target Prob')
        ax.fill_between(steps, probs, alpha=0.3, color=COLORS['correct'])

        # Mark the flip point (where prediction changed)
        # Assume flip happens when prob crosses 0.5 or becomes max
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')

        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Target Probability')
        ax.set_ylim(0, 1.05)

        # Rich title
        target = case.get('target_token', '?')
        before = case.get('pred_before', '?')
        after = case.get('pred_after', '?')
        title = f"'{before}' -> '{after}' (target: '{target}')"
        ax.set_title(title, fontweight='bold', color=COLORS['correct'])

        ax.legend(loc='lower right', fontsize=8)

    # Hide empty subplots
    for idx in range(n_cases, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Flipped Cases: Wrong -> Correct ({exp_name})', fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = f'{output_dir}/{exp_name}_flipped_cases.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_summary_stats(results: Dict,
                            output_dir: str,
                            exp_name: str):
    """
    Create summary statistics visualization.

    Shows key metrics in a clean, publication-ready format.
    """
    setup_academic_style()

    summary = results.get('summary', {})
    if not summary:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create table data
    acc_before = summary.get('accuracy_before', 0) * 100
    acc_after = summary.get('accuracy_after', 0) * 100
    improvement = summary.get('improvement', 0) * 100
    total_tokens = summary.get('total_tokens', 0)
    flipped = summary.get('flipped_count', 0)
    time_min = summary.get('elapsed_time_min', 0)

    config = summary.get('config', {})

    # Bar chart for accuracy comparison
    categories = ['Before TTA', 'After TTA']
    values = [acc_before, acc_after]
    colors_bar = [COLORS['neutral'], COLORS['correct']]

    bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold')

    # Add improvement annotation
    if improvement > 0:
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(1.5, max(values) * 0.9),
                   fontsize=12, fontweight='bold',
                   color=COLORS['correct'],
                   ha='center')

    ax.set_ylabel('Accuracy (%)', fontweight='medium')
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title(f'ANE-TTA Results: {exp_name}', fontweight='bold', pad=15)

    # Add text box with details
    info_text = (f"Total tokens: {total_tokens}\n"
                f"Flipped (wrong->correct): {flipped}\n"
                f"Time: {time_min:.1f} min\n"
                f"Steps: {config.get('steps', '?')}\n"
                f"Target: {config.get('update_target', '?')}")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=props)

    plt.tight_layout()

    output_path = f'{output_dir}/{exp_name}_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_all(results: Dict,
                  output_dir: str,
                  exp_name: str,
                  tokenizer):
    """
    Generate all visualizations for an experiment.

    v3 improvements:
    - Paginated heatmaps (all tokens visible)
    - Flipped cases visualization
    - Summary statistics
    - Removed accuracy bar plot (replaced with summary)
    """
    setup_academic_style()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations for {exp_name}...")

    # 1. Paginated probability heatmap
    visualize_prob_heatmap_paginated(results, output_dir, exp_name)

    # 2. Paginated rank heatmap
    visualize_rank_heatmap_paginated(results, output_dir, exp_name)

    # 3. Subspace evolution (6 cases, prioritize flipped)
    visualize_subspace_evolution(results, output_dir, exp_name, n_cases=6)

    # 4. Top-10 evolution (6 cases)
    visualize_top10_evolution(results, output_dir, exp_name, tokenizer, n_cases=6)

    # 5. Dedicated flipped cases visualization
    visualize_flipped_cases(results, output_dir, exp_name)

    # 6. Summary statistics
    visualize_summary_stats(results, output_dir, exp_name)

    print(f"All visualizations saved to {output_dir}/")
