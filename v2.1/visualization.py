"""
Academic-Quality Visualization for NAE Experiments
v2.1: Publication-ready figures for ICML submission

Features:
- Red-Green diverging palette for rank heatmap
- Professional blue sequential palette for probability heatmap
- Numerical Subspace Evolution visualization (replaces NAE dynamics)
- Enhanced Top-10 token evolution with target highlighting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Academic Styling Configuration
# ============================================================

def setup_academic_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.framealpha': 0.95,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.linewidth': 0.8,
    })

# Call setup on import
setup_academic_style()


# ============================================================
# Custom Academic Color Palettes
# ============================================================

def create_rank_cmap():
    """
    Create Red-Green diverging colormap for rank heatmap.
    Low rank (good) = Green, High rank (bad) = Red
    Based on ColorBrewer RdYlGn palette (colorblind-safe)
    """
    colors = [
        '#1a9850',  # Dark green (rank 1, excellent)
        '#91cf60',  # Light green
        '#d9ef8b',  # Yellow-green
        '#fee08b',  # Light yellow
        '#fc8d59',  # Orange
        '#d73027',  # Dark red (high rank, poor)
    ]
    return LinearSegmentedColormap.from_list('RdYlGn_rank', colors, N=256)


def create_prob_cmap():
    """
    Create professional blue sequential colormap for probability heatmap.
    Based on ColorBrewer Blues palette
    """
    colors = [
        '#f7fbff',  # Almost white (prob ≈ 0)
        '#deebf7',  # Very light blue
        '#c6dbef',  # Light blue
        '#9ecae1',  # Medium-light blue
        '#6baed6',  # Medium blue
        '#4292c6',  # Medium-dark blue
        '#2171b5',  # Dark blue
        '#084594',  # Very dark blue (prob ≈ 1)
    ]
    return LinearSegmentedColormap.from_list('Blues_prob', colors, N=256)


# Academic color constants
COLORS = {
    'target': '#d62728',      # Red for target token
    'correct': '#2ca02c',     # Green for correct
    'incorrect': '#d62728',   # Red for incorrect
    'neutral': '#7f7f7f',     # Gray
    'primary': '#1f77b4',     # Blue primary
    'secondary': '#ff7f0e',   # Orange secondary
}


# ============================================================
# Visualization Functions
# ============================================================

def visualize_subspace_evolution(results: List[Dict],
                                  output_path: str,
                                  sample_idx: int = 0,
                                  steps_to_show: List[int] = None):
    """
    Visualize numerical subspace probability evolution across optimization steps.

    Shows how probability mass concentrates during NAE minimization.
    Replaces the old NAE dynamics plot.

    Args:
        results: List of experiment results
        output_path: Path to save figure
        sample_idx: Which sample to visualize
        steps_to_show: List of step indices to show (default: [0, 5, 10, 15, 19])
    """
    if not results or not results[sample_idx]['tokens']:
        return

    if steps_to_show is None:
        steps_to_show = [0, 4, 9, 14, 19]  # 5 panels

    token_data = results[sample_idx]['tokens'][0]
    target_token = token_data.get('target_token', '?')

    # Get snapshots
    snapshots = token_data['metrics'].get('top10_tokens_snapshots', [])
    if not snapshots:
        return

    n_panels = min(len(steps_to_show), len(snapshots))
    if n_panels < 2:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    # Color palette for bars
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.8, 10))
    target_color = COLORS['target']

    for ax_idx, step_idx in enumerate(steps_to_show[:n_panels]):
        ax = axes[ax_idx]

        # Find closest snapshot
        snap_idx = min(step_idx // 5, len(snapshots) - 1)
        snap = snapshots[snap_idx]
        tokens = snap['tokens'][:10]

        # Extract data
        token_labels = [t['token'][:6] if len(t['token']) > 6 else t['token'] for t in tokens]
        probs = [t['prob'] for t in tokens]

        # Identify target
        is_target = [target_token in t['token'] for t in tokens]
        colors = [target_color if it else bar_colors[i] for i, it in enumerate(is_target)]

        # Create bar chart
        bars = ax.barh(range(len(tokens)), probs, color=colors, edgecolor='white', linewidth=0.5)

        # Styling
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([f"'{lbl}'" for lbl in token_labels], fontsize=8)
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_title(f'Step {snap["step"]}', fontsize=11, fontweight='bold')
        ax.invert_yaxis()

        # Add probability labels
        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.2f}', va='center', fontsize=7)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=target_color, edgecolor='white', label=f"Target: '{target_token}'"),
        mpatches.Patch(facecolor=bar_colors[5], edgecolor='white', label='Other tokens')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_prob_heatmap(results: List[Dict],
                           output_path: str,
                           max_tokens: int = 50,
                           annotate: bool = True):
    """
    Visualize target token probability evolution as heatmap.

    Features:
    - Professional blue colormap
    - Probability annotations (2 decimal places)
    - Sample boundaries marked
    """
    # Collect probability trajectories
    all_probs = []
    sample_boundaries = [0]

    for r in results:
        for t in r['tokens']:
            prob_curve = t['metrics']['target_prob']
            # Replace NaN with 0 for visualization
            prob_curve = [0.0 if (isinstance(p, float) and np.isnan(p)) else p for p in prob_curve]
            all_probs.append(prob_curve)
        sample_boundaries.append(len(all_probs))

    if not all_probs:
        return

    n_tokens = min(max_tokens, len(all_probs))
    n_steps = len(all_probs[0]) if all_probs else 0

    if n_steps == 0:
        return

    prob_mat = np.array([p[:n_steps] for p in all_probs[:n_tokens]])

    # Create figure
    fig_height = max(4, n_tokens * 0.15)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Use custom probability colormap
    cmap = create_prob_cmap()

    # Determine annotation settings
    do_annotate = annotate and n_tokens <= 35 and n_steps <= 25
    annot_fontsize = max(5, min(7, 180 // max(n_tokens, n_steps)))

    if do_annotate:
        # Annotate with 2 decimal places
        annot_matrix = np.vectorize(lambda x: f'{x:.2f}' if x > 0.005 else '')(prob_mat)
        sns.heatmap(prob_mat, cmap=cmap, vmin=0, vmax=1,
                    annot=annot_matrix, fmt='',
                    cbar_kws={'label': 'Probability', 'shrink': 0.6},
                    linewidths=0.3, linecolor='white',
                    ax=ax, annot_kws={'fontsize': annot_fontsize, 'color': '#333333'})
    else:
        sns.heatmap(prob_mat, cmap=cmap, vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability', 'shrink': 0.6},
                    linewidths=0, ax=ax)

    # Mark sample boundaries
    for b in sample_boundaries[1:-1]:
        if b < n_tokens:
            ax.axhline(b, color='#333333', lw=1.2, linestyle='-', alpha=0.7)

    ax.set_xlabel('Optimization Step', fontsize=11)
    ax.set_ylabel('Token Index', fontsize=11)
    ax.set_title('Target Token Probability Evolution', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_rank_heatmap(results: List[Dict],
                           output_path: str,
                           max_tokens: int = 50,
                           annotate: bool = True):
    """
    Visualize target token rank evolution as heatmap.

    Features:
    - Red-Green diverging colormap (Green=good/low rank, Red=bad/high rank)
    - Rank annotations
    - Sample boundaries marked
    """
    # Collect rank trajectories
    all_ranks = []
    sample_boundaries = [0]

    for r in results:
        for t in r['tokens']:
            rank_curve = t['metrics']['target_rank']
            all_ranks.append(rank_curve)
        sample_boundaries.append(len(all_ranks))

    if not all_ranks:
        return

    n_tokens = min(max_tokens, len(all_ranks))
    n_steps = len(all_ranks[0]) if all_ranks else 0

    if n_steps == 0:
        return

    rank_mat = np.array([r[:n_steps] for r in all_ranks[:n_tokens]])

    # Create figure
    fig_height = max(4, n_tokens * 0.15)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Use custom Red-Green colormap
    cmap = create_rank_cmap()

    # Determine max rank for normalization (cap at 50 for better color distribution)
    max_rank = min(50, np.nanmax(rank_mat))

    # Annotation settings
    do_annotate = annotate and n_tokens <= 35 and n_steps <= 25
    annot_fontsize = max(5, min(7, 180 // max(n_tokens, n_steps)))

    if do_annotate:
        sns.heatmap(rank_mat, cmap=cmap, vmin=1, vmax=max_rank,
                    annot=True, fmt='.0f',
                    cbar_kws={'label': 'Rank', 'shrink': 0.6},
                    linewidths=0.3, linecolor='white',
                    ax=ax, annot_kws={'fontsize': annot_fontsize, 'color': '#333333'})
    else:
        sns.heatmap(rank_mat, cmap=cmap, vmin=1, vmax=max_rank,
                    cbar_kws={'label': 'Rank', 'shrink': 0.6},
                    linewidths=0, ax=ax)

    # Mark sample boundaries
    for b in sample_boundaries[1:-1]:
        if b < n_tokens:
            ax.axhline(b, color='#333333', lw=1.2, linestyle='-', alpha=0.7)

    ax.set_xlabel('Optimization Step', fontsize=11)
    ax.set_ylabel('Token Index', fontsize=11)
    ax.set_title('Target Token Rank Evolution', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def select_representative_cases(results: List[Dict], n_cases: int = 3) -> List[int]:
    """
    Select representative cases based on initial rank diversity.

    Returns indices of:
    - Hard case: High initial rank (>30)
    - Medium case: Medium initial rank (10-30)
    - Easy case: Low initial rank (<=5)
    """
    cases = []

    # Get initial ranks for first token of each sample
    init_ranks = []
    for i, r in enumerate(results):
        if r['tokens']:
            ranks = r['tokens'][0]['metrics']['target_rank']
            if ranks and len(ranks) > 0:
                rank = ranks[0]
                if not (isinstance(rank, float) and np.isnan(rank)):
                    init_ranks.append((i, rank))

    if not init_ranks:
        return list(range(min(n_cases, len(results))))

    init_ranks.sort(key=lambda x: x[1])

    # Find hard case (high rank)
    for idx, rank in reversed(init_ranks):
        if rank > 20:
            cases.append(idx)
            break

    # Find medium case
    for idx, rank in init_ranks:
        if 5 < rank <= 20 and idx not in cases:
            cases.append(idx)
            break

    # Find easy case (low rank)
    for idx, rank in init_ranks:
        if rank <= 5 and idx not in cases:
            cases.append(idx)
            break

    # Fill remaining slots
    for idx, _ in init_ranks:
        if len(cases) >= n_cases:
            break
        if idx not in cases:
            cases.append(idx)

    return cases[:n_cases]


def visualize_top10_evolution(results: List[Dict],
                              output_path: str,
                              tokenizer=None,
                              n_cases: int = 3):
    """
    Visualize top-10 token probability evolution for representative cases.

    Features:
    - Decoded token labels in legend
    - Target token highlighted with red dashed line + star markers
    - Clear case descriptions (Easy/Medium/Hard)
    """
    case_indices = select_representative_cases(results, n_cases)

    if not case_indices:
        return

    n_cases_actual = len(case_indices)
    fig, axes = plt.subplots(1, n_cases_actual, figsize=(5.5 * n_cases_actual, 4.5))
    if n_cases_actual == 1:
        axes = [axes]

    # Professional color palette (tab10 subset)
    colors = plt.cm.tab10(np.linspace(0, 0.9, 10))

    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        result = results[case_idx]
        token_data = result['tokens'][0]

        top10_probs = np.array(token_data['metrics']['top10_probs'])
        target_prob = token_data['metrics']['target_prob']
        target_token_str = token_data.get('target_token', '?')
        init_rank = token_data['metrics']['target_rank'][0]

        n_steps = len(target_prob)

        # Determine case difficulty
        if init_rank <= 5:
            difficulty = "Easy"
            diff_color = COLORS['correct']
        elif init_rank <= 20:
            difficulty = "Medium"
            diff_color = COLORS['secondary']
        else:
            difficulty = "Hard"
            diff_color = COLORS['incorrect']

        # Get token strings from snapshots
        token_strs = []
        if 'top10_tokens_snapshots' in token_data['metrics']:
            snapshots = token_data['metrics']['top10_tokens_snapshots']
            if snapshots:
                token_strs = [t['token'] for t in snapshots[0]['tokens']]

        if not token_strs:
            token_strs = [f'T{i}' for i in range(10)]

        # Plot each of top-10 token trajectories
        legend_handles = []
        legend_labels = []

        for rank in range(min(10, top10_probs.shape[1])):
            line, = ax.plot(range(n_steps), top10_probs[:, rank],
                           color=colors[rank], lw=1.5, alpha=0.7)

            if rank < len(token_strs):
                tok_label = token_strs[rank]
                if len(tok_label) > 8:
                    tok_label = tok_label[:6] + '..'
                legend_handles.append(line)
                legend_labels.append(f"'{tok_label}'")

        # Highlight target token with distinct style
        target_line, = ax.plot(range(n_steps), target_prob,
                               color=COLORS['target'],
                               lw=2.5, linestyle='--', alpha=0.95, zorder=10)

        # Add star markers at key points
        marker_steps = [0, n_steps//2, n_steps-1]
        for ms in marker_steps:
            if ms < len(target_prob):
                ax.scatter(ms, target_prob[ms], color=COLORS['target'], s=100,
                          marker='*', zorder=11, edgecolors='white', linewidths=0.5)

        legend_handles.append(target_line)
        legend_labels.append(f"Target: '{target_token_str}'")

        # Title with difficulty
        ax.set_title(f"{difficulty} Case (Init Rank: {int(init_rank)})",
                    fontsize=11, color=diff_color, fontweight='bold')

        ax.set_xlabel('Optimization Step', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.5, n_steps - 0.5)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Legend
        ax.legend(legend_handles, legend_labels,
                  fontsize=7, loc='upper right',
                  frameon=True, framealpha=0.95,
                  ncol=2 if len(legend_handles) > 6 else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_accuracy_comparison(before_acc: float,
                                  after_acc: float,
                                  output_path: str,
                                  exp_name: str = ""):
    """
    Create accuracy comparison bar chart.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    x = ['Before TTA', 'After TTA']
    heights = [before_acc * 100, after_acc * 100]
    bar_colors = [COLORS['neutral'], COLORS['primary']]

    bars = ax.bar(x, heights, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.6)

    # Add value labels on bars
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Accuracy Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, max(heights) * 1.25)

    # Improvement annotation
    improvement = (after_acc - before_acc) * 100
    sign = '+' if improvement >= 0 else ''
    color = COLORS['correct'] if improvement > 0 else COLORS['incorrect']
    ax.text(0.5, 0.92, f'Change: {sign}{improvement:.2f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_all(data: Dict,
                  output_dir: str,
                  exp_name: str,
                  tokenizer=None):
    """
    Generate all visualization figures for an experiment.

    Args:
        data: Full experiment data dict with 'summary' and 'results'
        output_dir: Output directory path
        exp_name: Experiment name for file naming
        tokenizer: Optional tokenizer for decoding tokens
    """
    results = data['results']
    summary = data['summary']

    print(f"\nGenerating visualizations for: {exp_name}")

    # 1. Numerical Subspace Evolution (replaces NAE dynamics)
    visualize_subspace_evolution(
        results,
        f'{output_dir}/{exp_name}_subspace_evolution.png'
    )

    # 2. Probability Heatmap
    visualize_prob_heatmap(
        results,
        f'{output_dir}/{exp_name}_prob_heatmap.png'
    )

    # 3. Rank Heatmap
    visualize_rank_heatmap(
        results,
        f'{output_dir}/{exp_name}_rank_heatmap.png'
    )

    # 4. Top-10 Case Studies
    visualize_top10_evolution(
        results,
        f'{output_dir}/{exp_name}_top10_cases.png',
        tokenizer=tokenizer
    )

    # 5. Accuracy Comparison
    visualize_accuracy_comparison(
        summary['accuracy_before'],
        summary['accuracy_after'],
        f'{output_dir}/{exp_name}_accuracy.png',
        exp_name=exp_name
    )

    print(f"All visualizations completed for {exp_name}\n")


def create_summary_figure(all_experiments: Dict[str, Dict],
                          output_path: str):
    """
    Create a summary figure comparing multiple experiments.
    """
    n_exp = len(all_experiments)
    if n_exp == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    exp_names = list(all_experiments.keys())
    before_accs = [all_experiments[n]['summary']['accuracy_before'] * 100 for n in exp_names]
    after_accs = [all_experiments[n]['summary']['accuracy_after'] * 100 for n in exp_names]

    x = np.arange(n_exp)
    width = 0.35

    # Accuracy comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, before_accs, width, label='Before TTA',
                   color=COLORS['neutral'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, after_accs, width, label='After TTA',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Improvement chart
    ax = axes[1]
    improvements = [a - b for a, b in zip(after_accs, before_accs)]
    bar_colors = [COLORS['correct'] if imp >= 0 else COLORS['incorrect'] for imp in improvements]
    ax.bar(exp_names, improvements, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Accuracy Change', fontsize=12, fontweight='bold')
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved summary figure: {output_path}")
