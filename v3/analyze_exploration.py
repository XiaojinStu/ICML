"""
Analyze parameter exploration results and recommend optimal configuration.
"""

import json
import os
import glob
from typing import Dict, List

def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON results from directory."""
    results = {}
    for path in glob.glob(f"{results_dir}/*.json"):
        exp_name = os.path.basename(path).replace('.json', '')
        with open(path) as f:
            results[exp_name] = json.load(f)
    return results

def analyze_exploration(results_dir: str = "results_exploration"):
    """Analyze exploration results and print summary."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print("=" * 70)
    print("ANE Parameter Exploration Analysis")
    print("=" * 70)

    # Group by exploration phase
    phases = {
        'target': [],
        'lr': [],
        'opt': [],
        'layers': [],
        'sched': [],
        'steps': []
    }

    for exp_name, data in results.items():
        summary = data.get('summary', {})
        for phase in phases:
            if f'explore_{phase}' in exp_name:
                param_value = exp_name.split(f'explore_{phase}_')[1] if f'explore_{phase}_' in exp_name else exp_name
                phases[phase].append({
                    'param': param_value,
                    'before': summary.get('accuracy_before', 0),
                    'after': summary.get('accuracy_after', 0),
                    'improvement': summary.get('improvement', 0),
                    'time': summary.get('elapsed_time_min', 0),
                    'flipped': summary.get('flipped_count', 0)
                })

    # Print analysis for each phase
    for phase_name, phase_results in phases.items():
        if not phase_results:
            continue

        print(f"\n--- {phase_name.upper()} Analysis ---")
        print(f"{'Parameter':<15} {'Before':>10} {'After':>10} {'Improve':>10} {'Flipped':>8} {'Time':>8}")
        print("-" * 65)

        # Sort by improvement
        phase_results.sort(key=lambda x: x['improvement'], reverse=True)

        for r in phase_results:
            print(f"{r['param']:<15} {r['before']*100:>9.2f}% {r['after']*100:>9.2f}% "
                  f"{r['improvement']*100:>+9.2f}% {r['flipped']:>8} {r['time']:>7.1f}m")

        # Best configuration
        best = phase_results[0]
        print(f"\n  Best: {best['param']} (improvement: {best['improvement']*100:+.2f}%)")

    # Overall recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    best_config = {}
    for phase_name, phase_results in phases.items():
        if phase_results:
            phase_results.sort(key=lambda x: x['improvement'], reverse=True)
            best_config[phase_name] = phase_results[0]['param']
            print(f"  {phase_name}: {phase_results[0]['param']}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results_exploration')
    args = parser.parse_args()

    analyze_exploration(args.results_dir)
