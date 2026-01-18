"""
Numerical Angular Entropy Test-Time Adaptation (NAE-TTA)
v2: Numerically stable implementation with proper subspace constraint

Key improvements over v1.2:
1. Float32 NAE computation for numerical stability
2. Safe normalization to prevent NaN
3. Hard masking for numerical subspace constraint
4. NaN/Inf detection and graceful handling
5. Academic-quality visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Local imports
from nae_core import (
    angular_entropy_stable,
    angular_entropy_supervised,
    safe_normalize,
    check_numerical_health,
    compute_gradient_stats
)
from numerical_utils import (
    get_numerical_tokens,
    apply_numerical_mask,
    numerical_softmax
)
from visualization import visualize_all, setup_academic_style

# Prompt template for arithmetic
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the numerical answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def configure_trainable_params(model,
                               target: str = 'mlp',
                               n_layers: int = 2) -> List[torch.nn.Parameter]:
    """
    Configure which parameters are trainable during TTA.

    Args:
        model: The language model
        target: Which modules to train ('mlp', 'ln', 'mlp+ln')
        n_layers: Number of layers from the end to train ('all' for all layers)

    Returns:
        List of trainable parameters
    """
    # Disable all gradients first
    model.requires_grad_(False)

    total_layers = len(model.model.layers)

    if n_layers == 'all':
        layer_indices = range(total_layers)
    else:
        layer_indices = range(max(0, total_layers - n_layers), total_layers)

    params = []
    for idx in layer_indices:
        layer = model.model.layers[idx]

        if target in ['mlp', 'mlp+ln']:
            for p in layer.mlp.parameters():
                p.requires_grad = True
                params.append(p)

        if target in ['ln', 'mlp+ln']:
            for name, p in layer.named_parameters():
                if 'norm' in name.lower():
                    p.requires_grad = True
                    params.append(p)

    return params


def tta_single_token(model,
                     input_ids: torch.Tensor,
                     num_idx: List[int],
                     target_token_idx: int,
                     tokenizer,
                     args) -> Tuple[int, Dict]:
    """
    Perform TTA optimization for a single token position.

    Key features:
    - Numerically stable NAE computation
    - Hard masking for numerical subspace
    - NaN detection and recovery
    - Comprehensive metrics tracking

    Args:
        model: Language model
        input_ids: Input token IDs
        num_idx: Numerical token indices
        target_token_idx: Target token index in vocabulary
        tokenizer: Tokenizer for decoding
        args: Experiment arguments

    Returns:
        pred_idx: Final predicted token index
        metrics: Dictionary with optimization metrics
    """
    embed_layer = model.get_input_embeddings()
    params = configure_trainable_params(model, args.update_target, args.num_layers)

    # Store original parameters for restoration
    orig_state = {id(p): p.data.clone() for p in params}

    model.train()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    # Metrics tracking
    metrics = {
        'nae': [],
        'target_prob': [],
        'target_rank': [],
        'top10_probs': [],
        'top10_tokens_snapshots': [],
        'grad_norms': [],
        'health_status': []
    }

    # Convert numerical indices to tensor once
    num_idx_tensor = torch.tensor(num_idx, device=input_ids.device, dtype=torch.long)

    for step in range(args.steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Apply numerical mask for subspace constraint
        # This ensures optimization only affects numerical token predictions
        masked_logits = apply_numerical_mask(logits, num_idx)

        # Compute NAE loss on masked logits
        # The mask ensures gradients only flow through numerical tokens
        loss = angular_entropy_stable(
            masked_logits,
            num_idx,
            embed_layer,
            use_float32=True,
            eps=1e-4
        )

        # Check for NaN/Inf before backward
        if torch.isnan(loss) or torch.isinf(loss):
            metrics['health_status'].append('loss_nan')
            # Skip this step but continue optimization
            continue

        # Backward pass
        loss.backward()

        # Check gradient health
        is_healthy, health_msg = check_numerical_health(loss, params)
        metrics['health_status'].append(health_msg)

        if not is_healthy:
            # Clear bad gradients and skip step
            optimizer.zero_grad()
            continue

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        metrics['grad_norms'].append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

        # Update parameters
        optimizer.step()

        if args.scheduler:
            scheduler.step()

        # Record metrics (with fresh forward pass for accuracy)
        with torch.no_grad():
            # Get logits again after parameter update
            fresh_logits = model(input_ids).logits[0, -1, :]

            # Apply mask for probability computation
            masked_fresh = apply_numerical_mask(fresh_logits, num_idx)

            # Compute probabilities over all tokens (for ranking)
            full_probs = F.softmax(fresh_logits.float(), dim=-1)

            # Compute probabilities over numerical tokens only
            num_probs = F.softmax(masked_fresh[num_idx_tensor].float(), dim=-1)

            # Target probability (in full vocab)
            target_prob = full_probs[target_token_idx].item()

            # Target rank (in full vocab)
            target_rank = int((full_probs > full_probs[target_token_idx]).sum().item() + 1)

            # Top-10 tokens (from full vocab, but masked will be -inf)
            top10_probs, top10_indices = torch.topk(full_probs, 10)

            metrics['nae'].append(loss.item())
            metrics['target_prob'].append(target_prob)
            metrics['target_rank'].append(target_rank)
            metrics['top10_probs'].append(top10_probs.cpu().tolist())

            # Detailed snapshots every 5 steps
            if step % 5 == 0:
                tok_info = []
                for idx, prob in zip(top10_indices.cpu().tolist(), top10_probs.cpu().tolist()):
                    try:
                        tok_str = tokenizer.decode([idx])
                    except:
                        tok_str = f'<{idx}>'
                    tok_info.append({'token': tok_str, 'prob': float(prob), 'idx': idx})
                metrics['top10_tokens_snapshots'].append({
                    'step': step,
                    'tokens': tok_info
                })

    # Final prediction
    model.eval()
    with torch.no_grad():
        final_logits = model(input_ids).logits[0, -1, :]

        # Apply numerical mask for final prediction
        masked_final = apply_numerical_mask(final_logits, num_idx)

        # Predict from masked logits (guaranteed numerical output)
        pred_idx = torch.argmax(masked_final).item()

    # Restore original parameters
    for p in params:
        p.data.copy_(orig_state[id(p)])
        p.requires_grad = False

    return pred_idx, metrics


def run_experiment(args) -> Dict:
    """
    Run the full NAE-TTA experiment.

    Args:
        args: Experiment arguments

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {args.exp_name}")
    print(f"{'='*70}")

    # Load data
    with open(args.data_path) as f:
        data = json.load(f)

    # Load model
    model_path = args.models[0]
    model_name = model_path.split('/')[-1]
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Get numerical tokens
    num_idx, num_tok = get_numerical_tokens(tokenizer)
    print(f"Numerical tokens: {len(num_idx)}/{tokenizer.vocab_size}")
    print(f"Sample numerical tokens: {num_tok[:20]}")

    # Main experiment loop
    results = []
    total_correct_before = 0
    total_correct_after = 0
    total_tokens = 0

    for item in tqdm(data[:args.num_samples], desc="Processing samples"):
        question = item['question']
        answer = item['answer']

        # Tokenize ground truth answer
        gt_tokens = tokenizer.encode(str(answer), add_special_tokens=False)

        # Prepare input
        prompt = PROMPT_TEMPLATE.format(question=question)
        input_encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

        token_results = []

        for pos in range(len(gt_tokens)):
            # Build input up to current position
            if pos == 0:
                input_ids = input_encoding['input_ids']
            else:
                prefix_tokens = torch.tensor([gt_tokens[:pos]], device=model.device)
                input_ids = torch.cat([input_encoding['input_ids'], prefix_tokens], dim=1)

            target_idx = gt_tokens[pos]

            # Baseline prediction (before TTA)
            model.eval()
            with torch.no_grad():
                baseline_logits = model(input_ids).logits[0, -1, :]
                # For baseline, also apply mask to ensure fair comparison
                masked_baseline = apply_numerical_mask(baseline_logits, num_idx)
                pred_before = torch.argmax(masked_baseline).item()

            # TTA with NAE
            pred_after, metrics = tta_single_token(
                model, input_ids, num_idx, target_idx, tokenizer, args
            )

            correct_before = pred_before == target_idx
            correct_after = pred_after == target_idx

            total_correct_before += correct_before
            total_correct_after += correct_after
            total_tokens += 1

            token_results.append({
                'position': pos,
                'target': target_idx,
                'target_token': tokenizer.decode([target_idx]),
                'pred_before': tokenizer.decode([pred_before]),
                'pred_after': tokenizer.decode([pred_after]),
                'correct_before': correct_before,
                'correct_after': correct_after,
                'metrics': metrics
            })

        results.append({
            'question': question,
            'answer': answer,
            'tokens': token_results
        })

    # Compute final accuracies
    acc_before = total_correct_before / total_tokens if total_tokens > 0 else 0
    acc_after = total_correct_after / total_tokens if total_tokens > 0 else 0

    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")
    print(f"Total tokens: {total_tokens}")
    print(f"Accuracy Before TTA: {acc_before:.4f} ({total_correct_before}/{total_tokens})")
    print(f"Accuracy After TTA:  {acc_after:.4f} ({total_correct_after}/{total_tokens})")
    print(f"Improvement: {acc_after - acc_before:+.4f}")
    print(f"{'='*70}")

    # Prepare output
    output = {
        'summary': {
            'model': model_name,
            'accuracy_before': float(acc_before),
            'accuracy_after': float(acc_after),
            'improvement': float(acc_after - acc_before),
            'total_tokens': total_tokens,
            'config': {
                'models': args.models,
                'data_path': args.data_path,
                'num_samples': args.num_samples,
                'update_target': args.update_target,
                'num_layers': args.num_layers,
                'steps': args.steps,
                'lr': args.lr,
                'scheduler': args.scheduler
            }
        },
        'results': results
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.exp_name}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_all(output, args.output_dir, args.exp_name, tokenizer)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(
        description='NAE Test-Time Adaptation v2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data
    parser.add_argument('--models', nargs='+', default=[
        "/home/jinsk/Models/Llama-3.1-8B-Instruct"
    ], help='Model path(s)')
    parser.add_argument('--data_path', default='../data/addition_problems_dataset(1-50)(1).json',
                       help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to process')

    # TTA configuration
    parser.add_argument('--update_target', choices=['mlp', 'ln', 'mlp+ln'], default='mlp',
                       help='Which layers to update')
    parser.add_argument('--num_layers', default=2,
                       help='Number of layers to update (from end), or "all"')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--scheduler', action='store_true',
                       help='Use cosine annealing scheduler')

    # Output
    parser.add_argument('--output_dir', default='results_nae',
                       help='Output directory')
    parser.add_argument('--exp_name', required=True,
                       help='Experiment name')

    args = parser.parse_args()

    # Process num_layers
    if args.num_layers != 'all':
        args.num_layers = int(args.num_layers)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
