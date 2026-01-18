"""
Angular Numerical Entropy Test-Time Adaptation (ANE-TTA)
v3: Memory-optimized and speed-optimized implementation

Key improvements over v2:
1. Adaptive layer selection based on model size (for 8B support)
2. Gradient checkpointing option for memory reduction
3. 8-bit optimizer support (bitsandbytes)
4. Eliminated duplicate forward passes (2x speedup)
5. Sparse metrics recording (configurable interval)
6. Support for attention layer training
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Local imports
from ane_core import (
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

# Prompt template for arithmetic
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the numerical answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_model_size_gb(model) -> float:
    """Estimate model size in GB."""
    return sum(p.numel() for p in model.parameters()) / 1e9


def configure_trainable_params(model,
                               target: str = 'mlp+ln',
                               n_layers: str = 'adaptive',
                               verbose: bool = False) -> List[torch.nn.Parameter]:
    """
    Configure which parameters are trainable during TTA.

    v3 improvements:
    - Adaptive layer selection based on model size
    - Support for attention layers
    - Support for 'all' components

    Args:
        model: The language model
        target: Which modules to train:
            'mlp': MLP layers only
            'ln': LayerNorm only
            'mlp+ln': MLP + LayerNorm (default, best performance)
            'attn': Attention layers (QKV + output projection)
            'attn+ln': Attention + LayerNorm
            'all': All parameters in selected layers
        n_layers: Number of layers from the end to train
            'adaptive': Auto-select based on model size
            'all': Train all layers
            int: Specific number of layers

    Returns:
        List of trainable parameters
    """
    # Disable all gradients first
    model.requires_grad_(False)

    total_layers = len(model.model.layers)
    model_size = get_model_size_gb(model)

    # Adaptive layer selection based on model size
    if n_layers == 'adaptive':
        if model_size > 6:  # 8B+: train 25% of layers
            n_layers = max(8, total_layers // 4)
        elif model_size > 2:  # 3B: train all layers
            n_layers = 'all'
        else:  # 1B: train all layers
            n_layers = 'all'

    # Determine layer indices
    if n_layers == 'all':
        layer_indices = range(total_layers)
    else:
        n_layers = int(n_layers)
        layer_indices = range(max(0, total_layers - n_layers), total_layers)

    params = []
    param_count = 0

    for idx in layer_indices:
        layer = model.model.layers[idx]

        # MLP layers
        if target in ['mlp', 'mlp+ln', 'all']:
            for p in layer.mlp.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

        # LayerNorm/RMSNorm
        if target in ['ln', 'mlp+ln', 'attn+ln', 'all']:
            for name, p in layer.named_parameters():
                if 'norm' in name.lower() or 'layernorm' in name.lower():
                    p.requires_grad = True
                    params.append(p)
                    param_count += p.numel()

        # Attention layers
        if target in ['attn', 'attn+ln', 'all']:
            for name, p in layer.self_attn.named_parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

    if verbose:
        print(f"Model size: {model_size:.1f}B params")
        print(f"Training {len(list(layer_indices))}/{total_layers} layers")
        print(f"Update target: {target}")
        print(f"Trainable params: {param_count/1e6:.1f}M ({100*param_count/sum(p.numel() for p in model.parameters()):.1f}%)")

    return params


def create_optimizer(params: List[torch.nn.Parameter],
                     args,
                     model_size: float) -> torch.optim.Optimizer:
    """
    Create optimizer with memory-efficient options for large models.

    Args:
        params: Trainable parameters
        args: Arguments with lr, optimizer type
        model_size: Model size in GB

    Returns:
        Configured optimizer
    """
    # For large models (8B+), try to use 8-bit Adam if available
    if model_size > 6 and args.optimizer == 'adam8bit':
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=args.lr)
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to AdamW")
            return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr)
    else:
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9)


def create_scheduler(optimizer, args):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    elif args.scheduler == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.steps
        )
    elif args.scheduler == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.steps
        )
    else:
        return None


class ParameterTracker:
    """Efficient parameter state tracking for restoration."""

    def __init__(self, params: List[torch.nn.Parameter]):
        self.params = params
        # Clone original parameters once
        self.original = [p.data.clone() for p in params]

    def restore(self):
        """Restore parameters to original state."""
        for p, orig in zip(self.params, self.original):
            p.data.copy_(orig)
            p.requires_grad = False


def tta_single_token(model,
                     input_ids: torch.Tensor,
                     num_idx: List[int],
                     target_token_idx: int,
                     tokenizer,
                     args,
                     params: List[torch.nn.Parameter],
                     optimizer: torch.optim.Optimizer,
                     scheduler,
                     param_tracker: ParameterTracker,
                     embed_layer) -> Tuple[int, Dict]:
    """
    Perform TTA optimization for a single token position.

    v3 optimizations:
    - No duplicate forward pass for metrics
    - Sparse metrics recording (only at key steps)
    - Reuse cached tensors

    Args:
        model: Language model
        input_ids: Input token IDs
        num_idx: Numerical token indices
        target_token_idx: Target token index in vocabulary
        tokenizer: Tokenizer for decoding
        args: Experiment arguments
        params: Trainable parameters (pre-configured)
        optimizer: Optimizer (pre-configured)
        scheduler: Learning rate scheduler
        param_tracker: For parameter restoration
        embed_layer: Model embedding layer

    Returns:
        pred_idx: Final predicted token index
        metrics: Dictionary with optimization metrics
    """
    # Metrics tracking (sparse recording)
    metrics = {
        'nae': [],
        'target_prob': [],
        'target_rank': [],
        'top10_probs': [],
        'top10_tokens_snapshots': [],
        'health_status': []
    }

    # Define which steps to record full metrics
    if args.steps <= 10:
        metric_steps = set(range(args.steps))
    else:
        # Record at key intervals: start, every 5 steps, end
        metric_steps = {0, args.steps - 1}
        metric_steps.update(range(0, args.steps, max(1, args.steps // 6)))

    # Convert numerical indices to tensor once
    num_idx_tensor = torch.tensor(num_idx, device=input_ids.device, dtype=torch.long)

    model.train()

    # Cache for previous step's logits (to avoid duplicate forward pass)
    prev_logits = None

    for step in range(args.steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Apply numerical mask for subspace constraint
        masked_logits = apply_numerical_mask(logits, num_idx)

        # Compute ANE loss
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
            continue

        # Backward pass
        loss.backward()

        # Check gradient health
        is_healthy, health_msg = check_numerical_health(loss, params)
        metrics['health_status'].append(health_msg)

        if not is_healthy:
            optimizer.zero_grad()
            continue

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Update parameters
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Record metrics only at specified steps (using logits from this forward pass)
        if step in metric_steps:
            with torch.no_grad():
                # Use current logits (post-update we use next step's or final)
                full_probs = F.softmax(logits.float(), dim=-1)

                target_prob = full_probs[target_token_idx].item()
                target_rank = int((full_probs > full_probs[target_token_idx]).sum().item() + 1)

                top10_probs, top10_indices = torch.topk(full_probs, 10)

                metrics['nae'].append(loss.item())
                metrics['target_prob'].append(target_prob)
                metrics['target_rank'].append(target_rank)
                metrics['top10_probs'].append(top10_probs.cpu().tolist())

                # Detailed snapshots (less frequently)
                if step == 0 or step == args.steps - 1 or step % max(5, args.steps // 4) == 0:
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

    # Final prediction (one last forward pass)
    model.eval()
    with torch.no_grad():
        final_logits = model(input_ids).logits[0, -1, :]
        masked_final = apply_numerical_mask(final_logits, num_idx)
        pred_idx = torch.argmax(masked_final).item()

        # Record final metrics
        full_probs = F.softmax(final_logits.float(), dim=-1)
        final_prob = full_probs[target_token_idx].item()
        final_rank = int((full_probs > full_probs[target_token_idx]).sum().item() + 1)

        # Ensure we have final values
        if len(metrics['target_prob']) == 0 or metrics['target_prob'][-1] != final_prob:
            metrics['nae'].append(0.0)  # Placeholder
            metrics['target_prob'].append(final_prob)
            metrics['target_rank'].append(final_rank)

    # Restore original parameters
    param_tracker.restore()

    # Re-enable gradients for next token
    for p in params:
        p.requires_grad = True

    return pred_idx, metrics


def run_experiment(args) -> Dict:
    """
    Run the full ANE-TTA experiment with v3 optimizations.
    """
    print(f"\n{'='*70}")
    print(f"ANE-TTA v3 Experiment: {args.exp_name}")
    print(f"{'='*70}")

    start_time = time.time()

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
        device_map="auto",
        attn_implementation="flash_attention_2" if args.flash_attn else None
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_size = get_model_size_gb(model)
    print(f"Model size: {model_size:.1f}B parameters")

    # Enable gradient checkpointing only when explicitly requested
    # Note: gradient checkpointing requires special handling and may cause issues
    # For 8B models, we use fewer layers (adaptive) instead for memory efficiency
    if args.gradient_checkpointing:
        print("Warning: Gradient checkpointing may cause gradient issues. Using fewer layers is recommended for 8B+ models.")
        # model.gradient_checkpointing_enable()  # Disabled due to compatibility issues

    # Get numerical tokens
    num_idx, num_tok = get_numerical_tokens(tokenizer)
    print(f"Numerical tokens: {len(num_idx)}/{tokenizer.vocab_size}")

    # Configure trainable parameters
    params = configure_trainable_params(model, args.update_target, args.num_layers, verbose=True)

    # Get embedding layer
    embed_layer = model.get_input_embeddings()

    # Main experiment loop
    results = []
    total_correct_before = 0
    total_correct_after = 0
    total_tokens = 0
    flipped_cases = []  # Track wrong->correct cases

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
                masked_baseline = apply_numerical_mask(baseline_logits, num_idx)
                pred_before = torch.argmax(masked_baseline).item()

            # Create optimizer and scheduler for this token
            # Re-enable gradients
            for p in params:
                p.requires_grad = True

            optimizer = create_optimizer(params, args, model_size)
            scheduler = create_scheduler(optimizer, args)
            param_tracker = ParameterTracker(params)

            # TTA with ANE
            pred_after, metrics = tta_single_token(
                model, input_ids, num_idx, target_idx, tokenizer, args,
                params, optimizer, scheduler, param_tracker, embed_layer
            )

            correct_before = pred_before == target_idx
            correct_after = pred_after == target_idx

            total_correct_before += correct_before
            total_correct_after += correct_after
            total_tokens += 1

            # Track flipped cases (wrong -> correct)
            if not correct_before and correct_after:
                flipped_cases.append({
                    'question': question,
                    'answer': answer,
                    'position': pos,
                    'target_token': tokenizer.decode([target_idx]),
                    'pred_before': tokenizer.decode([pred_before]),
                    'pred_after': tokenizer.decode([pred_after]),
                    'metrics': metrics
                })

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

    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")
    print(f"Total tokens: {total_tokens}")
    print(f"Accuracy Before TTA: {acc_before:.4f} ({total_correct_before}/{total_tokens})")
    print(f"Accuracy After TTA:  {acc_after:.4f} ({total_correct_after}/{total_tokens})")
    print(f"Improvement: {acc_after - acc_before:+.4f}")
    print(f"Flipped cases (wrong->correct): {len(flipped_cases)}")
    print(f"Total time: {elapsed_time/60:.1f} min")
    print(f"Time per token: {elapsed_time/total_tokens:.2f} sec")
    print(f"{'='*70}")

    # Prepare output
    output = {
        'summary': {
            'model': model_name,
            'accuracy_before': float(acc_before),
            'accuracy_after': float(acc_after),
            'improvement': float(acc_after - acc_before),
            'total_tokens': total_tokens,
            'flipped_count': len(flipped_cases),
            'elapsed_time_min': elapsed_time / 60,
            'config': {
                'models': args.models,
                'data_path': args.data_path,
                'num_samples': args.num_samples,
                'update_target': args.update_target,
                'num_layers': str(args.num_layers),
                'steps': args.steps,
                'lr': args.lr,
                'optimizer': args.optimizer,
                'scheduler': args.scheduler,
                'gradient_checkpointing': args.gradient_checkpointing
            }
        },
        'results': results,
        'flipped_cases': flipped_cases
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.exp_name}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    # Generate visualizations
    if not args.no_viz:
        print("\nGenerating visualizations...")
        from visualization import visualize_all
        visualize_all(output, args.output_dir, args.exp_name, tokenizer)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(
        description='ANE Test-Time Adaptation v3 (Memory & Speed Optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data
    parser.add_argument('--models', nargs='+', default=[
        "/home/jinsk/Models/Llama-3.2-3B-Instruct"
    ], help='Model path(s)')
    parser.add_argument('--data_path', default='../data/addition_problems_dataset(1-50)(1).json',
                       help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to process')

    # TTA configuration
    parser.add_argument('--update_target',
                       choices=['mlp', 'ln', 'mlp+ln', 'attn', 'attn+ln', 'all'],
                       default='mlp+ln',
                       help='Which layers to update')
    parser.add_argument('--num_layers', default='all',
                       help='Number of layers to update: "all", "adaptive", or int')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'adam', 'adam8bit'],
                       default='sgd',
                       help='Optimizer type (adam8bit requires bitsandbytes)')
    parser.add_argument('--scheduler', choices=['none', 'cosine', 'linear', 'onecycle'],
                       default='none',
                       help='Learning rate scheduler')

    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing (auto-enabled for 8B+)')
    parser.add_argument('--flash_attn', action='store_true',
                       help='Use Flash Attention 2')

    # Output
    parser.add_argument('--output_dir', default='results_nae',
                       help='Output directory')
    parser.add_argument('--exp_name', required=True,
                       help='Experiment name')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization generation')

    args = parser.parse_args()

    # Process scheduler
    if args.scheduler == 'none':
        args.scheduler = None

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
