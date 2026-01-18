"""
Angular Numerical Entropy Test-Time Adaptation (ANE-TTA) v4
Optimized for performance with v2.1 proven configuration

Key changes from v3:
1. Restored optimal config: mlp+ln + lr=0.001 (not attn+ln + 0.0005)
2. All layers for all models including 8B
3. Record metrics at ALL steps (full visualization)
4. Memory optimization via cache clearing for 8B
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
from typing import List, Dict, Tuple

from ane_core import angular_entropy_stable, check_numerical_health
from numerical_utils import get_numerical_tokens, apply_numerical_mask

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the numerical answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_model_size_gb(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


def configure_trainable_params(model, target: str = 'mlp+ln', n_layers: str = 'all') -> List[torch.nn.Parameter]:
    """Configure trainable parameters. v4 uses all layers for all models."""
    model.requires_grad_(False)

    total_layers = len(model.model.layers)

    # v4: Always use all layers (效果优先)
    if n_layers == 'all':
        layer_indices = range(total_layers)
    else:
        n = int(n_layers)
        layer_indices = range(max(0, total_layers - n), total_layers)

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

        # LayerNorm
        if target in ['ln', 'mlp+ln', 'attn+ln', 'all']:
            for name, p in layer.named_parameters():
                if 'norm' in name.lower():
                    p.requires_grad = True
                    params.append(p)
                    param_count += p.numel()

        # Attention
        if target in ['attn', 'attn+ln', 'all']:
            for p in layer.self_attn.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

    print(f"Training {len(list(layer_indices))}/{total_layers} layers, target: {target}")
    print(f"Trainable params: {param_count/1e6:.1f}M ({100*param_count/sum(p.numel() for p in model.parameters()):.1f}%)")

    return params


def tta_single_token(model, input_ids, num_idx, target_idx, tokenizer, args,
                     params, embed_layer) -> Tuple[int, Dict]:
    """TTA for single token. v4: Records ALL steps for full visualization."""

    metrics = {
        'ane': [],
        'target_prob': [],
        'target_rank': [],
        'top10_probs': [],
        'top10_snapshots': []
    }

    # Save original parameters
    original_params = [p.data.clone() for p in params]

    # Create optimizer
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    num_idx_tensor = torch.tensor(num_idx, device=input_ids.device, dtype=torch.long)

    model.train()

    for step in range(args.steps):
        optimizer.zero_grad()

        # Forward
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Apply numerical mask
        masked_logits = apply_numerical_mask(logits, num_idx)

        # Compute ANE loss
        loss = angular_entropy_stable(masked_logits, num_idx, embed_layer, use_float32=True, eps=1e-4)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        # Check gradient health
        is_healthy, _ = check_numerical_health(loss, params)
        if not is_healthy:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # Record metrics at EVERY step (v4 improvement)
        with torch.no_grad():
            full_probs = F.softmax(logits.float(), dim=-1)
            target_prob = full_probs[target_idx].item()
            target_rank = int((full_probs > full_probs[target_idx]).sum().item() + 1)
            top10_probs, top10_indices = torch.topk(full_probs, 10)

            metrics['ane'].append(loss.item())
            metrics['target_prob'].append(target_prob)
            metrics['target_rank'].append(target_rank)
            metrics['top10_probs'].append(top10_probs.cpu().tolist())

            # Detailed snapshots at key steps
            if step == 0 or step == args.steps - 1 or step % 5 == 0:
                tok_info = []
                for idx_val, prob in zip(top10_indices.cpu().tolist(), top10_probs.cpu().tolist()):
                    try:
                        tok_str = tokenizer.decode([idx_val])
                    except:
                        tok_str = f'<{idx_val}>'
                    tok_info.append({'token': tok_str, 'prob': float(prob), 'idx': idx_val})
                metrics['top10_snapshots'].append({'step': step, 'tokens': tok_info})

    # Final prediction
    model.eval()
    with torch.no_grad():
        final_logits = model(input_ids).logits[0, -1, :]
        masked_final = apply_numerical_mask(final_logits, num_idx)
        pred_idx = torch.argmax(masked_final).item()

        # Final metrics
        full_probs = F.softmax(final_logits.float(), dim=-1)
        final_prob = full_probs[target_idx].item()
        final_rank = int((full_probs > full_probs[target_idx]).sum().item() + 1)

        if len(metrics['target_prob']) == 0 or metrics['target_prob'][-1] != final_prob:
            metrics['ane'].append(0.0)
            metrics['target_prob'].append(final_prob)
            metrics['target_rank'].append(final_rank)

    # Restore parameters
    for p, orig in zip(params, original_params):
        p.data.copy_(orig)

    return pred_idx, metrics


def run_experiment(args) -> Dict:
    """Run ANE-TTA experiment with v4 optimizations."""

    print(f"\n{'='*60}")
    print(f"ANE-TTA v4: {args.exp_name}")
    print(f"{'='*60}")

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
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_size = get_model_size_gb(model)
    print(f"Model size: {model_size:.1f}B")

    # Get numerical tokens
    num_idx, num_tok = get_numerical_tokens(tokenizer)
    print(f"Numerical tokens: {len(num_idx)}")

    # Configure trainable parameters
    params = configure_trainable_params(model, args.update_target, args.num_layers)
    embed_layer = model.get_input_embeddings()

    # Main loop
    results = []
    total_correct_before = 0
    total_correct_after = 0
    total_tokens = 0
    flipped_cases = []

    for item in tqdm(data[:args.num_samples], desc="Processing"):
        question = item['question']
        answer = item['answer']
        gt_tokens = tokenizer.encode(str(answer), add_special_tokens=False)

        prompt = PROMPT_TEMPLATE.format(question=question)
        input_encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

        token_results = []

        for pos in range(len(gt_tokens)):
            if pos == 0:
                input_ids = input_encoding['input_ids']
            else:
                prefix = torch.tensor([gt_tokens[:pos]], device=model.device)
                input_ids = torch.cat([input_encoding['input_ids'], prefix], dim=1)

            target_idx = gt_tokens[pos]

            # Baseline
            model.eval()
            with torch.no_grad():
                baseline_logits = model(input_ids).logits[0, -1, :]
                masked_baseline = apply_numerical_mask(baseline_logits, num_idx)
                pred_before = torch.argmax(masked_baseline).item()

            # Re-enable gradients
            for p in params:
                p.requires_grad = True

            # TTA
            pred_after, metrics = tta_single_token(
                model, input_ids, num_idx, target_idx, tokenizer, args, params, embed_layer
            )

            correct_before = pred_before == target_idx
            correct_after = pred_after == target_idx

            total_correct_before += correct_before
            total_correct_after += correct_after
            total_tokens += 1

            # Track flipped
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

        # Memory cleanup for 8B
        if model_size > 6:
            torch.cuda.empty_cache()

    # Summary
    acc_before = total_correct_before / total_tokens if total_tokens > 0 else 0
    acc_after = total_correct_after / total_tokens if total_tokens > 0 else 0
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Results: {total_tokens} tokens")
    print(f"Before: {acc_before*100:.2f}% ({total_correct_before}/{total_tokens})")
    print(f"After:  {acc_after*100:.2f}% ({total_correct_after}/{total_tokens})")
    print(f"Improve: {(acc_after-acc_before)*100:+.2f}%")
    print(f"Flipped: {len(flipped_cases)}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    output = {
        'summary': {
            'model': model_name,
            'accuracy_before': float(acc_before),
            'accuracy_after': float(acc_after),
            'improvement': float(acc_after - acc_before),
            'total_tokens': total_tokens,
            'flipped_count': len(flipped_cases),
            'elapsed_time_min': elapsed / 60,
            'config': {
                'update_target': args.update_target,
                'num_layers': str(args.num_layers),
                'steps': args.steps,
                'lr': args.lr
            }
        },
        'results': results,
        'flipped_cases': flipped_cases
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.exp_name}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")

    # Visualize
    if not args.no_viz:
        print("\nGenerating visualizations...")
        from visualization import visualize_all
        visualize_all(output, args.output_dir, args.exp_name, tokenizer)

    del model
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description='ANE-TTA v4')

    parser.add_argument('--models', nargs='+', default=["/home/jinsk/Models/Llama-3.2-3B-Instruct"])
    parser.add_argument('--data_path', default='../data/addition_problems_dataset(1-50)(1).json')
    parser.add_argument('--num_samples', type=int, default=50)

    # v4 optimal config (restored from v2.1)
    parser.add_argument('--update_target', default='mlp+ln',
                       choices=['mlp', 'ln', 'mlp+ln', 'attn', 'attn+ln', 'all'])
    parser.add_argument('--num_layers', default='all')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)  # v2.1 optimal

    parser.add_argument('--output_dir', default='results_ane')
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--no_viz', action='store_true')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
