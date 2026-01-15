"""
Numerical Angular Entropy for Test-Time Adaptation
基于embedding几何结构的数值推理优化
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 学术配色方案
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300
})

PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the numerical answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def get_numerical_tokens(tokenizer):
    """提取纯数字token"""
    num_idx, num_tok = [], []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.decode([i])
        if token and all(c in '0123456789' for c in token):
            num_idx.append(i)
            num_tok.append(token)
    return num_idx, num_tok

def configure_params(model, target, n_layers):
    """配置可训练参数"""
    model.requires_grad_(False)
    total = len(model.model.layers)
    layers = range(total) if n_layers == 'all' else range(max(0, total - n_layers), total)
    
    params = []
    for idx in layers:
        layer = model.model.layers[idx]
        if target in ['mlp', 'mlp+ln']:
            params.extend(layer.mlp.parameters())
        if target in ['ln', 'mlp+ln']:
            params.extend([p for n, p in layer.named_parameters() if 'norm' in n.lower()])
    
    for p in params:
        p.requires_grad = True
    return params

def angular_entropy(logits, num_idx, embed_layer, eps=1e-8):
    """
    Numerical Angular Entropy (NAE)
    
    H_∠(p) = Σ p(t) · arccos(⟨φ(t), φ̄(p)⟩)
    
    其中 φ̄(p) = Σ p(t)φ(t) 是概率加权anchor
    """
    # 数值token的logits和概率
    num_logits = logits[num_idx]
    probs = F.softmax(num_logits, dim=-1)  # (N_num,)
    
    # 获取数值token embeddings
    num_embeds = embed_layer.weight[num_idx]  # (N_num, d)
    num_embeds_norm = F.normalize(num_embeds, p=2, dim=-1)
    
    # 计算anchor：概率加权质心
    anchor = (probs.unsqueeze(-1) * num_embeds).sum(dim=0)  # (d,)
    anchor_norm = F.normalize(anchor, p=2, dim=-1)
    
    # 计算余弦相似度 -> 角度
    cos_sim = torch.matmul(num_embeds_norm, anchor_norm)  # (N_num,)
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    angles = torch.acos(cos_sim)  # [0, π]
    
    # 加权平均角度
    H_angular = (probs * angles).sum()
    
    return H_angular

def tta_token(model, ids, num_idx, target_idx, tokenizer, args):
    """单token的TTA优化"""
    embed_layer = model.get_input_embeddings()
    params = configure_params(model, args.update_target, args.num_layers)
    orig_state = {id(p): p.data.clone() for p in params}
    
    model.train()
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    if args.scheduler:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    
    # 记录指标
    metrics = {
        'nae': [],
        'target_prob': [],
        'target_rank': [],
        'top10_probs': [],
        'top10_tokens_snapshots': []  # 每5步记录一次
    }
    
    for step in range(args.steps):
        opt.zero_grad()
        
        logits = model(ids).logits[0, -1, :]
        loss = angular_entropy(logits, num_idx, embed_layer)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if args.scheduler:
            sched.step()
        
        # 记录
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            top10_p, top10_i = torch.topk(probs, 10)
            
            target_p = probs[target_idx].item()
            target_r = int((probs > probs[target_idx]).sum().item() + 1)
            
            metrics['nae'].append(loss.item())
            metrics['target_prob'].append(target_p)
            metrics['target_rank'].append(target_r)
            metrics['top10_probs'].append(top10_p.cpu().tolist())
            
            # 每5步记录top10 token详情
            if step % 5 == 0:
                tok_info = []
                for i, p in zip(top10_i.cpu().tolist(), top10_p.cpu().tolist()):
                    tok_str = tokenizer.decode([i])
                    tok_info.append({'token': tok_str, 'prob': float(p)})
                metrics['top10_tokens_snapshots'].append({
                    'step': step,
                    'tokens': tok_info
                })
    
    # 最终预测
    model.eval()
    with torch.no_grad():
        final_logits = model(ids).logits[0, -1, :]
        pred_idx = torch.argmax(final_logits).item()
    
    # 恢复参数
    for p in params:
        p.data.copy_(orig_state[id(p)])
        p.requires_grad = False
    
    return pred_idx, metrics

def run_experiment(args):
    """主实验流程"""
    print(f"\n{'='*70}\nExperiment: {args.exp_name}\n{'='*70}")
    
    with open(args.data_path) as f:
        data = json.load(f)
    
    # 加载模型
    model_path = args.models[0]
    model_name = model_path.split('/')[-1]
    print(f"Model: {model_name}")
    
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok.pad_token = tok.eos_token
    
    num_idx, num_tok = get_numerical_tokens(tok)
    print(f"Numerical tokens: {len(num_idx)}/{tok.vocab_size}")
    
    # 实验主循环
    results = []
    for item in tqdm(data[:args.num_samples], desc="Processing"):
        q, ans = item['question'], item['answer']
        gt_tokens = tok.encode(str(ans), add_special_tokens=False)
        inp = tok(PROMPT.format(question=q), return_tensors="pt").to(model.device)
        
        token_results = []
        for pos in range(len(gt_tokens)):
            # 构造输入
            if pos == 0:
                ids = inp['input_ids']
            else:
                ids = torch.cat([
                    inp['input_ids'],
                    torch.tensor([gt_tokens[:pos]], device=model.device)
                ], dim=1)
            
            # Baseline
            model.eval()
            with torch.no_grad():
                baseline_logits = model(ids).logits[0, -1, :]
                pred_before = torch.argmax(baseline_logits).item()
            
            # TTA with NAE
            pred_after, metrics = tta_token(model, ids, num_idx, gt_tokens[pos], tok, args)
            
            token_results.append({
                'position': pos,
                'target_token': tok.decode([gt_tokens[pos]]),
                'pred_before': tok.decode([pred_before]),
                'pred_after': tok.decode([pred_after]),
                'correct_before': pred_before == gt_tokens[pos],
                'correct_after': pred_after == gt_tokens[pos],
                'metrics': metrics
            })
        
        results.append({
            'question': q,
            'answer': ans,
            'tokens': token_results
        })
    
    # 统计
    acc_before = np.mean([t['correct_before'] for r in results for t in r['tokens']])
    acc_after = np.mean([t['correct_after'] for r in results for t in r['tokens']])
    
    print(f"\n{'='*70}")
    print(f"Results: {acc_before:.3f} → {acc_after:.3f} (Δ={acc_after-acc_before:+.3f})")
    print(f"{'='*70}")
    
    # 保存
    output = {
        'summary': {
            'model': model_name,
            'accuracy_before': float(acc_before),
            'accuracy_after': float(acc_after),
            'improvement': float(acc_after - acc_before),
            'config': vars(args)
        },
        'results': results
    }
    
    output_path = f'{args.output_dir}/{args.exp_name}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Results saved to {output_path}")
    
    del model
    torch.cuda.empty_cache()
    
    return output

def visualize(data, args):
    """学术级可视化"""
    results = data['results']
    out_dir = args.output_dir
    exp_name = args.exp_name
    
    # === Figure 1: NAE Dynamics (Multi-sample Overlay) ===
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    n_show = min(20, len(results))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_show))
    
    for i in range(n_show):
        if results[i]['tokens']:
            nae_curve = results[i]['tokens'][0]['metrics']['nae']
            ax.plot(nae_curve, alpha=0.5, lw=1.2, color=colors[i])
    
    # 平均曲线
    all_nae = [results[i]['tokens'][0]['metrics']['nae'] 
               for i in range(min(len(results), 50)) if results[i]['tokens']]
    if all_nae:
        mean_nae = np.mean(all_nae, axis=0)
        ax.plot(mean_nae, lw=2.5, color='red', label='Mean', zorder=10)
    
    ax.set_xlabel('Optimization Step', fontweight='bold')
    ax.set_ylabel('NAE (radians)', fontweight='bold')
    ax.set_title('Angular Entropy Minimization Dynamics', fontweight='bold', pad=12)
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{exp_name}_nae_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved NAE dynamics")
    
    # === Figure 2 & 3: Heatmaps ===
    # 准备数据
    all_prob, all_rank, sample_labels = [], [], []
    for sid, r in enumerate(results):
        for tid, t in enumerate(r['tokens']):
            all_prob.append(t['metrics']['target_prob'])
            all_rank.append(t['metrics']['target_rank'])
            sample_labels.append(f"S{sid+1}-T{tid+1}")
    
    n_tokens = min(80, len(all_prob))
    prob_mat = np.array(all_prob[:n_tokens])
    rank_mat = np.array(all_rank[:n_tokens])
    
    # Probability Heatmap (蓝=高，红=低)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(prob_mat, cmap='RdYlBu', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Probability', 'shrink': 0.85},
                linewidths=0, ax=ax)
    
    ax.set_xlabel('Optimization Step', fontweight='bold', fontsize=12)
    ax.set_ylabel('Token Index', fontweight='bold', fontsize=12)
    ax.set_title('Target Token Probability Evolution', fontweight='bold', fontsize=13, pad=15)
    
    # 标注sample分界
    token_counts = [len(r['tokens']) for r in results]
    boundaries = np.cumsum([0] + token_counts)
    for b in boundaries[1:-1]:
        if b < n_tokens:
            ax.axhline(b, color='black', lw=1.2, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{exp_name}_prob_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved probability heatmap")
    
    # Rank Heatmap
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(rank_mat, annot=True, fmt='.0f', cmap='YlOrRd_r',
                cbar_kws={'label': 'Rank', 'shrink': 0.85},
                linewidths=0.2, linecolor='lightgray', ax=ax,
                annot_kws={'fontsize': 6})
    
    ax.set_xlabel('Optimization Step', fontweight='bold', fontsize=12)
    ax.set_ylabel('Token Index', fontweight='bold', fontsize=12)
    ax.set_title('Target Token Rank Evolution', fontweight='bold', fontsize=13, pad=15)
    
    for b in boundaries[1:-1]:
        if b < n_tokens:
            ax.axhline(b, color='darkblue', lw=1.2, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{exp_name}_rank_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved rank heatmap")
    
    # === Figure 4: Top-10 Token Probability Case Studies ===
    # 选择3个代表性case
    case_indices = []
    init_ranks = [r['tokens'][0]['metrics']['target_rank'][0] for r in results if r['tokens']]
    
    # 高rank, 中rank, 低rank
    for threshold in [50, 20, 5]:
        for i, rank in enumerate(init_ranks):
            if len(case_indices) < 3:
                if threshold == 50 and rank > 50:
                    case_indices.append(i)
                    break
                elif threshold == 20 and 15 < rank < 30:
                    case_indices.append(i)
                    break
                elif threshold == 5 and rank <= 5:
                    case_indices.append(i)
                    break
    
    if len(case_indices) < 3:
        case_indices = list(range(min(3, len(results))))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for ax_idx, case_idx in enumerate(case_indices):
        ax = axes[ax_idx]
        token_data = results[case_idx]['tokens'][0]
        top10_probs = np.array(token_data['metrics']['top10_probs'])  # (steps, 10)
        target_curve = token_data['metrics']['target_prob']
        
        # 画top10概率演化
        for rank in range(10):
            ax.plot(top10_probs[:, rank], lw=1.2, alpha=0.6)
        
        # 高亮target
        ax.plot(target_curve, lw=3, color='red', linestyle='--',
                label='Target', zorder=10, alpha=0.9)
        
        ax.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax.set_ylabel('Probability', fontweight='bold', fontsize=10)
        ax.set_title(f'Case {ax_idx+1}: Answer={results[case_idx]["answer"]}',
                     fontweight='bold', fontsize=11)
        ax.grid(alpha=0.2, linestyle=':', linewidth=0.6)
        ax.legend(fontsize=8, loc='best', frameon=True)
        ax.set_ylim([-0.05, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{exp_name}_top10_cases.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved top-10 case studies")
    
    print(f"\n✓ All visualizations completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAE Test-Time Adaptation')
    
    parser.add_argument('--models', nargs='+', default=[
        "/home/jinsk/Models/Llama-3.1-8B-Instruct"
    ])
    parser.add_argument('--data_path', default='../data/addition_problems_dataset(1-50)(1).json')
    parser.add_argument('--num_samples', type=int, default=50)
    
    parser.add_argument('--update_target', choices=['mlp', 'ln', 'mlp+ln'], default='mlp')
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', action='store_true')
    
    parser.add_argument('--output_dir', default='results_nae')
    parser.add_argument('--exp_name', required=True)
    
    args = parser.parse_args()
    
    if args.num_layers != 'all':
        args.num_layers = int(args.num_layers)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = run_experiment(args)
    visualize(data, args)