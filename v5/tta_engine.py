"""Test-time adaptation engine for ANE-TTA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ane_core import AngularEntropy, check_numerical_health
from numerical_utils import mask_logits_to_num, numerical_softmax


@dataclass
class TrainableStats:
    trainable_params: int
    trainable_pct: float
    layer_count: int
    total_layers: int


def resolve_layer_indices(total_layers: int, num_layers: str, layer_stride: int) -> List[int]:
    if num_layers == "all":
        return list(range(0, total_layers, layer_stride))
    count = int(num_layers)
    start = max(0, total_layers - count * layer_stride)
    return list(range(start, total_layers, layer_stride))


def configure_trainable_params(
    model: torch.nn.Module,
    target: str,
    num_layers: str,
    layer_stride: int,
) -> Tuple[List[torch.nn.Parameter], TrainableStats]:
    model.requires_grad_(False)
    total_layers = len(model.model.layers)
    layer_indices = resolve_layer_indices(total_layers, num_layers, layer_stride)

    params: List[torch.nn.Parameter] = []
    param_count = 0

    for idx in layer_indices:
        layer = model.model.layers[idx]

        if target in ["mlp", "mlp+ln", "all"]:
            for p in layer.mlp.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

        if target in ["ln", "mlp+ln", "attn+ln", "all"]:
            for name, p in layer.named_parameters():
                if "norm" in name.lower():
                    p.requires_grad = True
                    params.append(p)
                    param_count += p.numel()

        if target in ["attn", "attn+ln", "all"]:
            for p in layer.self_attn.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

    total_params = sum(p.numel() for p in model.parameters())
    stats = TrainableStats(
        trainable_params=param_count,
        trainable_pct=100.0 * param_count / total_params,
        layer_count=len(layer_indices),
        total_layers=total_layers,
    )
    return params, stats


def make_optimizer(params: List[torch.nn.Parameter], args) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def make_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def compute_rank_prob(logits: torch.Tensor, target_idx: int) -> Tuple[float, int, torch.Tensor]:
    logits_f = logits.float()
    target_logit = logits_f[target_idx]
    lse = torch.logsumexp(logits_f, dim=-1)
    prob = torch.exp(target_logit - lse).item()
    rank = int((logits_f > target_logit).sum().item() + 1)
    return prob, rank, lse


def pad_list(values: List, target_len: int) -> List:
    if not values:
        return values
    if len(values) >= target_len:
        return values[:target_len]
    values.extend([values[-1]] * (target_len - len(values)))
    return values


def maybe_to_cpu(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    return tensor.cpu() if enabled else tensor


def backup_params(params: List[torch.nn.Parameter], to_cpu: bool) -> List[torch.Tensor]:
    return [maybe_to_cpu(p.detach().clone(), to_cpu) for p in params]


def restore_params(params: List[torch.nn.Parameter], backup: List[torch.Tensor], from_cpu: bool) -> None:
    for p, saved in zip(params, backup):
        if from_cpu:
            saved = saved.to(p.device)
        p.data.copy_(saved)


def build_topk_snapshot(
    num_probs: torch.Tensor,
    num_idx_tensor: torch.Tensor,
    num_tokens: List[str],
    topk: int,
) -> List[Dict[str, float | int | str]]:
    k = min(topk, num_probs.shape[0])
    top_probs, top_pos = torch.topk(num_probs, k)
    snapshot = []
    for prob, pos in zip(top_probs.tolist(), top_pos.tolist()):
        idx = num_idx_tensor[pos].item()
        tok = num_tokens[pos]
        snapshot.append({"token": tok, "idx": int(idx), "prob": float(prob)})
    return snapshot


class TTAEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        num_mask: torch.Tensor,
        num_idx_tensor: torch.Tensor,
        num_tokens: List[str],
        params: List[torch.nn.Parameter],
        ane_loss: AngularEntropy,
        args,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_mask = num_mask
        self.num_idx_tensor = num_idx_tensor
        self.num_tokens = num_tokens
        self.params = params
        self.ane_loss = ane_loss
        self.args = args

    def run_token(self, input_ids: torch.Tensor, target_idx: int) -> Tuple[int, int, Dict]:
        metrics = {
            "ane": [],
            "target_prob": [],
            "target_rank": [],
            "step_status": [],
            "tracked_topk": {"indices": [], "tokens": [], "probs": []},
            "num_topk_snapshots": [],
        }

        self.model.eval()
        with torch.no_grad():
            baseline_logits = self.model(input_ids).logits[0, -1, :]
            pred_before = torch.argmax(mask_logits_to_num(baseline_logits, self.num_mask)).item()
            base_prob, base_rank, base_lse = compute_rank_prob(baseline_logits, target_idx)
            base_loss = self.ane_loss(baseline_logits)

            metrics["ane"].append(float(base_loss.item()))
            metrics["target_prob"].append(float(base_prob))
            metrics["target_rank"].append(int(base_rank))
            metrics["step_status"].append("baseline")

            if self.args.tracked_topk > 0:
                top_vals, top_idx = torch.topk(baseline_logits, self.args.tracked_topk)
                metrics["tracked_topk"]["indices"] = top_idx.tolist()
                metrics["tracked_topk"]["tokens"] = [self.tokenizer.decode([i]) for i in top_idx.tolist()]
                metrics["tracked_topk"]["probs"].append(torch.exp(top_vals.float() - base_lse).tolist())

            if 0 in self.args.snapshot_steps:
                num_probs = numerical_softmax(baseline_logits, self.num_idx_tensor)
                metrics["num_topk_snapshots"].append(
                    {"step": 0, "tokens": build_topk_snapshot(num_probs, self.num_idx_tensor, self.num_tokens, self.args.num_topk)}
                )

        if self.args.skip_if_correct and pred_before == target_idx:
            total_len = self.args.steps + 1
            pad_list(metrics["ane"], total_len)
            pad_list(metrics["target_prob"], total_len)
            pad_list(metrics["target_rank"], total_len)
            pad_list(metrics["step_status"], total_len)
            pad_list(metrics["tracked_topk"]["probs"], total_len)
            if not metrics["num_topk_snapshots"] or metrics["num_topk_snapshots"][-1]["step"] != self.args.steps:
                num_probs = numerical_softmax(baseline_logits, self.num_idx_tensor)
                metrics["num_topk_snapshots"].append(
                    {"step": self.args.steps, "tokens": build_topk_snapshot(num_probs, self.num_idx_tensor, self.num_tokens, self.args.num_topk)}
                )
            return pred_before, pred_before, metrics

        optimizer = make_optimizer(self.params, self.args)
        scheduler = make_scheduler(optimizer, self.args)

        last_logits = baseline_logits.detach()
        best_rank = base_rank
        stale = 0

        for step in range(1, self.args.steps + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            loss = self.ane_loss(logits)

            if torch.isnan(loss) or torch.isinf(loss):
                metrics["step_status"].append("loss_nan")
                metrics["ane"].append(float(metrics["ane"][-1]))
                metrics["target_prob"].append(float(metrics["target_prob"][-1]))
                metrics["target_rank"].append(int(metrics["target_rank"][-1]))
                if self.args.tracked_topk > 0:
                    metrics["tracked_topk"]["probs"].append(metrics["tracked_topk"]["probs"][-1])
                continue

            loss.backward()
            healthy, status = check_numerical_health(loss, self.params, max_grad_norm=self.args.max_grad_norm)
            if not healthy:
                optimizer.zero_grad(set_to_none=True)
                metrics["step_status"].append(status)
                metrics["ane"].append(float(metrics["ane"][-1]))
                metrics["target_prob"].append(float(metrics["target_prob"][-1]))
                metrics["target_rank"].append(int(metrics["target_rank"][-1]))
                if self.args.tracked_topk > 0:
                    metrics["tracked_topk"]["probs"].append(metrics["tracked_topk"]["probs"][-1])
                continue

            torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if self.args.fast_eval:
                eval_logits = logits.detach()
            else:
                self.model.eval()
                with torch.no_grad():
                    eval_logits = self.model(input_ids).logits[0, -1, :]

            last_logits = eval_logits
            step_prob, step_rank, step_lse = compute_rank_prob(eval_logits, target_idx)

            metrics["ane"].append(float(loss.item()))
            metrics["target_prob"].append(float(step_prob))
            metrics["target_rank"].append(int(step_rank))
            metrics["step_status"].append("ok")

            if self.args.tracked_topk > 0:
                idx = torch.tensor(metrics["tracked_topk"]["indices"], device=eval_logits.device)
                tracked_logits = eval_logits[idx].float()
                metrics["tracked_topk"]["probs"].append(torch.exp(tracked_logits - step_lse).tolist())

            if step in self.args.snapshot_steps:
                num_probs = numerical_softmax(eval_logits, self.num_idx_tensor)
                metrics["num_topk_snapshots"].append(
                    {"step": step, "tokens": build_topk_snapshot(num_probs, self.num_idx_tensor, self.num_tokens, self.args.num_topk)}
                )

            if step_rank < best_rank:
                best_rank = step_rank
                stale = 0
            else:
                stale += 1

            if self.args.early_stop:
                if step_rank <= self.args.early_stop_rank:
                    if self.args.early_stop_prob is None or step_prob >= self.args.early_stop_prob:
                        break
                if self.args.patience > 0 and stale >= self.args.patience:
                    break

        total_len = self.args.steps + 1
        pad_list(metrics["ane"], total_len)
        pad_list(metrics["target_prob"], total_len)
        pad_list(metrics["target_rank"], total_len)
        pad_list(metrics["step_status"], total_len)
        if self.args.tracked_topk > 0:
            pad_list(metrics["tracked_topk"]["probs"], total_len)

        if not metrics["num_topk_snapshots"] or metrics["num_topk_snapshots"][-1]["step"] != self.args.steps:
            num_probs = numerical_softmax(last_logits, self.num_idx_tensor)
            metrics["num_topk_snapshots"].append(
                {"step": self.args.steps, "tokens": build_topk_snapshot(num_probs, self.num_idx_tensor, self.num_tokens, self.args.num_topk)}
            )

        pred_after = torch.argmax(mask_logits_to_num(last_logits, self.num_mask)).item()
        return pred_before, pred_after, metrics
