"""TTA engine for v7 (ANE minimization + rich metric logging).

Key design:
- Loss is unsupervised (ANE) on the numerical sub-vocab logits.
- Evaluation metrics (target prob/rank, pass@k) use gold target token ids.
- Forward-count optimized: ~steps+1 forwards per token (reuses updated logits).
- Records snapshots for subspace evolution plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ane_core import check_numerical_health
from numerical_utils import compute_num_prob_rank, mask_logits_to_num, numerical_softmax


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
    """Select trainable parameters for TTA."""
    model.requires_grad_(False)

    params: List[torch.nn.Parameter] = []
    param_count = 0

    train_lm_head = False
    base_target = target

    if base_target == "lm_head":
        train_lm_head = True
        base_target = ""
    elif base_target.endswith("+lm_head"):
        train_lm_head = True
        base_target = base_target[: -len("+lm_head")]

    total_layers = len(model.model.layers)
    layer_indices = resolve_layer_indices(total_layers, num_layers, layer_stride) if base_target else []

    for idx in layer_indices:
        layer = model.model.layers[idx]

        if base_target in ["mlp", "mlp+ln", "all"]:
            for p in layer.mlp.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

        if base_target in ["ln", "mlp+ln", "attn+ln", "all"]:
            for name, p in layer.named_parameters():
                if "norm" in name.lower():
                    p.requires_grad = True
                    params.append(p)
                    param_count += p.numel()

        if base_target in ["attn", "attn+ln", "all"]:
            for p in layer.self_attn.parameters():
                p.requires_grad = True
                params.append(p)
                param_count += p.numel()

    if train_lm_head:
        for p in model.lm_head.parameters():
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


def make_optimizer(
    params: List[torch.nn.Parameter],
    lr: float,
    optimizer: str = "sgd",
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    if optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def backup_params(params: List[torch.nn.Parameter], to_cpu: bool) -> List[torch.Tensor]:
    backup = []
    for p in params:
        if to_cpu:
            t = p.detach().to("cpu", copy=True)
        else:
            t = p.detach().clone()
        backup.append(t)
    return backup


def restore_params(params: List[torch.nn.Parameter], backup: List[torch.Tensor], from_cpu: bool) -> None:
    for p, saved in zip(params, backup):
        p.data.copy_(saved)


def _pad_list(values: List, target_len: int) -> None:
    if not values:
        return
    if len(values) >= target_len:
        del values[target_len:]
        return
    values.extend([values[-1]] * (target_len - len(values)))


def _build_topk_snapshot(
    num_probs: torch.Tensor,
    num_idx_tensor: torch.Tensor,
    num_tokens: List[str],
    topk: int,
) -> List[Dict]:
    k = min(topk, int(num_probs.shape[0]))
    top_probs, top_pos = torch.topk(num_probs, k)
    snapshot = []
    for prob, pos in zip(top_probs.tolist(), top_pos.tolist()):
        idx = int(num_idx_tensor[pos].item())
        tok = num_tokens[pos] if 0 <= pos < len(num_tokens) else str(idx)
        snapshot.append({"token": tok, "idx": idx, "prob": float(prob)})
    return snapshot


class TTAEngine:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer,
        params: List[torch.nn.Parameter],
        num_mask: torch.Tensor,
        num_idx_tensor: torch.Tensor,
        num_tokens: List[str],
        id_to_pos: torch.Tensor,
        loss_fn,
        steps: int,
        lr: float,
        optimizer: str = "sgd",
        momentum: float = 0.0,
        grad_clip: float = 1.0,
        max_grad_norm: float = 1e6,
        num_topk: int = 10,
        tracked_topk: int = 10,
        snapshot_stride: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.params = params
        self.num_mask = num_mask
        self.num_idx_tensor = num_idx_tensor
        self.num_tokens = num_tokens
        self.id_to_pos = id_to_pos
        self.loss_fn = loss_fn
        self.steps = int(steps)
        self.lr = float(lr)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.grad_clip = float(grad_clip)
        self.max_grad_norm = float(max_grad_norm)
        self.num_topk = int(num_topk)
        self.tracked_topk = int(tracked_topk)
        self.snapshot_stride = max(1, int(snapshot_stride))

    def _snapshot_steps(self) -> List[int]:
        steps = list(range(0, self.steps + 1, self.snapshot_stride))
        if steps[-1] != self.steps:
            steps.append(self.steps)
        return steps

    def run_token(self, input_ids: torch.Tensor, target_idx: int) -> Tuple[int, int, Dict]:
        metrics: Dict = {
            "ane": [],
            "target_prob": [],
            "target_rank": [],
            "num_topk_probs": [],
            "step_status": [],
            "tracked_topk": {"ids": [], "tokens": [], "probs": []},
            "num_topk_snapshots": [],
        }

        snapshot_steps = set(self._snapshot_steps())

        # Step 0: baseline forward WITH grad (reuse for step-1 update)
        self.model.eval()
        logits = self.model(input_ids).logits[0, -1, :]

        with torch.no_grad():
            ane0 = float(self.loss_fn(logits).item())

        pred_before = int(torch.argmax(mask_logits_to_num(logits.detach(), self.num_mask)).item())
        prob0, rank0 = compute_num_prob_rank(logits.detach(), target_idx, self.num_idx_tensor, self.id_to_pos)

        num_probs0 = numerical_softmax(logits.detach(), self.num_idx_tensor)
        topk0 = _build_topk_snapshot(num_probs0, self.num_idx_tensor, self.num_tokens, self.num_topk)
        last_topk = topk0

        # tracked tokens: fixed from baseline numerical distribution
        tracked_ids = [t["idx"] for t in topk0[: self.tracked_topk]]
        tracked_tokens = [t["token"] for t in topk0[: self.tracked_topk]]
        metrics["tracked_topk"]["ids"] = tracked_ids
        metrics["tracked_topk"]["tokens"] = tracked_tokens

        def _tracked_probs(num_probs: torch.Tensor) -> List[float]:
            probs = []
            for tid in tracked_ids:
                pos = int(self.id_to_pos[tid].item()) if 0 <= tid < self.id_to_pos.shape[0] else -1
                probs.append(float(num_probs[pos].item()) if pos >= 0 else 0.0)
            return probs

        metrics["ane"].append(ane0)
        metrics["target_prob"].append(float(prob0))
        metrics["target_rank"].append(int(rank0))
        metrics["num_topk_probs"].append([t["prob"] for t in topk0])
        metrics["step_status"].append("baseline")
        metrics["tracked_topk"]["probs"].append(_tracked_probs(num_probs0))

        if 0 in snapshot_steps:
            metrics["num_topk_snapshots"].append({"step": 0, "tokens": topk0})

        if self.steps <= 0 or not self.params:
            total_len = self.steps + 1
            for k in ["ane", "target_prob", "target_rank", "num_topk_probs", "step_status"]:
                _pad_list(metrics[k], total_len)
            _pad_list(metrics["tracked_topk"]["probs"], total_len)
            if not metrics["num_topk_snapshots"] or metrics["num_topk_snapshots"][-1]["step"] != self.steps:
                metrics["num_topk_snapshots"].append({"step": self.steps, "tokens": last_topk})
            return pred_before, pred_before, metrics

        optimizer = make_optimizer(self.params, self.lr, self.optimizer, self.momentum)

        logits_current = logits

        for step in range(1, self.steps + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            loss = self.loss_fn(logits_current)
            if torch.isnan(loss) or torch.isinf(loss):
                # keep previous metrics
                metrics["ane"].append(metrics["ane"][-1])
                metrics["target_prob"].append(metrics["target_prob"][-1])
                metrics["target_rank"].append(metrics["target_rank"][-1])
                metrics["num_topk_probs"].append(metrics["num_topk_probs"][-1])
                metrics["tracked_topk"]["probs"].append(metrics["tracked_topk"]["probs"][-1])
                metrics["step_status"].append("loss_nan")
                if step in snapshot_steps:
                    metrics["num_topk_snapshots"].append({"step": int(step), "tokens": last_topk})
                continue

            loss.backward()
            healthy, status = check_numerical_health(loss, self.params, self.max_grad_norm)
            if not healthy:
                optimizer.zero_grad(set_to_none=True)
                metrics["ane"].append(metrics["ane"][-1])
                metrics["target_prob"].append(metrics["target_prob"][-1])
                metrics["target_rank"].append(metrics["target_rank"][-1])
                metrics["num_topk_probs"].append(metrics["num_topk_probs"][-1])
                metrics["tracked_topk"]["probs"].append(metrics["tracked_topk"]["probs"][-1])
                metrics["step_status"].append(status)
                if step in snapshot_steps:
                    metrics["num_topk_snapshots"].append({"step": int(step), "tokens": last_topk})
                continue

            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            optimizer.step()

            # Forward once after update: used for (a) recording this step and (b) next step's update
            if step < self.steps:
                self.model.eval()
                logits_current = self.model(input_ids).logits[0, -1, :]
            else:
                self.model.eval()
                with torch.no_grad():
                    logits_current = self.model(input_ids).logits[0, -1, :]

            with torch.no_grad():
                ane_v = float(self.loss_fn(logits_current).item())

            prob, rank = compute_num_prob_rank(logits_current.detach(), target_idx, self.num_idx_tensor, self.id_to_pos)
            num_probs = numerical_softmax(logits_current.detach(), self.num_idx_tensor)
            topk = _build_topk_snapshot(num_probs, self.num_idx_tensor, self.num_tokens, self.num_topk)
            last_topk = topk

            metrics["ane"].append(ane_v)
            metrics["target_prob"].append(float(prob))
            metrics["target_rank"].append(int(rank))
            metrics["num_topk_probs"].append([t["prob"] for t in topk])
            metrics["tracked_topk"]["probs"].append(_tracked_probs(num_probs))
            metrics["step_status"].append("ok")

            if step in snapshot_steps:
                metrics["num_topk_snapshots"].append({"step": int(step), "tokens": topk})

        total_len = self.steps + 1
        for k in ["ane", "target_prob", "target_rank", "num_topk_probs", "step_status"]:
            _pad_list(metrics[k], total_len)
        _pad_list(metrics["tracked_topk"]["probs"], total_len)

        if not metrics["num_topk_snapshots"] or metrics["num_topk_snapshots"][-1]["step"] != self.steps:
            metrics["num_topk_snapshots"].append({"step": self.steps, "tokens": last_topk})

        pred_after = int(torch.argmax(mask_logits_to_num(logits_current.detach(), self.num_mask)).item())
        return pred_before, pred_after, metrics
