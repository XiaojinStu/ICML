"""TTA engine (v5.1) supporting multiple losses and trainable-parameter ablations.

重点优化：复用“更新后”的 logits 作为下一步的“更新前”输入，避免每步多做一次 forward。
在不改变优化轨迹/最终性能的前提下，forward 次数从约 2*steps+2 降为 steps+1。
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
            # Copy directly to CPU to avoid a GPU-side clone peak (important for 8B+).
            t = p.detach().to("cpu", copy=True)
        else:
            t = p.detach().clone()
        backup.append(t)
    return backup


def restore_params(params: List[torch.nn.Parameter], backup: List[torch.Tensor], from_cpu: bool) -> None:
    # `copy_` supports CPU->GPU without allocating an intermediate GPU tensor.
    for p, saved in zip(params, backup):
        p.data.copy_(saved)


def _topk_probs(num_probs: torch.Tensor, num_idx_tensor: torch.Tensor, topk: int) -> Tuple[List[float], List[int]]:
    topk_probs, topk_pos = torch.topk(num_probs, min(topk, num_probs.shape[0]))
    probs = [float(p) for p in topk_probs.tolist()]
    ids = [int(num_idx_tensor[p].item()) for p in topk_pos.tolist()]
    return probs, ids


def _record_step(metrics: Dict, logits: torch.Tensor, target_idx: int, num_mask, num_idx_tensor, id_to_pos, loss_value: float, status: str, snapshot: bool, num_topk: int):
    pred = int(torch.argmax(mask_logits_to_num(logits, num_mask)).item())
    prob, rank = compute_num_prob_rank(logits, target_idx, num_idx_tensor, id_to_pos)

    num_probs = numerical_softmax(logits, num_idx_tensor)
    top_probs, top_ids = _topk_probs(num_probs, num_idx_tensor, num_topk)

    metrics["loss"].append(float(loss_value))
    metrics["target_prob"].append(float(prob))
    metrics["target_rank"].append(int(rank))
    metrics["num_topk_probs"].append(top_probs)
    metrics["status"].append(status)

    if snapshot:
        metrics["num_topk_snapshots"].append(
            {"step": len(metrics["status"]) - 1, "topk": [{"idx": i, "prob": p} for i, p in zip(top_ids, top_probs)]}
        )

    return pred


class TTAEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        params: List[torch.nn.Parameter],
        num_mask: torch.Tensor,
        num_idx_tensor: torch.Tensor,
        id_to_pos: torch.Tensor,
        loss_fn,
        steps: int,
        lr: float,
        grad_clip: float = 1.0,
        max_grad_norm: float = 1e6,
        optimizer: str = "sgd",
        momentum: float = 0.0,
        snapshot_steps: List[int] | None = None,
        num_topk: int = 10,
    ) -> None:
        self.model = model
        self.params = params
        self.num_mask = num_mask
        self.num_idx_tensor = num_idx_tensor
        self.id_to_pos = id_to_pos
        self.loss_fn = loss_fn
        self.steps = steps
        self.lr = lr
        self.grad_clip = grad_clip
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.momentum = momentum
        self.snapshot_steps = sorted(set(snapshot_steps or [0, steps]))
        self.num_topk = num_topk

    def run_token(self, input_ids: torch.Tensor, target_idx: int) -> Tuple[int, int, Dict]:
        metrics = {
            "loss": [],
            "target_prob": [],
            "target_rank": [],
            "num_topk_probs": [],
            "num_topk_snapshots": [],
            "status": [],
        }

        # Step 0: baseline forward WITH grad (reuse for step-1 update)
        self.model.eval()
        logits = self.model(input_ids).logits[0, -1, :]

        with torch.no_grad():
            loss0 = float(self.loss_fn(logits).item())

        pred_before = _record_step(
            metrics,
            logits.detach(),
            target_idx,
            self.num_mask,
            self.num_idx_tensor,
            self.id_to_pos,
            loss0,
            status="baseline",
            snapshot=0 in self.snapshot_steps,
            num_topk=self.num_topk,
        )

        if self.steps <= 0:
            return pred_before, pred_before, metrics

        optimizer = make_optimizer(self.params, self.lr, self.optimizer, self.momentum)

        # Reuse updated logits for the next iteration (avoid double forward)
        logits_current = logits

        for step in range(1, self.steps + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            loss = self.loss_fn(logits_current)
            if torch.isnan(loss) or torch.isinf(loss):
                # keep previous metrics
                _record_step(
                    metrics,
                    logits_current.detach(),
                    target_idx,
                    self.num_mask,
                    self.num_idx_tensor,
                    self.id_to_pos,
                    metrics["loss"][-1],
                    status="loss_nan",
                    snapshot=False,
                    num_topk=self.num_topk,
                )
                continue

            loss.backward()
            healthy, status = check_numerical_health(loss, self.params, self.max_grad_norm)
            if not healthy:
                optimizer.zero_grad(set_to_none=True)
                _record_step(
                    metrics,
                    logits_current.detach(),
                    target_idx,
                    self.num_mask,
                    self.num_idx_tensor,
                    self.id_to_pos,
                    metrics["loss"][-1],
                    status=status,
                    snapshot=False,
                    num_topk=self.num_topk,
                )
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

            _record_step(
                metrics,
                logits_current.detach(),
                target_idx,
                self.num_mask,
                self.num_idx_tensor,
                self.id_to_pos,
                float(loss.item()),
                status="ok",
                snapshot=step in self.snapshot_steps,
                num_topk=self.num_topk,
            )

        pred_after = int(torch.argmax(mask_logits_to_num(logits_current, self.num_mask)).item())
        return pred_before, pred_after, metrics
