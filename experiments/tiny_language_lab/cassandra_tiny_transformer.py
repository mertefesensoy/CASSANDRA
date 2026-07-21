from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DEFAULT_CORPUS = Path(__file__).with_name("corpus") / "tiny_seed.txt"
COUNT_NGRAM_CACHE: dict[tuple[object, ...], tuple[object, dict[str, object]]] = {}
CURRICULUM_SCORE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, dict[str, object]]] = {}
RECENCY_WEIGHT_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
CURRICULUM_TOP_POOL_FRACTION = 0.10
DYNAMIC_CURRICULUM_HIGH_LOSS_FRACTION = 0.10
DYNAMIC_CURRICULUM_DELTA_EMA = 0.50
DYNAMIC_CURRICULUM_DELTA_THRESHOLD = 1e-4
DEFAULT_RECENCY_TAU = 96.0
DEFAULT_RECENCY_LAMBDA = 0.0
BASE_PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class Codec:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


@dataclass(frozen=True)
class SparseBackoffNgramPrior:
    lower_logits: torch.Tensor
    ctx_keys: torch.Tensor
    ctx_logits: torch.Tensor
    vocab_size: int
    order: int
    min_count: int

    def to(self, device: str | torch.device) -> "SparseBackoffNgramPrior":
        return SparseBackoffNgramPrior(
            lower_logits=self.lower_logits.to(device),
            ctx_keys=self.ctx_keys.to(device),
            ctx_logits=self.ctx_logits.to(device),
            vocab_size=self.vocab_size,
            order=self.order,
            min_count=self.min_count,
        )

    def cpu(self) -> "SparseBackoffNgramPrior":
        return self.to("cpu")

    def numel(self) -> int:
        return int(self.lower_logits.numel() + self.ctx_keys.numel() + self.ctx_logits.numel())

    def tensor_bytes(self) -> dict[str, int]:
        lower_bytes = int(self.lower_logits.numel() * self.lower_logits.element_size())
        key_bytes = int(self.ctx_keys.numel() * self.ctx_keys.element_size())
        logit_bytes = int(self.ctx_logits.numel() * self.ctx_logits.element_size())
        return {
            "lower_logits_bytes": lower_bytes,
            "ctx_keys_bytes": key_bytes,
            "ctx_logits_bytes": logit_bytes,
            "total_bytes": lower_bytes + key_bytes + logit_bytes,
        }


PriorLogits = torch.Tensor | SparseBackoffNgramPrior


def prior_vocab_size(base_logits: PriorLogits) -> int:
    if isinstance(base_logits, SparseBackoffNgramPrior):
        return base_logits.vocab_size
    return int(base_logits.size(-1))


def build_codec(text: str, vocab_chars: str = "") -> Codec:
    if vocab_chars:
        chars = list(dict.fromkeys(vocab_chars))
        missing = sorted(set(text) - set(chars))
        if missing:
            rendered = "".join(missing).encode("unicode_escape").decode("ascii")
            raise ValueError(f"--vocab-chars is missing corpus characters: {rendered}")
    else:
        chars = sorted(set(text))
    return Codec(
        chars=chars,
        stoi={ch: idx for idx, ch in enumerate(chars)},
        itos={idx: ch for idx, ch in enumerate(chars)},
    )


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) < 20:
        raise ValueError(f"Corpus is too short: {path}")
    return text


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    if matrix.ndim < 2:
        raise ValueError("Muon orthogonalization expects a matrix or batched matrices")
    if steps < 1:
        raise ValueError("--muon-ns-steps must be positive")

    wide = matrix.size(-2) <= matrix.size(-1)
    x = matrix.bfloat16()
    if not wide:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    if not wide:
        x = x.mT
    return x


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float,
    ns_steps: int,
    nesterov: bool,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape).to(dtype=grad.dtype)


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_corrected = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, object]]) -> None:
        if not param_groups:
            raise ValueError("Muon optimizer received no parameter groups")
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Muon parameter groups require use_muon")
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda parameter: parameter.numel(), reverse=True)
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        float(group["momentum"]),
                        int(group["ns_steps"]),
                        bool(group["nesterov"]),
                    )
                    parameter.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
                    parameter.add_(update, alpha=-float(group["lr"]))
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        int(state["step"]),
                        group["betas"],
                        float(group["eps"]),
                    )
                    parameter.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
                    parameter.add_(update, alpha=-float(group["lr"]))
        return loss


def split_at_index(ids_len: int, val_fraction: float, block_size: int) -> int:
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("--val-fraction must be between 0.05 and 0.5")

    split_at = max(block_size + 2, int(ids_len * (1.0 - val_fraction)))
    split_at = min(split_at, ids_len - block_size - 2)
    return split_at


def split_ids(ids: list[int], val_fraction: float, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    split_at = split_at_index(len(ids), val_fraction, block_size)
    if split_at <= block_size or len(ids) - split_at <= block_size:
        raise ValueError("Corpus is too short for this --block-size and --val-fraction")

    train = torch.tensor(ids[:split_at], dtype=torch.long, device=device)
    val = torch.tensor(ids[split_at:], dtype=torch.long, device=device)
    return train, val


class ShardedBatchSampler:
    def __init__(self, paths: list[Path], codec: Codec, block_size: int, device: str) -> None:
        if not paths:
            raise ValueError("No shard files found")
        self.paths = paths
        self.codec = codec
        self.block_size = block_size
        self.device = device
        self.shards = [path.read_text(encoding="utf-8") for path in paths]
        self.lengths = [len(shard) for shard in self.shards]
        self.start_counts = [max(length - block_size, 0) for length in self.lengths]
        self.cumulative_starts: list[int] = []
        total = 0
        for count in self.start_counts:
            total += count
            self.cumulative_starts.append(total)
        self.total_starts = total
        self.total_chars = sum(self.lengths)
        if self.total_starts <= 0:
            raise ValueError("Shard corpus is too short for the requested block size")

    @classmethod
    def from_dir(cls, shard_dir: Path, codec: Codec, block_size: int, device: str) -> "ShardedBatchSampler":
        paths = sorted(shard_dir.glob("train_*.txt"))
        if not paths:
            paths = sorted(path for path in shard_dir.glob("*.txt") if path.name.lower() != "val.txt")
        return cls(paths, codec, block_size, device)

    def _read_window(self, shard_index: int, local_start: int) -> torch.Tensor:
        chunk = self.shards[shard_index][local_start : local_start + self.block_size + 1]
        if len(chunk) != self.block_size + 1:
            raise ValueError(f"Short read from shard {self.paths[shard_index]}")
        ids = self.codec.encode(chunk)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_batch(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        global_starts = torch.randint(
            self.total_starts,
            (batch_size,),
            generator=generator,
            device=self.device,
        )
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for global_start_tensor in global_starts:
            global_start = int(global_start_tensor.item())
            shard_index = bisect.bisect_right(self.cumulative_starts, global_start)
            previous_total = 0 if shard_index == 0 else self.cumulative_starts[shard_index - 1]
            local_start = global_start - previous_total
            ids = self._read_window(shard_index, local_start)
            xs.append(ids[:-1])
            ys.append(ids[1:])
        return torch.stack(xs), torch.stack(ys)

    def eval_tensor(self, max_chars: int) -> torch.Tensor:
        if max_chars <= self.block_size + 1:
            raise ValueError("--stream-train-eval-chars must be larger than --block-size")
        chunks: list[str] = []
        remaining = max_chars
        for path in self.paths:
            if remaining <= 0:
                break
            text = path.read_text(encoding="utf-8")[:remaining]
            chunks.append(text)
            remaining -= len(text)
        text = "".join(chunks)
        if len(text) <= self.block_size + 1:
            raise ValueError("Shard evaluation prefix is too short for the requested block size")
        return torch.tensor(self.codec.encode(text), dtype=torch.long, device=self.device)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")

    starts = torch.randint(num_starts, (batch_size,), generator=generator, device=device)
    return make_batch(data, starts, block_size)


def make_batch(
    data: torch.Tensor,
    starts: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([data[int(i) : int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + block_size + 1] for i in starts])
    return x, y


def lookup_sparse_backoff_logits(idx: torch.Tensor, prior: SparseBackoffNgramPrior) -> torch.Tensor:
    logits = lookup_base_logits(idx, prior.lower_logits).clone()
    batch, tokens = idx.shape
    order = prior.order
    if order != 5:
        raise ValueError("Sparse backoff prior currently supports order 5 only")
    if tokens < order or prior.ctx_keys.numel() == 0:
        return logits

    rows = tokens - order + 1
    context_keys = torch.zeros((batch, rows), dtype=torch.long, device=idx.device)
    for offset in range(order):
        context_keys = context_keys * prior.vocab_size + idx[:, offset : offset + rows]
    flat_keys = context_keys.reshape(-1)

    positions = torch.searchsorted(prior.ctx_keys, flat_keys)
    safe_positions = positions.clamp(max=max(int(prior.ctx_keys.numel()) - 1, 0))
    hits = (positions < prior.ctx_keys.numel()) & (prior.ctx_keys[safe_positions] == flat_keys)
    if not bool(hits.any()):
        return logits

    replacement = logits[:, order - 1 :, :].clone().view(-1, prior.vocab_size)
    replacement[hits] = prior.ctx_logits[safe_positions[hits]].to(dtype=logits.dtype)
    logits[:, order - 1 :, :] = replacement.view(batch, rows, prior.vocab_size)
    return logits


def lookup_base_logits(idx: torch.Tensor, base_logits: PriorLogits) -> torch.Tensor:
    if isinstance(base_logits, SparseBackoffNgramPrior):
        return lookup_sparse_backoff_logits(idx, base_logits)
    if base_logits.dim() == 2:
        return base_logits[idx]
    if base_logits.dim() < 3:
        raise ValueError("base_logits must be rank 2 or greater")

    _batch, tokens = idx.shape
    prior_order = base_logits.dim() - 1
    context_indices = []
    for dim_index in range(prior_order - 1):
        history_offset = prior_order - 1 - dim_index
        start_token = base_logits.size(dim_index) - 1
        context = torch.empty_like(idx)
        context[:, :history_offset] = start_token
        if tokens > history_offset:
            context[:, history_offset:] = idx[:, :-history_offset]
        context_indices.append(context)
    context_indices.append(idx)
    return base_logits[tuple(context_indices)]


def mix_recency_base_logits(
    idx: torch.Tensor,
    count_base_logits: torch.Tensor,
    recency_tau: float,
    recency_lambda: float,
) -> torch.Tensor:
    if recency_lambda <= 0:
        return count_base_logits
    if not 0 <= recency_lambda <= 1:
        raise ValueError("recency_lambda must be in [0, 1]")
    if recency_tau <= 0:
        raise ValueError("recency_tau must be positive")

    _batch, tokens = idx.shape
    if tokens <= 1:
        return count_base_logits

    vocab_size = count_base_logits.size(-1)
    dtype = count_base_logits.dtype
    device = idx.device
    cache_key = (tokens, float(recency_tau), str(device), dtype)
    weights = RECENCY_WEIGHT_CACHE.get(cache_key)
    if weights is None:
        positions = torch.arange(tokens, device=device)
        distances = positions.view(1, tokens) - positions.view(tokens, 1) + 1
        history_mask = distances > 0
        weights = torch.zeros((tokens, tokens), dtype=dtype, device=device)
        weights[history_mask] = torch.exp(-distances[history_mask].to(dtype) / recency_tau)
        RECENCY_WEIGHT_CACHE[cache_key] = weights

    one_hot = F.one_hot(idx, num_classes=vocab_size).to(dtype)
    recency_scores = torch.einsum("biv,it->btv", one_hot, weights)
    recency_totals = recency_scores.sum(dim=-1, keepdim=True)
    has_history = recency_totals > 0
    recency_probs = recency_scores / recency_totals.clamp_min(BASE_PROB_FLOOR)
    count_probs = count_base_logits.exp()
    mixed_probs = (1.0 - recency_lambda) * count_probs + recency_lambda * recency_probs
    mixed_probs = torch.where(has_history, mixed_probs, count_probs)
    return mixed_probs.clamp_min(BASE_PROB_FLOOR).log()


@torch.no_grad()
def score_windows_by_prior_loss(
    data: torch.Tensor,
    block_size: int,
    base_logits: PriorLogits,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")

    scores = torch.empty(num_starts, dtype=torch.float32, device=device)
    windows = data.unfold(0, block_size + 1, 1)
    score_batch_size = max(batch_size, 8192)
    vocab_size = prior_vocab_size(base_logits)
    for offset in range(0, num_starts, score_batch_size):
        stop = min(offset + score_batch_size, num_starts)
        batch = windows[offset:stop]
        xb = batch[:, :-1]
        yb = batch[:, 1:]
        logits = lookup_base_logits(xb, base_logits)
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
            reduction="none",
        ).view_as(yb)
        scores[offset:stop] = token_loss.mean(dim=1)
    return scores


def build_prior_loss_curriculum(
    data: torch.Tensor,
    block_size: int,
    base_logits: PriorLogits,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    start = time.perf_counter()
    scores = score_windows_by_prior_loss(data, block_size, base_logits, batch_size, device)
    num_starts = int(scores.numel())
    pool_starts = max(batch_size, round(num_starts * CURRICULUM_TOP_POOL_FRACTION))
    pool_starts = max(1, min(num_starts, pool_starts))
    pool_scores, pool_indices = torch.topk(scores, k=pool_starts)
    elapsed = time.perf_counter() - start
    report = {
        "curriculum_pool_fraction": round(pool_starts / max(num_starts, 1), 6),
        "curriculum_pool_starts": pool_starts,
        "curriculum_total_starts": num_starts,
        "curriculum_score_mean": round(float(scores.mean().item()), 6),
        "curriculum_score_min": round(float(scores.min().item()), 6),
        "curriculum_score_max": round(float(scores.max().item()), 6),
        "curriculum_pool_score_mean": round(float(pool_scores.mean().item()), 6),
        "curriculum_pool_score_min": round(float(pool_scores.min().item()), 6),
        "curriculum_pool_score_max": round(float(pool_scores.max().item()), 6),
        "curriculum_score_seconds": round(elapsed, 4),
    }
    return pool_indices, report


def get_prior_loss_filtered_batch(
    data: torch.Tensor,
    high_loss_starts: torch.Tensor,
    block_size: int,
    batch_size: int,
    curriculum_fraction: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")
    if len(high_loss_starts) <= 0:
        raise ValueError("No high-loss starts available for curriculum filter")

    high_count = round(batch_size * curriculum_fraction)
    high_count = max(1, min(batch_size, high_count))
    random_count = batch_size - high_count
    high_picks = torch.randint(len(high_loss_starts), (high_count,), generator=generator, device=device)
    high_starts = high_loss_starts[high_picks]
    if random_count > 0:
        random_starts = torch.randint(num_starts, (random_count,), generator=generator, device=device)
        starts = torch.cat([high_starts, random_starts])
    else:
        starts = high_starts
    return make_batch(data, starts, block_size)


def build_dynamic_curriculum_pool(
    data: torch.Tensor,
    block_size: int,
    pool_size: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")
    if pool_size <= 0:
        raise ValueError("--curriculum-pool-size must be positive")

    pool_count = min(pool_size, num_starts)
    pool_generator = torch.Generator(device=device)
    pool_generator.manual_seed(seed + 1_000_003)
    if pool_count == num_starts:
        starts = torch.arange(num_starts, dtype=torch.long, device=device)
    else:
        starts = torch.randperm(num_starts, generator=pool_generator, device=device)[:pool_count]
    report = {
        "curriculum_pool_starts": int(starts.numel()),
        "curriculum_total_starts": num_starts,
        "curriculum_pool_fraction": round(float(starts.numel()) / max(num_starts, 1), 6),
    }
    return starts, report


@torch.no_grad()
def score_windows_by_model_loss(
    model: nn.Module,
    data: torch.Tensor,
    starts: torch.Tensor,
    block_size: int,
    batch_size: int,
    base_logits: PriorLogits,
    residual_scale: float,
    device: str,
) -> torch.Tensor:
    if len(starts) <= 0:
        raise ValueError("Cannot score an empty dynamic curriculum pool")

    was_training = model.training
    model.eval()
    windows = data.unfold(0, block_size + 1, 1)
    losses = torch.empty(len(starts), dtype=torch.float32, device=device)
    score_batch_size = max(batch_size, 1024)
    vocab_size = prior_vocab_size(base_logits)
    for offset in range(0, len(starts), score_batch_size):
        stop = min(offset + score_batch_size, len(starts))
        batch = windows[starts[offset:stop]]
        xb = batch[:, :-1]
        yb = batch[:, 1:]
        logits, _ = model(xb, base_logits=base_logits, residual_scale=residual_scale)
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
            reduction="none",
        ).view_as(yb)
        losses[offset:stop] = token_loss.mean(dim=1)
    if was_training:
        model.train()
    return losses


def select_dynamic_reducible_starts(
    pool_starts: torch.Tensor,
    current_losses: torch.Tensor,
    smoothed_delta: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    high_loss_count = max(batch_size, round(len(pool_starts) * DYNAMIC_CURRICULUM_HIGH_LOSS_FRACTION))
    high_loss_count = max(1, min(len(pool_starts), high_loss_count))
    high_loss_values, high_loss_indices = torch.topk(current_losses, k=high_loss_count)
    selected_mask = smoothed_delta[high_loss_indices] > DYNAMIC_CURRICULUM_DELTA_THRESHOLD
    selected_indices = high_loss_indices[selected_mask]
    selected_starts = pool_starts[selected_indices]
    if len(selected_indices) > 0:
        selected_current_mean = float(current_losses[selected_indices].mean().item())
        selected_delta_mean = float(smoothed_delta[selected_indices].mean().item())
    else:
        selected_current_mean = 0.0
        selected_delta_mean = 0.0
    report = {
        "curriculum_high_loss_pool_starts": high_loss_count,
        "curriculum_selected_starts": int(selected_starts.numel()),
        "curriculum_selected_fraction": round(int(selected_starts.numel()) / max(len(pool_starts), 1), 6),
        "curriculum_current_loss_mean": round(float(current_losses.mean().item()), 6),
        "curriculum_current_loss_min": round(float(current_losses.min().item()), 6),
        "curriculum_current_loss_max": round(float(current_losses.max().item()), 6),
        "curriculum_high_loss_score_min": round(float(high_loss_values.min().item()), 6),
        "curriculum_delta_mean": round(float(smoothed_delta.mean().item()), 6),
        "curriculum_delta_min": round(float(smoothed_delta.min().item()), 6),
        "curriculum_delta_max": round(float(smoothed_delta.max().item()), 6),
        "curriculum_selected_loss_mean": round(selected_current_mean, 6),
        "curriculum_selected_delta_mean": round(selected_delta_mean, 6),
    }
    return selected_starts, report


def build_marker_weights(text: str, marker: str, limit: int, device: str) -> torch.Tensor:
    weights = torch.zeros(limit, dtype=torch.float32, device=device)
    if not marker:
        return weights

    search_at = 0
    while True:
        marker_at = text.find(marker, search_at)
        if marker_at < 0:
            break
        target_at = marker_at + len(marker)
        if target_at < limit:
            weights[target_at] = 1.0
        search_at = marker_at + 1
    return weights


def build_position_weights(positions: list[int], limit: int, device: str) -> torch.Tensor:
    weights = torch.zeros(limit, dtype=torch.float32, device=device)
    for position in positions:
        if 0 <= position < limit:
            weights[position] = 1.0
    return weights


def build_choice_tensors(
    text: str,
    positions: list[int],
    codec: Codec,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    candidate_ids = sorted({codec.stoi[text[position]] for position in positions if text[position] in codec.stoi})
    if len(candidate_ids) < 2:
        raise ValueError("Copy choice loss needs at least two verified answer candidates")

    choice_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    lookup = torch.full((len(codec.chars),), -1, dtype=torch.long, device=device)
    for choice_index, token_id in enumerate(candidate_ids):
        lookup[token_id] = choice_index
    choice_chars = [codec.itos[token_id] for token_id in candidate_ids]
    return choice_ids, lookup, choice_chars


def find_verified_copy_targets(
    text: str,
    marker: str,
    limit: int,
    verify_mode: str,
    key_marker: str = "key=",
) -> list[int]:
    if not marker:
        return []

    targets: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        key_at = stripped.find(key_marker)
        marker_at = stripped.find(marker)
        if key_at >= 0 and marker_at >= 0:
            key_value_at = key_at + len(key_marker)
            answer_value_at = marker_at + len(marker)
            valid_pair = key_value_at < len(stripped) and answer_value_at < len(stripped)
            if verify_mode == "identity":
                valid_pair = valid_pair and stripped[key_value_at] == stripped[answer_value_at]
            elif verify_mode == "key-answer":
                valid_pair = valid_pair
            else:
                raise ValueError(f"Unknown copy verify mode: {verify_mode}")
            if valid_pair:
                target_at = offset + answer_value_at
                if target_at < limit:
                    targets.append(target_at)
        offset += len(line)
    return targets


def build_answer_starts(
    target_positions: list[int],
    block_size: int,
    data_len: int,
    device: str,
) -> torch.Tensor:
    starts, _ = build_answer_windows(target_positions, block_size, data_len, device)
    return starts


def build_answer_windows(
    target_positions: list[int],
    block_size: int,
    data_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts, offsets, _ = build_answer_window_records(target_positions, block_size, data_len, device)
    return starts, offsets


def build_answer_window_records(
    target_positions: list[int],
    block_size: int,
    data_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_starts = data_len - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")

    starts: list[int] = []
    offsets: list[int] = []
    positions: list[int] = []
    for target_at in target_positions:
        if target_at <= 0 or target_at >= data_len:
            continue
        start = max(0, target_at - block_size)
        start = min(start, num_starts - 1)
        if start < target_at <= start + block_size:
            starts.append(start)
            offsets.append(target_at - start - 1)
            positions.append(target_at)

    if not starts:
        raise ValueError("No usable answer-anchored starts found for copy sampler")
    return (
        torch.tensor(starts, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
    )


def make_weight_batch(
    marker_weights: torch.Tensor,
    starts: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    return torch.stack([marker_weights[int(i) + 1 : int(i) + block_size + 1] for i in starts])


def get_weighted_batch(
    data: torch.Tensor,
    marker_weights: torch.Tensor | None,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")

    starts = torch.randint(num_starts, (batch_size,), generator=generator, device=device)
    xb, yb = make_batch(data, starts, block_size)
    wb = make_weight_batch(marker_weights, starts, block_size) if marker_weights is not None else None
    return xb, yb, wb


def get_answer_sampled_batch(
    data: torch.Tensor,
    marker_weights: torch.Tensor,
    answer_starts: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    picks = torch.randint(len(answer_starts), (batch_size,), generator=generator, device=device)
    starts = answer_starts[picks]
    xb, yb = make_batch(data, starts, block_size)
    wb = make_weight_batch(marker_weights, starts, block_size)
    return xb, yb, wb


def get_mixed_sampled_batch(
    data: torch.Tensor,
    marker_weights: torch.Tensor,
    answer_starts: torch.Tensor,
    block_size: int,
    batch_size: int,
    answer_fraction: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")

    answer_count = round(batch_size * answer_fraction)
    answer_count = max(1, min(batch_size, answer_count))
    random_count = batch_size - answer_count

    answer_picks = torch.randint(len(answer_starts), (answer_count,), generator=generator, device=device)
    starts = [answer_starts[answer_picks]]
    if random_count > 0:
        starts.append(torch.randint(num_starts, (random_count,), generator=generator, device=device))

    all_starts = torch.cat(starts)
    xb, yb = make_batch(data, all_starts, block_size)
    wb = make_weight_batch(marker_weights, all_starts, block_size)
    return xb, yb, wb


def line_bounds(text: str, position: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end < 0:
        end = len(text)
    return start, end


def extract_key_value(line: str, key_marker: str = "key=") -> str | None:
    key_at = line.find(key_marker)
    if key_at < 0:
        return None
    key_value_at = key_at + len(key_marker)
    if key_value_at >= len(line):
        return None
    return line[key_value_at]


def extract_marker_value(line: str, marker: str) -> str | None:
    marker_at = line.find(marker)
    if marker_at < 0:
        return None
    value_at = marker_at + len(marker)
    if value_at >= len(line):
        return None
    return line[value_at]


def build_copy_memory(
    text: str,
    marker: str,
    limit: int,
    key_marker: str = "key=",
) -> tuple[dict[str, str], dict[str, int]]:
    observations: dict[str, Counter[str]] = {}
    for line in text[:limit].splitlines():
        key = extract_key_value(line, key_marker)
        answer = extract_marker_value(line, marker)
        if key is None or answer is None:
            continue
        observations.setdefault(key, Counter())[answer] += 1

    memory: dict[str, str] = {}
    conflicts = 0
    for key, counts in observations.items():
        if len(counts) > 1:
            conflicts += 1
        memory[key] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    return memory, {
        "entries": len(memory),
        "observations": sum(sum(counts.values()) for counts in observations.values()),
        "conflicts": conflicts,
    }


def correction_line(text: str, position: int, marker: str, template: str) -> str:
    target = text[position]
    start, end = line_bounds(text, position)
    line = text[start:end]
    key = extract_key_value(line) or target
    if template == "compact":
        return f"key={key} {marker}{target}\n"

    if template == "full":
        return f"{line}\n"
    if template == "prefix":
        return f"{line[: position - start + 1]}\n"
    if template == "focus":
        noise_at = line.find("noise=")
        noise_part = ""
        if noise_at >= 0 and noise_at + len("noise=") < len(line):
            noise_part = f" noise={line[noise_at + len('noise=')]}"
        return f"key={key}{noise_part} {marker}{target}\n"

    raise ValueError(f"Unknown correction template: {template}")


def retrieval_training_line(text: str, position: int, marker: str, template: str) -> str:
    target = text[position]
    start, end = line_bounds(text, position)
    line = text[start:end]
    marker_at = line.find(marker)
    if marker_at < 0:
        return ""
    prefix = line[: marker_at + len(marker)]
    retrieval = copy_probe_retrieval_context(line, prefix, target, marker, template, extract_key_value(line))
    return f"{retrieval}{prefix}{target}\n"


def build_correction_data(
    text: str,
    target_positions: torch.Tensor,
    codec: Codec,
    marker: str,
    template: str,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    positions = [int(position) for position in target_positions.tolist()]
    lines = [
        correction_line(text, position, marker, template)
        for position in positions
        if text[position] in codec.stoi
    ]
    if not lines:
        raise ValueError("No correction examples could be built")

    correction_text = "".join(lines)
    correction_ids = codec.encode(correction_text)
    while len(correction_ids) <= block_size:
        correction_ids.extend(correction_ids)

    data = torch.tensor(correction_ids, dtype=torch.long, device=device)
    weights = build_marker_weights(correction_text, marker, len(correction_text), device)
    if len(weights) < len(data):
        repeat_count = math.ceil(len(data) / len(weights))
        weights = weights.repeat(repeat_count)[: len(data)]
    return data, weights, len(lines), len(correction_text)


def build_retrieval_data(
    text: str,
    target_positions: torch.Tensor,
    codec: Codec,
    marker: str,
    template: str,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    positions = [int(position) for position in target_positions.tolist()]
    lines = [
        retrieval_training_line(text, position, marker, template)
        for position in positions
        if text[position] in codec.stoi
    ]
    lines = [line for line in lines if line and all(ch in codec.stoi for ch in line)]
    if not lines:
        raise ValueError("No retrieval-use examples could be built")

    retrieval_text = "".join(lines)
    retrieval_ids = codec.encode(retrieval_text)
    while len(retrieval_ids) <= block_size:
        retrieval_ids.extend(retrieval_ids)

    data = torch.tensor(retrieval_ids, dtype=torch.long, device=device)
    weights = build_marker_weights(retrieval_text, marker, len(retrieval_text), device)
    if len(weights) < len(data):
        repeat_count = math.ceil(len(data) / len(weights))
        weights = weights.repeat(repeat_count)[: len(data)]
    return data, weights, len(lines), len(retrieval_text)


def get_correction_mixed_batch(
    train_data: torch.Tensor,
    train_marker_weights: torch.Tensor,
    correction_data: torch.Tensor,
    correction_weights: torch.Tensor,
    block_size: int,
    batch_size: int,
    correction_fraction: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correction_count = round(batch_size * correction_fraction)
    correction_count = max(1, min(batch_size, correction_count))
    random_count = batch_size - correction_count

    xb_parts: list[torch.Tensor] = []
    yb_parts: list[torch.Tensor] = []
    wb_parts: list[torch.Tensor] = []

    xb_corr, yb_corr, wb_corr = get_weighted_batch(
        correction_data,
        correction_weights,
        block_size,
        correction_count,
        generator,
        device,
    )
    if wb_corr is None:
        raise RuntimeError("Expected correction weights during correction sampling")
    xb_parts.append(xb_corr)
    yb_parts.append(yb_corr)
    wb_parts.append(wb_corr)

    if random_count > 0:
        xb_train, yb_train, wb_train = get_weighted_batch(
            train_data,
            train_marker_weights,
            block_size,
            random_count,
            generator,
            device,
        )
        if wb_train is None:
            raise RuntimeError("Expected train weights during mixed correction sampling")
        xb_parts.append(xb_train)
        yb_parts.append(yb_train)
        wb_parts.append(wb_train)

    return torch.cat(xb_parts), torch.cat(yb_parts), torch.cat(wb_parts)


def get_dual_aux_mixed_batch(
    train_data: torch.Tensor,
    train_marker_weights: torch.Tensor,
    first_data: torch.Tensor,
    first_weights: torch.Tensor,
    second_data: torch.Tensor,
    second_weights: torch.Tensor,
    block_size: int,
    batch_size: int,
    aux_fraction: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_aux_count = 2 if batch_size >= 2 else 1
    aux_count = round(batch_size * aux_fraction)
    aux_count = min(batch_size, max(min_aux_count, aux_count))
    first_count = max(1, aux_count // 2)
    second_count = max(1, aux_count - first_count)
    if first_count + second_count > batch_size:
        second_count = batch_size - first_count
    random_count = batch_size - first_count - second_count

    xb_parts: list[torch.Tensor] = []
    yb_parts: list[torch.Tensor] = []
    wb_parts: list[torch.Tensor] = []

    for data, weights, count in (
        (first_data, first_weights, first_count),
        (second_data, second_weights, second_count),
    ):
        if count <= 0:
            continue
        xb_aux, yb_aux, wb_aux = get_weighted_batch(
            data,
            weights,
            block_size,
            count,
            generator,
            device,
        )
        if wb_aux is None:
            raise RuntimeError("Expected auxiliary weights during dual mixed sampling")
        xb_parts.append(xb_aux)
        yb_parts.append(yb_aux)
        wb_parts.append(wb_aux)

    if random_count > 0:
        xb_train, yb_train, wb_train = get_weighted_batch(
            train_data,
            train_marker_weights,
            block_size,
            random_count,
            generator,
            device,
        )
        if wb_train is None:
            raise RuntimeError("Expected train weights during dual mixed sampling")
        xb_parts.append(xb_train)
        yb_parts.append(yb_train)
        wb_parts.append(wb_train)

    return torch.cat(xb_parts), torch.cat(yb_parts), torch.cat(wb_parts)


def get_rehearsal_mixed_batch(
    train_data: torch.Tensor,
    train_marker_weights: torch.Tensor,
    correction_data: torch.Tensor,
    correction_weights: torch.Tensor,
    retrieval_data: torch.Tensor,
    retrieval_weights: torch.Tensor,
    block_size: int,
    batch_size: int,
    retrieval_fraction: float,
    rehearsal_fraction: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    retrieval_count = round(batch_size * retrieval_fraction)
    retrieval_count = max(1, min(batch_size, retrieval_count))
    rehearsal_count = 0
    available_for_rehearsal = batch_size - retrieval_count
    if rehearsal_fraction > 0 and available_for_rehearsal > 0:
        rehearsal_count = round(batch_size * rehearsal_fraction)
        rehearsal_count = max(1, min(available_for_rehearsal, rehearsal_count))
    random_count = batch_size - retrieval_count - rehearsal_count

    xb_parts: list[torch.Tensor] = []
    yb_parts: list[torch.Tensor] = []
    wb_parts: list[torch.Tensor] = []

    for data, weights, count in (
        (retrieval_data, retrieval_weights, retrieval_count),
        (correction_data, correction_weights, rehearsal_count),
    ):
        if count <= 0:
            continue
        xb_aux, yb_aux, wb_aux = get_weighted_batch(
            data,
            weights,
            block_size,
            count,
            generator,
            device,
        )
        if wb_aux is None:
            raise RuntimeError("Expected auxiliary weights during rehearsal sampling")
        xb_parts.append(xb_aux)
        yb_parts.append(yb_aux)
        wb_parts.append(wb_aux)

    if random_count > 0:
        xb_train, yb_train, wb_train = get_weighted_batch(
            train_data,
            train_marker_weights,
            block_size,
            random_count,
            generator,
            device,
        )
        if wb_train is None:
            raise RuntimeError("Expected train weights during rehearsal sampling")
        xb_parts.append(xb_train)
        yb_parts.append(yb_train)
        wb_parts.append(wb_train)

    return torch.cat(xb_parts), torch.cat(yb_parts), torch.cat(wb_parts)


@torch.no_grad()
def mine_failed_copy_starts(
    model: "TinyTransformer",
    data: torch.Tensor,
    target_positions: list[int],
    block_size: int,
    batch_size: int,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
) -> dict[str, object]:
    answer_starts, answer_offsets, answer_positions = build_answer_window_records(
        target_positions,
        block_size,
        len(data),
        device,
    )
    failed_chunks: list[torch.Tensor] = []
    failed_position_chunks: list[torch.Tensor] = []
    correct = 0
    total = 0
    was_training = model.training
    model.eval()

    for offset in range(0, len(answer_starts), batch_size):
        batch_starts = answer_starts[offset : offset + batch_size]
        batch_offsets = answer_offsets[offset : offset + batch_size]
        batch_positions = answer_positions[offset : offset + batch_size]
        xb, yb = make_batch(data, batch_starts, block_size)
        logits, _ = model(xb, base_logits=base_logits, residual_scale=residual_scale)
        rows = torch.arange(len(batch_starts), device=device)
        preds = torch.argmax(logits[rows, batch_offsets], dim=-1)
        targets = yb[rows, batch_offsets]
        matches = preds == targets
        correct += int(matches.sum().item())
        total += int(matches.numel())
        failed = batch_starts[~matches]
        if len(failed) > 0:
            failed_chunks.append(failed)
            failed_position_chunks.append(batch_positions[~matches])

    if was_training:
        model.train()

    failed_starts = torch.cat(failed_chunks) if failed_chunks else answer_starts[:0]
    failed_positions = torch.cat(failed_position_chunks) if failed_position_chunks else answer_positions[:0]
    accuracy = correct / total if total else 0.0
    return {
        "answer_starts": answer_starts,
        "answer_positions": answer_positions,
        "failed_starts": failed_starts,
        "failed_positions": failed_positions,
        "train_copy_cases": total,
        "train_copy_correct": correct,
        "train_copy_accuracy": accuracy,
        "train_copy_failures": int(failed_starts.numel()),
    }


def build_count_logits(ids: torch.Tensor, vocab_size: int, alpha: float) -> torch.Tensor:
    counts = torch.full((vocab_size, vocab_size), alpha, dtype=torch.float32, device=ids.device)
    for prev_id, next_id in zip(ids[:-1].tolist(), ids[1:].tolist()):
        counts[prev_id, next_id] += 1.0
    return counts.log()


def build_count_logits_trigram(
    ids: torch.Tensor,
    vocab_size: int,
    alpha: float,
    backoff: float,
) -> torch.Tensor:
    if backoff < 0:
        raise ValueError("--trigram-backoff must be non-negative")

    unigram_counts = torch.full((vocab_size,), alpha, dtype=torch.float32, device=ids.device)
    for token_id in ids.tolist():
        unigram_counts[token_id] += 1.0
    unigram_probs = unigram_counts / unigram_counts.sum().clamp_min(1.0)

    bigram_counts = torch.full((vocab_size, vocab_size), alpha, dtype=torch.float32, device=ids.device)
    bigram_context_counts = torch.zeros((vocab_size,), dtype=torch.float32, device=ids.device)
    for prev_id, next_id in zip(ids[:-1].tolist(), ids[1:].tolist()):
        bigram_counts[prev_id, next_id] += 1.0
        bigram_context_counts[prev_id] += 1.0
    bigram_probs = bigram_counts / bigram_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)

    trigram_counts = torch.full(
        (vocab_size, vocab_size, vocab_size),
        alpha,
        dtype=torch.float32,
        device=ids.device,
    )
    trigram_context_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=ids.device)
    for prev2_id, prev1_id, next_id in zip(ids[:-2].tolist(), ids[1:-1].tolist(), ids[2:].tolist()):
        trigram_counts[prev2_id, prev1_id, next_id] += 1.0
        trigram_context_counts[prev2_id, prev1_id] += 1.0
    trigram_probs = trigram_counts / trigram_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)

    lower_probs = 0.75 * bigram_probs + 0.25 * unigram_probs.view(1, vocab_size)
    if backoff == 0:
        context_mass = (trigram_context_counts > 0).to(torch.float32)
    else:
        context_mass = trigram_context_counts / (trigram_context_counts + backoff)
    mixed_probs = (
        context_mass.unsqueeze(-1) * trigram_probs
        + (1.0 - context_mass).unsqueeze(-1) * lower_probs.unsqueeze(0)
    )

    start_row = bigram_probs.unsqueeze(0)
    probs = torch.cat([mixed_probs, start_row], dim=0)
    return probs.clamp_min(1e-12).log()


def build_count_logits_ngram(
    ids: torch.Tensor,
    vocab_size: int,
    order: int,
    alpha: float,
    backoff: float,
    probe_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if order < 1 or order > 4:
        raise ValueError("--prior-order currently supports 1, 2, 3, or 4")
    if backoff < 0:
        raise ValueError("--ngram-backoff must be non-negative")

    probs_by_order: dict[int, torch.Tensor] = {}
    coverage: list[dict[str, float | int]] = []
    ids_long = ids.to(dtype=torch.long)
    probe_long = probe_ids.to(dtype=torch.long) if probe_ids is not None else None

    for current_order in range(1, order + 1):
        count_shape = (vocab_size,) * current_order + (vocab_size,)
        rows = max(int(ids_long.numel()) - current_order, 0)
        context_shape = (vocab_size,) * current_order
        if rows > 0:
            context_codes = torch.zeros(rows, dtype=torch.long, device=ids.device)
            for offset in range(current_order):
                context_codes = context_codes * vocab_size + ids_long[offset : offset + rows]
            next_ids = ids_long[current_order : current_order + rows]
            flat_counts = torch.bincount(
                context_codes * vocab_size + next_ids,
                minlength=vocab_size ** (current_order + 1),
            ).to(torch.float32)
            counts = flat_counts.view(count_shape)
            context_counts = torch.bincount(
                context_codes,
                minlength=vocab_size**current_order,
            ).to(torch.float32).view(context_shape)
        else:
            counts = torch.zeros(count_shape, dtype=torch.float32, device=ids.device)
            context_counts = torch.zeros(context_shape, dtype=torch.float32, device=ids.device)
        counts = counts + alpha

        empirical_probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if current_order == 1:
            mixed_probs = empirical_probs
        else:
            lower_probs = probs_by_order[current_order - 1].unsqueeze(0).expand(count_shape)
            if backoff == 0:
                context_mass = (context_counts > 0).to(torch.float32)
            else:
                context_mass = context_counts / (context_counts + backoff)
            mixed_probs = (
                context_mass.unsqueeze(-1) * empirical_probs
                + (1.0 - context_mass).unsqueeze(-1) * lower_probs
            )

        probs_by_order[current_order] = mixed_probs.clamp_min(1e-12)
        observed_contexts = int((context_counts > 0).sum().item())
        total_contexts = int(context_counts.numel())
        if observed_contexts:
            observed_values = context_counts[context_counts > 0]
            mean_observed_count = float(observed_values.mean().item())
            min_observed_count = float(observed_values.min().item())
        else:
            mean_observed_count = 0.0
            min_observed_count = 0.0
        probe_contexts = max(int(probe_long.numel()) - current_order, 0) if probe_long is not None else 0
        probe_observed_contexts = 0
        if probe_contexts:
            probe_context_codes = torch.zeros(probe_contexts, dtype=torch.long, device=ids.device)
            for offset in range(current_order):
                probe_context_codes = probe_context_codes * vocab_size + probe_long[offset : offset + probe_contexts]
            observed_flat = context_counts.reshape(-1) > 0
            probe_observed_contexts = int(observed_flat[probe_context_codes].sum().item())
        coverage.append(
            {
                "order": current_order,
                "contexts": total_contexts,
                "observed_contexts": observed_contexts,
                "coverage": observed_contexts / max(total_contexts, 1),
                "mean_observed_count": mean_observed_count,
                "min_observed_count": min_observed_count,
                "validation_contexts": probe_contexts,
                "validation_observed_contexts": probe_observed_contexts,
                "validation_coverage": probe_observed_contexts / max(probe_contexts, 1),
            }
        )

    if order == 1:
        probs = probs_by_order[1]
    else:
        base_shape = (vocab_size + 1,) * (order - 1) + (vocab_size, vocab_size)
        probs = probs_by_order[1].view((1,) * (order - 1) + (vocab_size, vocab_size)).expand(base_shape).clone()
        start_token = vocab_size
        for lower_order in range(2, order + 1):
            leading_starts = order - lower_order
            index = [start_token] * leading_starts
            index.extend([slice(0, vocab_size)] * (lower_order - 1))
            index.extend([slice(None), slice(None)])
            probs[tuple(index)] = probs_by_order[lower_order]

    report = {
        "prior_order": order,
        "ngram_backoff": backoff,
        "ngram_smoothing": "recursive add-alpha interpolation with context_mass=count/(count+backoff)",
        "ngram_context_coverage": coverage,
    }
    return probs.clamp_min(1e-12).log(), report


def count_probe_context_hits(
    context_counts: torch.Tensor,
    probe_ids: torch.Tensor | None,
    current_order: int,
    vocab_size: int,
    chunk_contexts: int = 2_000_000,
) -> tuple[int, int]:
    if probe_ids is None:
        return 0, 0
    probe_long = probe_ids.detach().to(device="cpu", dtype=torch.long)
    probe_contexts = max(int(probe_long.numel()) - current_order, 0)
    if probe_contexts <= 0:
        return 0, 0

    observed_flat = context_counts.reshape(-1) > 0
    observed = 0
    for start in range(0, probe_contexts, chunk_contexts):
        rows = min(chunk_contexts, probe_contexts - start)
        probe_context_codes = torch.zeros(rows, dtype=torch.long)
        for offset in range(current_order):
            probe_context_codes = (
                probe_context_codes * vocab_size
                + probe_long[start + offset : start + offset + rows]
            )
        observed += int(observed_flat[probe_context_codes].sum().item())
    return probe_contexts, observed


def count_probe_sparse_context_hits(
    ctx_keys: torch.Tensor,
    probe_ids: torch.Tensor | None,
    current_order: int,
    vocab_size: int,
    chunk_contexts: int = 2_000_000,
) -> tuple[int, int]:
    if probe_ids is None:
        return 0, 0
    probe_long = probe_ids.detach().to(device="cpu", dtype=torch.long)
    sorted_keys = ctx_keys.detach().to(device="cpu", dtype=torch.long)
    probe_contexts = max(int(probe_long.numel()) - current_order, 0)
    if probe_contexts <= 0 or sorted_keys.numel() == 0:
        return probe_contexts, 0

    observed = 0
    for start in range(0, probe_contexts, chunk_contexts):
        rows = min(chunk_contexts, probe_contexts - start)
        probe_context_codes = torch.zeros(rows, dtype=torch.long)
        for offset in range(current_order):
            probe_context_codes = (
                probe_context_codes * vocab_size
                + probe_long[start + offset : start + offset + rows]
            )
        positions = torch.searchsorted(sorted_keys, probe_context_codes)
        safe_positions = positions.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
        hits = (positions < sorted_keys.numel()) & (sorted_keys[safe_positions] == probe_context_codes)
        observed += int(hits.sum().item())
    return probe_contexts, observed


def build_count_logits_ngram5_backoff_from_shards(
    paths: list[Path],
    codec: Codec,
    vocab_size: int,
    alpha: float,
    backoff: float,
    prior5_min_count: int,
    probe_ids: torch.Tensor | None = None,
) -> tuple[SparseBackoffNgramPrior, dict[str, object]]:
    if backoff < 0:
        raise ValueError("--ngram-backoff must be non-negative")
    if prior5_min_count < 1:
        raise ValueError("--prior5-min-count must be at least 1")
    if not paths:
        raise ValueError("No shard paths were provided for count-ngram prior construction")

    build_start = time.perf_counter()
    lower_order = 4
    order = 5
    counts_by_order = {
        current_order: torch.zeros(vocab_size ** (current_order + 1), dtype=torch.float32)
        for current_order in range(1, lower_order + 1)
    }
    context_counts_by_order = {
        current_order: torch.zeros(vocab_size**current_order, dtype=torch.float32)
        for current_order in range(1, lower_order + 1)
    }
    order5_counts: Counter[int] = Counter()
    order5_count_chunk_rows = 2_000_000

    carry = ""
    total_chars = 0
    for path in paths:
        shard_text = path.read_text(encoding="utf-8")
        chunk_text = carry + shard_text
        prefix_len = len(carry)
        ids = torch.tensor(codec.encode(chunk_text), dtype=torch.long)
        total_chars += len(shard_text)

        for current_order in range(1, lower_order + 1):
            start_index = max(0, prefix_len - current_order)
            rows = max(int(ids.numel()) - current_order - start_index, 0)
            if rows <= 0:
                continue
            context_codes = torch.zeros(rows, dtype=torch.long)
            for offset in range(current_order):
                context_codes = (
                    context_codes * vocab_size
                    + ids[start_index + offset : start_index + offset + rows]
                )
            next_ids = ids[start_index + current_order : start_index + current_order + rows]
            counts_by_order[current_order] += torch.bincount(
                context_codes * vocab_size + next_ids,
                minlength=vocab_size ** (current_order + 1),
            ).to(torch.float32)
            context_counts_by_order[current_order] += torch.bincount(
                context_codes,
                minlength=vocab_size**current_order,
            ).to(torch.float32)

        start_index = max(0, prefix_len - order)
        rows = max(int(ids.numel()) - order - start_index, 0)
        if rows > 0:
            stop_index = start_index + rows
            for chunk_start in range(start_index, stop_index, order5_count_chunk_rows):
                chunk_rows = min(order5_count_chunk_rows, stop_index - chunk_start)
                context_codes = torch.zeros(chunk_rows, dtype=torch.long)
                for offset in range(order):
                    context_codes = (
                        context_codes * vocab_size
                        + ids[chunk_start + offset : chunk_start + offset + chunk_rows]
                    )
                next_ids = ids[chunk_start + order : chunk_start + order + chunk_rows]
                keys = context_codes * vocab_size + next_ids
                unique_keys, unique_counts = torch.unique(keys, return_counts=True)
                order5_counts.update(
                    {
                        int(key): int(count)
                        for key, count in zip(unique_keys.tolist(), unique_counts.tolist())
                    }
                )

        carry = chunk_text[-order:]

    probs_by_order: dict[int, torch.Tensor] = {}
    coverage: list[dict[str, float | int]] = []
    for current_order in range(1, lower_order + 1):
        count_shape = (vocab_size,) * current_order + (vocab_size,)
        context_shape = (vocab_size,) * current_order
        counts = counts_by_order[current_order].view(count_shape) + alpha
        context_counts = context_counts_by_order[current_order].view(context_shape)

        empirical_probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if current_order == 1:
            mixed_probs = empirical_probs
        else:
            lower_probs = probs_by_order[current_order - 1].unsqueeze(0).expand(count_shape)
            if backoff == 0:
                context_mass = (context_counts > 0).to(torch.float32)
            else:
                context_mass = context_counts / (context_counts + backoff)
            mixed_probs = (
                context_mass.unsqueeze(-1) * empirical_probs
                + (1.0 - context_mass).unsqueeze(-1) * lower_probs
            )

        probs_by_order[current_order] = mixed_probs.clamp_min(1e-12)
        observed_contexts = int((context_counts > 0).sum().item())
        total_contexts = int(context_counts.numel())
        if observed_contexts:
            observed_values = context_counts[context_counts > 0]
            mean_observed_count = float(observed_values.mean().item())
            min_observed_count = float(observed_values.min().item())
        else:
            mean_observed_count = 0.0
            min_observed_count = 0.0
        probe_contexts, probe_observed_contexts = count_probe_context_hits(
            context_counts,
            probe_ids,
            current_order,
            vocab_size,
        )
        coverage.append(
            {
                "order": current_order,
                "contexts": total_contexts,
                "observed_contexts": observed_contexts,
                "coverage": observed_contexts / max(total_contexts, 1),
                "mean_observed_count": mean_observed_count,
                "min_observed_count": min_observed_count,
                "validation_contexts": probe_contexts,
                "validation_observed_contexts": probe_observed_contexts,
                "validation_coverage": probe_observed_contexts / max(probe_contexts, 1),
            }
        )

    base_shape = (vocab_size + 1,) * (lower_order - 1) + (vocab_size, vocab_size)
    lower_probs = probs_by_order[1].view((1,) * (lower_order - 1) + (vocab_size, vocab_size)).expand(base_shape).clone()
    start_token = vocab_size
    for current_order in range(2, lower_order + 1):
        leading_starts = lower_order - current_order
        index = [start_token] * leading_starts
        index.extend([slice(0, vocab_size)] * (current_order - 1))
        index.extend([slice(None), slice(None)])
        lower_probs[tuple(index)] = probs_by_order[current_order]
    lower_logits = lower_probs.clamp_min(1e-12).log()

    if order5_counts:
        keys = torch.tensor(list(order5_counts.keys()), dtype=torch.long)
        counts = torch.tensor(list(order5_counts.values()), dtype=torch.float32)
        ctx_codes = torch.div(keys, vocab_size, rounding_mode="floor")
        next_ids = (keys % vocab_size).to(dtype=torch.long)
        observed_ctx_keys, inverse = torch.unique(ctx_codes, sorted=True, return_inverse=True)
        ctx_totals = torch.zeros(int(observed_ctx_keys.numel()), dtype=torch.float32)
        ctx_totals.scatter_add_(0, inverse, counts)
        keep_context_mask = ctx_totals >= float(prior5_min_count)
        kept_ctx_keys = observed_ctx_keys[keep_context_mask]
        row_for_context = torch.full((int(observed_ctx_keys.numel()),), -1, dtype=torch.long)
        row_for_context[keep_context_mask] = torch.arange(int(kept_ctx_keys.numel()), dtype=torch.long)
        kept_key_mask = row_for_context[inverse] >= 0
        ctx_logits = torch.full((int(kept_ctx_keys.numel()), vocab_size), alpha, dtype=torch.float32)
        if bool(kept_key_mask.any()):
            row_indices = row_for_context[inverse[kept_key_mask]]
            ctx_logits[row_indices, next_ids[kept_key_mask]] += counts[kept_key_mask]
        observed_order5_contexts = int(observed_ctx_keys.numel())
        observed_order5_values = ctx_totals
        kept_context_total_counts = ctx_totals[keep_context_mask]
    else:
        observed_ctx_keys = torch.empty(0, dtype=torch.long)
        kept_ctx_keys = torch.empty(0, dtype=torch.long)
        ctx_logits = torch.empty((0, vocab_size), dtype=torch.float32)
        observed_order5_contexts = 0
        observed_order5_values = torch.empty(0, dtype=torch.float32)
        kept_context_total_counts = torch.empty(0, dtype=torch.float32)

    probe_contexts, probe_observed_contexts = count_probe_sparse_context_hits(
        observed_ctx_keys,
        probe_ids,
        order,
        vocab_size,
    )
    _probe_contexts, probe_kept_contexts = count_probe_sparse_context_hits(
        kept_ctx_keys,
        probe_ids,
        order,
        vocab_size,
    )
    order5_possible_contexts = vocab_size**order
    coverage.append(
        {
            "order": order,
            "contexts": order5_possible_contexts,
            "observed_contexts": observed_order5_contexts,
            "coverage": observed_order5_contexts / max(order5_possible_contexts, 1),
            "mean_observed_count": float(observed_order5_values.mean().item()) if observed_order5_contexts else 0.0,
            "min_observed_count": float(observed_order5_values.min().item()) if observed_order5_contexts else 0.0,
            "kept_contexts": int(kept_ctx_keys.numel()),
            "kept_coverage": int(kept_ctx_keys.numel()) / max(order5_possible_contexts, 1),
            "mean_kept_count": float(kept_context_total_counts.mean().item()) if kept_ctx_keys.numel() else 0.0,
            "min_kept_count": float(kept_context_total_counts.min().item()) if kept_ctx_keys.numel() else 0.0,
            "validation_contexts": probe_contexts,
            "validation_observed_contexts": probe_observed_contexts,
            "validation_coverage": probe_observed_contexts / max(probe_contexts, 1),
            "validation_kept_contexts": probe_kept_contexts,
            "validation_kept_coverage": probe_kept_contexts / max(probe_contexts, 1),
        }
    )

    prior = SparseBackoffNgramPrior(
        lower_logits=lower_logits,
        ctx_keys=kept_ctx_keys.to(dtype=torch.long).contiguous(),
        ctx_logits=ctx_logits.clamp_min(1e-12).log().to(dtype=torch.float16).contiguous(),
        vocab_size=vocab_size,
        order=order,
        min_count=prior5_min_count,
    )
    byte_report = prior.tensor_bytes()
    build_seconds = time.perf_counter() - build_start
    validation_backoff_rate = 1.0 - (probe_kept_contexts / max(probe_contexts, 1))
    report = {
        "prior_order": order,
        "ngram_backoff": backoff,
        "ngram_smoothing": "order-5 add-alpha sparse rows with dense recursive order-4 backoff",
        "ngram_context_coverage": coverage,
        "ngram_count_source": "train_shards",
        "ngram_train_shard_files": [str(path) for path in paths],
        "ngram_train_shard_chars": total_chars,
        "prior5_min_count": prior5_min_count,
        "prior5_observed_contexts": observed_order5_contexts,
        "prior5_kept_contexts": int(prior.ctx_keys.numel()),
        "prior5_possible_contexts": order5_possible_contexts,
        "prior5_validation_contexts": probe_contexts,
        "prior5_validation_kept_contexts": probe_kept_contexts,
        "prior5_validation_backoff_rate": validation_backoff_rate,
        "prior5_count_chunk_rows": order5_count_chunk_rows,
        "prior5_lower_logits_bytes": byte_report["lower_logits_bytes"],
        "prior5_ctx_keys_bytes": byte_report["ctx_keys_bytes"],
        "prior5_ctx_logits_bytes": byte_report["ctx_logits_bytes"],
        "prior5_total_bytes": byte_report["total_bytes"],
        "prior5_build_seconds": round(build_seconds, 4),
    }
    return prior, report


def build_count_logits_ngram_from_shards(
    paths: list[Path],
    codec: Codec,
    vocab_size: int,
    order: int,
    alpha: float,
    backoff: float,
    prior5_min_count: int = 10,
    probe_ids: torch.Tensor | None = None,
) -> tuple[PriorLogits, dict[str, object]]:
    if order == 5:
        return build_count_logits_ngram5_backoff_from_shards(
            paths,
            codec,
            vocab_size,
            alpha,
            backoff,
            prior5_min_count,
            probe_ids=probe_ids,
        )
    if order < 1 or order > 4:
        raise ValueError("--prior-order currently supports 1, 2, 3, or 4")
    if backoff < 0:
        raise ValueError("--ngram-backoff must be non-negative")
    if not paths:
        raise ValueError("No shard paths were provided for count-ngram prior construction")

    counts_by_order = {
        current_order: torch.zeros(vocab_size ** (current_order + 1), dtype=torch.float32)
        for current_order in range(1, order + 1)
    }
    context_counts_by_order = {
        current_order: torch.zeros(vocab_size**current_order, dtype=torch.float32)
        for current_order in range(1, order + 1)
    }

    carry = ""
    total_chars = 0
    for path in paths:
        shard_text = path.read_text(encoding="utf-8")
        chunk_text = carry + shard_text
        prefix_len = len(carry)
        ids = torch.tensor(codec.encode(chunk_text), dtype=torch.long)
        total_chars += len(shard_text)

        for current_order in range(1, order + 1):
            start_index = max(0, prefix_len - current_order)
            rows = max(int(ids.numel()) - current_order - start_index, 0)
            if rows <= 0:
                continue
            context_codes = torch.zeros(rows, dtype=torch.long)
            for offset in range(current_order):
                context_codes = (
                    context_codes * vocab_size
                    + ids[start_index + offset : start_index + offset + rows]
                )
            next_ids = ids[start_index + current_order : start_index + current_order + rows]
            counts_by_order[current_order] += torch.bincount(
                context_codes * vocab_size + next_ids,
                minlength=vocab_size ** (current_order + 1),
            ).to(torch.float32)
            context_counts_by_order[current_order] += torch.bincount(
                context_codes,
                minlength=vocab_size**current_order,
            ).to(torch.float32)

        carry = chunk_text[-order:]

    probs_by_order: dict[int, torch.Tensor] = {}
    coverage: list[dict[str, float | int]] = []
    for current_order in range(1, order + 1):
        count_shape = (vocab_size,) * current_order + (vocab_size,)
        context_shape = (vocab_size,) * current_order
        counts = counts_by_order[current_order].view(count_shape) + alpha
        context_counts = context_counts_by_order[current_order].view(context_shape)

        empirical_probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if current_order == 1:
            mixed_probs = empirical_probs
        else:
            lower_probs = probs_by_order[current_order - 1].unsqueeze(0).expand(count_shape)
            if backoff == 0:
                context_mass = (context_counts > 0).to(torch.float32)
            else:
                context_mass = context_counts / (context_counts + backoff)
            mixed_probs = (
                context_mass.unsqueeze(-1) * empirical_probs
                + (1.0 - context_mass).unsqueeze(-1) * lower_probs
            )

        probs_by_order[current_order] = mixed_probs.clamp_min(1e-12)
        observed_contexts = int((context_counts > 0).sum().item())
        total_contexts = int(context_counts.numel())
        if observed_contexts:
            observed_values = context_counts[context_counts > 0]
            mean_observed_count = float(observed_values.mean().item())
            min_observed_count = float(observed_values.min().item())
        else:
            mean_observed_count = 0.0
            min_observed_count = 0.0
        probe_contexts, probe_observed_contexts = count_probe_context_hits(
            context_counts,
            probe_ids,
            current_order,
            vocab_size,
        )
        coverage.append(
            {
                "order": current_order,
                "contexts": total_contexts,
                "observed_contexts": observed_contexts,
                "coverage": observed_contexts / max(total_contexts, 1),
                "mean_observed_count": mean_observed_count,
                "min_observed_count": min_observed_count,
                "validation_contexts": probe_contexts,
                "validation_observed_contexts": probe_observed_contexts,
                "validation_coverage": probe_observed_contexts / max(probe_contexts, 1),
            }
        )

    if order == 1:
        probs = probs_by_order[1]
    else:
        base_shape = (vocab_size + 1,) * (order - 1) + (vocab_size, vocab_size)
        probs = probs_by_order[1].view((1,) * (order - 1) + (vocab_size, vocab_size)).expand(base_shape).clone()
        start_token = vocab_size
        for lower_order in range(2, order + 1):
            leading_starts = order - lower_order
            index = [start_token] * leading_starts
            index.extend([slice(0, vocab_size)] * (lower_order - 1))
            index.extend([slice(None), slice(None)])
            probs[tuple(index)] = probs_by_order[lower_order]

    report = {
        "prior_order": order,
        "ngram_backoff": backoff,
        "ngram_smoothing": "recursive add-alpha interpolation with context_mass=count/(count+backoff)",
        "ngram_context_coverage": coverage,
        "ngram_count_source": "train_shards",
        "ngram_train_shard_files": [str(path) for path in paths],
        "ngram_train_shard_chars": total_chars,
    }
    return probs.clamp_min(1e-12).log(), report


def count_ngram_prior_cache_path(
    cache_dir: Path | None,
    corpus: Path,
    train_shard_dir: Path | None,
    train_chars: int,
    vocab_size: int,
    order: int,
    alpha: float,
    backoff: float,
    prior5_min_count: int | None = None,
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {
        "corpus": str(corpus.resolve()),
        "train_shard_dir": str(train_shard_dir.resolve()) if train_shard_dir is not None else "",
        "train_chars": train_chars,
        "vocab_size": vocab_size,
        "order": order,
        "alpha": alpha,
        "backoff": backoff,
    }
    if prior5_min_count is not None:
        payload["prior5_min_count"] = prior5_min_count
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"count_ngram_{digest}.pt"


def load_count_ngram_prior_from_cache(cached: dict[str, object], device: str) -> PriorLogits:
    if cached.get("prior_kind") == "sparse_backoff_ngram":
        return SparseBackoffNgramPrior(
            lower_logits=cached["lower_logits"].to(device),  # type: ignore[index, union-attr]
            ctx_keys=cached["ctx_keys"].to(device),  # type: ignore[index, union-attr]
            ctx_logits=cached["ctx_logits"].to(device),  # type: ignore[index, union-attr]
            vocab_size=int(cached["vocab_size"]),
            order=int(cached["order"]),
            min_count=int(cached["min_count"]),
        )
    return cached["base_logits"].to(device)  # type: ignore[index, union-attr]


def count_ngram_prior_cache_payload(
    base_logits: PriorLogits,
    ngram_report: dict[str, object],
) -> dict[str, object]:
    if isinstance(base_logits, SparseBackoffNgramPrior):
        cpu_prior = base_logits.cpu()
        return {
            "prior_kind": "sparse_backoff_ngram",
            "lower_logits": cpu_prior.lower_logits,
            "ctx_keys": cpu_prior.ctx_keys,
            "ctx_logits": cpu_prior.ctx_logits,
            "vocab_size": cpu_prior.vocab_size,
            "order": cpu_prior.order,
            "min_count": cpu_prior.min_count,
            "ngram_report": ngram_report,
        }
    return {
        "base_logits": base_logits.cpu(),
        "ngram_report": ngram_report,
    }


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        super().__init__()
        if lora_rank < 0:
            raise ValueError("lora_rank must be non-negative")
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank if lora_rank > 0 else 0.0
        self.lora_dropout = nn.Dropout(lora_dropout)

        if lora_rank > 0:
            self.lora_a = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_b = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_rank > 0 and self.lora_a is not None and self.lora_b is not None:
            lora_hidden = F.linear(self.lora_dropout(x), self.lora_a)
            out = out + self.lora_scale * F.linear(lora_hidden, self.lora_b)
        return out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[:, :, : x.size(-2), :]
    sin = sin[:, :, : x.size(-2), :]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        pos_encoding: str,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("--n-embd must be divisible by --n-head")

        self.n_head = n_head
        self.pos_encoding = pos_encoding
        head_size = n_embd // n_head
        if pos_encoding == "rope" and head_size % 2 != 0:
            raise ValueError("--pos-encoding rope requires an even head size")
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError(f"Unknown positional encoding: {pos_encoding}")
        self.c_attn = LoRALinear(n_embd, 3 * n_embd, True, lora_rank, lora_alpha, lora_dropout)
        self.c_proj = LoRALinear(n_embd, n_embd, True, lora_rank, lora_alpha, lora_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        if pos_encoding == "rope":
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
            positions = torch.arange(block_size, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            self.register_buffer("rope_cos", freqs.cos().view(1, 1, block_size, head_size // 2))
            self.register_buffer("rope_sin", freqs.sin().view(1, 1, block_size, head_size // 2))
        else:
            self.register_buffer("rope_cos", torch.empty(0), persistent=False)
            self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        head_size = channels // self.n_head

        q = q.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        if self.pos_encoding == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(self.mask[:, :, :tokens, :tokens] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.resid_dropout(self.c_proj(y))


class ResidualAdapter(nn.Module):
    def __init__(self, n_embd: int, adapter_rank: int, dropout: float) -> None:
        super().__init__()
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.down = nn.Linear(n_embd, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up(F.gelu(self.down(x))))


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        adapter_rank: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        pos_encoding: str,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout,
            lora_rank,
            lora_alpha,
            lora_dropout,
            pos_encoding,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            LoRALinear(n_embd, 4 * n_embd, True, lora_rank, lora_alpha, lora_dropout),
            nn.GELU(),
            LoRALinear(4 * n_embd, n_embd, True, lora_rank, lora_alpha, lora_dropout),
            nn.Dropout(dropout),
        )
        self.adapter = ResidualAdapter(n_embd, adapter_rank, dropout) if adapter_rank > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.adapter is not None:
            x = x + self.adapter(x)
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        adapter_rank: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        pos_encoding: str,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.pos_encoding = pos_encoding
        self.activation_checkpoint = activation_checkpoint
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd) if pos_encoding == "learned" else None
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_head,
                    block_size,
                    dropout,
                    adapter_rank,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                    pos_encoding,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        for block in self.blocks:
            if block.adapter is not None:
                nn.init.normal_(block.adapter.down.weight, mean=0.0, std=0.02)
                nn.init.zeros_(block.adapter.up.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        base_logits: PriorLogits | None = None,
        residual_scale: float = 1.0,
        recency_tau: float = DEFAULT_RECENCY_TAU,
        recency_lambda: float = DEFAULT_RECENCY_LAMBDA,
        residual_l2: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens = idx.shape
        if tokens > self.block_size:
            raise ValueError("Input sequence is longer than block_size")

        positions = torch.arange(tokens, device=idx.device)
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            if self.activation_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        residual_logits = self.lm_head(x)
        logits = residual_logits
        if base_logits is not None:
            base = lookup_base_logits(idx, base_logits)
            base = mix_recency_base_logits(idx, base, recency_tau, recency_lambda)
            logits = base + residual_scale * residual_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch * tokens, -1), targets.view(batch * tokens))
            if residual_l2 > 0:
                loss = loss + residual_l2 * residual_logits.pow(2).mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        generator: torch.Generator,
        base_logits: PriorLogits | None = None,
        residual_scale: float = 1.0,
        recency_tau: float = DEFAULT_RECENCY_TAU,
        recency_lambda: float = DEFAULT_RECENCY_LAMBDA,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("--temperature must be greater than zero")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(
                idx_cond,
                base_logits=base_logits,
                residual_scale=residual_scale,
                recency_tau=recency_tau,
                recency_lambda=recency_lambda,
            )
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


@torch.no_grad()
def zero_transformer_blocks(model: TinyTransformer) -> None:
    for block in model.blocks:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.fill_(1.0)
                module.bias.zero_()


@torch.no_grad()
def zero_output_head(model: TinyTransformer) -> None:
    model.lm_head.weight.zero_()
    model.lm_head.bias.zero_()


@torch.no_grad()
def apply_count_bigram_init(
    model: TinyTransformer,
    train_data: torch.Tensor,
    vocab_size: int,
    alpha: float,
    embedding_scale: float,
) -> dict[str, object]:
    n_embd = model.token_embedding.embedding_dim
    if n_embd < vocab_size:
        raise ValueError("--init count-bigram requires --n-embd to be at least vocab_size")

    count_logits = build_count_logits(train_data, vocab_size, alpha)
    zero_transformer_blocks(model)
    if model.position_embedding is not None:
        model.position_embedding.weight.zero_()

    token_codes = torch.zeros((vocab_size, n_embd), dtype=torch.float32, device=train_data.device)
    token_codes[:, :vocab_size] = torch.eye(vocab_size, device=train_data.device) * embedding_scale
    model.token_embedding.weight.copy_(token_codes)

    token_ids = torch.arange(vocab_size, dtype=torch.long, device=train_data.device)
    hidden = model.ln_f(model.token_embedding(token_ids))
    ones = torch.ones((vocab_size, 1), dtype=hidden.dtype, device=hidden.device)
    features = torch.cat([hidden, ones], dim=1)

    solution = torch.linalg.lstsq(features, count_logits).solution
    model.lm_head.weight.copy_(solution[:-1].T)
    model.lm_head.bias.copy_(solution[-1])

    fitted_logits = features @ solution
    max_abs_error = float((fitted_logits - count_logits).abs().max().item())
    seeded_parameters = (
        model.token_embedding.weight.numel()
        + model.lm_head.weight.numel()
        + model.lm_head.bias.numel()
    )
    if model.position_embedding is not None:
        seeded_parameters += model.position_embedding.weight.numel()
    return {
        "count_alpha": alpha,
        "embedding_scale": embedding_scale,
        "seeded_parameters": seeded_parameters,
        "count_fit_max_abs_error": round(max_abs_error, 8),
    }


@torch.no_grad()
def full_split_loss(
    model: TinyTransformer,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
) -> float:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for full evaluation")

    total_loss = 0.0
    total_tokens = 0
    for offset in range(0, num_starts, batch_size):
        stop = min(offset + batch_size, num_starts)
        starts = torch.arange(offset, stop, device=device)
        xb, yb = make_batch(data, starts, block_size)
        _, loss = model(
            xb,
            yb,
            base_logits=base_logits,
            residual_scale=residual_scale,
            recency_tau=recency_tau,
            recency_lambda=recency_lambda,
        )
        if loss is None:
            raise RuntimeError("Expected loss during full evaluation")
        tokens = int(xb.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens

    return total_loss / total_tokens


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_mode: str,
    eval_batches: int,
    generator: torch.Generator,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        if eval_mode == "full":
            out[split] = full_split_loss(
                model,
                data,
                block_size,
                batch_size,
                device,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
            )
        elif eval_mode == "sampled":
            losses = []
            for _ in range(eval_batches):
                xb, yb = get_batch(data, block_size, batch_size, generator, device)
                _, loss = model(
                    xb,
                    yb,
                    base_logits=base_logits,
                    residual_scale=residual_scale,
                    recency_tau=recency_tau,
                    recency_lambda=recency_lambda,
                )
                if loss is None:
                    raise RuntimeError("Expected loss during sampled evaluation")
                losses.append(float(loss.item()))
            out[split] = sum(losses) / len(losses)
        else:
            raise ValueError(f"Unknown eval mode: {eval_mode}")
    model.train()
    return out


@torch.no_grad()
def estimate_data_loss(
    model: TinyTransformer,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_mode: str,
    eval_batches: int,
    generator: torch.Generator,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
) -> float:
    model.eval()
    try:
        if eval_mode == "full":
            return full_split_loss(
                model,
                data,
                block_size,
                batch_size,
                device,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
            )
        if eval_mode != "sampled":
            raise ValueError(f"Unknown eval mode: {eval_mode}")
        losses: list[float] = []
        for _ in range(eval_batches):
            xb, yb = get_batch(data, block_size, batch_size, generator, device)
            _, loss = model(
                xb,
                yb,
                base_logits=base_logits,
                residual_scale=residual_scale,
                recency_tau=recency_tau,
                recency_lambda=recency_lambda,
            )
            if loss is None:
                raise RuntimeError("Expected loss during auxiliary evaluation")
            losses.append(float(loss.item()))
        return sum(losses) / len(losses)
    finally:
        model.train()


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    params = model.parameters()
    if only_trainable:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def flatten_parameters(parameters: list[nn.Parameter]) -> torch.Tensor:
    if not parameters:
        raise ValueError("No parameters selected for residual search")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


@torch.no_grad()
def assign_flat_parameters(parameters: list[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(flat[offset : offset + count].view_as(parameter))
        offset += count
    if offset != flat.numel():
        raise ValueError("Flat parameter vector size did not match selected parameters")


def build_search_batches(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    search_batches: int,
    generator: torch.Generator,
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if search_batches <= 0:
        raise ValueError("--search-batches must be positive")
    return [
        get_batch(data, block_size, batch_size, generator, device)
        for _ in range(search_batches)
    ]


@torch.no_grad()
def residual_objective_loss(
    model: TinyTransformer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
    residual_l2: float,
) -> float:
    losses: list[float] = []
    for xb, yb in batches:
        _, loss = model(
            xb,
            yb,
            base_logits=base_logits,
            residual_scale=residual_scale,
            recency_tau=recency_tau,
            recency_lambda=recency_lambda,
            residual_l2=residual_l2,
        )
        if loss is None:
            raise RuntimeError("Expected loss during residual search")
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def evolution_strategy_residual_search(
    model: TinyTransformer,
    parameters: list[nn.Parameter],
    train_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    steps: int,
    search_batches: int,
    population: int,
    sigma: float,
    lr: float,
    eval_interval: int,
    log_every: int,
    generator: torch.Generator,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
    residual_l2: float,
) -> dict[str, object]:
    if population <= 0:
        raise ValueError("--search-population must be positive")
    if sigma <= 0:
        raise ValueError("--search-sigma must be positive")
    if lr <= 0:
        raise ValueError("--search-lr must be positive")

    model.train()
    batches = build_search_batches(train_data, block_size, batch_size, search_batches, generator, device)
    theta = flatten_parameters(parameters).to(device)
    forward_passes = 0
    initial_loss = residual_objective_loss(
        model,
        batches,
        base_logits,
        residual_scale,
        recency_tau,
        recency_lambda,
        residual_l2,
    )
    forward_passes += len(batches)
    loss_curve: list[dict[str, float | int]] = []

    for step in range(1, steps + 1):
        grad_estimate = torch.zeros_like(theta)
        for _ in range(population):
            noise = torch.randn(theta.shape, generator=generator, device=device, dtype=theta.dtype)
            assign_flat_parameters(parameters, theta + sigma * noise)
            positive_loss = residual_objective_loss(
                model,
                batches,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
                residual_l2,
            )
            assign_flat_parameters(parameters, theta - sigma * noise)
            negative_loss = residual_objective_loss(
                model,
                batches,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
                residual_l2,
            )
            forward_passes += 2 * len(batches)
            grad_estimate.add_((positive_loss - negative_loss) * noise)

        grad_estimate.div_(2.0 * sigma * population)
        theta = theta - lr * grad_estimate
        assign_flat_parameters(parameters, theta)

        should_eval = eval_interval > 0 and step % eval_interval == 0
        if should_eval or (log_every > 0 and step % log_every == 0):
            objective_loss = residual_objective_loss(
                model,
                batches,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
                residual_l2,
            )
            forward_passes += len(batches)
            if should_eval:
                loss_curve.append({"step": step, "search_objective_nll": round(objective_loss, 6)})
            if log_every > 0 and step % log_every == 0:
                print(f"step={step} search_objective_nll={objective_loss:.4f}")

    final_loss = residual_objective_loss(
        model,
        batches,
        base_logits,
        residual_scale,
        recency_tau,
        recency_lambda,
        residual_l2,
    )
    forward_passes += len(batches)
    return {
        "formation_steps": steps,
        "formation_forward_passes": forward_passes,
        "search_batches": search_batches,
        "search_population": population,
        "search_sigma": sigma,
        "search_lr": lr,
        "search_initial_objective_nll": round(initial_loss, 6),
        "search_final_objective_nll": round(final_loss, 6),
        "search_loss_curve": loss_curve,
    }


def coordinate_residual_search(
    model: TinyTransformer,
    parameters: list[nn.Parameter],
    train_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    steps: int,
    search_batches: int,
    step_size: float,
    eval_interval: int,
    log_every: int,
    generator: torch.Generator,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
    residual_l2: float,
) -> dict[str, object]:
    if step_size <= 0:
        raise ValueError("--coord-step-size must be positive")

    model.train()
    batches = build_search_batches(train_data, block_size, batch_size, search_batches, generator, device)
    theta = flatten_parameters(parameters).to(device)
    total_parameters = theta.numel()
    current_loss = residual_objective_loss(
        model,
        batches,
        base_logits,
        residual_scale,
        recency_tau,
        recency_lambda,
        residual_l2,
    )
    forward_passes = len(batches)
    initial_loss = current_loss
    accepted = 0
    loss_curve: list[dict[str, float | int]] = []

    for step in range(1, steps + 1):
        index = int(torch.randint(total_parameters, (1,), generator=generator, device=device).item())
        original_value = float(theta[index].item())
        best_value = original_value
        best_loss = current_loss

        for delta in (-step_size, step_size):
            candidate = theta.clone()
            candidate[index] = original_value + delta
            assign_flat_parameters(parameters, candidate)
            candidate_loss = residual_objective_loss(
                model,
                batches,
                base_logits,
                residual_scale,
                recency_tau,
                recency_lambda,
                residual_l2,
            )
            forward_passes += len(batches)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_value = original_value + delta

        theta[index] = best_value
        assign_flat_parameters(parameters, theta)
        if best_loss < current_loss:
            accepted += 1
            current_loss = best_loss

        if eval_interval > 0 and step % eval_interval == 0:
            loss_curve.append({"step": step, "search_objective_nll": round(current_loss, 6)})
        if log_every > 0 and step % log_every == 0:
            print(f"step={step} search_objective_nll={current_loss:.4f} accepted={accepted}")

    return {
        "formation_steps": steps,
        "formation_forward_passes": forward_passes,
        "search_batches": search_batches,
        "coord_step_size": step_size,
        "coord_accepted_updates": accepted,
        "coord_acceptance_rate": round(accepted / max(steps, 1), 6),
        "search_initial_objective_nll": round(initial_loss, 6),
        "search_final_objective_nll": round(current_loss, 6),
        "search_loss_curve": loss_curve,
    }


def configure_train_scope(model: TinyTransformer, train_scope: str) -> dict[str, object]:
    if train_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif train_scope == "head":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.lm_head.parameters():
            parameter.requires_grad_(True)
    elif train_scope == "adapters":
        adapter_count = 0
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks:
            if block.adapter is not None:
                adapter_count += 1
                for parameter in block.adapter.parameters():
                    parameter.requires_grad_(True)
        if adapter_count == 0:
            raise ValueError("--train-scope adapters requires --adapter-rank greater than 0")
        for parameter in model.lm_head.parameters():
            parameter.requires_grad_(True)
    elif train_scope == "lora":
        lora_count = 0
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                lora_count += 1
                parameter.requires_grad_(True)
        if lora_count == 0:
            raise ValueError("--train-scope lora requires --lora-rank greater than 0")
        for parameter in model.lm_head.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"Unknown train scope: {train_scope}")

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "train_scope": train_scope,
        "trainable_tensors": len(trainable_names),
        "trainable_tensor_names": trainable_names,
    }


def build_optimizer(
    model: TinyTransformer,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer | None, dict[str, object]]:
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if args.residual_optim != "adamw":
        return None, {
            "optimizer": "none",
            "muon_parameters": 0,
            "adam_parameters": 0,
            "optimizer_note": f"optimizer disabled for residual_optim={args.residual_optim}",
        }
    if args.optimizer == "adamw":
        parameters = [parameter for _name, parameter in trainable_named_parameters]
        return torch.optim.AdamW(
            parameters,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        ), {
            "optimizer": "adamw",
            "adam_parameters": sum(parameter.numel() for parameter in parameters),
            "muon_parameters": 0,
            "adam_tensors": len(parameters),
            "muon_tensors": 0,
        }
    if args.optimizer != "muon":
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    muon_parameters: list[nn.Parameter] = []
    adam_parameters: list[nn.Parameter] = []
    muon_names: list[str] = []
    adam_names: list[str] = []
    for name, parameter in trainable_named_parameters:
        if parameter.ndim >= 2 and name.startswith("blocks."):
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adam_parameters.append(parameter)
            adam_names.append(name)

    groups: list[dict[str, object]] = []
    if muon_parameters:
        groups.append(
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": args.muon_lr,
                "momentum": args.muon_momentum,
                "weight_decay": args.weight_decay,
                "ns_steps": args.muon_ns_steps,
                "nesterov": not args.no_muon_nesterov,
            }
        )
    if adam_parameters:
        groups.append(
            {
                "params": adam_parameters,
                "use_muon": False,
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            }
        )
    return SingleDeviceMuonWithAuxAdam(groups), {
        "optimizer": "muon",
        "muon_parameters": sum(parameter.numel() for parameter in muon_parameters),
        "adam_parameters": sum(parameter.numel() for parameter in adam_parameters),
        "muon_tensors": len(muon_parameters),
        "adam_tensors": len(adam_parameters),
        "muon_parameter_names": muon_names,
        "adam_parameter_names": adam_names,
        "muon_lr": args.muon_lr,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_nesterov": not args.no_muon_nesterov,
    }


def training_checkpoint_path(args: argparse.Namespace, step: int) -> Path | None:
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir is None:
        return None
    out_path = getattr(args, "out", None)
    stem = Path(out_path).stem if out_path else "checkpoint"
    return Path(checkpoint_dir) / f"{stem}_step{step:06d}.pt"


def precision_context(device: str, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unknown precision: {precision}")
    if device != "cuda":
        raise ValueError("--precision bf16 requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def lr_schedule_factor(step: int, total_steps: int, final_frac: float) -> float:
    if total_steps <= 1:
        return float(final_frac)
    progress = (step - 1) / (total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_frac + (1.0 - final_frac) * cosine)


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    step: int,
    schedule_total_steps: int,
) -> float:
    if optimizer is None:
        return 1.0
    if args.lr_schedule == "constant":
        return 1.0
    if args.lr_schedule != "cosine":
        raise ValueError(f"Unknown LR schedule: {args.lr_schedule}")
    factor = lr_schedule_factor(step, schedule_total_steps, args.lr_final_frac)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", float(group["lr"]))
        group["lr"] = float(group["initial_lr"]) * factor
    return factor


def prune_checkpoint_keep(path: Path, stem_hint: str, keep: int) -> list[str]:
    if keep <= 0 or not path.parent.exists():
        return []
    checkpoints = sorted(
        path.parent.glob(f"{stem_hint}_step*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pruned: list[str] = []
    for victim in checkpoints[keep:]:
        if victim == path:
            continue
        victim.unlink()
        pruned.append(victim.name)
        print(f"[checkpoint] pruned by keep={keep}: {victim.name}")
    return pruned


def _estimate_checkpoint_bytes(path: Path, model: nn.Module) -> int:
    siblings = sorted(
        path.parent.glob("*.pt"), key=lambda p: p.stat().st_size, reverse=True
    ) if path.parent.exists() else []
    if siblings:
        return int(siblings[0].stat().st_size * 1.3)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Model plus optimizer state plus container overhead.
    return int(param_bytes * 2.5)


def _prune_oldest_sibling_checkpoint(path: Path, stem_hint: str) -> bool:
    if not path.parent.exists():
        return False
    candidates = sorted(
        (
            p
            for p in path.parent.glob(f"{stem_hint}_step*.pt")
            if p != path
        ),
        key=lambda p: p.stat().st_mtime,
    )
    # Never prune the newest sibling: it is the resume anchor.
    if len(candidates) < 2:
        return False
    victim = candidates[0]
    try:
        victim.unlink()
        print(f"[checkpoint] pruned oldest sibling to free space: {victim.name}")
        return True
    except OSError:
        return False


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator,
    args: argparse.Namespace,
    codec: Codec,
    base_logits: PriorLogits | None,
    step: int,
    formation_forward_passes: int,
    loss_curve: list[dict[str, float | int]],
) -> bool:
    """Fail-safe checkpoint save. A save that cannot complete must never
    kill the run: two runs (Stage 56 seeds 7 and 19, 2026-07-08) crashed at
    their final save when the disk filled mid-write. Returns True on a
    durable save, False on a skipped or failed one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = _estimate_checkpoint_bytes(path, model)
    stem_hint = path.name.split("_step")[0]
    try:
        free = shutil.disk_usage(path.parent).free
    except OSError:
        free = needed
    if free < needed:
        _prune_oldest_sibling_checkpoint(path, stem_hint)
        free = shutil.disk_usage(path.parent).free
    if free < needed:
        print(
            f"[checkpoint] SKIPPED save at step {step}: "
            f"{free / 1e9:.2f} GB free, need about {needed / 1e9:.2f} GB"
        )
        return False
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "generator_state": generator.get_state(),
        "step": step,
        "formation_forward_passes": formation_forward_passes,
        "loss_curve": loss_curve,
        "chars": codec.chars,
        "args": vars(args),
        "base_logits": base_logits.cpu() if base_logits is not None else None,
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        prune_checkpoint_keep(path, stem_hint, int(getattr(args, "checkpoint_keep", 0)))
        return True
    except (OSError, RuntimeError) as error:
        print(f"[checkpoint] SAVE FAILED at step {step}: {error}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def archive_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown model-only dtype: {name}")


def save_model_only_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    codec: Codec,
    base_logits: PriorLogits | None,
    step: int,
    formation_forward_passes: int,
    loss_curve: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = archive_dtype(args.model_only_dtype)
    model_state = {
        name: tensor.detach().cpu().to(dtype=dtype) if tensor.is_floating_point() else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload = {
        "model_state": model_state,
        "optimizer_state": None,
        "generator_state": None,
        "step": step,
        "formation_forward_passes": formation_forward_passes,
        "loss_curve": loss_curve,
        "chars": codec.chars,
        "args": vars(args),
        "base_logits": base_logits.cpu() if base_logits is not None else None,
        "archive": {
            "model_only": True,
            "dtype": args.model_only_dtype,
        },
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    print(f"[model-only] wrote {path} dtype={args.model_only_dtype}")


def clean_prompt(prompt: str, codec: Codec, fallback: str) -> str:
    cleaned = "".join(ch for ch in prompt if ch in codec.stoi)
    if cleaned:
        return cleaned
    return fallback[0]


def copy_probe_retrieval_context(
    line: str,
    prefix: str,
    hint: str,
    marker: str,
    template: str,
    key_hint: str | None = None,
    key_marker: str = "key=",
) -> str:
    if template == "none":
        return ""
    key = key_hint or hint
    if template == "compact":
        return f"{key_marker}{key} {marker}{hint}\n"
    if template == "focus":
        noise_at = line.find("noise=")
        noise_part = ""
        if noise_at >= 0 and noise_at + len("noise=") < len(line):
            noise_part = f" noise={line[noise_at + len('noise=')]}"
        return f"{key_marker}{key}{noise_part} {marker}{hint}\n"
    if template == "prefix":
        return f"{prefix}{hint}\n"
    raise ValueError(f"Unknown copy probe retrieval template: {template}")


def corrupt_copy_probe_hint(hint: str, candidates: list[str]) -> str:
    options = sorted({candidate for candidate in candidates if candidate})
    if len(options) < 2:
        return hint
    if hint not in options:
        return next((candidate for candidate in options if candidate != hint), hint)
    index = options.index(hint)
    return options[(index + 1) % len(options)]


def normalize_key_chars(raw: str | list[str]) -> str:
    pieces = raw if isinstance(raw, list) else [raw]
    return "".join(ch for piece in pieces for ch in piece if ch.strip())


@torch.no_grad()
def copy_answer_probe(
    model: TinyTransformer,
    val_text: str,
    codec: Codec,
    block_size: int,
    device: str,
    base_logits: PriorLogits | None,
    residual_scale: float,
    recency_tau: float,
    recency_lambda: float,
    marker: str,
    max_cases: int,
    retrieval_template: str,
    retrieval_source: str,
    retrieval_corrupt: str,
    holdout_keys: str | list[str],
    retrieval_memory: dict[str, str] | None,
    retrieval_memory_report: dict[str, int] | None,
    key_marker: str = "key=",
) -> dict[str, object]:
    if not marker:
        return {}

    cases = 0
    correct = 0
    choice_cases = 0
    choice_correct_total = 0
    choice_mrr_total = 0.0
    total_nll = 0.0
    retrieval_hits = 0
    retrieval_misses = 0
    retrieval_corrupted = 0
    retrieval_candidates = sorted(set((retrieval_memory or {}).values()))
    normalized_holdout_keys = normalize_key_chars(holdout_keys)
    holdout_key_set = set(normalized_holdout_keys)
    choice_chars = sorted(
        {
            key
            for line in val_text.splitlines()
            if line.find(marker) >= 0
            for key in [extract_key_value(line, key_marker)]
            if key is not None and key in codec.stoi
        }
    )
    choice_ids = torch.tensor([codec.stoi[ch] for ch in choice_chars], dtype=torch.long, device=device)
    split_totals = {
        "seen": {
            "cases": 0,
            "correct": 0,
            "nll": 0.0,
            "choice_cases": 0,
            "choice_correct": 0,
            "choice_mrr": 0.0,
        },
        "heldout": {
            "cases": 0,
            "correct": 0,
            "nll": 0.0,
            "choice_cases": 0,
            "choice_correct": 0,
            "choice_mrr": 0.0,
        },
    }
    examples: list[dict[str, object]] = []
    model.eval()

    for line in val_text.splitlines():
        if cases >= max_cases:
            break
        key_pos = line.find(key_marker)
        marker_pos = line.find(marker)
        if key_pos < 0 or marker_pos < 0:
            continue

        key = extract_key_value(line, key_marker)
        target = extract_marker_value(line, marker)
        if key is None or target is None:
            continue
        prefix = line[: marker_pos + len(marker)]
        if target not in codec.stoi or any(ch not in codec.stoi for ch in prefix):
            continue

        retrieval_hint = target
        if retrieval_source == "memory" and retrieval_template != "none":
            if retrieval_memory is None:
                raise RuntimeError("Expected retrieval memory during memory-sourced copy probe")
            retrieved = retrieval_memory.get(key)
            if retrieved is None:
                retrieval_misses += 1
                retrieval_hint = ""
            else:
                retrieval_hits += 1
                retrieval_hint = retrieved
        elif retrieval_template != "none":
            retrieval_hits += 1

        if retrieval_hint and retrieval_corrupt == "wrong-answer":
            corrupted_hint = corrupt_copy_probe_hint(retrieval_hint, retrieval_candidates)
            if corrupted_hint != retrieval_hint:
                retrieval_corrupted += 1
            retrieval_hint = corrupted_hint

        retrieval_context = (
            copy_probe_retrieval_context(line, prefix, retrieval_hint, marker, retrieval_template, key, key_marker)
            if retrieval_hint
            else ""
        )
        if any(ch not in codec.stoi for ch in retrieval_context):
            continue
        probed_prefix = retrieval_context + prefix
        input_ids = codec.encode(probed_prefix)[-block_size:]
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits, _ = model(
            idx,
            base_logits=base_logits,
            residual_scale=residual_scale,
            recency_tau=recency_tau,
            recency_lambda=recency_lambda,
        )
        next_logits = logits[0, -1]
        target_id = codec.stoi[target]
        pred_id = int(torch.argmax(next_logits).item())
        pred = codec.itos[pred_id]
        nll = float(F.cross_entropy(next_logits.view(1, -1), torch.tensor([target_id], device=device)).item())
        is_correct = int(pred == target)
        choice_pred: str | None = None
        choice_rank: int | None = None
        choice_correct: int | None = None
        choice_mrr: float | None = None
        if len(choice_ids) >= 2:
            target_choice_positions = (choice_ids == target_id).nonzero(as_tuple=False)
            if target_choice_positions.numel() > 0:
                choice_logits = next_logits.index_select(0, choice_ids)
                choice_pred_index = int(torch.argmax(choice_logits).item())
                choice_pred_id = int(choice_ids[choice_pred_index].item())
                choice_pred = codec.itos[choice_pred_id]
                target_choice_index = int(target_choice_positions[0].item())
                target_choice_logit = choice_logits[target_choice_index]
                choice_rank = int((choice_logits > target_choice_logit).sum().item()) + 1
                choice_correct = int(choice_pred_id == target_id)
                choice_mrr = 1.0 / float(choice_rank)
        total_nll += nll
        correct += is_correct
        bucket = "heldout" if key in holdout_key_set else "seen"
        split_totals[bucket]["cases"] += 1
        split_totals[bucket]["correct"] += is_correct
        split_totals[bucket]["nll"] += nll
        if choice_correct is not None and choice_mrr is not None:
            choice_cases += 1
            choice_correct_total += choice_correct
            choice_mrr_total += choice_mrr
            split_totals[bucket]["choice_cases"] += 1
            split_totals[bucket]["choice_correct"] += choice_correct
            split_totals[bucket]["choice_mrr"] += choice_mrr
        if len(examples) < 5:
            examples.append(
                {
                    "prefix": prefix,
                    "retrieval_context": retrieval_context,
                    "target": target,
                    "prediction": pred,
                    "correct": pred == target,
                    "choice_prediction": choice_pred,
                    "choice_rank": choice_rank,
                    "choice_correct": choice_correct == 1 if choice_correct is not None else None,
                    "heldout_key": key in holdout_key_set,
                }
            )
        cases += 1

    def split_metric(bucket: str, field: str) -> int | float | None:
        total = split_totals[bucket]["cases"]
        if field == "cases":
            return int(total)
        if total == 0:
            return None
        if field == "accuracy":
            return round(float(split_totals[bucket]["correct"]) / float(total), 6)
        if field == "nll":
            return round(float(split_totals[bucket]["nll"]) / float(total), 6)
        if field == "choice_cases":
            return int(split_totals[bucket]["choice_cases"])
        choice_total = split_totals[bucket]["choice_cases"]
        if choice_total == 0:
            return None
        if field == "choice_accuracy":
            return round(float(split_totals[bucket]["choice_correct"]) / float(choice_total), 6)
        if field == "choice_mrr":
            return round(float(split_totals[bucket]["choice_mrr"]) / float(choice_total), 6)
        raise ValueError(f"Unknown split metric: {field}")

    if cases == 0:
        return {
            "copy_probe_marker": marker,
            "copy_probe_key_marker": key_marker,
            "copy_probe_retrieval_template": retrieval_template,
            "copy_probe_retrieval_source": retrieval_source,
            "copy_probe_retrieval_corrupt": retrieval_corrupt,
            "copy_probe_holdout_keys": normalized_holdout_keys,
            "copy_probe_choice_candidates": "".join(choice_chars),
            "copy_probe_retrieval_memory_entries": 0,
            "copy_probe_retrieval_memory_observations": 0,
            "copy_probe_retrieval_memory_conflicts": 0,
            "copy_probe_retrieval_hits": retrieval_hits,
            "copy_probe_retrieval_misses": retrieval_misses,
            "copy_probe_retrieval_corrupted": retrieval_corrupted,
            "copy_probe_cases": 0,
            "copy_probe_accuracy": None,
            "copy_probe_nll": None,
            "copy_probe_choice_cases": 0,
            "copy_probe_choice_accuracy": None,
            "copy_probe_choice_mrr": None,
            "copy_probe_seen_cases": 0,
            "copy_probe_seen_accuracy": None,
            "copy_probe_seen_nll": None,
            "copy_probe_seen_choice_cases": 0,
            "copy_probe_seen_choice_accuracy": None,
            "copy_probe_seen_choice_mrr": None,
            "copy_probe_heldout_cases": 0,
            "copy_probe_heldout_accuracy": None,
            "copy_probe_heldout_nll": None,
            "copy_probe_heldout_choice_cases": 0,
            "copy_probe_heldout_choice_accuracy": None,
            "copy_probe_heldout_choice_mrr": None,
            "copy_probe_examples": [],
        }

    return {
        "copy_probe_marker": marker,
        "copy_probe_key_marker": key_marker,
        "copy_probe_retrieval_template": retrieval_template,
        "copy_probe_retrieval_source": retrieval_source,
        "copy_probe_retrieval_corrupt": retrieval_corrupt,
        "copy_probe_holdout_keys": normalized_holdout_keys,
        "copy_probe_choice_candidates": "".join(choice_chars),
        "copy_probe_retrieval_memory_entries": (retrieval_memory_report or {}).get("entries", 0),
        "copy_probe_retrieval_memory_observations": (retrieval_memory_report or {}).get("observations", 0),
        "copy_probe_retrieval_memory_conflicts": (retrieval_memory_report or {}).get("conflicts", 0),
        "copy_probe_retrieval_hits": retrieval_hits,
        "copy_probe_retrieval_misses": retrieval_misses,
        "copy_probe_retrieval_corrupted": retrieval_corrupted,
        "copy_probe_cases": cases,
        "copy_probe_accuracy": round(correct / cases, 6),
        "copy_probe_nll": round(total_nll / cases, 6),
        "copy_probe_choice_cases": choice_cases,
        "copy_probe_choice_accuracy": round(choice_correct_total / choice_cases, 6) if choice_cases else None,
        "copy_probe_choice_mrr": round(choice_mrr_total / choice_cases, 6) if choice_cases else None,
        "copy_probe_seen_cases": split_metric("seen", "cases"),
        "copy_probe_seen_accuracy": split_metric("seen", "accuracy"),
        "copy_probe_seen_nll": split_metric("seen", "nll"),
        "copy_probe_seen_choice_cases": split_metric("seen", "choice_cases"),
        "copy_probe_seen_choice_accuracy": split_metric("seen", "choice_accuracy"),
        "copy_probe_seen_choice_mrr": split_metric("seen", "choice_mrr"),
        "copy_probe_heldout_cases": split_metric("heldout", "cases"),
        "copy_probe_heldout_accuracy": split_metric("heldout", "accuracy"),
        "copy_probe_heldout_nll": split_metric("heldout", "nll"),
        "copy_probe_heldout_choice_cases": split_metric("heldout", "choice_cases"),
        "copy_probe_heldout_choice_accuracy": split_metric("heldout", "choice_accuracy"),
        "copy_probe_heldout_choice_mrr": split_metric("heldout", "choice_mrr"),
        "copy_probe_examples": examples,
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    device = choose_device(args.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)

    text = read_text(args.corpus)
    if args.vocab_chars and args.vocab_chars_file is not None:
        raise ValueError("Pass only one of --vocab-chars or --vocab-chars-file")
    vocab_chars = args.vocab_chars
    if args.vocab_chars_file is not None:
        vocab_chars = args.vocab_chars_file.read_text(encoding="utf-8")
    codec = build_codec(text, vocab_chars)
    split_at = split_at_index(len(text), args.val_fraction, args.block_size)
    val_text = text[split_at:]
    stream_sampler: ShardedBatchSampler | None = None
    if args.train_shard_dir:
        if args.copy_train_marker:
            raise ValueError("--train-shard-dir is currently only supported for plain LM batches")
        if args.curriculum_filter != "off":
            raise ValueError("--train-shard-dir cannot be combined with curriculum filtering yet")
        if args.init != "random":
            raise ValueError("--train-shard-dir currently supports random-init runs only")
        if args.residual_base not in {"none", "count-ngram"}:
            raise ValueError("--train-shard-dir currently supports no residual base or count-ngram")
        if args.residual_optim in {"es", "coord"}:
            raise ValueError("--train-shard-dir does not support gradient-free residual search")
        stream_sampler = ShardedBatchSampler.from_dir(args.train_shard_dir, codec, args.block_size, device)
        train_chars_report = stream_sampler.total_chars
        val_data = torch.tensor(codec.encode(val_text), dtype=torch.long, device=device)
        train_data = stream_sampler.eval_tensor(args.stream_train_eval_chars)
    else:
        ids = codec.encode(text)
        train_data, val_data = split_ids(ids, args.val_fraction, args.block_size, device)
        train_chars_report = int(train_data.numel())

    retention_eval_data: torch.Tensor | None = None
    retention_eval_chars = 0
    if args.retention_eval_corpus is not None:
        retention_eval_text = read_text(args.retention_eval_corpus)
        retention_eval_ids = codec.encode(retention_eval_text)
        if len(retention_eval_ids) <= args.block_size:
            raise ValueError("--retention-eval-corpus is too short for the configured block size")
        retention_eval_data = torch.tensor(retention_eval_ids, dtype=torch.long, device=device)
        retention_eval_chars = len(retention_eval_ids)

    if args.copy_loss_weight <= 0:
        raise ValueError("--copy-loss-weight must be greater than zero")
    if args.copy_choice_weight < 0:
        raise ValueError("--copy-choice-weight must be non-negative")
    if not 0 < args.copy_sample_fraction <= 1:
        raise ValueError("--copy-sample-fraction must be in (0, 1]")
    if not 0 <= args.copy_rehearsal_fraction <= 1:
        raise ValueError("--copy-rehearsal-fraction must be in [0, 1]")
    if (
        args.copy_sampler == "correction_then_retrieval_rehearsal_mixed"
        and args.copy_sample_fraction + args.copy_rehearsal_fraction > 1
    ):
        raise ValueError("rehearsal sampler requires copy sample plus rehearsal fraction <= 1")
    if not 0 < args.copy_curriculum_switch_fraction < 1:
        raise ValueError("--copy-curriculum-switch-fraction must be in (0, 1)")
    if args.copy_mine_every < 0:
        raise ValueError("--copy-mine-every must be non-negative")
    if args.copy_probe_retrieval_source == "memory" and not args.copy_probe_marker:
        raise ValueError("--copy-probe-retrieval-source memory requires --copy-probe-marker")
    if args.copy_probe_memory_scope == "all" and args.copy_probe_retrieval_source != "memory":
        raise ValueError("--copy-probe-memory-scope all requires memory retrieval source")
    if args.copy_probe_retrieval_corrupt == "wrong-answer":
        if args.copy_probe_retrieval_source != "memory":
            raise ValueError("--copy-probe-retrieval-corrupt wrong-answer requires memory retrieval source")
        if args.copy_probe_retrieval_template == "none":
            raise ValueError("--copy-probe-retrieval-corrupt wrong-answer requires a retrieval template")
    if args.copy_sampler != "random" and not args.copy_train_marker:
        raise ValueError("--copy-sampler requires --copy-train-marker")
    if args.copy_choice_weight > 0 and not args.copy_train_marker:
        raise ValueError("--copy-choice-weight requires --copy-train-marker")
    if not 0 <= args.curriculum_fraction <= 1:
        raise ValueError("--curriculum-fraction must be in [0, 1]")
    if args.curriculum_rescore_every <= 0:
        raise ValueError("--curriculum-rescore-every must be positive")
    if args.curriculum_pool_size <= 0:
        raise ValueError("--curriculum-pool-size must be positive")
    if args.recency_tau <= 0:
        raise ValueError("--recency-tau must be positive")
    if not 0 <= args.recency_lambda <= 1:
        raise ValueError("--recency-lambda must be in [0, 1]")
    if args.recency_lambda > 0 and args.residual_base == "none":
        raise ValueError("--recency-lambda requires --residual-base")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    if args.grad_accum_steps > 1 and (args.copy_train_marker or args.curriculum_filter != "off"):
        raise ValueError("--grad-accum-steps greater than 1 is currently only supported for plain LM batches")
    if args.optimizer != "adamw" and args.residual_optim not in {"adamw", "none"}:
        raise ValueError("--optimizer only applies to gradient training with --residual-optim adamw")
    if args.precision == "bf16" and device != "cuda":
        raise ValueError("--precision bf16 requires CUDA")
    if args.lr_schedule == "cosine" and not 0 < args.lr_final_frac <= 1:
        raise ValueError("--lr-final-frac must be in (0, 1] for cosine scheduling")
    if args.lr_total_steps < 0:
        raise ValueError("--lr-total-steps must be non-negative")
    if args.checkpoint_keep < 0:
        raise ValueError("--checkpoint-keep must be non-negative")
    if args.optimizer == "muon":
        if args.muon_lr <= 0:
            raise ValueError("--muon-lr must be positive")
        if not 0 <= args.muon_momentum < 1:
            raise ValueError("--muon-momentum must be in [0, 1)")
        if args.muon_ns_steps <= 0:
            raise ValueError("--muon-ns-steps must be positive")
    if not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1:
        raise ValueError("--adam-beta1 and --adam-beta2 must be in (0, 1)")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be positive")
    if args.residual_optim in {"es", "coord", "none"} and args.residual_base == "none":
        raise ValueError("--residual-optim es, coord, or none requires --residual-base")
    if args.residual_optim in {"es", "coord"}:
        if args.copy_train_marker:
            raise ValueError("gradient-free residual search is only implemented for plain language-model batches")
        if args.curriculum_filter != "off":
            raise ValueError("gradient-free residual search cannot be combined with curriculum filtering")
    if args.search_batches <= 0:
        raise ValueError("--search-batches must be positive")
    if args.search_population <= 0:
        raise ValueError("--search-population must be positive")
    if args.search_sigma <= 0:
        raise ValueError("--search-sigma must be positive")
    if args.search_lr <= 0:
        raise ValueError("--search-lr must be positive")
    if args.coord_step_size <= 0:
        raise ValueError("--coord-step-size must be positive")
    train_marker_weights: torch.Tensor | None = None
    answer_starts: torch.Tensor | None = None
    failed_answer_starts: torch.Tensor | None = None
    correction_data: torch.Tensor | None = None
    correction_weights: torch.Tensor | None = None
    retrieval_data: torch.Tensor | None = None
    retrieval_weights: torch.Tensor | None = None
    copy_choice_ids: torch.Tensor | None = None
    copy_choice_lookup: torch.Tensor | None = None
    copy_choice_chars: list[str] = []
    verified_targets: list[int] = []
    copy_train_positions = 0
    copy_verified_positions = 0
    copy_sample_starts = 0
    copy_mine_events = 0
    copy_last_train_accuracy: float | None = None
    copy_last_failures = 0
    copy_correction_examples = 0
    copy_correction_chars = 0
    copy_retrieval_train_examples = 0
    copy_retrieval_train_chars = 0
    if args.copy_train_marker:
        train_marker_weights = build_marker_weights(text, args.copy_train_marker, split_at, device)
        copy_train_positions = int(train_marker_weights.sum().item())
        if (
            args.copy_sampler
            in {
                "answer",
                "mixed",
                "failed",
                "failed_mixed",
                "correction",
                "correction_mixed",
                "retrieval",
                "retrieval_mixed",
                "correction_retrieval_mixed",
                "correction_then_retrieval_mixed",
                "correction_then_retrieval_rehearsal_mixed",
            }
            or args.copy_choice_weight > 0
        ):
            verified_targets = find_verified_copy_targets(
                text,
                args.copy_train_marker,
                split_at,
                args.copy_verify_mode,
            )
            copy_verified_positions = len(verified_targets)
            if not verified_targets:
                raise ValueError(f"No verified key/answer pairs found for --copy-sampler {args.copy_sampler}")
            if args.copy_choice_weight > 0:
                copy_choice_ids, copy_choice_lookup, copy_choice_chars = build_choice_tensors(
                    text,
                    verified_targets,
                    codec,
                    device,
                )
            if args.copy_sampler in {
                "answer",
                "mixed",
                "failed",
                "failed_mixed",
                "correction",
                "correction_mixed",
                "retrieval",
                "retrieval_mixed",
                "correction_retrieval_mixed",
                "correction_then_retrieval_mixed",
                "correction_then_retrieval_rehearsal_mixed",
            }:
                train_marker_weights = build_position_weights(verified_targets, split_at, device)
                copy_train_positions = int(train_marker_weights.sum().item())
                answer_starts = build_answer_starts(verified_targets, args.block_size, len(train_data), device)
                failed_answer_starts = answer_starts
                copy_sample_starts = int(answer_starts.numel())
            if args.copy_sampler in {
                "retrieval",
                "retrieval_mixed",
                "correction_retrieval_mixed",
                "correction_then_retrieval_mixed",
                "correction_then_retrieval_rehearsal_mixed",
            }:
                retrieval_positions = torch.tensor(verified_targets, dtype=torch.long, device=device)
                retrieval_data, retrieval_weights, copy_retrieval_train_examples, copy_retrieval_train_chars = (
                    build_retrieval_data(
                        text,
                        retrieval_positions,
                        codec,
                        args.copy_train_marker,
                        args.copy_train_retrieval_template,
                        args.block_size,
                        device,
                    )
                )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    model = TinyTransformer(
        vocab_size=len(codec.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        adapter_rank=args.adapter_rank,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pos_encoding=args.pos_encoding,
        activation_checkpoint=args.activation_checkpoint,
    ).to(device)

    init_report: dict[str, object] = {}
    residual_report: dict[str, object] = {}
    base_logits: PriorLogits | None = None
    if args.residual_base != "none" and args.init != "random":
        raise ValueError("--residual-base cannot be combined with --init count-bigram")

    if args.init == "count-bigram":
        init_report = apply_count_bigram_init(
            model=model,
            train_data=train_data,
            vocab_size=len(codec.chars),
            alpha=args.count_alpha,
            embedding_scale=args.count_embedding_scale,
        )
    elif args.init != "random":
        raise ValueError(f"Unknown initialization: {args.init}")

    if args.residual_base == "count-bigram":
        base_logits = build_count_logits(train_data, len(codec.chars), args.count_alpha)
        if args.zero_residual_head:
            zero_output_head(model)
        residual_report = {
            "base": "count-bigram",
            "frozen_prior_parameters": int(base_logits.numel()),
            "count_alpha": args.count_alpha,
            "zero_residual_head": args.zero_residual_head,
            "residual_scale": args.residual_scale,
            "residual_l2": args.residual_l2,
        }
    elif args.residual_base == "count-trigram":
        base_logits = build_count_logits_trigram(
            train_data,
            len(codec.chars),
            args.count_alpha,
            args.trigram_backoff,
        )
        if args.zero_residual_head:
            zero_output_head(model)
        residual_report = {
            "base": "count-trigram",
            "frozen_prior_parameters": int(base_logits.numel()),
            "count_alpha": args.count_alpha,
            "trigram_backoff": args.trigram_backoff,
            "trigram_start_row": "bigram",
            "zero_residual_head": args.zero_residual_head,
            "residual_scale": args.residual_scale,
            "residual_l2": args.residual_l2,
        }
    elif args.residual_base == "count-ngram":
        cache_key = (
            "count-ngram",
            str(args.corpus.resolve()),
            str(args.train_shard_dir.resolve()) if args.train_shard_dir else "",
            train_chars_report,
            len(val_data),
            len(codec.chars),
            args.prior_order,
            args.count_alpha,
            args.ngram_backoff,
            args.prior5_min_count if args.prior_order == 5 else None,
            device,
        )
        cached_prior = COUNT_NGRAM_CACHE.get(cache_key)
        if cached_prior is None:
            prior_cache_path = count_ngram_prior_cache_path(
                args.prior_cache_dir,
                args.corpus,
                args.train_shard_dir,
                train_chars_report,
                len(codec.chars),
                args.prior_order,
                args.count_alpha,
                args.ngram_backoff,
                args.prior5_min_count if args.prior_order == 5 else None,
            )
            if prior_cache_path is not None and prior_cache_path.exists():
                cached = torch.load(prior_cache_path, map_location="cpu")
                base_logits = load_count_ngram_prior_from_cache(cached, device)
                ngram_report = dict(cached["ngram_report"])
                ngram_report["prior_cache_path"] = str(prior_cache_path)
                ngram_report["prior_cache_status"] = "hit"
            else:
                if stream_sampler is not None:
                    base_logits_cpu, ngram_report = build_count_logits_ngram_from_shards(
                        stream_sampler.paths,
                        codec,
                        len(codec.chars),
                        args.prior_order,
                        args.count_alpha,
                        args.ngram_backoff,
                        args.prior5_min_count,
                        probe_ids=val_data,
                    )
                    if prior_cache_path is not None:
                        prior_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            count_ngram_prior_cache_payload(base_logits_cpu, ngram_report),
                            prior_cache_path,
                        )
                        ngram_report["prior_cache_path"] = str(prior_cache_path)
                        ngram_report["prior_cache_status"] = "miss_saved"
                    base_logits = base_logits_cpu.to(device)
                else:
                    base_logits, ngram_report = build_count_logits_ngram(
                        train_data,
                        len(codec.chars),
                        args.prior_order,
                        args.count_alpha,
                        args.ngram_backoff,
                        probe_ids=val_data,
                    )
                    if prior_cache_path is not None:
                        prior_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            count_ngram_prior_cache_payload(base_logits, ngram_report),
                            prior_cache_path,
                        )
                        ngram_report["prior_cache_path"] = str(prior_cache_path)
                        ngram_report["prior_cache_status"] = "miss_saved"
            COUNT_NGRAM_CACHE[cache_key] = (base_logits, ngram_report)
        else:
            base_logits, ngram_report = cached_prior
        if args.zero_residual_head:
            zero_output_head(model)
        residual_report = {
            "base": "count-ngram",
            "frozen_prior_parameters": int(base_logits.numel()),
            "count_alpha": args.count_alpha,
            "zero_residual_head": args.zero_residual_head,
            "residual_scale": args.residual_scale,
            "residual_l2": args.residual_l2,
            "recency_tau": args.recency_tau,
            "recency_lambda": args.recency_lambda,
            "recency_interpolation": "count-plus-exponential-cache" if args.recency_lambda > 0 else "off",
            **ngram_report,
        }
    elif args.residual_base != "none":
        raise ValueError(f"Unknown residual base: {args.residual_base}")

    curriculum_starts: torch.Tensor | None = None
    dynamic_pool_starts: torch.Tensor | None = None
    dynamic_previous_losses: torch.Tensor | None = None
    dynamic_smoothed_delta: torch.Tensor | None = None
    dynamic_selected_starts: torch.Tensor | None = None
    curriculum_report: dict[str, object] = {
        "curriculum_pool_fraction": 0.0,
        "curriculum_pool_starts": 0,
        "curriculum_total_starts": 0,
        "curriculum_score_seconds": 0.0,
        "curriculum_rescore_events": 0,
        "curriculum_rescore_every": args.curriculum_rescore_every,
        "curriculum_pool_size": args.curriculum_pool_size,
        "curriculum_delta_ema": DYNAMIC_CURRICULUM_DELTA_EMA,
        "curriculum_delta_threshold": DYNAMIC_CURRICULUM_DELTA_THRESHOLD,
    }
    if args.curriculum_filter in {"prior-loss", "dynamic-reducible"}:
        if base_logits is None:
            raise ValueError(f"--curriculum-filter {args.curriculum_filter} requires --residual-base")
        if not 0 < args.curriculum_fraction <= 1:
            raise ValueError("--curriculum-fraction must be in (0, 1] when filtering")
        if train_marker_weights is not None:
            raise ValueError(f"--curriculum-filter {args.curriculum_filter} is only defined for plain language-model batches")
    if args.curriculum_filter == "prior-loss":
        score_key = (
            "prior-loss",
            str(args.corpus.resolve()),
            len(train_data),
            len(val_data),
            len(codec.chars),
            args.block_size,
            args.residual_base,
            args.prior_order,
            args.count_alpha,
            args.ngram_backoff,
            device,
        )
        cached_scores = CURRICULUM_SCORE_CACHE.get(score_key)
        if cached_scores is None:
            curriculum_starts, curriculum_report = build_prior_loss_curriculum(
                train_data,
                args.block_size,
                base_logits,
                args.batch_size,
                device,
            )
            curriculum_report.update(
                {
                    "curriculum_rescore_events": 0,
                    "curriculum_rescore_every": args.curriculum_rescore_every,
                    "curriculum_pool_size": args.curriculum_pool_size,
                    "curriculum_delta_ema": DYNAMIC_CURRICULUM_DELTA_EMA,
                    "curriculum_delta_threshold": DYNAMIC_CURRICULUM_DELTA_THRESHOLD,
                }
            )
            CURRICULUM_SCORE_CACHE[score_key] = (curriculum_starts, curriculum_report)
        else:
            curriculum_starts, curriculum_report = cached_scores
    elif args.curriculum_filter == "dynamic-reducible":
        dynamic_pool_starts, pool_report = build_dynamic_curriculum_pool(
            train_data,
            args.block_size,
            args.curriculum_pool_size,
            args.seed,
            device,
        )
        score_start = time.perf_counter()
        dynamic_previous_losses = score_windows_by_model_loss(
            model,
            train_data,
            dynamic_pool_starts,
            args.block_size,
            args.batch_size,
            base_logits,
            args.residual_scale,
            device,
        )
        score_seconds = time.perf_counter() - score_start
        dynamic_smoothed_delta = torch.zeros_like(dynamic_previous_losses)
        dynamic_selected_starts = dynamic_pool_starts[:0]
        initial_high_loss_count = max(
            args.batch_size,
            round(len(dynamic_pool_starts) * DYNAMIC_CURRICULUM_HIGH_LOSS_FRACTION),
        )
        initial_high_loss_count = max(1, min(len(dynamic_pool_starts), initial_high_loss_count))
        curriculum_report.update(pool_report)
        curriculum_report.update(
            {
                "curriculum_score_seconds": round(score_seconds, 4),
                "curriculum_rescore_events": 0,
                "curriculum_high_loss_pool_starts": initial_high_loss_count,
                "curriculum_selected_starts": 0,
                "curriculum_selected_fraction": 0.0,
                "curriculum_current_loss_mean": round(float(dynamic_previous_losses.mean().item()), 6),
                "curriculum_current_loss_min": round(float(dynamic_previous_losses.min().item()), 6),
                "curriculum_current_loss_max": round(float(dynamic_previous_losses.max().item()), 6),
                "curriculum_delta_mean": 0.0,
                "curriculum_delta_min": 0.0,
                "curriculum_delta_max": 0.0,
                "curriculum_selected_loss_mean": 0.0,
                "curriculum_selected_delta_mean": 0.0,
            }
        )
    elif args.curriculum_filter != "off":
        raise ValueError(f"Unknown curriculum filter: {args.curriculum_filter}")

    train_scope_report = configure_train_scope(model, args.train_scope)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    formation_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    if args.residual_optim == "none":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainable_parameters = []
    elif not trainable_parameters:
        raise ValueError("No trainable parameters selected")

    optimizer, optimizer_report = build_optimizer(model, args)

    loss_curve: list[dict[str, float | int]] = []
    residual_optim_report: dict[str, object] = {}
    formation_forward_passes = 0
    formation_steps = 0
    resume_step = 0
    resume_report: dict[str, object] = {
        "resume_from": str(args.resume_from) if args.resume_from else "",
        "resume_loaded": False,
        "resume_step": 0,
    }
    last_checkpoint_path = ""
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_every > 0 and args.checkpoint_dir is None:
        raise ValueError("--checkpoint-every requires --checkpoint-dir")
    if args.resume_from is not None:
        if args.residual_optim != "adamw":
            raise ValueError("--resume-from currently supports gradient training only")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if checkpoint.get("chars") != codec.chars:
            raise ValueError("Resume checkpoint vocabulary does not match this corpus")
        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        generator_state = checkpoint.get("generator_state")
        if generator_state is not None:
            if isinstance(generator_state, torch.Tensor):
                generator_state = generator_state.detach().cpu()
            generator.set_state(generator_state)
        resume_step = int(checkpoint.get("step", 0))
        formation_steps = resume_step
        formation_forward_passes = int(checkpoint.get("formation_forward_passes", 0))
        raw_loss_curve = checkpoint.get("loss_curve", [])
        if isinstance(raw_loss_curve, list):
            loss_curve = raw_loss_curve
        resume_report.update(
            {
                "resume_loaded": True,
                "resume_step": resume_step,
                "resume_checkpoint": str(args.resume_from),
            }
        )

    resumed_eval_row = None
    if resume_step > 0:
        for row in loss_curve:
            if int(row.get("step", -1)) == resume_step and "train_nll" in row and "val_nll" in row:
                resumed_eval_row = row
                break
    if resumed_eval_row is not None:
        initial_losses = {
            "train": float(resumed_eval_row["train_nll"]),
            "val": float(resumed_eval_row["val_nll"]),
        }
    else:
        initial_losses = estimate_loss(
            model,
            train_data,
            val_data,
            args.block_size,
            args.batch_size,
            args.eval_mode,
            args.eval_batches,
            generator,
            device,
            base_logits,
            args.residual_scale,
            args.recency_tau,
            args.recency_lambda,
        )
        if resume_step > 0 and args.eval_interval > 0 and resume_step % args.eval_interval == 0:
            loss_curve.append(
                {
                    "step": resume_step,
                    "train_nll": round(initial_losses["train"], 6),
                    "val_nll": round(initial_losses["val"], 6),
                }
            )
    progress_enabled = args.log_every > 0
    log_progress(
        progress_enabled,
        (
            "[run] "
            f"corpus={args.corpus} device={device} seed={args.seed} "
            f"chars={len(text)} train={train_chars_report} val={int(val_data.numel())} "
            f"vocab={len(codec.chars)} block={args.block_size} batch={args.batch_size} "
            f"accum={args.grad_accum_steps} "
            f"model=L{args.n_layer} H{args.n_head} D{args.n_embd} "
            f"pos={args.pos_encoding} checkpoint={args.activation_checkpoint} "
            f"params={count_parameters(model)} trainable={count_parameters(model, only_trainable=True)} "
            f"prior={args.residual_base} scope={args.train_scope} optimizer={args.optimizer}"
        ),
    )
    log_progress(
        progress_enabled,
        (
            "[eval] "
            f"step={resume_step} train_nll={initial_losses['train']:.6f} "
            f"val_nll={initial_losses['val']:.6f} "
            f"bits={initial_losses['val'] / math.log(2):.6f} "
            f"elapsed={time.perf_counter() - start:.1f}s"
        ),
    )
    copy_curriculum_switch_step = 0
    staged_samplers = {
        "correction_then_retrieval_mixed",
        "correction_then_retrieval_rehearsal_mixed",
    }
    if args.copy_sampler in staged_samplers:
        copy_curriculum_switch_step = max(1, min(args.steps - 1, round(args.steps * args.copy_curriculum_switch_fraction)))

    if args.residual_optim == "es":
        residual_optim_report = evolution_strategy_residual_search(
            model=model,
            parameters=trainable_parameters,
            train_data=train_data,
            block_size=args.block_size,
            batch_size=args.batch_size,
            steps=args.steps,
            search_batches=args.search_batches,
            population=args.search_population,
            sigma=args.search_sigma,
            lr=args.search_lr,
            eval_interval=args.eval_interval,
            log_every=args.log_every,
            generator=generator,
            device=device,
            base_logits=base_logits,
            residual_scale=args.residual_scale,
            recency_tau=args.recency_tau,
            recency_lambda=args.recency_lambda,
            residual_l2=args.residual_l2,
        )
        formation_forward_passes = int(residual_optim_report.get("formation_forward_passes", 0))
        formation_steps = int(residual_optim_report.get("formation_steps", 0))
    elif args.residual_optim == "coord":
        residual_optim_report = coordinate_residual_search(
            model=model,
            parameters=trainable_parameters,
            train_data=train_data,
            block_size=args.block_size,
            batch_size=args.batch_size,
            steps=args.steps,
            search_batches=args.search_batches,
            step_size=args.coord_step_size,
            eval_interval=args.eval_interval,
            log_every=args.log_every,
            generator=generator,
            device=device,
            base_logits=base_logits,
            residual_scale=args.residual_scale,
            recency_tau=args.recency_tau,
            recency_lambda=args.recency_lambda,
            residual_l2=args.residual_l2,
        )
        formation_forward_passes = int(residual_optim_report.get("formation_forward_passes", 0))
        formation_steps = int(residual_optim_report.get("formation_steps", 0))
    elif args.residual_optim == "none":
        residual_optim_report = {
            "formation_steps": 0,
            "formation_forward_passes": 0,
            "search_batches": 0,
        }
    elif args.residual_optim != "adamw":
        raise ValueError(f"Unknown residual optimizer: {args.residual_optim}")

    planned_training_steps = args.steps
    training_steps = planned_training_steps if args.residual_optim == "adamw" else 0
    if args.resume_from is not None and args.resume_steps_additional and args.residual_optim == "adamw":
        training_steps = resume_step + planned_training_steps
    if resume_step > training_steps:
        raise ValueError("--resume-from checkpoint step is greater than --steps")
    lr_total_steps = args.lr_total_steps if args.lr_total_steps > 0 else training_steps
    if args.lr_schedule == "cosine" and lr_total_steps < training_steps:
        raise ValueError("--lr-total-steps must be at least the final training step for cosine scheduling")
    last_lr_factor = 1.0
    for step in range(resume_step + 1, training_steps + 1):
        already_stepped = False
        in_correction_phase = args.copy_sampler not in staged_samplers or step <= copy_curriculum_switch_step
        if args.copy_sampler in {
            "failed",
            "failed_mixed",
            "correction",
            "correction_mixed",
            "correction_retrieval_mixed",
            "correction_then_retrieval_mixed",
            "correction_then_retrieval_rehearsal_mixed",
        } and in_correction_phase and (
            step == 1 or (args.copy_mine_every > 0 and (step - 1) % args.copy_mine_every == 0)
        ):
            if train_marker_weights is None or not verified_targets:
                raise RuntimeError("Expected verified targets during failed-copy mining")
            mined = mine_failed_copy_starts(
                model=model,
                data=train_data,
                target_positions=verified_targets,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
                base_logits=base_logits,
                residual_scale=args.residual_scale,
            )
            mined_failed_starts = mined["failed_starts"]
            mined_answer_starts = mined["answer_starts"]
            mined_failed_positions = mined["failed_positions"]
            mined_answer_positions = mined["answer_positions"]
            if (
                not isinstance(mined_failed_starts, torch.Tensor)
                or not isinstance(mined_answer_starts, torch.Tensor)
                or not isinstance(mined_failed_positions, torch.Tensor)
                or not isinstance(mined_answer_positions, torch.Tensor)
            ):
                raise RuntimeError("Expected mined start tensors during failed-copy mining")
            failed_answer_starts = mined_failed_starts if len(mined_failed_starts) > 0 else mined_answer_starts
            correction_positions = (
                mined_failed_positions if len(mined_failed_positions) > 0 else mined_answer_positions
            )
            correction_data, correction_weights, copy_correction_examples, copy_correction_chars = build_correction_data(
                text,
                correction_positions,
                codec,
                args.copy_train_marker,
                args.copy_correction_template,
                args.block_size,
                device,
            )
            copy_mine_events += 1
            copy_last_train_accuracy = float(mined["train_copy_accuracy"])
            copy_last_failures = int(mined["train_copy_failures"])

        if args.curriculum_filter == "dynamic-reducible" and step % args.curriculum_rescore_every == 0:
            if dynamic_pool_starts is None or dynamic_previous_losses is None or dynamic_smoothed_delta is None:
                raise RuntimeError("Expected dynamic curriculum state during re-scoring")
            if base_logits is None:
                raise RuntimeError("Expected base logits during dynamic curriculum re-scoring")
            score_start = time.perf_counter()
            current_losses = score_windows_by_model_loss(
                model,
                train_data,
                dynamic_pool_starts,
                args.block_size,
                args.batch_size,
                base_logits,
                args.residual_scale,
                device,
            )
            score_seconds = time.perf_counter() - score_start
            raw_delta = dynamic_previous_losses - current_losses
            dynamic_smoothed_delta = (
                DYNAMIC_CURRICULUM_DELTA_EMA * raw_delta
                + (1.0 - DYNAMIC_CURRICULUM_DELTA_EMA) * dynamic_smoothed_delta
            )
            dynamic_selected_starts, dynamic_report = select_dynamic_reducible_starts(
                dynamic_pool_starts,
                current_losses,
                dynamic_smoothed_delta,
                args.batch_size,
            )
            dynamic_previous_losses = current_losses
            curriculum_report.update(dynamic_report)
            curriculum_report["curriculum_score_seconds"] = round(
                float(curriculum_report.get("curriculum_score_seconds", 0.0)) + score_seconds,
                4,
            )
            curriculum_report["curriculum_rescore_events"] = int(
                curriculum_report.get("curriculum_rescore_events", 0)
            ) + 1

        if train_marker_weights is None:
            if args.grad_accum_steps > 1:
                if optimizer is None:
                    raise RuntimeError("Expected optimizer during gradient training")
                optimizer.zero_grad(set_to_none=True)
                total_loss_value = 0.0
                for _micro_step in range(args.grad_accum_steps):
                    if stream_sampler is not None:
                        xb, yb = stream_sampler.get_batch(args.batch_size, generator)
                    else:
                        xb, yb = get_batch(train_data, args.block_size, args.batch_size, generator, device)
                    with precision_context(device, args.precision):
                        _, micro_loss = model(
                            xb,
                            yb,
                            base_logits=base_logits,
                            residual_scale=args.residual_scale,
                            recency_tau=args.recency_tau,
                            recency_lambda=args.recency_lambda,
                            residual_l2=args.residual_l2,
                        )
                    if micro_loss is None:
                        raise RuntimeError("Expected micro loss during gradient accumulation")
                    total_loss_value += float(micro_loss.item())
                    (micro_loss / args.grad_accum_steps).backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                last_lr_factor = apply_lr_schedule(optimizer, args, step, lr_total_steps)
                optimizer.step()
                loss = torch.tensor(total_loss_value / args.grad_accum_steps, device=device)
                already_stepped = True
            else:
                if args.curriculum_filter == "prior-loss":
                    if curriculum_starts is None:
                        raise RuntimeError("Expected curriculum starts during prior-loss filtering")
                    xb, yb = get_prior_loss_filtered_batch(
                        train_data,
                        curriculum_starts,
                        args.block_size,
                        args.batch_size,
                        args.curriculum_fraction,
                        generator,
                        device,
                    )
                elif args.curriculum_filter == "dynamic-reducible":
                    if dynamic_selected_starts is not None and len(dynamic_selected_starts) > 0:
                        xb, yb = get_prior_loss_filtered_batch(
                            train_data,
                            dynamic_selected_starts,
                            args.block_size,
                            args.batch_size,
                            args.curriculum_fraction,
                            generator,
                            device,
                        )
                    else:
                        if stream_sampler is not None:
                            xb, yb = stream_sampler.get_batch(args.batch_size, generator)
                        else:
                            xb, yb = get_batch(train_data, args.block_size, args.batch_size, generator, device)
                else:
                    if stream_sampler is not None:
                        xb, yb = stream_sampler.get_batch(args.batch_size, generator)
                    else:
                        xb, yb = get_batch(train_data, args.block_size, args.batch_size, generator, device)
                with precision_context(device, args.precision):
                    _, loss = model(
                        xb,
                        yb,
                        base_logits=base_logits,
                        residual_scale=args.residual_scale,
                        recency_tau=args.recency_tau,
                        recency_lambda=args.recency_lambda,
                        residual_l2=args.residual_l2,
                    )
        else:
            if args.copy_sampler == "answer":
                if answer_starts is None:
                    raise RuntimeError("Expected answer starts during answer-sampled training")
                xb, yb, wb = get_answer_sampled_batch(
                    train_data,
                    train_marker_weights,
                    answer_starts,
                    args.block_size,
                    args.batch_size,
                    generator,
                    device,
                )
            elif args.copy_sampler == "mixed":
                if answer_starts is None:
                    raise RuntimeError("Expected answer starts during mixed copy-sampled training")
                xb, yb, wb = get_mixed_sampled_batch(
                    train_data,
                    train_marker_weights,
                    answer_starts,
                    args.block_size,
                    args.batch_size,
                    args.copy_sample_fraction,
                    generator,
                    device,
                )
            elif args.copy_sampler == "failed":
                if failed_answer_starts is None:
                    raise RuntimeError("Expected failed starts during failed-copy replay")
                xb, yb, wb = get_answer_sampled_batch(
                    train_data,
                    train_marker_weights,
                    failed_answer_starts,
                    args.block_size,
                    args.batch_size,
                    generator,
                    device,
                )
            elif args.copy_sampler == "failed_mixed":
                if failed_answer_starts is None:
                    raise RuntimeError("Expected failed starts during mixed failed-copy replay")
                xb, yb, wb = get_mixed_sampled_batch(
                    train_data,
                    train_marker_weights,
                    failed_answer_starts,
                    args.block_size,
                    args.batch_size,
                    args.copy_sample_fraction,
                    generator,
                    device,
                )
            elif args.copy_sampler == "correction":
                if correction_data is None or correction_weights is None:
                    raise RuntimeError("Expected correction data during correction replay")
                xb, yb, wb = get_weighted_batch(
                    correction_data,
                    correction_weights,
                    args.block_size,
                    args.batch_size,
                    generator,
                    device,
                )
                if wb is None:
                    raise RuntimeError("Expected correction weights during correction replay")
            elif args.copy_sampler == "correction_mixed":
                if correction_data is None or correction_weights is None:
                    raise RuntimeError("Expected correction data during mixed correction replay")
                xb, yb, wb = get_correction_mixed_batch(
                    train_data,
                    train_marker_weights,
                    correction_data,
                    correction_weights,
                    args.block_size,
                    args.batch_size,
                    args.copy_sample_fraction,
                    generator,
                    device,
                )
            elif args.copy_sampler == "retrieval":
                if retrieval_data is None or retrieval_weights is None:
                    raise RuntimeError("Expected retrieval-use data during retrieval training")
                xb, yb, wb = get_weighted_batch(
                    retrieval_data,
                    retrieval_weights,
                    args.block_size,
                    args.batch_size,
                    generator,
                    device,
                )
                if wb is None:
                    raise RuntimeError("Expected retrieval weights during retrieval training")
            elif args.copy_sampler == "retrieval_mixed":
                if retrieval_data is None or retrieval_weights is None:
                    raise RuntimeError("Expected retrieval-use data during mixed retrieval training")
                xb, yb, wb = get_correction_mixed_batch(
                    train_data,
                    train_marker_weights,
                    retrieval_data,
                    retrieval_weights,
                    args.block_size,
                    args.batch_size,
                    args.copy_sample_fraction,
                    generator,
                    device,
                )
            elif args.copy_sampler == "correction_retrieval_mixed":
                if correction_data is None or correction_weights is None:
                    raise RuntimeError("Expected correction data during mixed correction retrieval training")
                if retrieval_data is None or retrieval_weights is None:
                    raise RuntimeError("Expected retrieval-use data during mixed correction retrieval training")
                xb, yb, wb = get_dual_aux_mixed_batch(
                    train_data,
                    train_marker_weights,
                    correction_data,
                    correction_weights,
                    retrieval_data,
                    retrieval_weights,
                    args.block_size,
                    args.batch_size,
                    args.copy_sample_fraction,
                    generator,
                    device,
                )
            elif args.copy_sampler == "correction_then_retrieval_mixed":
                if step <= copy_curriculum_switch_step:
                    if correction_data is None or correction_weights is None:
                        raise RuntimeError("Expected correction data during staged correction training")
                    xb, yb, wb = get_correction_mixed_batch(
                        train_data,
                        train_marker_weights,
                        correction_data,
                        correction_weights,
                        args.block_size,
                        args.batch_size,
                        args.copy_sample_fraction,
                        generator,
                        device,
                    )
                else:
                    if retrieval_data is None or retrieval_weights is None:
                        raise RuntimeError("Expected retrieval-use data during staged retrieval training")
                    xb, yb, wb = get_correction_mixed_batch(
                        train_data,
                        train_marker_weights,
                        retrieval_data,
                        retrieval_weights,
                        args.block_size,
                        args.batch_size,
                        args.copy_sample_fraction,
                        generator,
                        device,
                    )
            elif args.copy_sampler == "correction_then_retrieval_rehearsal_mixed":
                if step <= copy_curriculum_switch_step:
                    if correction_data is None or correction_weights is None:
                        raise RuntimeError("Expected correction data during staged rehearsal correction training")
                    xb, yb, wb = get_correction_mixed_batch(
                        train_data,
                        train_marker_weights,
                        correction_data,
                        correction_weights,
                        args.block_size,
                        args.batch_size,
                        args.copy_sample_fraction,
                        generator,
                        device,
                    )
                else:
                    if correction_data is None or correction_weights is None:
                        raise RuntimeError("Expected correction data during staged rehearsal retrieval training")
                    if retrieval_data is None or retrieval_weights is None:
                        raise RuntimeError("Expected retrieval-use data during staged rehearsal retrieval training")
                    xb, yb, wb = get_rehearsal_mixed_batch(
                        train_data,
                        train_marker_weights,
                        correction_data,
                        correction_weights,
                        retrieval_data,
                        retrieval_weights,
                        args.block_size,
                        args.batch_size,
                        args.copy_sample_fraction,
                        args.copy_rehearsal_fraction,
                        generator,
                        device,
                    )
            else:
                xb, yb, wb = get_weighted_batch(
                    train_data,
                    train_marker_weights,
                    args.block_size,
                    args.batch_size,
                    generator,
                    device,
                )
                if wb is None:
                    raise RuntimeError("Expected marker weights during copy-aware training")
            with precision_context(device, args.precision):
                logits, _ = model(
                    xb,
                    base_logits=base_logits,
                    residual_scale=args.residual_scale,
                    recency_tau=args.recency_tau,
                    recency_lambda=args.recency_lambda,
                )
                token_loss = F.cross_entropy(
                    logits.view(-1, len(codec.chars)),
                    yb.view(-1),
                    reduction="none",
                ).view_as(yb)
                loss_weights = torch.ones_like(token_loss) + wb * (args.copy_loss_weight - 1.0)
                loss = (token_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
                if args.copy_choice_weight > 0:
                    if copy_choice_ids is None or copy_choice_lookup is None:
                        raise RuntimeError("Expected copy choice tensors during choice-loss training")
                    choice_mask = wb > 0
                    if choice_mask.any():
                        choice_logits = logits[choice_mask].index_select(-1, copy_choice_ids)
                        choice_targets = copy_choice_lookup[yb[choice_mask]]
                        valid_choice_targets = choice_targets >= 0
                        if valid_choice_targets.any():
                            choice_loss = F.cross_entropy(
                                choice_logits[valid_choice_targets],
                                choice_targets[valid_choice_targets],
                            )
                            loss = loss + args.copy_choice_weight * choice_loss
                if args.residual_l2 > 0:
                    residual_logits, _ = model(xb)
                    loss = loss + args.residual_l2 * residual_logits.pow(2).mean()
        if loss is None:
            raise RuntimeError("Expected loss during training")
        formation_forward_passes += args.grad_accum_steps if already_stepped else 1
        formation_steps = step
        if not already_stepped:
            if optimizer is None:
                raise RuntimeError("Expected optimizer during gradient training")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            last_lr_factor = apply_lr_schedule(optimizer, args, step, lr_total_steps)
            optimizer.step()

        if args.log_every > 0 and step % args.log_every == 0:
            print(
                "[train] "
                f"step={step}/{training_steps} "
                f"batch_nll={float(loss.item()):.6f} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.block_size,
                args.batch_size,
                args.eval_mode,
                args.eval_batches,
                generator,
                device,
                base_logits,
                args.residual_scale,
                args.recency_tau,
                args.recency_lambda,
            )
            loss_curve.append(
                {
                    "step": step,
                    "train_nll": round(losses["train"], 6),
                    "val_nll": round(losses["val"], 6),
                }
            )
            log_progress(
                progress_enabled,
                (
                    "[eval] "
                    f"step={step}/{training_steps} "
                    f"train_nll={losses['train']:.6f} "
                    f"val_nll={losses['val']:.6f} "
                    f"bits={losses['val'] / math.log(2):.6f} "
                    f"elapsed={time.perf_counter() - start:.1f}s"
                ),
            )

        checkpoint_path = training_checkpoint_path(args, step)
        if args.checkpoint_every > 0 and checkpoint_path is not None and step % args.checkpoint_every == 0:
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                generator,
                args,
                codec,
                base_logits,
                step,
                formation_forward_passes,
                loss_curve,
            )
            last_checkpoint_path = str(checkpoint_path)
            log_progress(
                progress_enabled,
                f"[checkpoint] step={step}/{training_steps} path={checkpoint_path}",
            )

    final_losses = estimate_loss(
        model,
        train_data,
        val_data,
        args.block_size,
        args.batch_size,
        args.eval_mode,
        args.eval_batches,
        generator,
        device,
        base_logits,
        args.residual_scale,
        args.recency_tau,
        args.recency_lambda,
    )

    retention_eval_seed = args.seed + 59_000
    retention_val_nll: float | None = None
    if retention_eval_data is not None:
        retention_generator = torch.Generator(device=device)
        retention_generator.manual_seed(retention_eval_seed)
        retention_val_nll = estimate_data_loss(
            model,
            retention_eval_data,
            args.block_size,
            args.batch_size,
            args.eval_mode,
            args.eval_batches,
            retention_generator,
            device,
            base_logits,
            args.residual_scale,
            args.recency_tau,
            args.recency_lambda,
        )

    model.eval()
    used_prompt = clean_prompt(args.prompt, codec, text)
    prompt_ids = torch.tensor([codec.encode(used_prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        generator=generator,
        base_logits=base_logits,
        residual_scale=args.residual_scale,
        recency_tau=args.recency_tau,
        recency_lambda=args.recency_lambda,
    )[0].tolist()
    retrieval_memory: dict[str, str] | None = None
    retrieval_memory_report: dict[str, int] | None = None
    if args.copy_probe_retrieval_source == "memory":
        memory_limit = len(text) if args.copy_probe_memory_scope == "all" else split_at
        retrieval_memory, retrieval_memory_report = build_copy_memory(
            text,
            args.copy_probe_marker,
            memory_limit,
            args.copy_probe_key_marker,
        )
    probe_report = copy_answer_probe(
        model=model,
        val_text=val_text,
        codec=codec,
        block_size=args.block_size,
        device=device,
        base_logits=base_logits,
        residual_scale=args.residual_scale,
        recency_tau=args.recency_tau,
        recency_lambda=args.recency_lambda,
        marker=args.copy_probe_marker,
        max_cases=args.copy_probe_max_cases,
        retrieval_template=args.copy_probe_retrieval_template,
        retrieval_source=args.copy_probe_retrieval_source,
        retrieval_corrupt=args.copy_probe_retrieval_corrupt,
        holdout_keys=args.copy_probe_holdout_keys,
        retrieval_memory=retrieval_memory,
        retrieval_memory_report=retrieval_memory_report,
        key_marker=args.copy_probe_key_marker,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "generator_state": generator.get_state(),
                "step": formation_steps,
                "formation_forward_passes": formation_forward_passes,
                "loss_curve": loss_curve,
                "chars": codec.chars,
                "args": vars(args),
                "base_logits": base_logits.cpu() if base_logits is not None else None,
            },
            args.out,
        )
    if args.model_only_out is not None:
        save_model_only_checkpoint(
            args.model_only_out,
            model,
            args,
            codec,
            base_logits,
            formation_steps,
            formation_forward_passes,
            loss_curve,
        )

    elapsed = time.perf_counter() - start
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    peak_cuda_memory_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    log_progress(
        progress_enabled,
        (
            "[done] "
            f"steps={formation_steps} val_nll={final_losses['val']:.6f} "
            f"bits={final_losses['val'] / math.log(2):.6f} "
            f"seconds={elapsed:.1f}"
        ),
    )
    return {
        "experiment": "tiny_transformer",
        "method": f"{args.residual_optim}_residual" if base_logits is not None else "gradient",
        "init": args.init,
        "init_report": init_report,
        "residual_base": args.residual_base,
        "residual_report": residual_report,
        "device": device,
        "device_name": device_name,
        "corpus": str(args.corpus),
        "corpus_chars": len(text),
        "train_chars": train_chars_report,
        "train_eval_chars": int(train_data.numel()),
        "train_shard_dir": str(args.train_shard_dir) if args.train_shard_dir else "",
        "train_shard_files": [str(path) for path in stream_sampler.paths] if stream_sampler is not None else [],
        "val_chars": int(val_data.numel()),
        "retention_eval_corpus": str(args.retention_eval_corpus) if args.retention_eval_corpus else "",
        "retention_eval_chars": retention_eval_chars,
        "retention_eval_seed": retention_eval_seed if retention_eval_data is not None else None,
        "retention_eval_mode": args.eval_mode if retention_eval_data is not None else "",
        "retention_eval_batches": args.eval_batches if retention_eval_data is not None else 0,
        "vocab_size": len(codec.chars),
        "vocab_chars_override": bool(vocab_chars),
        "vocab_chars": "".join(codec.chars),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, only_trainable=True),
        "frozen_prior_parameters": int(base_logits.numel()) if base_logits is not None else 0,
        "train_scope": args.train_scope,
        "train_scope_report": train_scope_report,
        "residual_optim": args.residual_optim,
        "optimizer": args.optimizer,
        "optimizer_report": optimizer_report,
        "precision": args.precision,
        "lr_schedule": args.lr_schedule,
        "lr_final_frac": args.lr_final_frac,
        "lr_total_steps": lr_total_steps,
        "lr_last_factor": round(last_lr_factor, 8),
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else "",
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_keep": args.checkpoint_keep,
        "last_checkpoint_path": last_checkpoint_path,
        "resume_steps_additional": args.resume_steps_additional,
        "planned_training_steps": planned_training_steps,
        "training_target_steps": training_steps,
        "model_only_out": str(args.model_only_out) if args.model_only_out else "",
        "model_only_dtype": args.model_only_dtype,
        **resume_report,
        "formation_parameters": formation_parameter_count,
        "formation_steps": formation_steps,
        "formation_forward_passes": formation_forward_passes,
        **residual_optim_report,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "pos_encoding": args.pos_encoding,
        "activation_checkpoint": args.activation_checkpoint,
        "adapter_rank": args.adapter_rank,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dropout": args.dropout,
        "residual_scale": args.residual_scale,
        "residual_l2": args.residual_l2,
        "recency_tau": args.recency_tau,
        "recency_lambda": args.recency_lambda,
        "prior_order": args.prior_order,
        "ngram_backoff": args.ngram_backoff,
        **{
            key: value
            for key, value in residual_report.items()
            if key.startswith("prior5_") or key in {"prior_cache_path", "prior_cache_status"}
        },
        "curriculum_filter": args.curriculum_filter,
        "curriculum_fraction": args.curriculum_fraction,
        **curriculum_report,
        "copy_train_marker": args.copy_train_marker,
        "copy_sampler": args.copy_sampler,
        "copy_sample_fraction": args.copy_sample_fraction,
        "copy_rehearsal_fraction": args.copy_rehearsal_fraction,
        "copy_loss_weight": args.copy_loss_weight,
        "copy_choice_weight": args.copy_choice_weight,
        "copy_verify_mode": args.copy_verify_mode,
        "copy_choice_candidates": copy_choice_chars,
        "copy_mine_every": args.copy_mine_every,
        "copy_mine_events": copy_mine_events,
        "copy_curriculum_switch_fraction": args.copy_curriculum_switch_fraction,
        "copy_curriculum_switch_step": copy_curriculum_switch_step,
        "copy_correction_template": args.copy_correction_template,
        "copy_train_retrieval_template": args.copy_train_retrieval_template,
        "copy_probe_memory_scope": args.copy_probe_memory_scope,
        "copy_probe_key_marker": args.copy_probe_key_marker,
        "copy_last_train_copy_accuracy": round(copy_last_train_accuracy, 6)
        if copy_last_train_accuracy is not None
        else None,
        "copy_last_train_copy_failures": copy_last_failures,
        "copy_correction_examples": copy_correction_examples,
        "copy_correction_chars": copy_correction_chars,
        "copy_retrieval_train_examples": copy_retrieval_train_examples,
        "copy_retrieval_train_chars": copy_retrieval_train_chars,
        "copy_train_positions": copy_train_positions,
        "copy_verified_positions": copy_verified_positions,
        "copy_sample_starts": copy_sample_starts,
        "steps": args.steps,
        "eval_mode": args.eval_mode,
        "eval_batches": args.eval_batches,
        "seconds": round(elapsed, 4),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
        "peak_cuda_memory_mib": round(peak_cuda_memory_bytes / (1024 * 1024), 4),
        "initial_train_nll": round(initial_losses["train"], 6),
        "initial_val_nll": round(initial_losses["val"], 6),
        "train_nll": round(final_losses["train"], 6),
        "val_nll": round(final_losses["val"], 6),
        "val_bits_per_char": round(final_losses["val"] / math.log(2), 6),
        "retention_val_nll": round(retention_val_nll, 6) if retention_val_nll is not None else None,
        "retention_val_bits_per_char": (
            round(retention_val_nll / math.log(2), 6) if retention_val_nll is not None else None
        ),
        "loss_curve": loss_curve,
        "used_prompt": used_prompt,
        "sample": codec.decode(generated_ids),
        **probe_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cassandra tiny causal transformer baseline")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--init", choices=["random", "count-bigram"], default="random")
    parser.add_argument("--residual-base", choices=["none", "count-bigram", "count-trigram", "count-ngram"], default="none")
    parser.add_argument("--count-alpha", type=float, default=0.1)
    parser.add_argument("--trigram-backoff", type=float, default=1.0)
    parser.add_argument("--prior-order", type=int, default=1)
    parser.add_argument("--ngram-backoff", type=float, default=1.0)
    parser.add_argument("--prior5-min-count", type=int, default=10)
    parser.add_argument("--recency-tau", type=float, default=DEFAULT_RECENCY_TAU)
    parser.add_argument("--recency-lambda", type=float, default=DEFAULT_RECENCY_LAMBDA)
    parser.add_argument("--curriculum-filter", choices=["off", "prior-loss", "dynamic-reducible"], default="off")
    parser.add_argument("--curriculum-fraction", type=float, default=0.5)
    parser.add_argument("--curriculum-rescore-every", type=int, default=25)
    parser.add_argument("--curriculum-pool-size", type=int, default=4096)
    parser.add_argument("--count-embedding-scale", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-l2", type=float, default=0.0)
    parser.add_argument(
        "--zero-residual-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--residual-optim", choices=["adamw", "es", "coord", "none"], default="adamw")
    parser.add_argument("--search-batches", type=int, default=1)
    parser.add_argument("--search-population", type=int, default=8)
    parser.add_argument("--search-sigma", type=float, default=0.02)
    parser.add_argument("--search-lr", type=float, default=0.05)
    parser.add_argument("--coord-step-size", type=float, default=0.02)
    parser.add_argument("--train-scope", choices=["all", "head", "adapters", "lora"], default="all")
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--train-shard-dir", type=Path, default=None)
    parser.add_argument("--stream-train-eval-chars", type=int, default=200_000)
    parser.add_argument("--retention-eval-corpus", type=Path, default=None)
    parser.add_argument("--prior-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-keep", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-steps-additional", action="store_true")
    parser.add_argument("--model-only-out", type=Path, default=None)
    parser.add_argument("--model-only-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--vocab-chars", type=str, default="")
    parser.add_argument("--vocab-chars-file", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--pos-encoding", choices=["learned", "rope"], default="learned")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-final-frac", type=float, default=0.1)
    parser.add_argument("--lr-total-steps", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--no-muon-nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--eval-mode", choices=["full", "sampled"], default="full")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="cassandra ")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--copy-train-marker", type=str, default="")
    parser.add_argument(
        "--copy-sampler",
        choices=[
            "random",
            "answer",
            "mixed",
            "failed",
            "failed_mixed",
            "correction",
            "correction_mixed",
            "retrieval",
            "retrieval_mixed",
            "correction_retrieval_mixed",
            "correction_then_retrieval_mixed",
            "correction_then_retrieval_rehearsal_mixed",
        ],
        default="random",
    )
    parser.add_argument("--copy-sample-fraction", type=float, default=0.5)
    parser.add_argument("--copy-rehearsal-fraction", type=float, default=0.05)
    parser.add_argument("--copy-curriculum-switch-fraction", type=float, default=0.5)
    parser.add_argument("--copy-loss-weight", type=float, default=1.0)
    parser.add_argument("--copy-choice-weight", type=float, default=0.0)
    parser.add_argument("--copy-verify-mode", choices=["identity", "key-answer"], default="identity")
    parser.add_argument("--copy-mine-every", type=int, default=50)
    parser.add_argument(
        "--copy-correction-template",
        choices=["compact", "focus", "prefix", "full"],
        default="compact",
    )
    parser.add_argument(
        "--copy-train-retrieval-template",
        choices=["compact", "focus", "prefix"],
        default="compact",
    )
    parser.add_argument("--copy-probe-marker", type=str, default="")
    parser.add_argument("--copy-probe-key-marker", type=str, default="key=")
    parser.add_argument("--copy-probe-max-cases", type=int, default=200)
    parser.add_argument(
        "--copy-probe-retrieval-template",
        choices=["none", "compact", "focus", "prefix"],
        default="none",
    )
    parser.add_argument(
        "--copy-probe-retrieval-source",
        choices=["target", "memory"],
        default="target",
    )
    parser.add_argument(
        "--copy-probe-retrieval-corrupt",
        choices=["none", "wrong-answer"],
        default="none",
    )
    parser.add_argument(
        "--copy-probe-memory-scope",
        choices=["train", "all"],
        default="train",
    )
    parser.add_argument("--copy-probe-holdout-keys", nargs="*", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
