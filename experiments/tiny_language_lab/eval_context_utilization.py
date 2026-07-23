"""Context-utilization probe for Cassandra checkpoints (H027, Stage 62).

Measures whether a model predicts a target segment better under its TRUE
preceding context than under a RANDOM unrelated context, resolved by distance
into the target. This isolates long-range context use, which plain NLL hides
(a locally fluent, globally drifting model has good NLL and near-zero deep
context use).

Definitions, per held-out text8 passage of length L_c + L_t:
  TRUE   : score NLL of the target chars given the passage's own first L_c chars.
  RANDOM : score the SAME target chars given a different passage's first L_c chars.
  U = NLL_random - NLL_true, per target offset. Positive => context helped.

The deciding quantity is U in the DEEP target bucket (offsets 33..L_t), because a
model that only uses local boundary continuity shows U concentrated in the first
few offsets and ~0 deep. Bucket edges, a synthetic self-copy sensitivity anchor,
an L_c dose curve, and a double-random control are all produced here; see
docs/hypotheses/027-context-utilization-and-coherence.md for the decision rule.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\eval_context_utilization.py `
    --checkpoint C:\\cassandra_runs\\...\\flagship.pt `
    --split test --n 4096 --context-len 192 --target-len 64 `
    --bucket-edges 4 8 16 32 --seed 20260723 --device cuda `
    --synthetic-anchor --context-len-sweep 64 128 192 --control `
    --out runs\\stage62_context_util_flagship.json `
    --summary runs\\stage62_context_util_flagship.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eval_text8 import ensure_text8
from flagship_eval_lib import LOG2E, encode_fast, free_model, load_model

LAB_DIR = Path(__file__).resolve().parent
RUN_DIR = LAB_DIR / "runs"

# text8 canonical split: train first 90M, valid next 5M, test last 5M.
SPLIT_RANGES = {"valid": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_text(split: str) -> str:
    lo, hi = SPLIT_RANGES[split]
    return ensure_text8()[lo:hi]


def draw_passages(text: str, n: int, passage_len: int, seed: int) -> list[str]:
    """N disjoint fixed passages of passage_len chars, deterministic by seed.

    Non-overlapping windows spread uniformly across the split, so coverage is
    representative and no character is reused between passages.
    """
    max_start = len(text) - passage_len
    stride = max_start // n
    if stride < passage_len:
        raise ValueError(
            f"split too small for {n} disjoint passages of {passage_len} chars"
        )
    rng = np.random.default_rng(seed)
    jitter = rng.integers(0, stride - passage_len + 1, size=n)
    starts = np.arange(n) * stride + jitter
    return [text[int(s) : int(s) + passage_len] for s in starts]


def sattolo_derangement(n: int, seed: int) -> np.ndarray:
    """A single cycle permutation (Sattolo): every index maps to a different one,
    so no passage is ever paired with its own context."""
    perm = np.arange(n)
    rng = np.random.default_rng(seed + 1)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))  # strictly below i
        perm[i], perm[j] = perm[j], perm[i]
    return perm


@torch.no_grad()
def target_ce(
    model: Any,
    context_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: str,
    batch_windows: int,
) -> np.ndarray:
    """Per-passage, per-offset cross-entropy (nats) on the target chars.

    context_ids : [N, L_c] the (true or random) preceding context per passage.
    target_ids  : [N, L_t] the SAME target chars whichever context is used.
    Returns [N, L_t]: CE of predicting each target char from the model's forward
    over [context ++ target]; the char at target offset o is predicted from
    logits at position L_c - 1 + o (its full window includes the whole context,
    since L_c + L_t <= block_size).
    """
    n, l_c = context_ids.shape
    l_t = target_ids.shape[1]
    out = np.empty((n, l_t), dtype=np.float64)
    for offset in range(0, n, batch_windows):
        ctx = context_ids[offset : offset + batch_windows]
        tgt = target_ids[offset : offset + batch_windows]
        xb = torch.cat([ctx, tgt], dim=1).to(device)  # [b, L_c+L_t]
        logits, _ = model(xb, base_logits=None)  # [b, L_c+L_t, V]
        # positions L_c-1 .. L_c+L_t-2 predict target offsets 0 .. L_t-1
        pred = logits[:, l_c - 1 : l_c - 1 + l_t, :]
        logp = F.log_softmax(pred.float(), dim=-1)
        gold = tgt.to(device).unsqueeze(-1)
        ce = -logp.gather(-1, gold).squeeze(-1)  # [b, L_t]
        out[offset : offset + xb.shape[0]] = ce.double().cpu().numpy()
    return out


def bucket_labels(l_t: int, edges: list[int]) -> list[tuple[str, int, int]]:
    """1-indexed target-char buckets from cut points, e.g. edges [4,8,16,32] ->
    (1-4)(5-8)(9-16)(17-32)(33-L_t). Returns (label, lo0, hi0) with 0-indexed
    half-open [lo0, hi0)."""
    cuts = [0, *edges, l_t]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        out.append((f"{a + 1}-{b}", a, min(b, l_t)))
    return out


def bootstrap_ci(
    per_passage: np.ndarray, seed: int, resamples: int = 2000
) -> tuple[float, float, float]:
    """Mean plus 95 percent bootstrap CI of a per-passage scalar (nats)."""
    rng = np.random.default_rng(seed + 7)
    n = per_passage.shape[0]
    means = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        idx = rng.integers(0, n, size=n)
        means[r] = per_passage[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(per_passage.mean()), float(lo), float(hi)


def run_setting(
    model: Any,
    codec: Any,
    text: str,
    n: int,
    l_c: int,
    l_t: int,
    edges: list[int],
    seed: int,
    device: str,
    batch_windows: int,
    mode: str,
) -> dict[str, Any]:
    """One (L_c, L_t) measurement. mode: 'text8' | 'synthetic' | 'control'."""
    passages = draw_passages(text, n, l_c + l_t, seed)
    ctx_true = torch.stack([encode_fast(p[:l_c], codec) for p in passages])
    if mode == "synthetic":
        # deep-copy anchor: the target IS the first L_t chars of the true context,
        # so predicting it requires copying from ~L_c positions back.
        tgt = torch.stack([encode_fast(p[:l_t], codec) for p in passages])
    else:
        tgt = torch.stack([encode_fast(p[l_c : l_c + l_t], codec) for p in passages])

    ce_true = target_ce(model, ctx_true, tgt, device, batch_windows)

    perm = sattolo_derangement(n, seed)
    ctx_rand = ctx_true[perm]
    ce_rand = target_ce(model, ctx_rand, tgt, device, batch_windows)

    if mode == "control":
        # second independent wrong context; U should collapse to ~0.
        perm2 = sattolo_derangement(n, seed + 13)
        ce_true = target_ce(model, ctx_true[perm2], tgt, device, batch_windows)

    u = ce_rand - ce_true  # [N, L_t], nats
    buckets = []
    for label, lo0, hi0 in bucket_labels(l_t, edges):
        per_passage = u[:, lo0:hi0].mean(axis=1)
        mean_nats, ci_lo, ci_hi = bootstrap_ci(per_passage, seed)
        buckets.append(
            {
                "label": label,
                "offset_lo": lo0 + 1,
                "offset_hi": hi0,
                "u_bits_per_char": mean_nats * LOG2E,
                "ci_lo_bits": ci_lo * LOG2E,
                "ci_hi_bits": ci_hi * LOG2E,
            }
        )
    deep = next(b for b in buckets if b["offset_lo"] >= 33) if l_t >= 33 else buckets[-1]
    return {
        "mode": mode,
        "n": n,
        "context_len": l_c,
        "target_len": l_t,
        "nll_true_bits": float(ce_true.mean() * LOG2E),
        "nll_random_bits": float(ce_rand.mean() * LOG2E),
        "buckets": buckets,
        "u_deep_bits": deep["u_bits_per_char"],
        "u_deep_ci_lo_bits": deep["ci_lo_bits"],
        "u_deep_ci_hi_bits": deep["ci_hi_bits"],
        "deep_bucket_label": deep["label"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H027 context-utilization probe")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["test", "valid"], default="test")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--context-len", type=int, default=192)
    p.add_argument("--target-len", type=int, default=64)
    p.add_argument("--bucket-edges", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-windows", type=int, default=64)
    p.add_argument("--synthetic-anchor", action="store_true")
    p.add_argument("--control", action="store_true")
    p.add_argument("--context-len-sweep", type=int, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    return p.parse_args()


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    prim = payload["primary"]
    lines = [
        "# Stage 62 Context-Utilization Probe (H027)",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        f"- parameters: {payload['parameters']:,}  block size: {payload['block_size']}",
        f"- split: text8 {payload['split']}  N: {prim['n']}  seed: {payload['seed']}",
        f"- setting: L_c={prim['context_len']} L_t={prim['target_len']}",
        "",
        "U = NLL_random - NLL_true (bits/char). Positive means the true preceding "
        "context helped predict the target. The deep bucket is the coherence signal; "
        "near-boundary buckets are the positive control.",
        "",
        "## Primary (text8) U by target-distance bucket",
        "",
        "| Target chars | U bits/char | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for b in prim["buckets"]:
        lines.append(
            f"| {b['label']} | {b['u_bits_per_char']:+.6f} | "
            f"[{b['ci_lo_bits']:+.6f}, {b['ci_hi_bits']:+.6f}] |"
        )
    lines += [
        "",
        f"- NLL_true: {prim['nll_true_bits']:.6f} bits/char; "
        f"NLL_random: {prim['nll_random_bits']:.6f} bits/char.",
        f"- Deep bucket ({prim['deep_bucket_label']}): U = "
        f"{prim['u_deep_bits']:+.6f} bits/char, CI "
        f"[{prim['u_deep_ci_lo_bits']:+.6f}, {prim['u_deep_ci_hi_bits']:+.6f}].",
        "",
    ]
    if payload.get("synthetic"):
        s = payload["synthetic"]
        lines += [
            "## Synthetic deep-copy sensitivity anchor",
            "",
            f"Target = copy of the context's first {s['target_len']} chars, so deep "
            "prediction requires copying from ~L_c back. A working long-range "
            "attention shows a large positive deep U here; a null indicts the "
            "model's deep attention, not the text8 result.",
            "",
            f"- Deep-bucket U: {s['u_deep_bits']:+.6f} bits/char, CI "
            f"[{s['u_deep_ci_lo_bits']:+.6f}, {s['u_deep_ci_hi_bits']:+.6f}].",
            "",
        ]
    if payload.get("control"):
        c = payload["control"]
        lines += [
            "## Double-random control (should be ~0)",
            "",
            f"- Deep-bucket U: {c['u_deep_bits']:+.6f} bits/char, CI "
            f"[{c['u_deep_ci_lo_bits']:+.6f}, {c['u_deep_ci_hi_bits']:+.6f}].",
            "",
        ]
    if payload.get("dose_curve"):
        lines += [
            "## L_c dose curve (deep-bucket U rising with more true context = "
            "internal evidence of long-range use)",
            "",
            "| L_c | deep U bits/char | 95% CI |",
            "| ---: | ---: | --- |",
        ]
        for d in payload["dose_curve"]:
            lines.append(
                f"| {d['context_len']} | {d['u_deep_bits']:+.6f} | "
                f"[{d['u_deep_ci_lo_bits']:+.6f}, {d['u_deep_ci_hi_bits']:+.6f}] |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.context_len + args.target_len > 256 and args.device == "cuda":
        # allowed for longer-context models; just a note, not a hard cap here
        pass
    text = split_text(args.split)
    model, codec, ckpt_args, meta = load_model(args.checkpoint, device=args.device)
    block_size = int(ckpt_args["block_size"])
    if args.context_len + args.target_len > block_size:
        raise ValueError(
            f"L_c+L_t={args.context_len + args.target_len} exceeds model block "
            f"size {block_size}; the whole passage must fit one window"
        )
    started = time.time()
    try:
        primary = run_setting(
            model, codec, text, args.n, args.context_len, args.target_len,
            args.bucket_edges, args.seed, args.device, args.batch_windows, "text8",
        )
        payload: dict[str, Any] = {
            "stage": 62,
            "hypothesis": "H027",
            "created_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "parameters": meta["parameters"],
            "block_size": block_size,
            "split": args.split,
            "seed": args.seed,
            "primary": primary,
        }
        if args.synthetic_anchor:
            payload["synthetic"] = run_setting(
                model, codec, text, args.n, args.context_len, args.target_len,
                args.bucket_edges, args.seed, args.device, args.batch_windows,
                "synthetic",
            )
        if args.control:
            payload["control"] = run_setting(
                model, codec, text, args.n, args.context_len, args.target_len,
                args.bucket_edges, args.seed, args.device, args.batch_windows,
                "control",
            )
        if args.context_len_sweep:
            dose = []
            for l_c in args.context_len_sweep:
                if l_c + args.target_len > block_size:
                    continue
                dose.append(
                    run_setting(
                        model, codec, text, args.n, l_c, args.target_len,
                        args.bucket_edges, args.seed, args.device,
                        args.batch_windows, "text8",
                    )
                )
            payload["dose_curve"] = dose
    finally:
        free_model(model)
    payload["seconds"] = round(time.time() - started, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    d = primary
    print(
        f"context_util deep U={d['u_deep_bits']:+.6f} bits/char "
        f"CI[{d['u_deep_ci_lo_bits']:+.6f},{d['u_deep_ci_hi_bits']:+.6f}] "
        f"NLL_true={d['nll_true_bits']:.4f} NLL_random={d['nll_random_bits']:.4f}"
    )


if __name__ == "__main__":
    main()
