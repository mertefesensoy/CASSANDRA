"""Shared loading, evaluation, and sampling helpers for Stage 55 close-out tools.

Used by phase4_validate.py, eval_text8.py, make_phase4_figures.py, and
playground.py. Sits beside cassandra_tiny_transformer.py so the sibling
import resolves (same convention as cassandra_compare.py).
"""

from __future__ import annotations

import gc
import unicodedata
from pathlib import Path

import numpy as np
import torch

from cassandra_tiny_transformer import Codec, TinyTransformer, split_at_index

LOG2E = 1.4426950408889634
LAB_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = LAB_DIR / "corpus" / "tinystories_char_seed.txt"

# The Phase 4 keep-set (RESULTS.md, Stage 55 closeout) plus the Stage 51
# 25M reference checkpoint. Keys are display names used across the tools.
FINAL_CHECKPOINTS: dict[str, Path] = {
    "flagship_200m_50k_seed7": LAB_DIR
    / "artifacts"
    / "phase4"
    / "checkpoints"
    / "stage55_flagship_200m_b50000_seed7"
    / "stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt",
    "replica_200m_20k_seed11": Path(
        r"C:\cassandra_runs\stage55_flagship_checkpoints"
        r"\stage55_flagship_200m_b20000_seed11_random_full_seed11_step020000.pt"
    ),
    "replica_200m_20k_seed19": Path(
        r"C:\cassandra_runs\stage55_flagship_checkpoints"
        r"\stage55_flagship_200m_b20000_seed19_random_full_seed19_step020000.pt"
    ),
    "stage51_25m_5k_seed7": LAB_DIR
    / "runs"
    / "stage51_checkpoints"
    / "stage51_coherence_25m_b5000_random_full_seed7.pt",
}


def build_codec(chars: list[str]) -> Codec:
    return Codec(
        chars=list(chars),
        stoi={ch: idx for idx, ch in enumerate(chars)},
        itos={idx: ch for idx, ch in enumerate(chars)},
    )


def load_model(
    checkpoint_path: Path | str, device: str = "cuda"
) -> tuple[TinyTransformer, Codec, dict, dict]:
    """Load a TinyTransformer checkpoint for inference.

    Returns (model, codec, checkpoint_args, meta). The optimizer state is
    dropped immediately to keep host memory at model scale. Only
    random_full-style checkpoints (no frozen prior) are supported here;
    a checkpoint carrying base_logits raises.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"Checkpoint has no usable args dict: {checkpoint_path}")
    chars = ckpt.get("chars")
    if not isinstance(chars, list) or not chars:
        raise ValueError(f"Checkpoint has no usable chars vocab: {checkpoint_path}")
    if ckpt.get("base_logits") is not None:
        raise ValueError("This loader only supports prior-free (random_full) checkpoints")

    model = TinyTransformer(
        vocab_size=len(chars),
        block_size=int(args["block_size"]),
        n_layer=int(args["n_layer"]),
        n_head=int(args["n_head"]),
        n_embd=int(args["n_embd"]),
        dropout=float(args.get("dropout") or 0.0),
        adapter_rank=int(args.get("adapter_rank") or 0),
        lora_rank=int(args.get("lora_rank") or 0),
        lora_alpha=float(args.get("lora_alpha") or 1.0),
        lora_dropout=float(args.get("lora_dropout") or 0.0),
        pos_encoding=str(args.get("pos_encoding") or "learned"),
        activation_checkpoint=False,
    )
    state = ckpt.get("model_state") if "model_state" in ckpt else ckpt.get("model")
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(state)
    model.to(device).eval()

    meta = {
        "checkpoint": str(checkpoint_path),
        "step": ckpt.get("step"),
        "formation_forward_passes": ckpt.get("formation_forward_passes"),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    codec = build_codec(chars)
    del ckpt, state
    gc.collect()
    return model, codec, args, meta


def encode_fast(text: str, codec: Codec) -> torch.Tensor:
    """Vectorized ASCII text -> id tensor. Raises if any char is outside the vocab."""
    lut = np.full(256, -1, dtype=np.int64)
    for ch, idx in codec.stoi.items():
        code = ord(ch)
        if code < 256:
            lut[code] = idx
    arr = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    ids = lut[arr]
    bad = int((ids < 0).sum())
    if bad:
        raise ValueError(f"{bad} characters fall outside the model alphabet")
    return torch.from_numpy(ids)


def normalize_to_alphabet(text: str, codec: Codec) -> str:
    """Mirror the corpus normalization: NFKD, lowercase, unsupported -> space,
    space runs collapsed. Used for free-form user prompts."""
    text = unicodedata.normalize("NFKD", text).lower()
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch not in codec.stoi:
            ch = " "
        if ch == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        out.append(ch)
    normalized = "".join(out)
    return normalized if normalized else " "


def load_val_ids(
    codec: Codec,
    corpus_path: Path = DEFAULT_CORPUS,
    val_fraction: float = 0.15,
    block_size: int = 256,
) -> torch.Tensor:
    """Reproduce the trainer's deterministic suffix validation split."""
    text = corpus_path.read_text(encoding="utf-8")
    split_at = split_at_index(len(text), val_fraction, block_size)
    return encode_fast(text[split_at:], codec)


@torch.no_grad()
def chunked_nll(
    model: TinyTransformer,
    ids: torch.Tensor,
    device: str,
    batch_windows: int = 32,
    max_eval_chars: int | None = None,
) -> dict[str, float]:
    """Deterministic chunked evaluation: non-overlapping block-size windows,
    every covered char predicted exactly once, context resets per window.

    With max_eval_chars set, windows are subsampled uniformly across the whole
    split so coverage stays representative. This is the conservative chunked
    convention (no sliding context), so results are directly comparable across
    models and slightly pessimistic versus sliding-window evaluation.
    """
    block_size = model.block_size
    n = int(ids.numel())
    starts = torch.arange(0, n - block_size - 1, block_size)
    if max_eval_chars is not None:
        wanted = max(1, int(max_eval_chars) // block_size)
        if wanted < starts.numel():
            keep = torch.linspace(0, starts.numel() - 1, wanted).round().long()
            starts = starts[keep]

    total_loss = 0.0
    total_tokens = 0
    for offset in range(0, int(starts.numel()), batch_windows):
        batch_starts = starts[offset : offset + batch_windows]
        xs = [ids[int(s) : int(s) + block_size] for s in batch_starts]
        ys = [ids[int(s) + 1 : int(s) + block_size + 1] for s in batch_starts]
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
        _, loss = model(xb, yb, base_logits=None)
        if loss is None:
            raise RuntimeError("Expected loss during chunked evaluation")
        tokens = int(xb.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens

    nll = total_loss / total_tokens
    return {
        "nll": nll,
        "bits_per_char": nll * LOG2E,
        "chars_evaluated": total_tokens,
        "windows": int(starts.numel()),
        "block_size": block_size,
    }


@torch.no_grad()
def sample_text(
    model: TinyTransformer,
    codec: Codec,
    prompt: str,
    max_new_chars: int = 400,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int = 7,
    device: str = "cuda",
) -> str:
    """Temperature with optional top-k, nucleus (top-p), and repetition-penalty
    sampling, seeded for reproducibility.

    Defaults reproduce the prior temperature-plus-top-k behavior: `top_p <= 0`
    or `>= 1` disables nucleus truncation, and `repetition_penalty == 1`
    disables the penalty. Filter order per step: repetition penalty on raw
    logits, then temperature, then top-k, then top-p. Repetition penalty is a
    CTRL-style rescale (positive logits divided, negative multiplied) over the
    tokens already in the window; at the character level it is a weak lever
    because almost every character recurs, so it is provided mainly for
    completeness. Nucleus and temperature are the effective controls for
    topical drift.
    """
    prompt_n = normalize_to_alphabet(prompt, codec)
    idx = torch.tensor([codec.encode(prompt_n)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    temperature = max(float(temperature), 1e-4)
    penalty = max(float(repetition_penalty), 1e-4)
    for _ in range(int(max_new_chars)):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond, base_logits=None)
        logits = logits[:, -1, :]
        if penalty != 1.0:
            seen = torch.unique(idx_cond)
            gathered = logits[:, seen]
            logits[:, seen] = torch.where(
                gathered > 0, gathered / penalty, gathered * penalty
            )
        logits = logits / temperature
        if top_k and top_k > 0:
            top_vals, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < top_vals[:, [-1]]] = -float("inf")
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_sorted = cum > top_p
            # shift so the token that crosses the threshold is kept
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False
            remove = torch.zeros_like(remove_sorted).scatter(
                -1, sorted_idx, remove_sorted
            )
            logits[remove] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat((idx, next_id), dim=1)
    return codec.decode(idx[0].tolist())


def free_model(model: TinyTransformer) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
