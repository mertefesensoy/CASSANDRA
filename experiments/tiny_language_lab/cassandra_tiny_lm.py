from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_CORPUS = Path(__file__).with_name("corpus") / "tiny_seed.txt"


@dataclass(frozen=True)
class Codec:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


def build_codec(text: str) -> Codec:
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


def split_pairs(
    ids: list[int],
    val_fraction: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("--val-fraction must be between 0.05 and 0.5")

    split_at = max(2, int(len(ids) * (1.0 - val_fraction)))
    split_at = min(split_at, len(ids) - 2)

    train_ids = ids[:split_at]
    val_ids = ids[split_at - 1 :]

    x_train = torch.tensor(train_ids[:-1], dtype=torch.long, device=device)
    y_train = torch.tensor(train_ids[1:], dtype=torch.long, device=device)
    x_val = torch.tensor(val_ids[:-1], dtype=torch.long, device=device)
    y_val = torch.tensor(val_ids[1:], dtype=torch.long, device=device)
    return x_train, y_train, x_val, y_val


def nll(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(logits[x], y).item())


def build_count_logits(x: torch.Tensor, y: torch.Tensor, vocab_size: int, alpha: float) -> torch.Tensor:
    counts = torch.full((vocab_size, vocab_size), alpha, dtype=torch.float32, device=x.device)
    for prev_id, next_id in zip(x.tolist(), y.tolist()):
        counts[prev_id, next_id] += 1.0
    return counts.log()


def build_zero_logits(vocab_size: int, device: str) -> torch.Tensor:
    return torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=device)


def coordinate_search(
    logits: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    steps: int,
    step_size: float,
    seed: int,
    log_every: int,
) -> tuple[torch.Tensor, int]:
    rng = random.Random(seed)
    current_loss = nll(logits, x_train, y_train)
    accepted = 0
    vocab_size = logits.shape[0]

    for step in range(1, steps + 1):
        row = rng.randrange(vocab_size)
        col = rng.randrange(vocab_size)
        original = float(logits[row, col].item())

        best_value = original
        best_loss = current_loss
        for delta in (-step_size, step_size):
            logits[row, col] = original + delta
            candidate_loss = nll(logits, x_train, y_train)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_value = original + delta

        logits[row, col] = best_value
        if best_loss < current_loss:
            accepted += 1
            current_loss = best_loss

        if log_every > 0 and step % log_every == 0:
            print(f"step={step} train_nll={current_loss:.4f} accepted={accepted}")

    return logits, accepted


def gradient_train(
    logits: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    steps: int,
    lr: float,
    log_every: int,
) -> tuple[torch.Tensor, int]:
    logits.requires_grad_(True)
    optimizer = torch.optim.AdamW([logits], lr=lr)

    for step in range(1, steps + 1):
        loss = F.cross_entropy(logits[x_train], y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if log_every > 0 and step % log_every == 0:
            print(f"step={step} train_nll={float(loss.item()):.4f}")

    logits.requires_grad_(False)
    return logits.detach(), steps


def clean_prompt(prompt: str, codec: Codec, fallback: str) -> str:
    cleaned = "".join(ch for ch in prompt if ch in codec.stoi)
    if cleaned:
        return cleaned
    return fallback[0]


def sample(
    logits: torch.Tensor,
    codec: Codec,
    prompt: str,
    corpus_text: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[str, str]:
    if temperature <= 0:
        raise ValueError("--temperature must be greater than zero")

    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)

    used_prompt = clean_prompt(prompt, codec, corpus_text)
    ids = codec.encode(used_prompt)
    for _ in range(max_new_tokens):
        prev_id = ids[-1]
        probs = torch.softmax(logits[prev_id] / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        ids.append(next_id)

    return used_prompt, codec.decode(ids)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def run(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    device = choose_device(args.device)
    text = read_text(args.corpus)
    codec = build_codec(text)
    ids = codec.encode(text)
    x_train, y_train, x_val, y_val = split_pairs(ids, args.val_fraction, device)
    vocab_size = len(codec.chars)

    if args.init == "count":
        logits = build_count_logits(x_train, y_train, vocab_size, args.alpha)
    else:
        logits = build_zero_logits(vocab_size, device)

    trainable_parameters = 0
    changed_parameters = 0
    accepted_coordinate_updates = 0
    if args.method == "count":
        logits = build_count_logits(x_train, y_train, vocab_size, args.alpha)
        changed_parameters = vocab_size * vocab_size
    elif args.method == "coordinate":
        trainable_parameters = vocab_size * vocab_size
        logits, accepted_coordinate_updates = coordinate_search(
            logits=logits,
            x_train=x_train,
            y_train=y_train,
            steps=args.steps,
            step_size=args.step_size,
            seed=args.seed,
            log_every=args.log_every,
        )
        changed_parameters = accepted_coordinate_updates
    elif args.method == "gradient":
        trainable_parameters = vocab_size * vocab_size
        logits, _ = gradient_train(
            logits=logits,
            x_train=x_train,
            y_train=y_train,
            steps=args.steps,
            lr=args.lr,
            log_every=args.log_every,
        )
        changed_parameters = trainable_parameters
    else:
        raise ValueError(f"Unknown method: {args.method}")

    train_nll = nll(logits, x_train, y_train)
    val_nll = nll(logits, x_val, y_val)
    used_prompt, generated = sample(
        logits=logits,
        codec=codec,
        prompt=args.prompt,
        corpus_text=text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": logits.cpu(),
                "chars": codec.chars,
                "method": args.method,
                "init": args.init,
            },
            args.out,
        )

    elapsed = time.perf_counter() - start
    return {
        "method": args.method,
        "init": args.init,
        "device": device,
        "corpus": str(args.corpus),
        "corpus_chars": len(text),
        "vocab_size": vocab_size,
        "parameters": vocab_size * vocab_size,
        "trainable_parameters": trainable_parameters,
        "changed_parameters": changed_parameters,
        "accepted_coordinate_updates": accepted_coordinate_updates,
        "steps": args.steps if args.method != "count" else 0,
        "seconds": round(elapsed, 4),
        "train_nll": round(train_nll, 6),
        "val_nll": round(val_nll, 6),
        "val_bits_per_char": round(val_nll / math.log(2), 6),
        "used_prompt": used_prompt,
        "sample": generated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cassandra tiny language-model formation lab")
    parser.add_argument("--method", choices=["count", "coordinate", "gradient"], default="count")
    parser.add_argument("--init", choices=["zeros", "count"], default="zeros")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--prompt", type=str, default="cassandra ")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
