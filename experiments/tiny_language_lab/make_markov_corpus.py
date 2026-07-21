from __future__ import annotations

import argparse
import json
import math
import random
from itertools import product
from pathlib import Path


DEFAULT_OUT = Path(__file__).with_name("corpus") / "markov_order2_seed.txt"
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("transition weights must have positive mass")
    return [weight / total for weight in weights]


def build_transition_table(
    order: int,
    vocab_size: int,
    concentration: float,
    rng: random.Random,
) -> dict[tuple[int, ...], list[float]]:
    if order < 1:
        raise ValueError("--order must be at least 1")
    if concentration <= 0:
        raise ValueError("--concentration must be greater than zero")

    table: dict[tuple[int, ...], list[float]] = {}
    for context in product(range(vocab_size), repeat=order):
        weights = [rng.gammavariate(concentration, 1.0) for _ in range(vocab_size)]
        table[context] = normalize(weights)
    return table


def sample_from_distribution(probs: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return index
    return len(probs) - 1


def sample_sequence(
    table: dict[tuple[int, ...], list[float]],
    order: int,
    vocab_size: int,
    length: int,
    rng: random.Random,
) -> tuple[list[int], float]:
    if length <= order:
        raise ValueError("corpus length must exceed order")

    ids = [rng.randrange(vocab_size) for _ in range(order)]
    total_nll = 0.0
    predicted = 0
    while len(ids) < length:
        context = tuple(ids[-order:])
        probs = table[context]
        next_id = sample_from_distribution(probs, rng)
        total_nll += -math.log(max(probs[next_id], 1e-12))
        predicted += 1
        ids.append(next_id)
    return ids, total_nll / max(predicted, 1)


def entropy(probs: list[float]) -> float:
    return -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)


def table_to_json(
    table: dict[tuple[int, ...], list[float]],
    chars: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for context, probs in table.items():
        context_text = "".join(chars[index] for index in context)
        out[context_text] = {chars[index]: round(prob, 10) for index, prob in enumerate(probs)}
    return out


def build_corpus(
    order: int,
    vocab_size: int,
    lines: int,
    line_length: int,
    seed: int,
    concentration: float,
) -> tuple[str, dict[str, object]]:
    if vocab_size < 2:
        raise ValueError("--vocab must be at least 2")
    if vocab_size > len(ALPHABET):
        raise ValueError(f"--vocab must be at most {len(ALPHABET)}")
    if lines < 1:
        raise ValueError("--lines must be at least 1")
    if line_length < 8:
        raise ValueError("--line-length must be at least 8")

    rng = random.Random(seed)
    chars = ALPHABET[:vocab_size]
    table = build_transition_table(order, vocab_size, concentration, rng)
    length = lines * line_length
    ids, generated_nll = sample_sequence(table, order, vocab_size, length, rng)
    text = "".join(chars[index] for index in ids)
    mean_context_entropy = sum(entropy(probs) for probs in table.values()) / len(table)
    metadata = {
        "generator": "make_markov_corpus.py",
        "order": order,
        "vocab": vocab_size,
        "chars": chars,
        "lines": lines,
        "line_length": line_length,
        "length": length,
        "seed": seed,
        "concentration": concentration,
        "mean_context_entropy_nats": mean_context_entropy,
        "mean_context_entropy_bits": mean_context_entropy / math.log(2),
        "sampled_source_nll_nats": generated_nll,
        "sampled_source_bits_per_char": generated_nll / math.log(2),
        "transition_table": table_to_json(table, chars),
    }
    return text, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pure order-k Markov character corpus")
    parser.add_argument("--order", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--lines", type=int, default=512)
    parser.add_argument("--line-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--concentration", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text, metadata = build_corpus(
        args.order,
        args.vocab,
        args.lines,
        args.line_length,
        args.seed,
        args.concentration,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    metadata_out = args.metadata_out or args.out.with_suffix(".meta.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")
    print(f"wrote {metadata_out}")


if __name__ == "__main__":
    main()
