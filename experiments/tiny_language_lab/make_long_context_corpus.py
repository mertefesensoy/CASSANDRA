from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_OUT = Path(__file__).with_name("corpus") / "long_context_seed.txt"

KEYS = "abcdefgh"
PAYLOAD_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
FILLERS = [
    "amber signal",
    "brisk method",
    "careful trace",
    "direct measure",
    "even budget",
    "frozen prior",
    "gentle update",
    "hidden copy",
]


def parse_key_set(raw: str | list[str]) -> list[str]:
    pieces = raw if isinstance(raw, list) else [raw]
    keys = [ch for piece in pieces for ch in piece if ch.strip()]
    unknown = sorted(set(keys) - set(KEYS))
    if unknown:
        raise ValueError(f"Unknown holdout keys: {''.join(unknown)}")
    return sorted(set(keys), key=KEYS.index)


def make_payload_alphabet(size: int) -> list[str]:
    if size < 2:
        raise ValueError("--payload-alphabet-size must be at least 2")
    if size > len(PAYLOAD_ALPHABET):
        raise ValueError(f"--payload-alphabet-size must be at most {len(PAYLOAD_ALPHABET)}")
    return list(PAYLOAD_ALPHABET[:size])


def make_line(rng: random.Random, index: int, key: str) -> str:
    distractor = KEYS[(index * 3 + 5) % len(KEYS)]
    filler_a = FILLERS[(index + rng.randrange(len(FILLERS))) % len(FILLERS)]
    filler_b = FILLERS[(index * 5 + rng.randrange(len(FILLERS))) % len(FILLERS)]
    filler_c = FILLERS[(index * 7 + rng.randrange(len(FILLERS))) % len(FILLERS)]
    return (
        f"case {index:04d} key={key} "
        f"noise={distractor} {filler_a}; {filler_b}; {filler_c}; "
        f"answer={key}\n"
    )


def build_corpus(
    lines: int,
    seed: int,
    holdout_keys: list[str],
    random_payload: bool = False,
    payload_alphabet_size: int = 16,
) -> str:
    rng = random.Random(seed)
    if random_payload and holdout_keys:
        raise ValueError("--random-payload cannot be combined with --holdout-keys")
    if random_payload:
        payload_keys = make_payload_alphabet(payload_alphabet_size)
        payload_text = "".join(payload_keys)
        header = (
            "cassandra long context corpus.\n"
            "each line asks the model to copy the key character after answer=.\n"
            "random_payload: true\n"
            f"payload_alphabet: {payload_text}\n"
            f"payload_alphabet_size: {payload_alphabet_size}\n"
            "holdout_keys: (none)\n\n"
        )
        rows = [make_line(rng, index, rng.choice(payload_keys)) for index in range(lines)]
        return header + "".join(rows)

    seen_keys = [key for key in KEYS if key not in holdout_keys]
    if not seen_keys:
        raise ValueError("At least one seen key is required")
    train_lines = lines
    val_lines = 0
    if holdout_keys:
        train_lines = int(lines * 0.9)
        val_lines = lines - train_lines
    seen_text = "".join(seen_keys) or "(none)"
    holdout_text = "".join(holdout_keys) or "(none)"
    header = (
        "cassandra long context corpus.\n"
        "each line asks the model to copy the key character after answer=.\n"
        f"seen_keys: {seen_text}\n"
        f"holdout_keys: {holdout_text}\n\n"
    )
    rows: list[str] = []
    for index in range(train_lines):
        key = seen_keys[index % len(seen_keys)] if holdout_keys else KEYS[index % len(KEYS)]
        rows.append(make_line(rng, index, key))
    for offset in range(val_lines):
        index = train_lines + offset
        key = KEYS[offset % len(KEYS)]
        rows.append(make_line(rng, index, key))
    return header + "".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic long-context copy corpus")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--holdout-keys", nargs="*", default=[])
    parser.add_argument("--random-payload", action="store_true")
    parser.add_argument("--payload-alphabet-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lines < 32:
        raise ValueError("--lines must be at least 32")
    holdout_keys = parse_key_set(args.holdout_keys)
    text = build_corpus(
        args.lines,
        args.seed,
        holdout_keys,
        random_payload=args.random_payload,
        payload_alphabet_size=args.payload_alphabet_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
