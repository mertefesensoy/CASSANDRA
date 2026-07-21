from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_OUT = Path(__file__).with_name("corpus") / "memory_mapping_seed.txt"

KEYS = "abcdefgh"
DEFAULT_MAPPING = {
    "a": "h",
    "b": "e",
    "c": "g",
    "d": "a",
    "e": "c",
    "f": "b",
    "g": "d",
    "h": "f",
}
FILLERS = [
    "amber signal",
    "brisk method",
    "careful trace",
    "direct measure",
    "even budget",
    "frozen prior",
    "gentle update",
    "hidden memory",
]


def parse_key_set(raw: str | list[str]) -> list[str]:
    pieces = raw if isinstance(raw, list) else [raw]
    keys = [ch for piece in pieces for ch in piece if ch.strip()]
    unknown = sorted(set(keys) - set(KEYS))
    if unknown:
        raise ValueError(f"Unknown holdout keys: {''.join(unknown)}")
    return sorted(set(keys), key=KEYS.index)


def build_mapping(holdout_keys: list[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_MAPPING)
    if not holdout_keys:
        return mapping

    seen_keys = [key for key in KEYS if key not in holdout_keys]
    if not seen_keys:
        raise ValueError("At least one seen key is required")
    for index, key in enumerate(holdout_keys):
        donor = seen_keys[index % len(seen_keys)]
        mapping[key] = mapping[donor]
    return mapping


def make_line(rng: random.Random, index: int, key: str, mapping: dict[str, str]) -> str:
    answer = mapping[key]
    distractor = KEYS[(index * 5 + 3) % len(KEYS)]
    filler_a = FILLERS[(index + rng.randrange(len(FILLERS))) % len(FILLERS)]
    filler_b = FILLERS[(index * 3 + rng.randrange(len(FILLERS))) % len(FILLERS)]
    filler_c = FILLERS[(index * 7 + rng.randrange(len(FILLERS))) % len(FILLERS)]
    return (
        f"case {index:04d} key={key} "
        f"noise={distractor} {filler_a}; {filler_b}; {filler_c}; "
        f"answer={answer}\n"
    )


def build_corpus(lines: int, seed: int, holdout_keys: list[str]) -> str:
    rng = random.Random(seed)
    mapping = build_mapping(holdout_keys)
    seen_keys = [key for key in KEYS if key not in holdout_keys]
    train_lines = lines
    val_lines = 0
    if holdout_keys:
        train_lines = int(lines * 0.9)
        val_lines = lines - train_lines
    mapping_text = " ".join(f"{key}->{mapping[key]}" for key in KEYS)
    seen_text = "".join(seen_keys) or "(none)"
    holdout_text = "".join(holdout_keys) or "(none)"
    header = (
        "cassandra non identity memory corpus.\n"
        "each line asks the model to map key to answer using a fixed memory table.\n"
        f"mapping: {mapping_text}\n"
        f"seen_keys: {seen_text}\n"
        f"holdout_keys: {holdout_text}\n\n"
    )
    rows: list[str] = []
    for index in range(train_lines):
        key = seen_keys[index % len(seen_keys)] if holdout_keys else KEYS[index % len(KEYS)]
        rows.append(make_line(rng, index, key, mapping))
    for offset in range(val_lines):
        index = train_lines + offset
        key = KEYS[offset % len(KEYS)]
        rows.append(make_line(rng, index, key, mapping))
    return header + "".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic non-identity memory corpus")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--holdout-keys", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lines < 32:
        raise ValueError("--lines must be at least 32")
    holdout_keys = parse_key_set(args.holdout_keys)
    text = build_corpus(args.lines, args.seed, holdout_keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
