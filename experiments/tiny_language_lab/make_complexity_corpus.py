from __future__ import annotations

import argparse
import random
from pathlib import Path

from make_long_context_corpus import make_line as make_long_line
from make_synthetic_corpus import make_line as make_structured_line


DEFAULT_OUT = Path(__file__).with_name("corpus") / "complexity_mix_seed.txt"


def build_corpus(lines: int, seed: int, long_fraction: float) -> str:
    if not 0 <= long_fraction <= 1:
        raise ValueError("--long-fraction must be in [0, 1]")

    rng = random.Random(seed)
    long_count = round(lines * long_fraction)
    long_indices = set(rng.sample(range(lines), long_count))
    header = (
        "cassandra corpus complexity sweep.\n"
        "structured lines are bigram-local; long lines require copying a key after intervening context.\n"
        f"lines: {lines}\n"
        f"seed: {seed}\n"
        f"long_fraction: {long_fraction:.4f}\n"
        f"structured_lines: {lines - long_count}\n"
        f"long_context_lines: {long_count}\n\n"
    )

    body: list[str] = []
    for index in range(lines):
        if index in long_indices:
            body.append(make_long_line(rng, index))
        else:
            body.append(make_structured_line(rng, index))
    return header + "".join(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic corpus-complexity sweep point")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--long-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lines < 32:
        raise ValueError("--lines must be at least 32")
    text = build_corpus(args.lines, args.seed, args.long_fraction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
