from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_OUT = Path(__file__).with_name("corpus") / "structured_seed.txt"


SUBJECTS = [
    "cassandra",
    "the student",
    "the small model",
    "the count prior",
    "the residual path",
    "the adapter",
    "the verifier",
    "the notebook",
]

VERBS = [
    "measures",
    "compares",
    "protects",
    "updates",
    "tests",
    "records",
    "keeps",
    "checks",
]

OBJECTS = [
    "the loss",
    "the tiny corpus",
    "the frozen table",
    "the sampled batch",
    "the next token",
    "the small correction",
    "the validation split",
    "the training budget",
]

QUALIFIERS = [
    "carefully",
    "slowly",
    "with one laptop",
    "with a fixed seed",
    "without hiding the failure",
    "before trusting the result",
    "under the same budget",
    "with fewer trainable weights",
]

LESSONS = [
    "counts can start the model",
    "random weights need a fair baseline",
    "small updates can still overfit",
    "validation decides the method",
    "a frozen prior can carry memory",
    "an adapter can learn a correction",
    "one seed is not enough",
    "a tiny result must be repeated",
]


def make_line(rng: random.Random, index: int) -> str:
    subject = SUBJECTS[(index + rng.randrange(len(SUBJECTS))) % len(SUBJECTS)]
    verb = VERBS[(index * 3 + rng.randrange(len(VERBS))) % len(VERBS)]
    obj = OBJECTS[(index * 5 + rng.randrange(len(OBJECTS))) % len(OBJECTS)]
    qualifier = QUALIFIERS[(index * 7 + rng.randrange(len(QUALIFIERS))) % len(QUALIFIERS)]
    lesson = LESSONS[(index * 11 + rng.randrange(len(LESSONS))) % len(LESSONS)]

    if index % 5 == 0:
        return f"{subject} {verb} {obj} {qualifier}. {lesson}.\n"
    if index % 5 == 1:
        return f"if {lesson}, then {subject} {verb} {obj}.\n"
    if index % 5 == 2:
        return f"{subject} asks whether {obj} changes {qualifier}.\n"
    if index % 5 == 3:
        return f"result {index:03d}: {lesson}; {subject} {verb} {obj}.\n"
    return f"{subject} {verb} {obj}, and cassandra records the cost.\n"


def build_corpus(lines: int, seed: int) -> str:
    rng = random.Random(seed)
    header = (
        "cassandra structured corpus.\n"
        "this file is generated for repeatable low hardware language model experiments.\n\n"
    )
    body = [make_line(rng, index) for index in range(lines)]
    return header + "".join(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic structured Cassandra corpus")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lines < 20:
        raise ValueError("--lines must be at least 20")
    text = build_corpus(args.lines, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
