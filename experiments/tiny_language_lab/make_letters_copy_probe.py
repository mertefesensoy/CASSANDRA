from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = LAB_DIR / "corpus" / "letters_copy_probe_seed.txt"

PAYLOAD_ALPHABET = "abcdefghijklmnop"
CASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
KEY_MARKER = "key "
ANSWER_MARKER = "answer "
ALLOWED_CHARS = set("\n !',.?abcdefghijklmnopqrstuvwxyz")
FILLERS = [
    "amber signal",
    "brisk method",
    "careful trace",
    "direct measure",
    "even budget",
    "frozen prior",
    "gentle update",
    "hidden memory",
    "quiet ladder",
    "steady mirror",
    "plain window",
    "patient record",
]


def spell_case_id(index: int, width: int = 4) -> str:
    if index < 0:
        raise ValueError("case index must be non-negative")
    base = len(CASE_ALPHABET)
    chars: list[str] = []
    value = index
    for _ in range(width):
        chars.append(CASE_ALPHABET[value % base])
        value //= base
    if value:
        raise ValueError(f"case index {index} exceeds width {width}")
    return "".join(reversed(chars))


def make_targets(lines: int, seed: int) -> list[str]:
    if lines < len(PAYLOAD_ALPHABET):
        raise ValueError(f"--lines must be at least {len(PAYLOAD_ALPHABET)}")
    targets = [PAYLOAD_ALPHABET[index % len(PAYLOAD_ALPHABET)] for index in range(lines)]
    rng = random.Random(seed)
    rng.shuffle(targets)
    return targets


def make_line(rng: random.Random, index: int, target: str) -> str:
    distractors = [ch for ch in PAYLOAD_ALPHABET if ch != target]
    distractor = distractors[(index * 7 + rng.randrange(len(distractors))) % len(distractors)]
    filler_a = FILLERS[(index + rng.randrange(len(FILLERS))) % len(FILLERS)]
    filler_b = FILLERS[(index * 5 + rng.randrange(len(FILLERS))) % len(FILLERS)]
    case_id = spell_case_id(index)
    return f"case {case_id} {KEY_MARKER}{target} noise {distractor} {filler_a}, {filler_b}. {ANSWER_MARKER}{target}\n"


def build_corpus(lines: int, seed: int) -> str:
    rng = random.Random(seed)
    rows = [make_line(rng, index, target) for index, target in enumerate(make_targets(lines, seed))]
    return "".join(rows)


def parse_value(line: str, marker: str) -> str | None:
    at = line.find(marker)
    if at < 0:
        return None
    value_at = at + len(marker)
    if value_at >= len(line):
        return None
    return line[value_at]


def validate_corpus(text: str) -> dict[str, object]:
    unknown = sorted(set(text) - ALLOWED_CHARS)
    if unknown:
        raise ValueError(f"Probe text contains unsupported chars: {unknown!r}")
    if any(ch.isdigit() for ch in text) or "=" in text:
        raise ValueError("Probe text must be letters-only: no digits and no equals sign")

    cases = 0
    counts = {ch: 0 for ch in PAYLOAD_ALPHABET}
    for line in text.splitlines():
        key = parse_value(line, KEY_MARKER)
        answer = parse_value(line, ANSWER_MARKER)
        if key is None or answer is None:
            continue
        if key != answer:
            raise ValueError(f"Unverified key/answer pair in line: {line!r}")
        if key not in PAYLOAD_ALPHABET:
            raise ValueError(f"Unexpected payload {key!r}")
        counts[key] += 1
        cases += 1

    if cases == 0:
        raise ValueError("Probe contains no parseable cases")
    if sorted(ch for ch, count in counts.items() if count) != list(PAYLOAD_ALPHABET):
        raise ValueError("Probe must contain every payload candidate at least once")
    return {
        "cases": cases,
        "payload_counts": counts,
        "choice_candidates": PAYLOAD_ALPHABET,
        "chance_accuracy": 1.0 / len(PAYLOAD_ALPHABET),
        "key_marker": KEY_MARKER,
        "answer_marker": ANSWER_MARKER,
        "verified_identity_pairs": True,
        "letters_only": True,
        "has_digits": False,
        "has_equals": False,
    }


def write_probe(out: Path, lines: int, seed: int) -> dict[str, object]:
    text = build_corpus(lines, seed)
    meta = validate_corpus(text)
    meta.update(
        {
            "path": str(out),
            "lines_requested": lines,
            "seed": seed,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({meta['cases']} cases, sha256={meta['sha256']})")
    print(f"wrote {meta_path}")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Phase 5 letters-only copy behavior probe")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_probe(args.out, args.lines, args.seed)


if __name__ == "__main__":
    main()
