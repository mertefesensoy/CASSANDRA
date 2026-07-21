"""Build deterministic text8 character shards for Phase 5 Stage 56.

The standard text8 split is:

- train: chars [0, 90M)
- valid: chars [90M, 95M)
- test: chars [95M, 100M)

This script writes train shards plus a validation file and a 95M-char seed
file whose suffix validation split can reproduce the standard valid range.
The test slice is never written into the seed or training shard directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
TEXT8_PATH = LAB_DIR / "corpus" / "text8" / "text8"
OUT_DIR = LAB_DIR / "corpus" / "text8_char_shards"
SEED_OUT = LAB_DIR / "corpus" / "text8_char_seed.txt"
META_OUT = LAB_DIR / "corpus" / "text8_char_shards.meta.json"

TEXT8_CHARS = 100_000_000
TRAIN_END = 90_000_000
VAL_END = 95_000_000
DEFAULT_SHARD_CHARS = 10_000_000
TEXT8_ALPHABET = set("abcdefghijklmnopqrstuvwxyz ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 56 text8 char shards")
    parser.add_argument("--text8", type=Path, default=TEXT8_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed-out", type=Path, default=SEED_OUT)
    parser.add_argument("--metadata-out", type=Path, default=META_OUT)
    parser.add_argument("--shard-chars", type=int, default=DEFAULT_SHARD_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_chars <= 0:
        raise ValueError("--shard-chars must be positive")
    if not args.text8.exists():
        raise FileNotFoundError(f"Missing text8 file: {args.text8}")

    text = args.text8.read_text(encoding="ascii")
    if len(text) != TEXT8_CHARS:
        raise ValueError(f"Expected {TEXT8_CHARS:,} chars, found {len(text):,}")
    extra_chars = sorted(set(text) - TEXT8_ALPHABET)
    if extra_chars:
        raise ValueError(f"text8 contains chars outside expected alphabet: {extra_chars!r}")

    train_text = text[:TRAIN_END]
    val_text = text[TRAIN_END:VAL_END]
    test_text = text[VAL_END:]
    seed_text = text[:VAL_END]
    if len(train_text) != TRAIN_END or len(val_text) != VAL_END - TRAIN_END:
        raise AssertionError("Internal split length mismatch")
    if len(test_text) != TEXT8_CHARS - VAL_END:
        raise AssertionError("Internal test length mismatch")
    if test_text in seed_text or test_text in train_text or test_text in val_text:
        raise AssertionError("text8 test range appears in generated train or seed text")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_train_files: set[Path] = set()
    shard_reports: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, TRAIN_END, args.shard_chars)):
        end = min(start + args.shard_chars, TRAIN_END)
        shard_path = args.out_dir / f"train_{shard_index:04d}.txt"
        write_text(shard_path, text[start:end])
        expected_train_files.add(shard_path.resolve())
        shard_reports.append(
            {
                "path": str(shard_path),
                "start": start,
                "end": end,
                "chars": end - start,
                "sha256": sha256_text(text[start:end]),
            }
        )

    for old_path in args.out_dir.glob("train_*.txt"):
        if old_path.resolve() not in expected_train_files:
            old_path.unlink()

    val_path = args.out_dir / "val.txt"
    write_text(val_path, val_text)
    write_text(args.seed_out, seed_text)

    metadata = {
        "source": str(args.text8),
        "source_chars": len(text),
        "alphabet": "".join(sorted(set(text))),
        "standard_split": {
            "train": [0, TRAIN_END],
            "valid": [TRAIN_END, VAL_END],
            "test": [VAL_END, TEXT8_CHARS],
        },
        "seed_file": str(args.seed_out),
        "seed_chars": len(seed_text),
        "seed_sha256": sha256_text(seed_text),
        "out_dir": str(args.out_dir),
        "shard_chars": args.shard_chars,
        "train_shards": shard_reports,
        "train_chars": len(train_text),
        "val_file": str(val_path),
        "val_chars": len(val_text),
        "val_sha256": sha256_text(val_text),
        "test_chars": len(test_text),
        "test_sha256": sha256_text(test_text),
        "contamination_assert": "exact 5M-char test segment absent from seed, train, and val text",
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        "[text8-shards] "
        f"train={len(train_text):,} val={len(val_text):,} "
        f"seed={len(seed_text):,} shards={len(shard_reports)}"
    )
    print(f"[text8-shards] wrote {args.out_dir}")
    print(f"[text8-shards] wrote {args.seed_out}")
    print(f"[text8-shards] wrote {args.metadata_out}")


if __name__ == "__main__":
    main()
