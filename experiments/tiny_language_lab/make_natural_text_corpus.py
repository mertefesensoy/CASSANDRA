from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path(__file__).with_name("corpus") / "tiny_shakespeare_raw.txt"
DEFAULT_OUT = Path(__file__).with_name("corpus") / "natural_text_seed.txt"
DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?'\n"


def normalize_text(text: str, alphabet: str) -> str:
    allowed = set(alphabet)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKD", text).lower()
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": " ",
        "\u201d": " ",
        "\t": " ",
    }

    chars: list[str] = []
    for char in text:
        if unicodedata.combining(char):
            continue
        char = replacements.get(char, char)
        if char in allowed:
            chars.append(char)
        elif char.isascii() and char.isalpha() and char in allowed:
            chars.append(char)
        else:
            chars.append(" " if " " in allowed else "")

    normalized = "".join(chars)
    if "\n" in allowed:
        normalized = re.sub(r"[ ]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    else:
        normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip() + "\n"


def split_at_index(text_len: int, val_fraction: float, block_size: int) -> int:
    split_at = max(block_size + 2, int(text_len * (1.0 - val_fraction)))
    split_at = min(split_at, text_len - block_size - 2)
    return split_at


def build_metadata(
    source: Path,
    source_url: str,
    raw_text: str,
    normalized_text: str,
    alphabet: str,
    val_fraction: float,
    block_size: int,
) -> dict[str, object]:
    counts = Counter(normalized_text)
    split_at = split_at_index(len(normalized_text), val_fraction, block_size)
    return {
        "generator": "make_natural_text_corpus.py",
        "source": str(source),
        "source_url": source_url,
        "input_chars": len(raw_text),
        "normalized_chars": len(normalized_text),
        "normalization": [
            "unicode NFKD",
            "lowercase",
            "unsupported characters mapped to space",
            "space runs collapsed",
            "blank-line runs capped at two newlines",
        ],
        "alphabet": alphabet,
        "observed_vocab": sorted(counts),
        "observed_vocab_size": len(counts),
        "observed_counts": {char: counts[char] for char in sorted(counts)},
        "val_fraction": val_fraction,
        "block_size_for_split_record": block_size,
        "split_protocol": "same deterministic prefix train and suffix validation split as cassandra_tiny_transformer.py",
        "train_chars_at_recorded_block_size": split_at,
        "val_chars_at_recorded_block_size": len(normalized_text) - split_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a natural text source for Stage 30")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-url", type=str, default=DEFAULT_SOURCE_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    parser.add_argument("--min-chars", type=int, default=1_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_chars < 1:
        raise ValueError("--min-chars must be positive")
    if not 0.05 <= args.val_fraction <= 0.5:
        raise ValueError("--val-fraction must be between 0.05 and 0.5")
    if args.block_size < 8:
        raise ValueError("--block-size must be at least 8")

    alphabet = args.alphabet.replace("\\n", "\n")
    raw_text = args.source.read_text(encoding="utf-8")
    normalized_text = normalize_text(raw_text, alphabet)
    if len(normalized_text) < args.min_chars:
        raise ValueError(
            f"Normalized corpus is too short: {len(normalized_text)} chars "
            f"after normalization, wanted at least {args.min_chars}"
        )

    metadata = build_metadata(
        args.source,
        args.source_url,
        raw_text,
        normalized_text,
        alphabet,
        args.val_fraction,
        args.block_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(normalized_text, encoding="utf-8")
    metadata_out = args.metadata_out or args.out.with_suffix(".meta.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({len(normalized_text)} chars, "
        f"vocab={metadata['observed_vocab_size']})"
    )
    print(f"wrote {metadata_out}")


if __name__ == "__main__":
    main()
