from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path(__file__).with_name("corpus") / "tinystories_char_seed.txt"
DEFAULT_OUT = Path(__file__).with_name("corpus") / "tinystories_bpe_v256_seed.txt"
DEFAULT_METADATA = Path(__file__).with_name("corpus") / "tinystories_bpe_v256_seed.meta.json"
PUA_BASE = 0xE000
PUA_LIMIT = 0xF8FF - PUA_BASE + 1


def merge_once(tokens: list[int], left: int, right: int, merged: int) -> list[int]:
    out: list[int] = []
    index = 0
    stop = len(tokens) - 1
    while index < len(tokens):
        if index < stop and tokens[index] == left and tokens[index + 1] == right:
            out.append(merged)
            index += 2
        else:
            out.append(tokens[index])
            index += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    min_pair_count: int,
) -> tuple[list[str], list[tuple[int, int, int, int]]]:
    initial_tokens = sorted(set(text))
    if vocab_size < len(initial_tokens):
        raise ValueError(f"--vocab-size {vocab_size} is smaller than base alphabet {len(initial_tokens)}")
    if vocab_size > PUA_LIMIT:
        raise ValueError(f"--vocab-size may not exceed {PUA_LIMIT} for private-use encoding")

    vocab = list(initial_tokens)
    token_ids = {token: index for index, token in enumerate(vocab)}
    tokens = [token_ids[ch] for ch in text]
    merges: list[tuple[int, int, int, int]] = []

    while len(vocab) < vocab_size:
        pair_counts = Counter(zip(tokens, tokens[1:]))
        if not pair_counts:
            break
        (left, right), count = pair_counts.most_common(1)[0]
        if count < min_pair_count:
            break
        merged = len(vocab)
        vocab.append(vocab[left] + vocab[right])
        tokens = merge_once(tokens, left, right, merged)
        merges.append((left, right, merged, count))

    return vocab, merges


def encode_bpe(text: str, base_vocab: list[str], merges: list[tuple[int, int, int, int]]) -> list[int]:
    token_ids = {token: index for index, token in enumerate(base_vocab)}
    base_count = 0
    while base_count < len(base_vocab) and len(base_vocab[base_count]) == 1:
        base_count += 1
    char_ids = {token: index for index, token in enumerate(base_vocab[:base_count])}
    missing = sorted({ch for ch in text if ch not in char_ids})
    if missing:
        raise ValueError(f"Text contains characters outside the BPE base alphabet: {missing[:10]}")

    tokens = [char_ids[ch] for ch in text]
    for left, right, merged, _count in merges:
        tokens = merge_once(tokens, left, right, merged)
    return tokens


def private_use_text(token_ids: list[int]) -> str:
    return "".join(chr(PUA_BASE + token_id) for token_id in token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small local BPE tokenizer and emit a token-as-char corpus")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--train-chars", type=int, default=500_000)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--min-pair-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_chars <= 0 or args.max_chars <= 0:
        raise ValueError("--train-chars and --max-chars must be positive")
    if args.train_chars > args.max_chars:
        raise ValueError("--train-chars cannot exceed --max-chars")

    source_text = args.source.read_text(encoding="utf-8")[: args.max_chars]
    if len(source_text) < 1000:
        raise ValueError("Source text is too short for BPE training")
    train_text = source_text[: args.train_chars]
    vocab, merges = train_bpe(train_text, args.vocab_size, args.min_pair_count)
    encoded = encode_bpe(source_text, vocab, merges)
    encoded_text = private_use_text(encoded)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded_text, encoding="utf-8")
    metadata = {
        "source": str(args.source),
        "out": str(args.out),
        "encoding": "private-use-codepoint-per-bpe-token",
        "private_use_base": PUA_BASE,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "source_chars": len(source_text),
        "train_chars": len(train_text),
        "encoded_tokens": len(encoded),
        "compression_chars_per_token": len(source_text) / max(len(encoded), 1),
        "min_pair_count": args.min_pair_count,
        "vocab": vocab,
        "merges": [
            {
                "left": left,
                "right": right,
                "merged": merged,
                "count_at_merge": count,
                "token": vocab[merged],
            }
            for left, right, merged, count in merges
        ],
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.metadata_out}")
    print(
        "bpe "
        f"vocab={len(vocab)} source_chars={len(source_text)} encoded_tokens={len(encoded)} "
        f"chars_per_token={metadata['compression_chars_per_token']:.3f}"
    )


if __name__ == "__main__":
    main()
