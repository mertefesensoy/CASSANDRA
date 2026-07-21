"""Deterministic text8 evaluation for Cassandra checkpoints.

text8 (Mahoney) is the classic character-level LM benchmark: 100M chars of
cleaned lowercase Wikipedia, alphabet a-z plus space, standard split
train=first 90M, valid=next 5M, test=last 5M. The lab alphabet (33 chars)
is a strict superset of older lab checkpoints, and Stage 56 checkpoints use
the native text8 alphabet. Published anchors use bits per character on the
test split.

Method here: deterministic chunked evaluation (non-overlapping block-size
windows, context resets per window), the same convention as
phase4_validate.py. Chunked evaluation is slightly pessimistic versus
sliding-window scoring, so treat results as conservative.

Writes a Markdown and JSON report at --out-stem.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\eval_text8.py
  python .\\experiments\\tiny_language_lab\\eval_text8.py --models flagship_200m_50k_seed7 --max-chars 500000
  python .\\experiments\\tiny_language_lab\\eval_text8.py --checkpoint C:\\cassandra_runs\\stage56_broadchar_checkpoints\\stage56.pt --checkpoint-name stage56_seed7
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

from flagship_eval_lib import (
    FINAL_CHECKPOINTS,
    LAB_DIR,
    chunked_nll,
    encode_fast,
    free_model,
    load_model,
)

TEXT8_URL = "https://mattmahoney.net/dc/text8.zip"
TEXT8_DIR = LAB_DIR / "corpus" / "text8"


def ensure_text8() -> str:
    TEXT8_DIR.mkdir(parents=True, exist_ok=True)
    raw = TEXT8_DIR / "text8"
    if not raw.exists():
        zip_path = TEXT8_DIR / "text8.zip"
        if not zip_path.exists():
            print(f"[text8] downloading {TEXT8_URL}")
            urllib.request.urlretrieve(TEXT8_URL, zip_path)
        print("[text8] extracting")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("text8", TEXT8_DIR)
    text = raw.read_text(encoding="ascii")
    if len(text) != 100_000_000:
        raise ValueError(f"Unexpected text8 length: {len(text):,}")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot text8 bits/char")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(FINAL_CHECKPOINTS.keys()),
        default=None,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="custom_checkpoint")
    parser.add_argument(
        "--out-stem", type=Path, default=LAB_DIR / "runs" / "text8_eval"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    text = ensure_text8()
    if args.split == "test":
        segment = text[95_000_000:]
    else:
        segment = text[90_000_000:95_000_000]

    if args.models is None:
        model_names = [] if args.checkpoint is not None else ["flagship_200m_50k_seed7", "stage51_25m_5k_seed7"]
    else:
        model_names = args.models

    model_items: list[tuple[str, Path]] = [(name, FINAL_CHECKPOINTS[name]) for name in model_names]
    if args.checkpoint is not None:
        model_items.append((args.checkpoint_name, args.checkpoint))
    if not model_items:
        raise ValueError("No models selected. Pass --models and/or --checkpoint.")

    report: dict = {
        "benchmark": "text8",
        "split": args.split,
        "segment_chars": len(segment),
        "method": "chunked non-overlapping windows, context resets per window",
        "models": {},
    }

    ids_by_vocab: dict[tuple[str, ...], torch.Tensor] = {}
    for name, path in model_items:
        print(f"[text8] loading {name}")
        model, codec, _, meta = load_model(path, device=args.device)
        vocab_key = tuple(codec.chars)
        ids = ids_by_vocab.get(vocab_key)
        if ids is None:
            ids = encode_fast(segment, codec)
            ids_by_vocab[vocab_key] = ids
        started = time.time()
        stats = chunked_nll(
            model,
            ids,
            device=args.device,
            batch_windows=args.batch_windows,
            max_eval_chars=args.max_chars,
        )
        stats["seconds"] = round(time.time() - started, 1)
        report["models"][name] = {"meta": meta, "result": stats}
        print(
            f"[text8] {name}: bits/char={stats['bits_per_char']:.4f} "
            f"({stats['chars_evaluated']:,} chars in {stats['seconds']}s)"
        )
        free_model(model)

    json_path = args.out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Cassandra text8 Evaluation", ""]
    lines.append(
        f"Split: {args.split} ({len(segment):,} chars) · chunked convention · "
        f"device {args.device}"
    )
    lines.append("")
    lines.append("| Model | Params | text8 bits/char | Chars evaluated |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, entry in report["models"].items():
        stats = entry["result"]
        lines.append(
            f"| {name} | {entry['meta']['parameters']:,} "
            f"| {stats['bits_per_char']:.4f} | {stats['chars_evaluated']:,} |"
        )
    lines.append("")
    lines.append(
        "Published anchors for context: GPT-2 zero-shot text8 "
        "1.17 bits/char at 117M and 0.98 at 1542M (Radford et al. 2019, "
        "Table 3); dedicated text8-trained models reach about 1.0 to 1.1. "
        "Interpretation depends on the checkpoint corpus: TinyStories "
        "checkpoints are out-of-domain, while Stage 56 broad-char checkpoints "
        "are in-domain train/valid text8 runs scored on the held-out TEST split."
    )
    md_path = args.out_stem.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[text8] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
