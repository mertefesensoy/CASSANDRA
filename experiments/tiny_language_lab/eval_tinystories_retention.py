from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from flagship_eval_lib import LAB_DIR, chunked_nll, encode_fast, free_model, load_model


DEFAULT_CORPUS = LAB_DIR / "corpus" / "tinystories_char_shards_500mb" / "val.txt"
DEFAULT_OUT_STEM = LAB_DIR / "runs" / "stage58_tinystories_retention"
STEP_RE = re.compile(r"_step(\d+)\.pt$")


def checkpoint_step(path: Path, meta: dict[str, object] | None = None) -> int:
    if meta is not None and meta.get("step") is not None:
        return int(meta["step"])
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    paths.extend(args.checkpoint or [])
    for directory in args.checkpoint_dir or []:
        paths.extend(sorted(Path(directory).glob("*_step*.pt")))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("Pass --checkpoint and/or --checkpoint-dir")
    return unique


def write_summary(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Stage 58 TinyStories Retention Evaluation",
        "",
        f"Corpus: `{report['corpus']}`",
        f"Device: `{report['device']}`",
        f"Method: {report['method']}",
        "",
        "| Checkpoint | Step | Bits/char | NLL | Chars evaluated | Seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = sorted(report["models"], key=lambda row: (row["step"], row["checkpoint"]))
    for row in rows:
        result = row["result"]
        lines.append(
            f"| `{Path(row['checkpoint']).name}` | {row['step']} | "
            f"{result['bits_per_char']:.6f} | {result['nll']:.6f} | "
            f"{result['chars_evaluated']:,} | {result['seconds']:.1f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    corpus_text = args.corpus.read_text(encoding="utf-8")
    report: dict[str, object] = {
        "suite": "stage58_tinystories_retention",
        "corpus": str(args.corpus),
        "device": args.device,
        "method": "chunked non-overlapping windows, context resets per window",
        "max_chars": args.max_chars,
        "models": [],
    }

    ids_by_vocab: dict[tuple[str, ...], torch.Tensor] = {}
    for path in collect_checkpoints(args):
        print(f"[retention] loading {path}")
        model, codec, _, meta = load_model(path, device=args.device)
        vocab_key = tuple(codec.chars)
        ids = ids_by_vocab.get(vocab_key)
        if ids is None:
            ids = encode_fast(corpus_text, codec)
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
        row = {
            "checkpoint": str(path),
            "step": checkpoint_step(path, meta),
            "meta": meta,
            "result": stats,
        }
        report["models"].append(row)
        print(
            f"[retention] {path.name}: step={row['step']} "
            f"bits/char={stats['bits_per_char']:.6f} chars={stats['chars_evaluated']:,}"
        )
        free_model(model)
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_stem.with_suffix(".json")
    md_path = args.out_stem.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(md_path, report)
    print(f"[retention] wrote {json_path} and {md_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score checkpoint series on TinyStories validation text")
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--checkpoint-dir", type=Path, action="append")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-chars", type=int, default=1_500_000)
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument("--out-stem", type=Path, default=DEFAULT_OUT_STEM)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
