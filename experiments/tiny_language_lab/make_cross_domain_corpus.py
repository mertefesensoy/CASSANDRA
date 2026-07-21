from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_natural_text_corpus import DEFAULT_ALPHABET, normalize_text


DEFAULT_TRAIN_SOURCE = Path(__file__).with_name("corpus") / "natural_text_seed.txt"
DEFAULT_OUT = Path(__file__).with_name("corpus") / "natural_text_crossdomain_seed.txt"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_markdown(root: Path) -> list[Path]:
    excluded_parts = {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
    }
    excluded_suffixes = (
        ("experiments", "tiny_language_lab", "runs"),
        ("experiments", "tiny_language_lab", "corpus"),
    )
    files: list[Path] = []
    for path in root.rglob("*.md"):
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in excluded_parts for part in parts):
            continue
        if any(parts[: len(suffix)] == suffix for suffix in excluded_suffixes):
            continue
        files.append(path)
    return sorted(files)


def coverage_report(train_text: str, val_text: str, vocab_size: int, max_order: int) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for order in range(1, max_order + 1):
        train_contexts = {
            train_text[index : index + order]
            for index in range(max(len(train_text) - order, 0))
        }
        val_total = max(len(val_text) - order, 0)
        val_hits = sum(
            1
            for index in range(val_total)
            if val_text[index : index + order] in train_contexts
        )
        report.append(
            {
                "order": order,
                "contexts": vocab_size**order,
                "observed_contexts": len(train_contexts),
                "table_coverage": len(train_contexts) / max(vocab_size**order, 1),
                "validation_contexts": val_total,
                "validation_observed_contexts": val_hits,
                "validation_coverage": val_hits / max(val_total, 1),
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-domain natural-text split for Stage 32")
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    parser.add_argument("--train-chars", type=int, default=935_612)
    parser.add_argument("--val-chars", type=int, default=165_109)
    parser.add_argument("--max-order", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_chars < 1000:
        raise ValueError("--train-chars must be at least 1000")
    if args.val_chars < 1000:
        raise ValueError("--val-chars must be at least 1000")
    if args.max_order < 1:
        raise ValueError("--max-order must be at least 1")

    alphabet = args.alphabet.replace("\\n", "\n")
    train_source = args.train_source.read_text(encoding="utf-8")
    if len(train_source) < args.train_chars:
        raise ValueError(
            f"Train source has {len(train_source)} chars, needs {args.train_chars}"
        )
    train_text = train_source[: args.train_chars]

    val_files = collect_markdown(args.val_root)
    val_raw = "\n\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in val_files)
    val_text = normalize_text(val_raw, alphabet)
    if len(val_text) < args.val_chars:
        raise ValueError(
            f"Validation source has {len(val_text)} normalized chars, needs {args.val_chars}"
        )
    val_text = val_text[: args.val_chars]

    text = train_text + val_text
    metadata = {
        "generator": "make_cross_domain_corpus.py",
        "train_source": str(args.train_source),
        "validation_root": str(args.val_root),
        "validation_source": "normalized markdown project prose",
        "validation_files": [str(path.relative_to(args.val_root)) for path in val_files],
        "alphabet": alphabet,
        "observed_vocab": sorted(set(text)),
        "observed_vocab_size": len(set(text)),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "total_chars": len(text),
        "val_fraction": len(val_text) / len(text),
        "split_protocol": "train prefix from normalized Tiny Shakespeare, validation suffix from normalized Cassandra project prose",
        "coverage": coverage_report(train_text, val_text, len(set(text)), args.max_order),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    metadata_out = args.metadata_out or args.out.with_suffix(".meta.json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({len(text)} chars, train={len(train_text)}, "
        f"val={len(val_text)}, vocab={metadata['observed_vocab_size']})"
    )
    print(f"wrote {metadata_out}")
    for row in metadata["coverage"]:
        print(
            "order={order} table={table_coverage:.6f} val_hit={validation_coverage:.6f}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
