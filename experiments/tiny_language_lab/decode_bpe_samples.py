from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METADATA = Path(__file__).with_name("corpus") / "tinystories_bpe_v256_seed.meta.json"
DEFAULT_RUN = Path(__file__).with_name("runs") / "phase2_tinystories_bpe_smoke.jsonl"
DEFAULT_OUT = Path(__file__).with_name("runs") / "phase2_tinystories_bpe_smoke_decoded_samples.md"


def decode_private_use(text: str, vocab: list[str], base: int) -> str:
    pieces: list[str] = []
    for char in text:
        token_id = ord(char) - base
        if 0 <= token_id < len(vocab):
            pieces.append(vocab[token_id])
        else:
            pieces.append("?")
    return "".join(pieces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode private-use BPE samples from Cassandra run JSONL")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", type=str, default="Phase 2 TinyStories BPE Decoded Samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    vocab = metadata["vocab"]
    base = int(metadata["private_use_base"])
    rows = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        f"# {args.title}",
        "",
        f"Run: `{args.run}`",
        f"Metadata: `{args.metadata}`",
        "",
    ]
    for row in rows:
        decoded = decode_private_use(str(row.get("sample", "")), vocab, base)
        lines.extend(
            [
                f"## {row.get('comparison_name')} seed={row.get('seed')}",
                "",
                f"val_nll: `{row.get('val_nll')}`",
                "",
                "```text",
                decoded,
                "```",
                "",
            ]
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
