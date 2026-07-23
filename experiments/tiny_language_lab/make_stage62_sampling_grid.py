"""Stage 62 Workstream 2: sampling-grid diagnostic for the Stage 61 flagship.

The flagship's samples drift off topic. Part of that is the model's finite
context window (measured by the H027 context-utilization probe), and part may
be the decoding choice: the review sheet used temperature 0.8 with no
truncation, which leaves the full derailing tail live at every character. This
script regenerates a few fixed prompts across a grid of decoding settings so the
same prompt can be read as the sampler tightens, separating genuine model drift
from a sampling artifact. It makes no coherence claim; the reader judges.

Deterministic: each (prompt, setting) uses a fixed seed derived from the prompt
index, so the grid reproduces exactly. Eval-only, no training.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\make_stage62_sampling_grid.py `
    --checkpoint C:\\cassandra_runs\\...\\flagship.pt `
    --out runs\\stage62_sampling_grid.json --summary runs\\stage62_sampling_grid.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flagship_eval_lib import free_model, load_model, sample_text

LAB_DIR = Path(__file__).resolve().parent
RUN_DIR = LAB_DIR / "runs"
DEFAULT_CHECKPOINT = Path(
    r"C:\cassandra_runs\stage61_pure_broad_200m_checkpoints"
    r"\stage61_pure_broad_200m_seed7_random_full_seed7.pt"
)
EXPECTED_PARAMETERS = 201_609_249

PROMPTS = (
    "the history of ",
    "the united states ",
    "the city of ",
)

# label, temperature, top_k, top_p, repetition_penalty
SETTINGS = (
    ("baseline (temp 0.8, no truncation)", 0.8, 0, 0.0, 1.0),
    ("nucleus (temp 0.8, top-p 0.9)", 0.8, 0, 0.9, 1.0),
    ("warm nucleus (temp 0.7, top-p 0.95)", 0.7, 0, 0.95, 1.0),
    ("cool nucleus (temp 0.4, top-p 0.95)", 0.4, 0, 0.95, 1.0),
    ("near-greedy (temp 0.2, top-p 0.95)", 0.2, 0, 0.95, 1.0),
    ("greedy (temp -> 0)", 0.0, 0, 0.0, 1.0),
    ("rep-penalty (temp 0.8, top-p 0.9, pen 1.3)", 0.8, 0, 0.9, 1.3),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 62 Sampling-Grid Diagnostic (H027 Workstream 2)",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        f"- parameters: {payload['parameters']:,}  block size: {payload['block_size']}",
        f"- max new characters: {payload['max_new_chars']}",
        "",
        "Fixed prompts across a decoding grid, so the same prompt can be read as "
        "the sampler tightens. Nucleus (top-p) and lower temperature commit the "
        "model to higher-probability continuations, which suppresses the "
        "low-probability topic jumps that read as drift; they do not extend the "
        "model's finite context window (H027, ADR 0019), so drift that survives "
        "tighter decoding is genuine model behavior, not a sampling artifact. "
        "This sheet makes no coherence claim.",
        "",
    ]
    for group in payload["prompts"]:
        lines.append(f"## Prompt: `{group['prompt']}`")
        lines.append("")
        for row in group["samples"]:
            lines.append(f"### {row['label']}")
            lines.append("")
            lines.append("```text")
            lines.append(row["text"])
            lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 62 sampling-grid diagnostic")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-chars", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=20260723)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "stage62_sampling_grid.json")
    parser.add_argument(
        "--summary", type=Path, default=RUN_DIR / "stage62_sampling_grid.md"
    )
    args = parser.parse_args()

    if not args.checkpoint.exists() or args.checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing flagship checkpoint: {args.checkpoint}")
    for path in (args.out, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing grid evidence: {path}")

    model, codec, ckpt_args, meta = load_model(args.checkpoint, device=args.device)
    try:
        parameters = int(meta["parameters"])
        if parameters != EXPECTED_PARAMETERS:
            raise ValueError(
                f"Checkpoint has {parameters:,} parameters, expected {EXPECTED_PARAMETERS:,}"
            )
        prompt_groups = []
        for p_index, prompt in enumerate(PROMPTS):
            seed = args.seed_base + p_index
            samples = []
            for label, temperature, top_k, top_p, penalty in SETTINGS:
                text = sample_text(
                    model,
                    codec,
                    prompt=prompt,
                    max_new_chars=args.max_new_chars,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=penalty,
                    seed=seed,
                    device=args.device,
                )
                samples.append(
                    {
                        "label": label,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": penalty,
                        "seed": seed,
                        "text": text,
                    }
                )
            prompt_groups.append({"prompt": prompt, "seed": seed, "samples": samples})
    finally:
        free_model(model)

    payload: dict[str, Any] = {
        "stage": 62,
        "hypothesis": "H027",
        "workstream": "2 (sampling grid)",
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "parameters": parameters,
        "block_size": int(ckpt_args["block_size"]),
        "max_new_chars": args.max_new_chars,
        "settings": [
            {"label": s[0], "temperature": s[1], "top_k": s[2], "top_p": s[3], "repetition_penalty": s[4]}
            for s in SETTINGS
        ],
        "prompts": prompt_groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    print(f"stage62_sampling_grid prompts={len(prompt_groups)} settings={len(SETTINGS)}")


if __name__ == "__main__":
    main()
