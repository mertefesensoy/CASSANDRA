"""Generate the fixed Stage 61 sample sheet for the user's final review.

The sheet is deliberately descriptive.  It does not score coherence or make a
pass decision: ADR 0018 reserves that judgment for the user.
"""

from __future__ import annotations

import argparse
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
    "in the early years ",
    "the first world war ",
    "the university of ",
    "the government of ",
    "the population of ",
)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 61 User Sample Review Sheet",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        f"- parameters: {payload['parameters']:,}",
        f"- sampling: temperature {payload['temperature']}, top-k {payload['top_k']}, {payload['max_new_chars']} new characters",
        "- review status: **PENDING_USER_REVIEW**",
        "",
        "These are fixed deterministic samples for the user's review. They do not constitute an automated coherence claim or a publication decision.",
        "",
    ]
    for item in payload["samples"]:
        lines.extend(
            [
                f"## Sample {item['index']} · seed {item['seed']}",
                "",
                f"Prompt: `{item['prompt']}`",
                "",
                "```text",
                item["text"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 61's deterministic user sample sheet")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "stage61_user_samples.json")
    parser.add_argument("--summary", type=Path, default=RUN_DIR / "stage61_user_samples.md")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-chars", type=int, default=400)
    args = parser.parse_args()

    for path in (args.checkpoint,):
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required Stage 61 checkpoint is absent or empty: {path}")
    for path in (args.out, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 61 user-review evidence: {path}")
    if args.temperature <= 0.0 or args.max_new_chars <= 0:
        raise ValueError("Sampling temperature and length must be positive")

    model, codec, _, meta = load_model(args.checkpoint, device="cuda")
    try:
        parameters = int(meta.get("parameters", -1))
        if parameters != EXPECTED_PARAMETERS:
            raise ValueError(
                f"Stage 61 sample checkpoint has {parameters:,} parameters, expected {EXPECTED_PARAMETERS:,}"
            )
        samples = [
            {
                "index": index,
                "prompt": prompt,
                "seed": 20_260_722 + index,
                "text": sample_text(
                    model,
                    codec,
                    prompt=prompt,
                    max_new_chars=args.max_new_chars,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=20_260_722 + index,
                    device="cuda",
                ),
            }
            for index, prompt in enumerate(PROMPTS, start=1)
        ]
    finally:
        free_model(model)

    payload: dict[str, Any] = {
        "stage": 61,
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "parameters": parameters,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_chars": args.max_new_chars,
        "review_status": "PENDING_USER_REVIEW",
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    print(f"stage61_user_samples={len(samples)} review_status=PENDING_USER_REVIEW")


if __name__ == "__main__":
    main()
