"""Validate and report Stage 61's pre-registered publication bars.

This is intentionally a verifier, not a launcher.  It consumes the immutable
training ladder, append-only instrumentation ledger, and deterministic text8
TEST report and refuses to turn a partial set of artifacts into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parent
RUN_DIR = LAB_DIR / "runs"
CHECKPOINT_DIR = Path(r"C:\cassandra_runs\stage61_pure_broad_200m_checkpoints")
PREFIX = "stage61_pure_broad_200m_seed7_random_full_seed7"
TEXT8_NAME = "stage61_pure_broad_200m_seed7"
TEXT8_BAR = 1.357318
SEED_NOISE = 0.003035
EXPECTED_PARAMETERS = 201_609_249
EXPECTED_TEXT8_CHARS = 4_999_936
EXPECTED_RETENTION_CHARS = 1_499_904
EXPECTED_PROBE_CASES = 1024


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def finite(value: object, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return parsed


def muon_learning_rate(row: dict[str, Any]) -> float:
    report = row.get("optimizer_report")
    if not isinstance(report, dict) or report.get("optimizer") != "muon":
        raise ValueError("Training ladder does not carry a Muon optimizer report")
    return finite(report.get("muon_lr"), "Muon lr")


def assert_training_row(row: dict[str, Any], current_step: int, target_steps: int) -> None:
    expected = {
        "comparison_name": "random_full",
        "seed": 7,
        "parameters": EXPECTED_PARAMETERS,
        "steps": 5_000,
        "formation_steps": current_step + 5_000,
        "block_size": 256,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "n_layer": 16,
        "n_head": 16,
        "n_embd": 1024,
        "pos_encoding": "rope",
        "activation_checkpoint": True,
        "optimizer": "muon",
        "precision": "fp32",
        "lr_schedule": "cosine",
        "lr_total_steps": target_steps,
        "checkpoint_every": 5_000,
        "checkpoint_keep": 12,
        "resume_step": current_step,
        "resume_loaded": current_step > 0,
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(
                f"Training ladder row for start {current_step} has {key}={row.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    if abs(muon_learning_rate(row) - 0.01) > 1e-12:
        raise ValueError("Training ladder does not use registered Muon lr=0.01")
    if abs(finite(row.get("lr_final_frac"), "cosine final fraction") - 0.1) > 1e-12:
        raise ValueError("Training ladder does not use registered cosine final fraction=0.1")
    for metric in ("seconds", "peak_cuda_memory_mib", "val_nll", "val_bits_per_char"):
        if finite(row.get(metric), metric) <= 0.0 and metric in {"seconds", "peak_cuda_memory_mib"}:
            raise ValueError(f"Training ladder has non-positive {metric}")


def verify_training(rows: list[dict[str, Any]], target_steps: int, checkpoint_dir: Path, prefix: str) -> dict[str, Any]:
    if target_steps < 30_000 or target_steps > 50_000 or target_steps % 5_000:
        raise ValueError(f"Invalid registered target steps: {target_steps}")
    expected_rows = target_steps // 5_000
    if len(rows) != expected_rows:
        raise ValueError(f"Training ladder has {len(rows)} rows, expected {expected_rows}")
    for index, row in enumerate(rows):
        current_step = index * 5_000
        assert_training_row(row, current_step, target_steps)
    step_paths = [checkpoint_dir / f"{prefix}_step{step:06d}.pt" for step in range(5_000, target_steps + 1, 5_000)]
    final_path = checkpoint_dir / f"{prefix}.pt"
    for path in [*step_paths, final_path]:
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty registered checkpoint: {path}")
    return {
        "rows": len(rows),
        "target_steps": target_steps,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": path_sha256(final_path),
        "step_checkpoints": [str(path.resolve()) for path in step_paths],
    }


def verify_instrumentation(
    rows: list[dict[str, Any]], checkpoint_dir: Path, prefix: str, target_steps: int
) -> dict[str, Any]:
    expected_paths = [
        str((checkpoint_dir / f"{prefix}_step{step:06d}.pt").resolve())
        for step in range(5_000, target_steps + 1, 5_000)
    ]
    expected_paths.append(str((checkpoint_dir / f"{prefix}.pt").resolve()))
    completed = [row for row in rows if row.get("status") == "complete"]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Completed instrumentation row lacks checkpoint object")
        path = checkpoint.get("path")
        if not isinstance(path, str):
            raise ValueError("Completed instrumentation row lacks checkpoint path")
        by_path.setdefault(str(Path(path).resolve()), []).append(row)
    for expected_path in expected_paths:
        matches = by_path.get(expected_path, [])
        if len(matches) != 1:
            raise ValueError(
                f"Instrumentation requires exactly one completed row for {Path(expected_path).name}; "
                f"found {len(matches)}"
            )
        row = matches[0]
        checkpoint = row["checkpoint"]
        actual = Path(expected_path)
        if checkpoint.get("sha256") != path_sha256(actual):
            raise ValueError(f"Instrumentation SHA-256 does not bind current checkpoint: {actual}")
        if int(checkpoint.get("bytes", -1)) != actual.stat().st_size:
            raise ValueError(f"Instrumentation byte count does not bind current checkpoint: {actual}")
        letters = row.get("letters_probe")
        retention = row.get("tinystories_retention")
        if not isinstance(letters, dict) or not isinstance(retention, dict):
            raise ValueError(f"Instrumentation row lacks probe or retention result: {actual}")
        if int(letters.get("choice_cases", -1)) != EXPECTED_PROBE_CASES:
            raise ValueError(f"Instrumentation row has wrong frozen-probe count: {actual}")
        if int(retention.get("chars_evaluated", -1)) != EXPECTED_RETENTION_CHARS:
            raise ValueError(f"Instrumentation row has wrong retention coverage: {actual}")
        for metric, value in (
            ("choice_accuracy", letters.get("choice_accuracy")),
            ("choice_mrr", letters.get("choice_mrr")),
            ("raw_accuracy", letters.get("raw_accuracy")),
            ("probe_nll", letters.get("nll")),
            ("retention_nll", retention.get("nll")),
            ("retention_bits", retention.get("bits_per_char")),
        ):
            finite(value, f"instrumentation {metric}")
    return {
        "complete_rows_for_registered_ladder": len(expected_paths),
        "expected_checkpoint_paths": expected_paths,
        "history_rows": len(rows),
        "error_history_rows": sum(1 for row in rows if row.get("status") == "error"),
    }


def verify_text8(report: dict[str, Any], final_checkpoint: str) -> dict[str, Any]:
    if report.get("benchmark") != "text8" or report.get("split") != "test":
        raise ValueError("Stage 61 text8 report is not a TEST report")
    models = report.get("models")
    if not isinstance(models, dict) or set(models) != {TEXT8_NAME}:
        raise ValueError("Stage 61 text8 report must contain exactly the registered final model")
    entry = models[TEXT8_NAME]
    if not isinstance(entry, dict):
        raise ValueError("Stage 61 text8 model entry is malformed")
    meta = entry.get("meta")
    result = entry.get("result")
    if not isinstance(meta, dict) or not isinstance(result, dict):
        raise ValueError("Stage 61 text8 model entry lacks metadata or result")
    if int(meta.get("parameters", -1)) != EXPECTED_PARAMETERS:
        raise ValueError("Stage 61 text8 report has wrong parameter count")
    if str(Path(str(meta.get("checkpoint", ""))).resolve()) != str(Path(final_checkpoint).resolve()):
        raise ValueError("Stage 61 text8 report is not bound to the final checkpoint")
    if int(result.get("chars_evaluated", -1)) != EXPECTED_TEXT8_CHARS:
        raise ValueError(
            f"Stage 61 text8 TEST must cover {EXPECTED_TEXT8_CHARS:,} chars, got "
            f"{result.get('chars_evaluated')!r}"
        )
    bpc = finite(result.get("bits_per_char"), "text8 bits/char")
    nll = finite(result.get("nll"), "text8 NLL")
    return {"bits_per_char": bpc, "nll": nll, "chars_evaluated": int(result["chars_evaluated"])}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    text8 = payload["text8_test"]
    bars = payload["bars"]
    lines = [
        "# Stage 61 Publication Bars",
        "",
        f"- final checkpoint: `{payload['training']['final_checkpoint']}`",
        f"- text8 TEST bits/char: {text8['bits_per_char']:.9f}",
        f"- strict bar: < {TEXT8_BAR:.6f}",
        f"- margin versus bar: {bars['text8_margin_vs_bar']:+.9f} bits/char",
        f"- known Stage 56 replica spread: {SEED_NOISE:.6f} bits/char",
        f"- text8 result: **{bars['text8_bar']}**",
        f"- instrumentation result: **{bars['instrumentation_bar']}**",
        f"- user sample review: **PENDING_USER_REVIEW**",
        f"- packaging status: **{bars['packaging_status']}**",
        "",
        "The text8 bar is fail-first. A failed text8 bar means no Stage 61 publication or replacement flagship in this Phase 6 branch. Instrumentation completion is repairable but blocks packaging until complete. The final sample review remains the user's decision.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Stage 61 publication bars")
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--training", type=Path, default=RUN_DIR / "stage61_pure_broad_200m_seed7.jsonl")
    parser.add_argument("--instrumentation", type=Path, default=RUN_DIR / "stage61_instrumentation.jsonl")
    parser.add_argument("--text8", type=Path, default=RUN_DIR / "stage61_text8_test.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--prefix", type=str, default=PREFIX)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "stage61_publish_bars.json")
    parser.add_argument("--summary", type=Path, default=RUN_DIR / "stage61_publish_bars.md")
    args = parser.parse_args()

    for path in (args.training, args.instrumentation, args.text8, args.checkpoint_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 61 verification input is absent: {path}")
    for path in (args.out, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 61 publication-bar evidence: {path}")

    training_rows = load_jsonl(args.training)
    training = verify_training(training_rows, args.target_steps, args.checkpoint_dir, args.prefix)
    instrumentation = verify_instrumentation(
        load_jsonl(args.instrumentation), args.checkpoint_dir, args.prefix, args.target_steps
    )
    text8 = verify_text8(load_json(args.text8), training["final_checkpoint"])
    margin = TEXT8_BAR - text8["bits_per_char"]
    text8_bar = "PASS" if text8["bits_per_char"] < TEXT8_BAR else "FAIL_NO_PUBLICATION"
    payload: dict[str, Any] = {
        "stage": 61,
        "created_utc": datetime.now(UTC).isoformat(),
        "training": training,
        "instrumentation": instrumentation,
        "text8_test": text8,
        "bars": {
            "text8_bar": text8_bar,
            "text8_threshold_bits_per_char": TEXT8_BAR,
            "text8_margin_vs_bar": margin,
            "known_seed_spread_bits_per_char": SEED_NOISE,
            "instrumentation_bar": "PASS",
            "user_sample_review": "PENDING_USER_REVIEW",
            "packaging_status": "BLOCKED_PENDING_USER_REVIEW" if text8_bar == "PASS" else "BLOCKED_TEXT8_FAIL",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    print(
        f"stage61_text8_bar={text8_bar} "
        f"bits_per_char={text8['bits_per_char']:.9f} margin={margin:+.9f}"
    )


if __name__ == "__main__":
    main()
