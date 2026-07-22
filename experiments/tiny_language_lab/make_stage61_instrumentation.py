"""Append crash-resilient Stage 61 checkpoint instrumentation rows.

The Stage 61 training process deliberately only writes checkpoints.  This tool
is the separate, resumable observer required by ADR 0018 D4: for every landed
checkpoint it records the frozen letters probe and deterministic TinyStories
retention result in an append-only JSONL ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from eval_letters_copy_probe import run as run_letters_probe
from eval_tinystories_retention import run as run_retention


LAB_DIR = Path(__file__).resolve().parent
RUN_DIR = LAB_DIR / "runs"
DEFAULT_CHECKPOINT_DIR = Path(r"C:\cassandra_runs\stage61_pure_broad_200m_checkpoints")
DEFAULT_PREFIX = "stage61_pure_broad_200m_seed7_random_full_seed7"
DEFAULT_PROBE = LAB_DIR / "corpus" / "letters_copy_probe_seed.txt"
DEFAULT_RETENTION = LAB_DIR / "corpus" / "tinystories_char_shards_500mb" / "val.txt"
EXPECTED_PROBE_CASES = 1024
EXPECTED_RETENTION_CHARS = 1_499_904
STEP_RE = re.compile(r"_step(\d+)\.pt$")


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_from(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return parsed


def checkpoint_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        step = payload.get("step")
        if step is None:
            raise ValueError(f"Final checkpoint has no stored step: {path}")
        return int(step)
    finally:
        del payload


def discover_checkpoints(
    directory: Path,
    prefix: str,
    minimum_age_seconds: float,
    include_final: bool,
) -> list[Path]:
    candidates = sorted(directory.glob(f"{prefix}*.pt"))
    if not candidates:
        return []
    now = time.time()
    ready: list[Path] = []
    for path in candidates:
        if not include_final and STEP_RE.search(path.name) is None:
            continue
        age = now - path.stat().st_mtime
        if age < minimum_age_seconds:
            print(f"[instrument] defer young checkpoint age={age:.1f}s path={path.name}")
            continue
        ready.append(path.resolve())
    return sorted(ready, key=lambda path: (checkpoint_step(path), path.name))


def read_complete_keys(path: Path) -> set[tuple[str, str]]:
    complete: set[tuple[str, str]] = set()
    if not path.exists():
        return complete
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Instrumentation row {line_number} is not an object")
        if value.get("status") == "complete":
            checkpoint = value.get("checkpoint", {})
            source = checkpoint.get("path")
            digest = checkpoint.get("sha256")
            if not isinstance(source, str) or not isinstance(digest, str):
                raise ValueError(f"Completed instrumentation row {line_number} lacks checkpoint identity")
            complete.add((str(Path(source).resolve()), digest))
    return complete


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def safe_label(path: Path, digest: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    return f"{raw}_{digest[:16]}"


def load_or_run_letters(checkpoint: Path, out: Path, summary: Path, probe: Path) -> dict[str, Any]:
    if out.exists():
        if not summary.exists():
            raise RuntimeError(f"Letters artifact is partial; preserve and inspect: {out}")
        payload = json_from(out)
    else:
        if summary.exists():
            raise RuntimeError(f"Letters summary exists without JSON; preserve and inspect: {summary}")
        payload = run_letters_probe(
            SimpleNamespace(
                checkpoint=checkpoint,
                probe=probe,
                lines=EXPECTED_PROBE_CASES,
                seed=20260709,
                max_cases=EXPECTED_PROBE_CASES,
                device="cuda",
                out=out,
                summary=summary,
            )
        )
    report = payload.get("copy_probe")
    derived = payload.get("derived")
    if not isinstance(report, dict) or not isinstance(derived, dict):
        raise ValueError(f"Letters artifact is incomplete: {out}")
    if int(report.get("copy_probe_choice_cases", -1)) != EXPECTED_PROBE_CASES:
        raise ValueError(f"Letters artifact has wrong scored-case count: {out}")
    return {
        "choice_accuracy": finite(derived.get("choice_accuracy"), "choice accuracy"),
        "choice_mrr": finite(report.get("copy_probe_choice_mrr"), "choice MRR"),
        "raw_accuracy": finite(report.get("copy_probe_accuracy"), "raw accuracy"),
        "nll": finite(report.get("copy_probe_nll"), "probe NLL"),
        "choice_cases": int(report["copy_probe_choice_cases"]),
        "artifact_json": str(out.resolve()),
        "artifact_summary": str(summary.resolve()),
    }


def load_or_run_retention(checkpoint: Path, out_stem: Path, corpus: Path) -> dict[str, Any]:
    out = out_stem.with_suffix(".json")
    summary = out_stem.with_suffix(".md")
    if out.exists():
        if not summary.exists():
            raise RuntimeError(f"Retention artifact is partial; preserve and inspect: {out}")
        payload = json_from(out)
    else:
        if summary.exists():
            raise RuntimeError(f"Retention summary exists without JSON; preserve and inspect: {summary}")
        payload = run_retention(
            SimpleNamespace(
                checkpoint=[checkpoint],
                checkpoint_dir=[],
                corpus=corpus,
                device="cuda",
                max_chars=1_500_000,
                batch_windows=32,
                out_stem=out_stem,
            )
        )
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError(f"Retention artifact does not contain exactly one checkpoint: {out}")
    model = models[0]
    result = model.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Retention artifact has no result: {out}")
    if int(result.get("chars_evaluated", -1)) != EXPECTED_RETENTION_CHARS:
        raise ValueError(
            f"Retention evaluation must cover {EXPECTED_RETENTION_CHARS:,} chars, got "
            f"{result.get('chars_evaluated')!r}"
        )
    return {
        "nll": finite(result.get("nll"), "retention NLL"),
        "bits_per_char": finite(result.get("bits_per_char"), "retention bits/char"),
        "chars_evaluated": int(result["chars_evaluated"]),
        "checkpoint_step": int(model.get("step", -1)),
        "checkpoint_meta": model.get("meta"),
        "artifact_json": str(out.resolve()),
        "artifact_summary": str(summary.resolve()),
    }


def score_checkpoint(args: argparse.Namespace, checkpoint: Path, digest: str, step: int) -> dict[str, Any]:
    label = safe_label(checkpoint, digest)
    letters_dir = args.artifact_dir / "letters"
    retention_dir = args.artifact_dir / "retention"
    letters_json = letters_dir / f"{label}.json"
    letters_summary = letters_dir / f"{label}.md"
    retention_stem = retention_dir / label
    letters = load_or_run_letters(checkpoint, letters_json, letters_summary, args.probe)
    retention = load_or_run_retention(checkpoint, retention_stem, args.retention_corpus)
    if retention["checkpoint_step"] != step:
        raise ValueError(
            f"Retention checkpoint step {retention['checkpoint_step']} disagrees with discovered step {step}"
        )
    return {
        "stage": 61,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "name": checkpoint.name,
            "sha256": digest,
            "bytes": checkpoint.stat().st_size,
            "step": step,
        },
        "letters_probe": letters,
        "tinystories_retention": retention,
        "contract": {
            "probe_cases": EXPECTED_PROBE_CASES,
            "probe_seed": 20260709,
            "retention_chars": EXPECTED_RETENTION_CHARS,
            "retention_method": "deterministic chunked non-overlapping windows",
        },
    }


def write_summary(path: Path, jsonl: Path) -> None:
    rows: list[dict[str, Any]] = []
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    complete = [row for row in rows if row.get("status") == "complete"]
    lines = [
        "# Stage 61 Instrumentation Ledger",
        "",
        f"Append-only source: `{jsonl}`",
        f"Completed checkpoint rows: {len(complete)}",
        "",
        "| Checkpoint | Step | Letters choice accuracy | Letters MRR | TinyStories bits/char | Retention chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(complete, key=lambda item: (int(item["checkpoint"]["step"]), item["checkpoint"]["name"])):
        checkpoint = row["checkpoint"]
        letters = row["letters_probe"]
        retention = row["tinystories_retention"]
        lines.append(
            f"| `{checkpoint['name']}` | {checkpoint['step']} | {letters['choice_accuracy']:.6f} | "
            f"{letters['choice_mrr']:.6f} | {retention['bits_per_char']:.6f} | "
            f"{retention['chars_evaluated']:,} |"
        )
    lines.append("")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Stage 61 checkpoint probe and retention rows")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--retention-corpus", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--jsonl", type=Path, default=RUN_DIR / "stage61_instrumentation.jsonl")
    parser.add_argument("--summary", type=Path, default=RUN_DIR / "stage61_instrumentation.md")
    parser.add_argument("--artifact-dir", type=Path, default=RUN_DIR / "stage61_instrumentation_rows")
    parser.add_argument("--minimum-age-seconds", type=float, default=60.0)
    parser.add_argument("--include-final", action="store_true")
    args = parser.parse_args()

    for path in (args.checkpoint_dir, args.probe, args.retention_corpus):
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 61 instrumentation input is absent: {path}")
    if not math.isfinite(args.minimum_age_seconds) or args.minimum_age_seconds < 0.0:
        raise ValueError("--minimum-age-seconds must be finite and non-negative")

    complete = read_complete_keys(args.jsonl)
    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        args.prefix,
        args.minimum_age_seconds,
        args.include_final,
    )
    if not checkpoints:
        print("[instrument] no mature Stage 61 checkpoints found")
        write_summary(args.summary, args.jsonl)
        return

    created = 0
    skipped = 0
    for checkpoint in checkpoints:
        digest = path_sha256(checkpoint)
        key = (str(checkpoint.resolve()), digest)
        if key in complete:
            skipped += 1
            continue
        step = checkpoint_step(checkpoint)
        print(f"[instrument] start checkpoint={checkpoint.name} step={step} sha256={digest}")
        try:
            row = score_checkpoint(args, checkpoint, digest, step)
        except Exception as error:
            row = {
                "stage": 61,
                "created_utc": datetime.now(UTC).isoformat(),
                "status": "error",
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "name": checkpoint.name,
                    "sha256": digest,
                    "bytes": checkpoint.stat().st_size,
                    "step": step,
                },
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            append_row(args.jsonl, row)
            write_summary(args.summary, args.jsonl)
            raise
        append_row(args.jsonl, row)
        complete.add(key)
        created += 1
        write_summary(args.summary, args.jsonl)
        print(
            f"[instrument] complete checkpoint={checkpoint.name} step={step} "
            f"choice_accuracy={row['letters_probe']['choice_accuracy']:.6f} "
            f"retention_bits={row['tinystories_retention']['bits_per_char']:.6f}"
        )

    write_summary(args.summary, args.jsonl)
    print(f"[instrument] created={created} skipped_complete={skipped} discovered={len(checkpoints)}")


if __name__ == "__main__":
    main()
