"""Freeze Stage 61's admissible step budget from its 5,000-step measurement.

ADR 0018 requires a sustained throughput measurement before the pure-broad
200M run.  This program is deliberately small and fail-closed: it accepts one
registered 5,000-step row, binds the decision to that row's SHA-256, and
chooses the largest 5,000-step multiple in the 30 to 40 GPU-hour envelope.
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
EXPECTED_PARAMETERS = 201_609_249
EXPECTED_THROUGHPUT_STEPS = 5_000
MIN_STEPS = 30_000
MAX_STEPS = 50_000
STEP_QUANTUM = 5_000
MAX_GPU_HOURS = 40.0


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_row(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Throughput row {line_number} is not a JSON object")
        rows.append(value)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one throughput row in {path}, found {len(rows)}")
    return rows[0]


def finite_number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Throughput row is missing {key}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Throughput row has non-finite {key}: {value!r}")
    return parsed


def muon_learning_rate(row: dict[str, Any]) -> float:
    report = row.get("optimizer_report")
    if not isinstance(report, dict) or report.get("optimizer") != "muon":
        raise ValueError("Throughput row does not carry a Muon optimizer report")
    return finite_number(report, "muon_lr")


def assert_registered_surface(row: dict[str, Any], expected_shards: Path) -> None:
    expected = {
        "comparison_name": "random_full",
        "seed": 7,
        "parameters": EXPECTED_PARAMETERS,
        "steps": EXPECTED_THROUGHPUT_STEPS,
        "formation_steps": EXPECTED_THROUGHPUT_STEPS,
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
        "checkpoint_every": STEP_QUANTUM,
        "checkpoint_keep": 12,
        "lr_total_steps": EXPECTED_THROUGHPUT_STEPS,
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(
                f"Throughput row has {key}={row.get(key)!r}; expected {expected_value!r}"
            )
    if abs(muon_learning_rate(row) - 0.01) > 1e-12:
        raise ValueError("Throughput row does not use registered Muon lr=0.01")
    if abs(finite_number(row, "lr_final_frac") - 0.1) > 1e-12:
        raise ValueError("Throughput row does not use registered cosine final fraction=0.1")
    if Path(str(row.get("train_shard_dir", ""))).resolve() != expected_shards.resolve():
        raise ValueError("Throughput row does not point at the registered pure text8 shard directory")
    if row.get("copy_train_marker") not in ("", None):
        raise ValueError("Throughput row must not carry a copy-training marker")
    if int(row.get("formation_forward_passes", 0)) <= 0:
        raise ValueError("Throughput row has no positive formation-forward-pass count")
    if finite_number(row, "seconds") <= 0.0:
        raise ValueError("Throughput row has non-positive elapsed seconds")
    if finite_number(row, "peak_cuda_memory_mib") <= 0.0:
        raise ValueError("Throughput row has no CUDA peak-memory measurement")


def decide(row: dict[str, Any], max_gpu_hours: float) -> dict[str, Any]:
    seconds = finite_number(row, "seconds")
    seconds_per_step = seconds / EXPECTED_THROUGHPUT_STEPS
    projected_30k_hours = seconds_per_step * MIN_STEPS / 3600.0
    projected_50k_hours = seconds_per_step * MAX_STEPS / 3600.0
    max_steps_by_hours = int(math.floor(max_gpu_hours * 3600.0 / seconds_per_step))
    target_steps = min(MAX_STEPS, (max_steps_by_hours // STEP_QUANTUM) * STEP_QUANTUM)
    if target_steps < MIN_STEPS:
        status = "STOP_AND_ASK"
        target_steps_out: int | None = None
        reason = (
            f"The measured rate projects {projected_30k_hours:.3f} GPU-hours for 30,000 steps, "
            f"above the registered {max_gpu_hours:.1f}-hour ceiling."
        )
    else:
        status = "CLEARED"
        target_steps_out = target_steps
        reason = (
            f"The largest 5,000-step multiple within {max_gpu_hours:.1f} GPU-hours is "
            f"{target_steps:,} steps."
        )
    return {
        "status": status,
        "reason": reason,
        "target_steps": target_steps_out,
        "seconds_per_step": seconds_per_step,
        "projected_30k_gpu_hours": projected_30k_hours,
        "projected_50k_gpu_hours": projected_50k_hours,
        "max_gpu_hours": max_gpu_hours,
        "minimum_steps": MIN_STEPS,
        "maximum_steps": MAX_STEPS,
        "step_quantum": STEP_QUANTUM,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    throughput = payload["throughput"]
    lines = [
        "# Stage 61 Throughput Budget Gate",
        "",
        f"- source row: `{payload['throughput_path']}`",
        f"- source SHA-256: `{payload['throughput_sha256']}`",
        f"- measured 5,000-step seconds: {throughput['seconds']:.4f}",
        f"- measured seconds/step: {decision['seconds_per_step']:.6f}",
        f"- projected 30,000-step GPU-hours: {decision['projected_30k_gpu_hours']:.3f}",
        f"- projected 50,000-step GPU-hours: {decision['projected_50k_gpu_hours']:.3f}",
        f"- decision: **{decision['status']}**",
        f"- target steps: {decision['target_steps'] if decision['target_steps'] is not None else 'none; user decision required'}",
        "",
        decision["reason"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage 61's measured compute budget")
    parser.add_argument("--throughput", type=Path, default=RUN_DIR / "stage61_throughput.jsonl")
    parser.add_argument("--text8-shards", type=Path, default=LAB_DIR / "corpus" / "text8_char_shards")
    parser.add_argument("--out", type=Path, default=RUN_DIR / "stage61_throughput_gate.json")
    parser.add_argument("--summary", type=Path, default=RUN_DIR / "stage61_throughput_gate.md")
    parser.add_argument("--max-gpu-hours", type=float, default=MAX_GPU_HOURS)
    args = parser.parse_args()

    for path in (args.throughput, args.text8_shards):
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 61 budget input is absent: {path}")
    for path in (args.out, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 61 budget evidence: {path}")
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours <= 0.0:
        raise ValueError("--max-gpu-hours must be a finite positive value")

    row = read_single_row(args.throughput)
    assert_registered_surface(row, args.text8_shards)
    decision = decide(row, args.max_gpu_hours)
    payload: dict[str, Any] = {
        "stage": 61,
        "title": "Stage 61 pure-broad 200M throughput budget gate",
        "created_utc": datetime.now(UTC).isoformat(),
        "throughput_path": str(args.throughput.resolve()),
        "throughput_sha256": sha256_path(args.throughput),
        "throughput": {
            "seconds": finite_number(row, "seconds"),
            "peak_cuda_memory_mib": finite_number(row, "peak_cuda_memory_mib"),
            "parameters": int(row["parameters"]),
            "formation_forward_passes": int(row["formation_forward_passes"]),
        },
        "decision": decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    print(
        f"stage61_budget_status={decision['status']} "
        f"target_steps={decision['target_steps']} "
        f"projected_30k_gpu_hours={decision['projected_30k_gpu_hours']:.3f}"
    )


if __name__ == "__main__":
    main()
