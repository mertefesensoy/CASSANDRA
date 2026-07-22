"""Build H026's frozen letters-copy checkpoint matrix without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import eval_letters_copy_probe as letters_probe


LAB_DIR = Path(__file__).resolve().parent
RUNS_DIR = LAB_DIR / "runs"
DEFAULT_PROBE = LAB_DIR / "corpus" / "letters_copy_probe_seed.txt"
CHECKPOINT_ROOT = Path(r"C:\cassandra_runs")
PRESENT_THRESHOLD = 0.1625
ABSENT_THRESHOLD = 0.1125
ANCHOR_ACCURACY = 0.194336
ANCHOR_NAME = "stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt"
PROBE_LINES = 1024
PROBE_SEED = 20260709
PROBE_CASES = 1024


# These are the checkpoint byte hashes already recorded in durable Stage 58
# evidence. All other surviving rungs are intentionally hash_unverified: H026
# requires recording, not silently excluding, those previously unhashed files.
RECORDED_CHECKPOINT_HASHES: dict[str, dict[str, str]] = {
    ANCHOR_NAME: {
        "sha256": "f7cd52fb4db5f7f61ceed27c31861af6e5923c4b264e146073893d2d2d3167eb",
        "source": "RESULTS.md Stage 59 Part 0",
    },
    "stage58_dev_cold_85m_b20000_seed19_random_full_seed19_step015000.pt": {
        "sha256": "c77537fa09e227c56de7e28f61df7ec2640a427cb5c1e90eb92f1ad275660365",
        "source": "RESULTS.md Stage 58 replica power-safety pause",
    },
    "stage58_dev_cold_85m_b20000_seed19_random_full_seed19.pt": {
        "sha256": "0a15419d0fa64165117fb9b251f341a9b2119ccac108343041c1805199a60771",
        "source": "RESULTS.md Stage 58 seed-19 recovery",
    },
    "stage58_dev_curriculum_phase1_85m_b5952_seed19_random_full_seed19.pt": {
        "sha256": "f8b40c1a76c068e30d999036c0dcf7177edb20e803ffd0e029cd7388897954ad",
        "source": "RESULTS.md Stage 58 seed-19 recovery",
    },
    "stage58_dev_curriculum_phase2_85m_b20000_seed19_random_full_seed19.pt": {
        "sha256": "2a1b209c8278992184322d947476d5bae38fd24dfa56aeb6ccd932fd253373c6",
        "source": "RESULTS.md Stage 58 seed-19 recovery",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step020000.pt": {
        "sha256": "62f54cc84df9faa870c642c8dd98f75b3b24fbe47b0d5b7c7e518a89bda3d8cd",
        "source": "RESULTS.md Stage 58 curriculum step-20,000",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step025000.pt": {
        "sha256": "9ef44d15539bd5d2f220f36a7f295f6912549e64591822d6f0ea8d3109ffd811",
        "source": "RESULTS.md Stage 58 curriculum step-25,000",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step030000.pt": {
        "sha256": "b443cfb7036bf59b3dc7b5a6b6f927aa0401cbe27b42a08929162abdeb5d23cb",
        "source": "RESULTS.md Stage 58 curriculum step-30,000",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step035000.pt": {
        "sha256": "6f0b714377f4ab3b5470f23779975c97f085b4f4c6a5ca9cd8d3f52e9da922de",
        "source": "RESULTS.md Stage 58 curriculum step-35,000",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step040000.pt": {
        "sha256": "f694a885e3746e972abc50effe4408e098270cb459fad75c75fe2c7cafa1e772",
        "source": "RESULTS.md Stage 58 curriculum step-40,000",
    },
    "stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7.pt": {
        "sha256": "f8a734503143764141c96924ed233f12fbdd9734d984e0df7886fc5f6dc71ccd",
        "source": "RESULTS.md Stage 58 curriculum seed-7 completion",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step005000.pt": {
        "sha256": "9dc7cf9abe077b269ef3a297973cbf3dafba7b44cd2cf2ebb6d0d03640f49497",
        "source": "RESULTS.md Stage 58 mixture step-5,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step010000.pt": {
        "sha256": "f8bb94bea11ef7475f602415fc6cbaaee19ab68ab983569c0bbc26f963eb4aec",
        "source": "RESULTS.md Stage 58 mixture step-10,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step015000.pt": {
        "sha256": "839281d62255455ba676c1e2810ad70acc4d58be47a4a368644e1e4533a64377",
        "source": "RESULTS.md Stage 58 mixture step-15,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step020000.pt": {
        "sha256": "9bc1ccf91a97dec5d77a3b9d74d08ef802bd52d8e51dc11bff4f79623285d685",
        "source": "RESULTS.md Stage 58 mixture step-20,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step025000.pt": {
        "sha256": "297f366edb6e53df8a824e00d6a27f7ec9ecdee7622d75506d74ba7c55237868",
        "source": "RESULTS.md Stage 58 mixture step-25,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step030000.pt": {
        "sha256": "2cad7aa59c831299133202553c7d2bc502b1b6ab6a741c8f79481495d19ccb84",
        "source": "RESULTS.md Stage 58 mixture step-30,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step035000.pt": {
        "sha256": "9b915c5f51c8828e49f65dcbc29ee0033be2ff002a9e5a748e637e7e30b7ce93",
        "source": "RESULTS.md Stage 58 mixture step-35,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step040000.pt": {
        "sha256": "dc5c0dcd79cebb48ac328d6867c3d50f9dea052d37b6855b10c6213cd80322e6",
        "source": "RESULTS.md Stage 58 mixture step-40,000",
    },
    "stage58_dev_mixture_85m_b42000_seed7_random_full_seed7.pt": {
        "sha256": "45d312f976916d86f8021d21c1b45987860fb18bf58b7020e9ee2b6546edc707",
        "source": "RESULTS.md Stage 58 mixture seed-7 completion",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(name: str) -> int:
    match = re.search(r"_step(\d{6})\.pt$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"_b(\d+)_seed\d+_random_full_seed\d+\.pt$", name)
    if not match:
        raise ValueError(f"Cannot recover global step from checkpoint name: {name}")
    return int(match.group(1))


def seed_from_name(name: str) -> int:
    match = re.search(r"_seed(\d+)_random_full_seed\d+", name)
    if not match:
        raise ValueError(f"Cannot recover seed from checkpoint name: {name}")
    return int(match.group(1))


def lineage_from_name(name: str) -> str:
    if name.startswith("stage56_broadchar_"):
        return "stage56_recipe_v1_broad"
    if name.startswith("stage58_dev_cold_"):
        return "cold_broad_only"
    if name.startswith("stage58_dev_curriculum_phase1_"):
        return "curriculum_phase1_tinystories_only"
    if name.startswith("stage58_dev_curriculum_phase2_"):
        return "curriculum_phase2_domain_shift"
    if name.startswith("stage58_dev_mixture_"):
        return "mixture_stage58_w298"
    if name.startswith("stage59_mixture_w10_"):
        return "mixture_stage59_w010"
    raise ValueError(f"Unknown Stage 56/58/59 checkpoint lineage: {name}")


def phase1_steps(seed: int) -> int:
    if seed == 7:
        return 12500
    if seed in (11, 19):
        return 5952
    raise ValueError(f"Unexpected curriculum seed {seed}")


def exposure_steps(lineage: str, seed: int, global_step: int) -> tuple[float, float]:
    if lineage in {"stage56_recipe_v1_broad", "cold_broad_only"}:
        return float(global_step), 0.0
    if lineage == "curriculum_phase1_tinystories_only":
        return 0.0, float(global_step)
    if lineage == "curriculum_phase2_domain_shift":
        tiny = phase1_steps(seed)
        if global_step < tiny:
            raise ValueError(f"Curriculum phase-2 checkpoint predates phase 1: seed {seed}, step {global_step}")
        return float(global_step - tiny), float(tiny)
    if lineage == "mixture_stage58_w298":
        return global_step * 59.0 / 84.0, global_step * 25.0 / 84.0
    if lineage == "mixture_stage59_w010":
        return global_step * 0.90, global_step * 0.10
    raise ValueError(f"Unknown lineage for exposure calculation: {lineage}")


def priority(entry: dict[str, Any]) -> tuple[int, int, int, str]:
    name = Path(str(entry["checkpoint"])).name
    lineage = str(entry["lineage"])
    if name == ANCHOR_NAME:
        return (0, 0, 0, name)
    if lineage == "curriculum_phase1_tinystories_only":
        return (10, int(entry["seed"]), int(entry["global_step"]), name)
    if lineage == "cold_broad_only":
        return (20, int(entry["seed"]), int(entry["global_step"]), name)
    if lineage == "curriculum_phase2_domain_shift":
        return (30, int(entry["seed"]), int(entry["global_step"]), name)
    if lineage == "mixture_stage58_w298":
        return (40, int(entry["seed"]), int(entry["global_step"]), name)
    if lineage == "mixture_stage59_w010":
        return (50, int(entry["seed"]), int(entry["global_step"]), name)
    return (60, int(entry["seed"]), int(entry["global_step"]), name)


def build_inventory() -> list[dict[str, Any]]:
    roots = [
        (CHECKPOINT_ROOT / "stage56_broadchar_checkpoints", "stage56"),
        (CHECKPOINT_ROOT / "stage58_dev_cold_checkpoints", "stage58"),
        (CHECKPOINT_ROOT / "stage58_dev_curriculum_checkpoints", "stage58"),
        (CHECKPOINT_ROOT / "stage58_dev_mixture_checkpoints", "stage58"),
        (CHECKPOINT_ROOT / "stage59_mixture_w10_checkpoints", "stage59"),
    ]
    inventory: list[dict[str, Any]] = []
    for root, stage in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Required H026 checkpoint directory is missing: {root}")
        for path in sorted(root.glob("*.pt")):
            global_step = checkpoint_step(path.name)
            seed = seed_from_name(path.name)
            lineage = lineage_from_name(path.name)
            broad_steps, tiny_steps = exposure_steps(lineage, seed, global_step)
            recorded = RECORDED_CHECKPOINT_HASHES.get(path.name)
            inventory.append(
                {
                    "checkpoint": str(path.resolve()),
                    "checkpoint_name": path.name,
                    "stage": stage,
                    "lineage": lineage,
                    "seed": seed,
                    "global_step": global_step,
                    "broad_steps_seen": broad_steps,
                    "tinystories_steps_seen": tiny_steps,
                    "checkpoint_kind": "intermediate" if "_step" in path.stem else "final",
                    "recorded_checkpoint_sha256": recorded["sha256"] if recorded else None,
                    "recorded_hash_source": recorded["source"] if recorded else None,
                }
            )
    inventory.sort(key=priority)
    stage58 = sum(1 for entry in inventory if entry["stage"] == "stage58")
    stage59 = sum(1 for entry in inventory if entry["stage"] == "stage59")
    stage56 = sum(1 for entry in inventory if entry["stage"] == "stage56")
    if (stage58, stage59, stage56) != (68, 10, 4):
        raise ValueError(
            "H026 inventory changed before launch: "
            f"Stage 58={stage58}, Stage 59={stage59}, Stage 56={stage56}; expected 68, 10, 4"
        )
    if inventory[0]["checkpoint_name"] != ANCHOR_NAME:
        raise ValueError("H026 anchor was not first in the frozen inventory")
    return inventory


def finite(value: object) -> bool:
    return value is not None and math.isfinite(float(value))


def classify_accuracy(value: object) -> str | None:
    if not finite(value):
        return None
    number = float(value)
    if number >= PRESENT_THRESHOLD:
        return "PRESENT"
    if number <= ABSENT_THRESHOLD:
        return "ABSENT"
    return "GRAY"


def probe_one(entry: dict[str, Any], args: argparse.Namespace, probe_sha256: str) -> dict[str, Any]:
    row = dict(entry)
    checkpoint = Path(str(entry["checkpoint"]))
    row.update(
        {
            "probe_path": str(args.probe.resolve()),
            "probe_sha256": probe_sha256,
            "probe_lines": PROBE_LINES,
            "probe_seed": PROBE_SEED,
            "probe_max_cases": PROBE_CASES,
            "status": "pending",
            "reason": None,
            "choice_accuracy": None,
            "choice_mrr": None,
            "raw_accuracy": None,
            "nll": None,
            "probe_seconds": None,
            "checkpoint_sha256": None,
            "hash_status": None,
            "probe_artifact": None,
            "probe_summary": None,
        }
    )
    if not checkpoint.is_file():
        row.update({"status": "missing", "reason": "checkpoint missing at probe time", "hash_status": "missing"})
        return row
    actual_hash = sha256_file(checkpoint)
    row["checkpoint_sha256"] = actual_hash
    recorded = row["recorded_checkpoint_sha256"]
    if recorded is not None and actual_hash != recorded:
        row.update(
            {
                "status": "hash_mismatch_excluded",
                "reason": "checkpoint SHA-256 differs from previously recorded durable evidence",
                "hash_status": "hash_mismatch",
            }
        )
        return row
    row["hash_status"] = "verified" if recorded is not None else "hash_unverified"
    artifact_stem = args.probe_dir / f"stage60_probe_{checkpoint.stem}"
    row["probe_artifact"] = str(artifact_stem.with_suffix(".json"))
    row["probe_summary"] = str(artifact_stem.with_suffix(".md"))
    probe_args = argparse.Namespace(
        checkpoint=checkpoint,
        probe=args.probe,
        lines=PROBE_LINES,
        seed=PROBE_SEED,
        max_cases=PROBE_CASES,
        device=args.device,
        out=artifact_stem.with_suffix(".json"),
        summary=artifact_stem.with_suffix(".md"),
    )
    try:
        payload = letters_probe.run(probe_args)
    except Exception as error:  # preserve incompatible and failed-load evidence as rows
        row.update({"status": "probe_error", "reason": f"{type(error).__name__}: {error}"})
        return row
    report = payload["copy_probe"]
    derived = payload["derived"]
    metrics = {
        "choice_accuracy": derived["choice_accuracy"],
        "choice_mrr": report["copy_probe_choice_mrr"],
        "raw_accuracy": report["copy_probe_accuracy"],
        "nll": report["copy_probe_nll"],
        "probe_seconds": payload["seconds"],
    }
    if not all(finite(value) for value in metrics.values()):
        row.update({"status": "probe_error", "reason": "probe returned a non-finite required metric"})
        return row
    row.update(metrics)
    row["classification"] = classify_accuracy(row["choice_accuracy"])
    row["status"] = "scored"
    return row


def add_cold_matches(rows: list[dict[str, Any]]) -> None:
    cold_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row["lineage"] == "cold_broad_only":
            cold_by_seed.setdefault(int(row["seed"]), []).append(row)
    for candidates in cold_by_seed.values():
        candidates.sort(key=lambda row: int(row["global_step"]))
    for row in rows:
        if row["lineage"] != "curriculum_phase2_domain_shift":
            row["nearest_cold_checkpoint"] = None
            row["nearest_cold_global_step"] = None
            row["nearest_cold_step_delta"] = None
            continue
        candidates = cold_by_seed.get(int(row["seed"]), [])
        if not candidates:
            row["nearest_cold_checkpoint"] = None
            row["nearest_cold_global_step"] = None
            row["nearest_cold_step_delta"] = None
            continue
        nearest = min(candidates, key=lambda cold: abs(float(cold["broad_steps_seen"]) - float(row["broad_steps_seen"])))
        row["nearest_cold_checkpoint"] = nearest["checkpoint"]
        row["nearest_cold_global_step"] = nearest["global_step"]
        row["nearest_cold_step_delta"] = float(row["broad_steps_seen"]) - float(nearest["broad_steps_seen"])


def screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    onset: list[dict[str, Any]] = []
    destruction: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["lineage"]), int(row["seed"])), []).append(row)
    for (lineage, seed), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["global_step"]))
        scored = [row for row in group if row.get("status") == "scored"]
        present = next((row for row in scored if row.get("classification") == "PRESENT"), None)
        onset.append(
            {
                "lineage": lineage,
                "seed": seed,
                "first_present_checkpoint": present["checkpoint"] if present else None,
                "first_present_global_step": present["global_step"] if present else None,
                "first_present_broad_steps_seen": present["broad_steps_seen"] if present else None,
            }
        )
        for previous, current in zip(scored, scored[1:]):
            drop = float(previous["choice_accuracy"]) - float(current["choice_accuracy"])
            if drop >= 0.05:
                destruction.append(
                    {
                        "lineage": lineage,
                        "seed": seed,
                        "from_checkpoint": previous["checkpoint"],
                        "to_checkpoint": current["checkpoint"],
                        "accuracy_drop": drop,
                        "note": "descriptive H026 destruction screen only; it fires no intervention",
                    }
                )
    return {"formation_onsets": onset, "destruction_candidates": destruction}


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    narrow = [row for row in rows if row["lineage"] == "curriculum_phase1_tinystories_only"]
    cold_seed7 = [row for row in rows if row["lineage"] == "cold_broad_only" and row["seed"] == 7]
    phase1_seed7 = [row for row in narrow if row["seed"] == 7 and row["checkpoint_kind"] == "final"]
    phase2_seed7 = [
        row for row in rows
        if row["lineage"] == "curriculum_phase2_domain_shift" and row["seed"] == 7 and row["checkpoint_kind"] == "final"
    ]
    narrow_present = [row for row in narrow if row.get("classification") == "PRESENT"]
    all_narrow_absent = bool(narrow) and all(row.get("classification") == "ABSENT" for row in narrow)
    cold_reaches_present = any(row.get("classification") == "PRESENT" for row in cold_seed7)
    # A completed lineage may retain an earlier unsuffixed recovery checkpoint
    # (the seed-7 phase-2 b12501 resume smoke). The H026 "final" is the
    # terminal final by global step, not merely the only unsuffixed file.
    phase1 = max(phase1_seed7, key=lambda row: int(row["global_step"])) if phase1_seed7 else None
    phase2 = max(phase2_seed7, key=lambda row: int(row["global_step"])) if phase2_seed7 else None
    phase2_gain = None
    phase2_clause = False
    if phase1 and phase2 and finite(phase1.get("choice_accuracy")) and finite(phase2.get("choice_accuracy")):
        phase2_gain = float(phase2["choice_accuracy"]) - float(phase1["choice_accuracy"])
        phase2_clause = phase2_gain >= 0.05 or phase2.get("classification") == "PRESENT"
    if narrow_present:
        line = "E-steps"
        explanation = "KILL: at least one TinyStories-only checkpoint is PRESENT."
    elif all_narrow_absent and cold_reaches_present and phase2_clause:
        line = "E-diverse"
        explanation = "CONFIRM: all narrow-only rows are ABSENT, COLD seed 7 reaches PRESENT, and the domain-shift clause passes."
        cold_replica_finals = [
            row for row in rows
            if row["lineage"] == "cold_broad_only" and row["seed"] in (11, 19) and row["checkpoint_kind"] == "final"
        ]
        absent_replica = [row for row in cold_replica_finals if row.get("classification") == "ABSENT"]
        if absent_replica:
            line = "E-gray"
            explanation = "INCONCLUSIVE: E-diverse conjunction passed on seed 7, but an ABSENT COLD replica triggers H026's downgrade rule."
    else:
        line = "E-gray"
        explanation = "INCONCLUSIVE: the E-steps and E-diverse precedence conditions do not both hold."
    return {
        "decision_line": line,
        "interpretation": explanation,
        "precedence": ["E-steps", "E-diverse", "E-gray"],
        "narrow_rows": len(narrow),
        "narrow_present_rows": [row["checkpoint"] for row in narrow_present],
        "all_narrow_absent": all_narrow_absent,
        "cold_seed7_reaches_present": cold_reaches_present,
        "seed7_phase1_final": phase1["checkpoint"] if phase1 else None,
        "seed7_phase2_final": phase2["checkpoint"] if phase2 else None,
        "seed7_domain_shift_accuracy_gain": phase2_gain,
        "seed7_domain_shift_clause_passed": phase2_clause,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 60 H026 Frozen Letters-Copy Circuit Matrix",
        "",
        f"- authority: `{payload['authority']}`",
        "- scope: identity-copy circuit only; this matrix makes no general reasoning claim.",
        f"- inventory: `{len(payload['rows'])}` checkpoint rows, frozen before probing",
        f"- probe byte SHA-256: `{payload['probe']['sha256']}`",
        f"- deterministic anchor: `{ANCHOR_NAME}` reproduced `{payload['anchor']['choice_accuracy']}`",
        f"- decision line: **{payload['verdict']['decision_line']}**",
        f"- interpretation: {payload['verdict']['interpretation']}",
        "",
        "| Lineage | Seed | Step | Broad steps | TinyStories steps | Choice accuracy | MRR | Class | Hash | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        def value(name: str) -> str:
            raw = row.get(name)
            return "n/a" if raw is None else f"{float(raw):.6f}"
        lines.append(
            f"| {row['lineage']} | {row['seed']} | {row['global_step']} | "
            f"{float(row['broad_steps_seen']):.3f} | {float(row['tinystories_steps_seen']):.3f} | "
            f"{value('choice_accuracy')} | {value('choice_mrr')} | {row.get('classification') or 'n/a'} | "
            f"{row['hash_status']} | {row['status']} |"
        )
    lines.extend(["", "## Screening Reads", ""])
    for onset in payload["screening"]["formation_onsets"]:
        lines.append(
            f"- {onset['lineage']} seed {onset['seed']}: first PRESENT checkpoint "
            f"`{onset['first_present_checkpoint']}` at global step `{onset['first_present_global_step']}`."
        )
    if payload["screening"]["destruction_candidates"]:
        lines.extend(["", "### Candidate destruction events", ""])
        for event in payload["screening"]["destruction_candidates"]:
            lines.append(
                f"- {event['lineage']} seed {event['seed']}: drop `{event['accuracy_drop']:.6f}` from "
                f"`{event['from_checkpoint']}` to `{event['to_checkpoint']}`. Screening only."
            )
    else:
        lines.append("- No within-lineage choice-accuracy drop at or above 0.05 was observed.")
    lines.extend(["", "## Hash and failed-row audit", ""])
    for row in payload["rows"]:
        if row["status"] != "scored" or row["hash_status"] != "verified":
            lines.append(
                f"- `{row['checkpoint']}`: status `{row['status']}`, hash `{row['hash_status']}`, reason `{row.get('reason') or 'none'}`."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H026's frozen Stage 60 circuit matrix without training")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "stage60_circuit_matrix.jsonl")
    parser.add_argument("--summary", type=Path, default=RUNS_DIR / "stage60_circuit_matrix.md")
    parser.add_argument("--inventory-out", type=Path, default=RUNS_DIR / "stage60_circuit_inventory.json")
    parser.add_argument("--probe-dir", type=Path, default=RUNS_DIR / "stage60_probe_rows")
    parser.add_argument("--analysis-from", type=Path)
    parser.add_argument("--analysis-out", type=Path)
    parser.add_argument("--analysis-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_from is not None:
        if args.analysis_out is None or args.analysis_summary is None:
            raise ValueError("--analysis-from requires both --analysis-out and --analysis-summary")
        if args.analysis_out.exists() or args.analysis_summary.exists():
            raise FileExistsError("Refusing to overwrite Stage 60 analysis evidence")
        rows = [json.loads(line) for line in args.analysis_from.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 82 or len({str(row["checkpoint"]) for row in rows}) != 82:
            raise ValueError("Stage 60 analysis input must contain the exact 82-row frozen inventory")
        anchor = rows[0]
        if (
            anchor.get("checkpoint_name") != ANCHOR_NAME
            or anchor.get("status") != "scored"
            or round(float(anchor.get("choice_accuracy")), 6) != ANCHOR_ACCURACY
        ):
            raise ValueError("Stage 60 analysis input fails the exact deterministic anchor check")
        add_cold_matches(rows)
        payload = {
            "title": "Stage 60 H026 corrected matrix analysis",
            "authority": "docs/hypotheses/026-diverse-data-circuit-formation.md",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_input": str(args.analysis_from.resolve()),
            "analysis_input_sha256": sha256_file(args.analysis_from),
            "probe": {
                "path": anchor["probe_path"],
                "sha256": anchor["probe_sha256"],
                "lines": PROBE_LINES,
                "seed": PROBE_SEED,
                "max_cases": PROBE_CASES,
            },
            "thresholds": {"present": PRESENT_THRESHOLD, "absent": ABSENT_THRESHOLD, "chance": 0.0625},
            "anchor": {"checkpoint": anchor["checkpoint"], "choice_accuracy": anchor["choice_accuracy"], "expected": ANCHOR_ACCURACY},
            "rows": rows,
            "screening": screening(rows),
            "verdict": verdict(rows),
            "correction": "Terminal phase-2 final is selected as the highest global-step unsuffixed checkpoint, excluding the retained b12501 resume smoke from final-row selection.",
        }
        args.analysis_out.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_markdown(args.analysis_summary, payload)
        print(f"analysis_decision_line={payload['verdict']['decision_line']}")
        print(f"analysis_phase2_gain={payload['verdict']['seed7_domain_shift_accuracy_gain']}")
        print(f"wrote {args.analysis_out}")
        print(f"wrote {args.analysis_summary}")
        return
    outputs = [args.out, args.summary, args.inventory_out]
    if any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"Refusing to overwrite Stage 60 evidence: {existing}")
    if not args.probe.is_file():
        raise FileNotFoundError(f"Frozen H026 probe is missing: {args.probe}")
    probe_sha256 = sha256_file(args.probe)
    inventory = build_inventory()
    args.probe_dir.mkdir(parents=True, exist_ok=False)
    inventory_payload = {
        "title": "Stage 60 H026 frozen checkpoint inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe": str(args.probe.resolve()),
        "probe_sha256": probe_sha256,
        "expected_counts": {"stage58": 68, "stage59": 10, "stage56": 4},
        "rows": inventory,
    }
    args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_out.write_text(json.dumps(inventory_payload, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(inventory):
        row = probe_one(entry, args, probe_sha256)
        rows.append(row)
        print(
            f"row={index + 1}/{len(inventory)} status={row['status']} "
            f"checkpoint={entry['checkpoint_name']} choice_accuracy={row['choice_accuracy']}"
        )
        if index == 0:
            if row["status"] != "scored" or round(float(row["choice_accuracy"]), 6) != ANCHOR_ACCURACY:
                payload = {
                    "title": "Stage 60 H026 frozen letters-copy circuit matrix",
                    "authority": "docs/hypotheses/026-diverse-data-circuit-formation.md",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "probe": {"path": str(args.probe.resolve()), "sha256": probe_sha256},
                    "anchor": {"checkpoint": entry["checkpoint"], "choice_accuracy": row["choice_accuracy"], "expected": ANCHOR_ACCURACY},
                    "rows": rows,
                    "verdict": {"decision_line": "ANCHOR_FAILURE", "interpretation": "No further row is trusted after a failed deterministic re-probe."},
                    "screening": {"formation_onsets": [], "destruction_candidates": []},
                }
                args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                write_markdown(args.summary, payload)
                raise RuntimeError("H026 anchor failed exact 0.194336 deterministic re-probe; stopped before remaining rows")
    add_cold_matches(rows)
    payload = {
        "title": "Stage 60 H026 frozen letters-copy circuit matrix",
        "authority": "docs/hypotheses/026-diverse-data-circuit-formation.md",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe": {
            "path": str(args.probe.resolve()),
            "sha256": probe_sha256,
            "lines": PROBE_LINES,
            "seed": PROBE_SEED,
            "max_cases": PROBE_CASES,
        },
        "thresholds": {"present": PRESENT_THRESHOLD, "absent": ABSENT_THRESHOLD, "chance": 0.0625},
        "anchor": {"checkpoint": inventory[0]["checkpoint"], "choice_accuracy": rows[0]["choice_accuracy"], "expected": ANCHOR_ACCURACY},
        "rows": rows,
        "screening": screening(rows),
        "verdict": verdict(rows),
    }
    args.out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    matrix_payload_path = args.out.with_suffix(".payload.json")
    matrix_payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.summary, payload)
    print(f"anchor_choice_accuracy={rows[0]['choice_accuracy']:.6f}")
    print(f"decision_line={payload['verdict']['decision_line']}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
