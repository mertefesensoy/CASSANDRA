from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = LAB_DIR / "runs"
PRIMARY_SEEDS = (11, 19)
EXPECTED_PARAMETERS = 85_106_721
CONFIRM_COST_MAX = 0.010
CONFIRM_RETENTION_MIN = 1.0
KILL_COST_MIN = 0.020
TEXT8_CHARS = 4_999_936
RETENTION_CHARS = 1_499_904


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    row = rows[0]
    if row.get("status") == "error":
        raise ValueError(f"Training artifact is an error row: {path}")
    return row


def validate_deterministic_result(result: object, expected_chars: int, path: Path) -> None:
    if not isinstance(result, dict):
        raise ValueError(f"Missing deterministic result object: {path}")
    expected_block = 256
    expected_windows, remainder = divmod(expected_chars, expected_block)
    if remainder:
        raise AssertionError("Registered deterministic character count is not block-aligned")
    expected = {
        "chars_evaluated": expected_chars,
        "block_size": expected_block,
        "windows": expected_windows,
    }
    mismatches = {
        key: {"expected": value, "observed": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Deterministic evaluation shape mismatch in {path}: {mismatches}")
    nll = float(result.get("nll", math.nan))
    bits = float(result.get("bits_per_char", math.nan))
    if not math.isfinite(nll) or not math.isfinite(bits):
        raise ValueError(f"Non-finite deterministic evaluation metric in {path}")
    if abs(bits - nll / math.log(2.0)) > 1e-12:
        raise ValueError(f"Deterministic NLL/bits conversion mismatch in {path}")

def only_text8_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("benchmark") != "text8" or payload.get("split") != "test":
        raise ValueError(f"Not a deterministic text8 TEST report: {path}")
    if payload.get("method") != "chunked non-overlapping windows, context resets per window":
        raise ValueError(f"Unexpected text8 evaluation method: {path}")
    models = payload.get("models")
    if not isinstance(models, dict) or len(models) != 1:
        raise ValueError(f"Expected exactly one text8 model in {path}")
    row = next(iter(models.values()))
    result = row.get("result", {})
    validate_deterministic_result(result, TEXT8_CHARS, path)
    return row, payload


def retention_final(path: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("method") != "chunked non-overlapping windows, context resets per window":
        raise ValueError(f"Unexpected retention evaluation method: {path}")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Expected a retention model list in {path}")
    candidates = []
    for row in models:
        checkpoint = Path(str(row.get("checkpoint", "")))
        if f"seed{seed}" not in checkpoint.name or "_step" in checkpoint.stem:
            continue
        if int(row.get("step", -1)) == 20_000:
            candidates.append(row)
    if len(candidates) != 1:
        raise ValueError(f"Expected one unsuffixed seed-{seed} final in {path}, found {len(candidates)}")
    result = candidates[0].get("result", {})
    validate_deterministic_result(result, RETENTION_CHARS, path)
    return candidates[0], payload


def validate_checkpoint_meta(meta: dict[str, Any], seed: int, label: str) -> None:
    if int(meta.get("step", -1)) != 20_000:
        raise ValueError(f"{label} seed {seed} is not a 20,000-step final")
    if int(meta.get("formation_forward_passes", -1)) != 40_000:
        raise ValueError(f"{label} seed {seed} does not have 40,000 formation forward passes")
    if int(meta.get("parameters", -1)) != EXPECTED_PARAMETERS:
        raise ValueError(f"{label} seed {seed} parameter count mismatch")
    checkpoint = Path(str(meta.get("checkpoint", "")))
    if f"seed{seed}" not in checkpoint.name or "_step" in checkpoint.stem:
        raise ValueError(f"{label} seed {seed} does not identify the same-seed unsuffixed final checkpoint")


def training_guard(row: dict[str, Any], seed: int) -> dict[str, Any]:
    expected = {
        "seed": seed,
        "steps": 20_000,
        "parameters": EXPECTED_PARAMETERS,
        "trainable_parameters": EXPECTED_PARAMETERS,
        "comparison_name": "random_full",
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "block_size": 256,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "optimizer": "muon",
        "precision": "fp32",
        "lr_schedule": "cosine",
        "lr_total_steps": 20_000,
        "checkpoint_every": 5_000,
        "checkpoint_keep": 0,
        "eval_mode": "sampled",
        "eval_batches": 16,
    }
    mismatches = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Seed {seed} training convention mismatch: {mismatches}")
    if not bool(row.get("activation_checkpoint")) or row.get("pos_encoding") != "rope":
        raise ValueError(f"Seed {seed} is not the registered RoPE plus activation-checkpoint recipe")
    if not bool(row.get("vocab_chars_override")) or int(row.get("vocab_size", -1)) != 33:
        raise ValueError(f"Seed {seed} is not using the registered 33-character union vocabulary")
    curve = row.get("loss_curve")
    if not isinstance(curve, list):
        raise ValueError(f"Seed {seed} has no sampled loss curve")
    step10 = [item for item in curve if int(item.get("step", -1)) == 10_000]
    step20 = [item for item in curve if int(item.get("step", -1)) == 20_000]
    if len(step10) != 1 or len(step20) != 1:
        raise ValueError(f"Seed {seed} loss curve must contain exactly one step-10k and step-20k row")
    step10_nll = float(step10[0]["val_nll"])
    step20_nll = float(step20[0]["val_nll"])
    final_nll = float(row["val_nll"])
    passed = final_nll < step10_nll
    return {
        "step10000_sampled_broad_val_nll": step10_nll,
        "step20000_sampled_broad_val_nll": step20_nll,
        "final_sampled_broad_val_nll": final_nll,
        "improvement_final_minus_step10000": final_nll - step10_nll,
        "passed": passed,
        "rule": "final sampled broad val NLL must be strictly lower than the arm's own step-10,000 value",
    }


def classify_seed(cost_delta: float, retention_gain: float) -> str:
    if cost_delta <= CONFIRM_COST_MAX and retention_gain >= CONFIRM_RETENTION_MIN:
        return "CONFIRM-side"
    if cost_delta >= KILL_COST_MIN:
        return "KILL-side"
    return "neither"


def partition(classes: list[str]) -> tuple[str, str]:
    confirm = classes.count("CONFIRM-side")
    kill = classes.count("KILL-side")
    if confirm == 1 and kill == 1:
        return "INCONCLUSIVE", "Run the registered seed-7 COLD plus MIXTURE pair; majority decides and remains seed-sensitive."
    if confirm == 2:
        return "E-cheap-rehearsal", "CONFIRM: both paired seeds meet the cost and retention conjunction."
    if kill == 2 or (kill == 1 and confirm == 0):
        return "E-costly-rehearsal", "KILL: both seeds are KILL-side, or one is KILL-side and the other is neither."
    return "E-partial", "GRADED: both costs remain below the KILL line, but the CONFIRM conjunction fails."


def transfer_read(fit: dict[str, Any], measured_mean: float) -> dict[str, Any]:
    predictions = fit["predictions"]
    predicted = float(predictions["predicted_85m_cost_at_w010"]["delta_bits_per_char"])
    if abs(measured_mean) <= 1e-15:
        ratio = 1.0 if abs(predicted) <= 1e-15 else math.inf
    else:
        ratio = abs(predicted) / abs(measured_mean)
    same_direction = predicted == 0.0 or measured_mean == 0.0 or predicted * measured_mean > 0.0
    factor_two = same_direction and 0.5 <= ratio <= 2.0
    correct_costly_side = (predicted >= KILL_COST_MIN) == (measured_mean >= KILL_COST_MIN)
    useful = factor_two and correct_costly_side
    return {
        "predicted_85m_cost_bits_per_char": predicted,
        "measured_mean_cost_bits_per_char": measured_mean,
        "absolute_magnitude_ratio_predicted_over_measured": ratio,
        "same_direction": same_direction,
        "within_factor_two": factor_two,
        "correct_side_of_costly_line": correct_costly_side,
        "read": "USEFUL" if useful else "NOT_DECISION_GRADE",
        "rule": "USEFUL iff prediction is within a factor of two of measured mean, has the same direction, and is on the same side of +0.020.",
    }


def make_payload(runs_dir: Path) -> dict[str, Any]:
    fit = read_json(runs_dir / "stage59_mixing_law_fit.json")
    if not fit.get("primary", {}).get("kind"):
        raise ValueError("Fit artifact has no frozen primary family")
    cold_retention_path = runs_dir / "stage59_cold_b20000_retention_baselines.json"
    seed_rows: list[dict[str, Any]] = []
    failed_guards: list[int] = []
    for seed in PRIMARY_SEEDS:
        training_path = runs_dir / f"stage59_mixture_w10_85m_b20000_seed{seed}.jsonl"
        mixture_text8_path = runs_dir / f"stage59_mixture_w10_85m_b20000_seed{seed}_text8_test.json"
        cold_text8_path = runs_dir / f"stage58_dev_cold_85m_b20000_seed{seed}_text8_test.json"
        mixture_retention_path = runs_dir / f"stage59_mixture_w10_85m_b20000_seed{seed}_retention.json"
        training = read_single_jsonl(training_path)
        guard = training_guard(training, seed)
        if not guard["passed"]:
            failed_guards.append(seed)
        mixture_text8, _ = only_text8_result(mixture_text8_path)
        cold_text8, _ = only_text8_result(cold_text8_path)
        mixture_retention, _ = retention_final(mixture_retention_path, seed)
        cold_retention, _ = retention_final(cold_retention_path, seed)
        validate_checkpoint_meta(mixture_text8.get("meta", {}), seed, "MIXTURE text8")
        validate_checkpoint_meta(cold_text8.get("meta", {}), seed, "COLD text8")
        validate_checkpoint_meta(mixture_retention.get("meta", {}), seed, "MIXTURE retention")
        validate_checkpoint_meta(cold_retention.get("meta", {}), seed, "COLD retention")
        mixture_bits = float(mixture_text8["result"]["bits_per_char"])
        cold_bits = float(cold_text8["result"]["bits_per_char"])
        mixture_retention_bits = float(mixture_retention["result"]["bits_per_char"])
        cold_retention_bits = float(cold_retention["result"]["bits_per_char"])
        cost_delta = mixture_bits - cold_bits
        retention_gain = cold_retention_bits - mixture_retention_bits
        seed_rows.append(
            {
                "seed": seed,
                "cold_text8_test_bits_per_char": cold_bits,
                "mixture_text8_test_bits_per_char": mixture_bits,
                "d_cost_bits_per_char": cost_delta,
                "cold_retention_bits_per_char": cold_retention_bits,
                "mixture_retention_bits_per_char": mixture_retention_bits,
                "r_retention_gain_bits_per_char": retention_gain,
                "classification": classify_seed(cost_delta, retention_gain),
                "instability_guard": guard,
                "sources": {
                    "training": str(training_path),
                    "cold_text8": str(cold_text8_path),
                    "mixture_text8": str(mixture_text8_path),
                    "cold_retention": str(cold_retention_path),
                    "mixture_retention": str(mixture_retention_path),
                },
            }
        )
    if failed_guards:
        raise ValueError(f"Instability guard failed for seeds {failed_guards}; rerun those arms before any verdict")
    classes = [str(row["classification"]) for row in seed_rows]
    verdict, interpretation = partition(classes)
    measured_mean = statistics.mean(float(row["d_cost_bits_per_char"]) for row in seed_rows)
    transfer = transfer_read(fit, measured_mean)
    w_star = fit["predictions"]["w_star"]
    w_star_dose = float(w_star["dose"])
    w_star_cost = float(w_star["predicted_broad_cost_delta_bits"])
    use_w_star = (
        verdict == "E-cheap-rehearsal"
        and transfer["read"] == "USEFUL"
        and w_star_cost <= CONFIRM_COST_MAX
        and abs(w_star_dose - 0.10) <= 0.10
    )
    return {
        "title": "Stage 59 H025 paired deterministic verdict",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md",
        "fit": str(runs_dir / "stage59_mixing_law_fit.json"),
        "fit_primary": fit["primary"],
        "thresholds": {
            "confirm_cost_max_bits_per_char": CONFIRM_COST_MAX,
            "confirm_retention_min_bits_per_char": CONFIRM_RETENTION_MIN,
            "kill_cost_min_bits_per_char": KILL_COST_MIN,
        },
        "deterministic_conventions": {
            "text8_test_chars_evaluated": TEXT8_CHARS,
            "retention_chars_evaluated": RETENTION_CHARS,
            "method": "chunked non-overlapping windows, context resets per window",
        },
        "seeds": seed_rows,
        "mean_measured_cost_bits_per_char": measured_mean,
        "verdict": verdict,
        "interpretation": interpretation,
        "precedence": ["INCONCLUSIVE", "E-cheap-rehearsal", "E-costly-rehearsal", "E-partial"],
        "transfer": transfer,
        "stage60_candidate": {
            "data_gate_fired": verdict == "E-cheap-rehearsal",
            "requires_claude_readback": True,
            "may_launch_before_readback": False,
            "dose_if_authorized": w_star_dose if use_w_star else 0.10,
            "w_star_rule_passed": use_w_star,
            "note": "This calculation does not constitute Claude's required read-back.",
        },
    }


def seed7_majority(primary_classes: list[str], seed7_class: str) -> tuple[str, str, bool]:
    if sorted(primary_classes) != ["CONFIRM-side", "KILL-side"]:
        raise ValueError("Seed-7 majority is only registered after one primary CONFIRM-side and one primary KILL-side seed")
    if seed7_class == "CONFIRM-side":
        return (
            "E-cheap-rehearsal",
            "CONFIRM, seed-sensitive: seed 7 gives CONFIRM-side a two-of-three majority.",
            True,
        )
    if seed7_class == "KILL-side":
        return (
            "E-costly-rehearsal",
            "KILL, seed-sensitive: seed 7 gives KILL-side a two-of-three majority.",
            True,
        )
    return (
        "INCONCLUSIVE-AFTER-ESCALATION",
        "Seed 7 is neither side, so H025's registered three-seed majority does not exist; stop and ask Claude/user rather than inventing a tie-break.",
        False,
    )


def make_seed7_majority_payload(runs_dir: Path) -> dict[str, Any]:
    initial_path = runs_dir / "stage59_verdict.json"
    initial = read_json(initial_path)
    if initial.get("verdict") != "INCONCLUSIVE":
        raise ValueError("Seed-7 majority requires the preserved two-seed INCONCLUSIVE verdict")
    initial_rows = initial.get("seeds")
    if not isinstance(initial_rows, list) or len(initial_rows) != 2:
        raise ValueError("Initial verdict must contain the two primary seed rows")
    primary_classes = [str(row.get("classification")) for row in initial_rows]
    fit = read_json(runs_dir / "stage59_mixing_law_fit.json")
    if not fit.get("primary", {}).get("kind"):
        raise ValueError("Fit artifact has no frozen primary family")

    seed = 7
    mixture_training_path = runs_dir / "stage59_mixture_w10_85m_b20000_seed7.jsonl"
    cold_training_path = runs_dir / "stage59_cold_85m_b20000_seed7.jsonl"
    mixture_text8_path = runs_dir / "stage59_mixture_w10_85m_b20000_seed7_text8_test.json"
    cold_text8_path = runs_dir / "stage59_cold_85m_b20000_seed7_text8_test.json"
    mixture_retention_path = runs_dir / "stage59_mixture_w10_85m_b20000_seed7_retention.json"
    cold_retention_path = runs_dir / "stage59_cold_85m_b20000_seed7_retention.json"

    mixture_guard = training_guard(read_single_jsonl(mixture_training_path), seed)
    cold_guard = training_guard(read_single_jsonl(cold_training_path), seed)
    failed = [label for label, guard in (("MIXTURE", mixture_guard), ("COLD", cold_guard)) if not guard["passed"]]
    if failed:
        raise ValueError(f"Seed-7 escalation instability guard failed for {failed}; rerun before any majority read")

    mixture_text8, _ = only_text8_result(mixture_text8_path)
    cold_text8, _ = only_text8_result(cold_text8_path)
    mixture_retention, _ = retention_final(mixture_retention_path, seed)
    cold_retention, _ = retention_final(cold_retention_path, seed)
    validate_checkpoint_meta(mixture_text8.get("meta", {}), seed, "MIXTURE text8")
    validate_checkpoint_meta(cold_text8.get("meta", {}), seed, "COLD text8")
    validate_checkpoint_meta(mixture_retention.get("meta", {}), seed, "MIXTURE retention")
    validate_checkpoint_meta(cold_retention.get("meta", {}), seed, "COLD retention")

    mixture_bits = float(mixture_text8["result"]["bits_per_char"])
    cold_bits = float(cold_text8["result"]["bits_per_char"])
    mixture_retention_bits = float(mixture_retention["result"]["bits_per_char"])
    cold_retention_bits = float(cold_retention["result"]["bits_per_char"])
    cost_delta = mixture_bits - cold_bits
    retention_gain = cold_retention_bits - mixture_retention_bits
    seed7_class = classify_seed(cost_delta, retention_gain)
    verdict, interpretation, majority_exists = seed7_majority(primary_classes, seed7_class)
    seed7_row = {
        "seed": seed,
        "cold_text8_test_bits_per_char": cold_bits,
        "mixture_text8_test_bits_per_char": mixture_bits,
        "d_cost_bits_per_char": cost_delta,
        "cold_retention_bits_per_char": cold_retention_bits,
        "mixture_retention_bits_per_char": mixture_retention_bits,
        "r_retention_gain_bits_per_char": retention_gain,
        "classification": seed7_class,
        "instability_guard": mixture_guard,
        "cold_instability_guard": cold_guard,
        "sources": {
            "mixture_training": str(mixture_training_path),
            "cold_training": str(cold_training_path),
            "cold_text8": str(cold_text8_path),
            "mixture_text8": str(mixture_text8_path),
            "cold_retention": str(cold_retention_path),
            "mixture_retention": str(mixture_retention_path),
        },
    }
    seed_rows = [*initial_rows, seed7_row]
    measured_mean = statistics.mean(float(row["d_cost_bits_per_char"]) for row in seed_rows)
    transfer = transfer_read(fit, measured_mean)
    w_star = fit["predictions"]["w_star"]
    w_star_dose = float(w_star["dose"])
    w_star_cost = float(w_star["predicted_broad_cost_delta_bits"])
    use_w_star = (
        verdict == "E-cheap-rehearsal"
        and transfer["read"] == "USEFUL"
        and w_star_cost <= CONFIRM_COST_MAX
        and abs(w_star_dose - 0.10) <= 0.10
    )
    return {
        "title": "Stage 59 H025 seed-7 majority verdict",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md",
        "initial_two_seed_verdict": str(initial_path),
        "fit": str(runs_dir / "stage59_mixing_law_fit.json"),
        "fit_primary": fit["primary"],
        "thresholds": initial["thresholds"],
        "deterministic_conventions": initial["deterministic_conventions"],
        "seeds": seed_rows,
        "mean_measured_cost_bits_per_char": measured_mean,
        "verdict": verdict,
        "interpretation": interpretation,
        "seed_sensitive": True,
        "registered_majority_exists": majority_exists,
        "precedence": ["initial two-seed split", "registered seed-7 pair", "two-of-three side majority"],
        "transfer": transfer,
        "stage60_candidate": {
            "data_gate_fired": verdict == "E-cheap-rehearsal" and majority_exists,
            "requires_claude_readback": True,
            "may_launch_before_readback": False,
            "dose_if_authorized": w_star_dose if use_w_star else 0.10,
            "w_star_rule_passed": use_w_star,
            "note": "This calculation does not constitute Claude's required read-back.",
        },
    }

def format_number(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.6f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['title']}",
        "",
        f"- decision line: **{payload['verdict']}**",
        f"- interpretation: {payload['interpretation']}",
        f"- fit primary frozen before Part 2: `{payload['fit_primary']['kind']}`",
        "- registered instability guards: passed",
        "- Stage 60 remains blocked until the required Claude read-back, even if the data gate fired.",
        "",
        "| Seed | COLD text8 TEST bpc | MIXTURE text8 TEST bpc | d | COLD retention bpc | MIXTURE retention bpc | r | Class |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["seeds"]:
        lines.append(
            f"| {row['seed']} | {format_number(row['cold_text8_test_bits_per_char'])} | "
            f"{format_number(row['mixture_text8_test_bits_per_char'])} | {format_number(row['d_cost_bits_per_char'])} | "
            f"{format_number(row['cold_retention_bits_per_char'])} | {format_number(row['mixture_retention_bits_per_char'])} | "
            f"{format_number(row['r_retention_gain_bits_per_char'])} | {row['classification']} |"
        )
    lines.extend(["", "## Instability Guards", ""])
    for row in payload["seeds"]:
        guard = row["instability_guard"]
        lines.append(
            f"- seed {row['seed']}: step-10k `{format_number(guard['step10000_sampled_broad_val_nll'])}`, "
            f"final `{format_number(guard['final_sampled_broad_val_nll'])}`, delta "
            f"`{format_number(guard['improvement_final_minus_step10000'])}`: **PASS**"
        )
        if row.get("cold_instability_guard") is not None:
            cold_guard = row["cold_instability_guard"]
            lines.append(
                f"- seed {row['seed']} COLD contingency: step-10k "
                f"`{format_number(cold_guard['step10000_sampled_broad_val_nll'])}`, final "
                f"`{format_number(cold_guard['final_sampled_broad_val_nll'])}`, delta "
                f"`{format_number(cold_guard['improvement_final_minus_step10000'])}`: **PASS**"
            )
    transfer = payload["transfer"]
    stage60 = payload["stage60_candidate"]
    lines.extend(
        [
            "",
            "## Secondary Mixing-Law Transfer Read",
            "",
            f"- predicted 85M `w=0.10` cost: `{format_number(transfer['predicted_85m_cost_bits_per_char'])}` bits/char",
            f"- measured mean cost: `{format_number(transfer['measured_mean_cost_bits_per_char'])}` bits/char",
            f"- magnitude ratio: `{format_number(transfer['absolute_magnitude_ratio_predicted_over_measured'])}`",
            f"- registered transfer read: **{transfer['read']}**",
            "",
            "## Conditional Stage 60 Gate",
            "",
            f"- Stage 59 data gate fired: **{stage60['data_gate_fired']}**",
            f"- candidate dose if Claude authorizes Stage 60: `{format_number(stage60['dose_if_authorized'])}`",
            f"- fit-derived `w*` rule passed: **{stage60['w_star_rule_passed']}**",
            "- Claude read-back is still required before any Stage 60 action.",
            "",
            "## Sources",
            "",
        ]
    )
    for row in payload["seeds"]:
        lines.append(f"### Seed {row['seed']}")
        lines.append("")
        lines.extend(f"- {label}: `{source}`" for label, source in row["sources"].items())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply H025's Stage 59 paired verdict partition")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--include-seed7", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_stem = "stage59_verdict_seed7_majority" if args.include_seed7 else "stage59_verdict"
    out = args.out or args.runs_dir / f"{default_stem}.json"
    summary = args.summary or args.runs_dir / f"{default_stem}.md"
    if out.exists() or summary.exists():
        raise FileExistsError("Refusing to overwrite existing Stage 59 verdict evidence")
    payload = make_seed7_majority_payload(args.runs_dir) if args.include_seed7 else make_payload(args.runs_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(summary, payload)
    print(f"decision_line={payload['verdict']}")
    print(f"transfer_read={payload['transfer']['read']}")
    print(f"stage60_data_gate_fired={payload['stage60_candidate']['data_gate_fired']}")
    print(f"wrote {out}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
