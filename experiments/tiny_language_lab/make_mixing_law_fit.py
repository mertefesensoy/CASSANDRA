from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = LAB_DIR / "runs" / "stage59_proxy_sweep.jsonl"
DEFAULT_OUT = LAB_DIR / "runs" / "stage59_mixing_law_fit.json"
DEFAULT_SUMMARY = LAB_DIR / "runs" / "stage59_mixing_law_fit.md"
EXPECTED_DOSES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
EXPECTED_SEEDS = (7, 11, 19)
EXPECTED_CONFIG = "stage59_proxy_random_full"
EXPECTED_PARAMETERS = 3_176_481
EXPECTED_STEPS = 5_000
REGISTERED_FIELDS: dict[str, object] = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "block_size": 256,
    "batch_size": 8,
    "grad_accum_steps": 2,
    "effective_batch_size": 16,
    "pos_encoding": "rope",
    "activation_checkpoint": True,
    "precision": "fp32",
    "optimizer": "muon",
    "lr_schedule": "cosine",
    "lr_final_frac": 0.1,
    "lr_total_steps": EXPECTED_STEPS,
    "vocab_size": 33,
    "vocab_chars_override": True,
    "parameters": EXPECTED_PARAMETERS,
    "trainable_parameters": EXPECTED_PARAMETERS,
    "frozen_prior_parameters": 0,
    "steps": EXPECTED_STEPS,
    "formation_steps": EXPECTED_STEPS,
    "formation_forward_passes": EXPECTED_STEPS * 2,
    "eval_mode": "sampled",
    "eval_batches": 16,
    "retention_eval_mode": "sampled",
    "retention_eval_batches": 16,
    "copy_train_marker": "",
}
DOSE_SCALE = 0.50


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
        if row.get("status") == "error":
            raise ValueError(f"Decision input contains an error row at line {line_number}")
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    expected = {(dose, seed) for dose in EXPECTED_DOSES for seed in EXPECTED_SEEDS}
    observed: set[tuple[float, int]] = set()
    parameters: set[int] = set()
    steps: set[int] = set()
    metric_names = (
        "val_nll",
        "val_bits_per_char",
        "retention_val_nll",
        "retention_val_bits_per_char",
    )
    for row in rows:
        if row.get("comparison_name") != EXPECTED_CONFIG:
            raise ValueError(f"Unexpected config in sweep: {row.get('comparison_name')!r}")
        if row.get("mixture_dose") is None:
            raise ValueError("Every sweep row must carry mixture_dose")
        dose = round(float(row["mixture_dose"]), 8)
        seed = int(row["seed"])
        key = (dose, seed)
        if key in observed:
            raise ValueError(f"Duplicate sweep row for dose={dose} seed={seed}")
        observed.add(key)
        parameters.add(int(row["parameters"]))
        steps.add(int(row["steps"]))
        mismatches = {
            name: {"expected": expected_value, "observed": row.get(name)}
            for name, expected_value in REGISTERED_FIELDS.items()
            if row.get(name) != expected_value
        }
        if mismatches:
            raise ValueError(f"Registered proxy surface mismatch for dose={dose} seed={seed}: {mismatches}")
        if int(row.get("retention_eval_seed") or -1) != seed + 59_000:
            raise ValueError(f"Auxiliary evaluation seed mismatch for dose={dose} seed={seed}")
        if not str(row.get("retention_eval_corpus") or ""):
            raise ValueError(f"Missing auxiliary retention corpus for dose={dose} seed={seed}")
        for metric in metric_names:
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Missing or non-finite {metric} for dose={dose} seed={seed}")
        broad_error = abs(float(row["val_bits_per_char"]) - float(row["val_nll"]) / math.log(2.0))
        retention_error = abs(
            float(row["retention_val_bits_per_char"])
            - float(row["retention_val_nll"]) / math.log(2.0)
        )
        if broad_error > 3e-6 or retention_error > 3e-6:
            raise ValueError(f"NLL/bits conversion mismatch for dose={dose} seed={seed}")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Sweep grid mismatch; missing={missing} extra={extra}")
    if parameters != {EXPECTED_PARAMETERS}:
        raise ValueError(f"Proxy parameter count mismatch: {sorted(parameters)}")
    if steps != {EXPECTED_STEPS}:
        raise ValueError(f"Proxy step budget mismatch: {sorted(steps)}")


def mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = {dose: [] for dose in EXPECTED_DOSES}
    for row in rows:
        grouped[round(float(row["mixture_dose"]), 8)].append(row)
    means: list[dict[str, object]] = []
    for dose in EXPECTED_DOSES:
        group = grouped[dose]
        broad = [float(row["val_nll"]) for row in group]
        broad_bits = [float(row["val_bits_per_char"]) for row in group]
        retention = [float(row["retention_val_nll"]) for row in group]
        retention_bits = [float(row["retention_val_bits_per_char"]) for row in group]
        means.append(
            {
                "dose": dose,
                "seeds": [int(row["seed"]) for row in group],
                "broad_val_nll_mean": statistics.mean(broad),
                "broad_val_nll_stdev": statistics.stdev(broad),
                "broad_val_bits_mean": statistics.mean(broad_bits),
                "retention_val_nll_mean": statistics.mean(retention),
                "retention_val_nll_stdev": statistics.stdev(retention),
                "retention_val_bits_mean": statistics.mean(retention_bits),
            }
        )
    return means


def linear_coefficients(z: list[float], y: list[float]) -> tuple[float, float, float]:
    z_mean = statistics.mean(z)
    y_mean = statistics.mean(y)
    denominator = sum((value - z_mean) ** 2 for value in z)
    if denominator <= 1e-24:
        raise ValueError("Degenerate curve basis")
    amplitude = sum((value - z_mean) * (target - y_mean) for value, target in zip(z, y)) / denominator
    intercept = y_mean - amplitude * z_mean
    sse = sum((target - (intercept + amplitude * value)) ** 2 for value, target in zip(z, y))
    return intercept, amplitude, sse


def exponential_basis(w: float, exponent: float) -> float:
    if abs(exponent) < 1e-9:
        return w / DOSE_SCALE
    return math.expm1(exponent * w) / math.expm1(exponent * DOSE_SCALE)


def power_basis(w: float, exponent: float) -> float:
    return (w / DOSE_SCALE) ** exponent


def fit_family(kind: str, x: list[float], y: list[float]) -> dict[str, object]:
    if kind == "exponential":
        lower, upper = -20.0, 20.0
        basis: Callable[[float, float], float] = exponential_basis
        function = "L(w) = intercept + amplitude * expm1(exponent*w) / expm1(exponent*0.5)"
    elif kind == "power":
        lower, upper = 0.05, 5.0
        basis = power_basis
        function = "L(w) = intercept + amplitude * (w/0.5)^exponent"
    else:
        raise ValueError(f"Unknown fit family: {kind}")

    def coefficients(exponent: float) -> tuple[float, float, float]:
        return linear_coefficients([basis(value, exponent) for value in x], y)

    grid_count = 4001
    grid = [lower + (upper - lower) * index / (grid_count - 1) for index in range(grid_count)]
    scored = [(coefficients(exponent)[2], exponent) for exponent in grid]
    best_index = min(range(len(scored)), key=lambda index: scored[index][0])
    best_exponent = scored[best_index][1]

    if 0 < best_index < len(grid) - 1:
        left = grid[best_index - 1]
        right = grid[best_index + 1]
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden * (right - left)
        x2 = left + golden * (right - left)
        f1 = coefficients(x1)[2]
        f2 = coefficients(x2)[2]
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden * (right - left)
                f1 = coefficients(x1)[2]
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden * (right - left)
                f2 = coefficients(x2)[2]
        best_exponent = (left + right) / 2.0

    intercept, amplitude, sse = coefficients(best_exponent)
    predictions = [intercept + amplitude * basis(value, best_exponent) for value in x]
    residuals = [target - prediction for target, prediction in zip(y, predictions)]
    rmse = math.sqrt(sse / len(x))
    target_mean = statistics.mean(y)
    total_squares = sum((target - target_mean) ** 2 for target in y)
    r_squared = 1.0 - sse / total_squares if total_squares > 0 else None
    return {
        "kind": kind,
        "function": function,
        "parameters": {
            "intercept": intercept,
            "amplitude": amplitude,
            "exponent": best_exponent,
            "dose_scale": DOSE_SCALE,
        },
        "sse": sse,
        "rmse": rmse,
        "r_squared": r_squared,
        "predictions": predictions,
        "residuals": residuals,
        "search_interval": [lower, upper],
    }


def predict(fit: dict[str, object], w: float) -> float:
    parameters = fit["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("Fit parameters must be a dictionary")
    intercept = float(parameters["intercept"])
    amplitude = float(parameters["amplitude"])
    exponent = float(parameters["exponent"])
    if fit["kind"] == "exponential":
        basis = exponential_basis(w, exponent)
    elif fit["kind"] == "power":
        basis = power_basis(w, exponent)
    else:
        raise ValueError(f"Unknown fit kind: {fit['kind']}")
    return intercept + amplitude * basis


def add_leave_one_out(fit: dict[str, object], x: list[float], y: list[float]) -> None:
    residuals: list[dict[str, float]] = []
    for heldout in range(len(x)):
        train_x = [value for index, value in enumerate(x) if index != heldout]
        train_y = [value for index, value in enumerate(y) if index != heldout]
        heldout_fit = fit_family(str(fit["kind"]), train_x, train_y)
        prediction = predict(heldout_fit, x[heldout])
        residuals.append(
            {
                "dose": x[heldout],
                "observed": y[heldout],
                "predicted": prediction,
                "residual": y[heldout] - prediction,
            }
        )
    fit["leave_one_out"] = {
        "rmse": math.sqrt(statistics.mean(item["residual"] ** 2 for item in residuals)),
        "residuals": residuals,
    }


def interpolate_retention(means: list[dict[str, object]], w: float) -> float:
    points = [(float(row["dose"]), float(row["retention_val_bits_mean"])) for row in means]
    if w <= points[0][0]:
        return points[0][1]
    if w >= points[-1][0]:
        return points[-1][1]
    for (left_w, left_value), (right_w, right_value) in zip(points, points[1:]):
        if left_w <= w <= right_w:
            fraction = (w - left_w) / (right_w - left_w)
            return left_value + fraction * (right_value - left_value)
    raise AssertionError("Retention interpolation failed")


def choose_w_star(
    primary_fit: dict[str, object],
    means: list[dict[str, object]],
    retention_bound: float,
) -> dict[str, float]:
    grid_count = 10001
    candidates: list[tuple[float, float, float]] = []
    for index in range(grid_count):
        w = DOSE_SCALE * index / (grid_count - 1)
        retention = interpolate_retention(means, w)
        if retention <= retention_bound:
            candidates.append((predict(primary_fit, w), w, retention))
    if not candidates:
        raise ValueError("No dose satisfies the provisional retention bound")
    broad_nll, w_star, retention = min(candidates)
    return {
        "dose": w_star,
        "predicted_broad_val_nll": broad_nll,
        "predicted_broad_val_bits": broad_nll / math.log(2.0),
        "interpolated_retention_bits": retention,
        "grid_resolution": DOSE_SCALE / (grid_count - 1),
    }


def make_payload(input_path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    validate_rows(rows)
    means = mean_rows(rows)
    x = [float(row["dose"]) for row in means]
    y = [float(row["broad_val_nll_mean"]) for row in means]
    fits = {kind: fit_family(kind, x, y) for kind in ("exponential", "power")}
    for fit in fits.values():
        add_leave_one_out(fit, x, y)
        fit["residual_table"] = [
            {
                "dose": dose,
                "observed": observed,
                "predicted": prediction,
                "residual": residual,
            }
            for dose, observed, prediction, residual in zip(
                x,
                y,
                fit["predictions"],
                fit["residuals"],
            )
        ]

    exponential_cv = float(fits["exponential"]["leave_one_out"]["rmse"])
    power_cv = float(fits["power"]["leave_one_out"]["rmse"])
    primary_kind = "exponential" if exponential_cv <= power_cv else "power"
    primary_fit = fits[primary_kind]
    predicted_w0 = predict(primary_fit, 0.0)
    predicted_w10 = predict(primary_fit, 0.10)
    predicted_delta_nll = predicted_w10 - predicted_w0
    retention_w30 = next(float(row["retention_val_bits_mean"]) for row in means if row["dose"] == 0.30)
    retention_bound = retention_w30 + 0.5
    w_star = choose_w_star(primary_fit, means, retention_bound)
    predicted_w_star_delta_nll = float(w_star["predicted_broad_val_nll"]) - predicted_w0
    w_star["predicted_broad_cost_delta_nll"] = predicted_w_star_delta_nll
    w_star["predicted_broad_cost_delta_bits"] = predicted_w_star_delta_nll / math.log(2.0)

    return {
        "title": "Stage 59 H025 proxy mixing-law fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "expected_doses": list(EXPECTED_DOSES),
        "expected_seeds": list(EXPECTED_SEEDS),
        "parameters": int(rows[0]["parameters"]),
        "steps_per_run": int(rows[0]["steps"]),
        "dose_means": means,
        "fits": fits,
        "primary": {
            "kind": primary_kind,
            "selection_rule": "Lower leave-one-dose-out RMSE; exponential wins an exact tie.",
            "paper_alignment": (
                "The exponential family is the two-domain reduction of Ye et al. Eq. 7. "
                "The power family is H025's registered robustness alternative."
            ),
        },
        "predictions": {
            "predicted_85m_cost_at_w010": {
                "assumption": "Direct transfer of the proxy fitted delta to 85M; descriptive until Part 2.",
                "predicted_proxy_nll_w000": predicted_w0,
                "predicted_proxy_nll_w010": predicted_w10,
                "delta_nll": predicted_delta_nll,
                "delta_bits_per_char": predicted_delta_nll / math.log(2.0),
            },
            "provisional_retention_bound": {
                "definition": "Mean proxy retention bits at w=0.30 plus 0.5 bits/char.",
                "w030_mean_retention_bits": retention_w30,
                "bound_bits_per_char": retention_bound,
                "role": "Part 1 reporting only; it does not feed the Part 2 verdict.",
            },
            "w_star": w_star,
        },
        "limitations": [
            "Six dose means cannot identify a universal functional family.",
            "The 85M prediction directly transfers a 3.2M proxy delta and is not nested across model size.",
            "Retention between measured doses is piecewise-linearly interpolated.",
            "In-run sampled validation metrics are monitoring-grade and are used here only because H025 registers them for the proxy fit.",
        ],
    }


def format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_summary(path: Path, payload: dict[str, object]) -> None:
    primary = payload["primary"]
    predictions = payload["predictions"]
    cost = predictions["predicted_85m_cost_at_w010"]
    bound = predictions["provisional_retention_bound"]
    w_star = predictions["w_star"]
    lines = [
        "# Stage 59 H025 Proxy Mixing-Law Fit",
        "",
        f"- input: `{payload['input']}`",
        f"- input SHA-256: `{payload['input_sha256']}`",
        f"- rows: {payload['rows']} (6 doses x 3 seeds)",
        f"- proxy parameters: {int(payload['parameters']):,}",
        f"- steps per run: {int(payload['steps_per_run']):,}",
        f"- registered primary: **{primary['kind']}**",
        f"- primary rule: {primary['selection_rule']}",
        "",
        "## Pre-registered Predictions Before Part 2",
        "",
        f"- predicted 85M broad cost at `w = 0.10`: **{format_float(cost['delta_bits_per_char'])} bits/char** ({format_float(cost['delta_nll'])} NLL)",
        f"- provisional retention bound: **{format_float(bound['bound_bits_per_char'])} bits/char** (`w = 0.30` mean plus `0.5`)",
        f"- predicted `w*`: **{format_float(w_star['dose'])}**",
        f"- predicted broad toll at `w*`: **{format_float(w_star['predicted_broad_cost_delta_bits'])} bits/char**",
        "",
        "The 85M number is a direct transfer of the proxy-fitted delta. It is frozen here before Part 2 and remains descriptive until measured.",
        "",
        "## Dose Means",
        "",
        "| w | Broad val NLL | Broad bits/char | Retention val NLL | Retention bits/char |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["dose_means"]:
        lines.append(
            f"| {float(row['dose']):.2f} | {format_float(row['broad_val_nll_mean'])} | "
            f"{format_float(row['broad_val_bits_mean'])} | {format_float(row['retention_val_nll_mean'])} | "
            f"{format_float(row['retention_val_bits_mean'])} |"
        )
    lines.extend(["", "## Fits and Residuals", ""])
    for kind in ("exponential", "power"):
        fit = payload["fits"][kind]
        params = fit["parameters"]
        lines.extend(
            [
                f"### {kind.title()}",
                "",
                f"- function: `{fit['function']}`",
                f"- intercept: {format_float(params['intercept'])}",
                f"- amplitude: {format_float(params['amplitude'])}",
                f"- exponent: {format_float(params['exponent'])}",
                f"- in-sample RMSE: {format_float(fit['rmse'])}",
                f"- leave-one-dose-out RMSE: {format_float(fit['leave_one_out']['rmse'])}",
                f"- R-squared: {format_float(fit['r_squared'])}",
                "",
                "| w | Observed broad NLL | Predicted | Residual |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in fit["residual_table"]:
            lines.append(
                f"| {float(row['dose']):.2f} | {format_float(row['observed'])} | "
                f"{format_float(row['predicted'])} | {format_float(row['residual'])} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Stage 59 proxy data-mixing laws")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    payload = make_payload(args.input, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    print(f"primary={payload['primary']['kind']}")
    print(
        "predicted_85m_w010_cost_bits="
        f"{payload['predictions']['predicted_85m_cost_at_w010']['delta_bits_per_char']:.6f}"
    )
    print(f"predicted_w_star={payload['predictions']['w_star']['dose']:.6f}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()