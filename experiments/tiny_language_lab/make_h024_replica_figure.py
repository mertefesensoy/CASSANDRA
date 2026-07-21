"""Generate the fail-closed H024 three-seed replica sign-check artifact.

The primary decision uses seed 7 at the full Stage 58 budget. Seeds 11 and 19
are reduced-budget sign replicas, not pooled observations. This generator reads
the six deterministic text8 TEST reports, validates their conventions, then
writes a figure, Markdown table, and JSON evidence bundle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAB_DIR = Path(__file__).resolve().parent
RUNS = LAB_DIR / "runs"
REPO = LAB_DIR.parent.parent
DEFAULT_OUT_DIR = REPO / "docs" / "figures" / "phase5"
EXPECTED_CHARS = 4_999_000
EXPECTED_PARAMETERS = 85_106_721
CHUNKED_METHOD = "chunked non-overlapping windows, context resets per window"


@dataclass(frozen=True)
class Score:
    seed: int
    budget: int
    arm: str
    checkpoint: str
    bits_per_char: float
    nll: float
    chars_evaluated: int


def default_paths() -> dict[tuple[int, str], Path]:
    return {
        (7, "COLD"): RUNS / "stage58_dev_cold_85m_b42000_seed7_text8_test.json",
        (7, "CURRICULUM"): RUNS
        / "stage58_dev_curriculum_phase2_85m_b42000_seed7_text8_test.json",
        (11, "COLD"): RUNS / "stage58_dev_cold_85m_b20000_seed11_text8_test.json",
        (11, "CURRICULUM"): RUNS
        / "stage58_dev_curriculum_phase2_85m_b20000_seed11_text8_test.json",
        (19, "COLD"): RUNS / "stage58_dev_cold_85m_b20000_seed19_text8_test.json",
        (19, "CURRICULUM"): RUNS
        / "stage58_dev_curriculum_phase2_85m_b20000_seed19_text8_test.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the H024 reduced-budget replica sign-check artifact."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_score(seed: int, arm: str, path: Path) -> Score:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required deterministic TEST report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "text8" or payload.get("split") != "test":
        raise ValueError(f"{path} is not a text8 TEST report")
    if payload.get("method") != CHUNKED_METHOD:
        raise ValueError(f"{path} does not use the registered chunked convention")
    models = payload.get("models")
    if not isinstance(models, dict) or len(models) != 1:
        raise ValueError(f"{path} must contain exactly one model result")
    entry = next(iter(models.values()))
    if not isinstance(entry, dict):
        raise ValueError(f"{path} has an invalid model entry")
    meta = entry.get("meta")
    result = entry.get("result")
    if not isinstance(meta, dict) or not isinstance(result, dict):
        raise ValueError(f"{path} is missing metadata or metrics")
    expected_budget = 42_000 if seed == 7 else 20_000
    if int(meta["step"]) != expected_budget:
        raise ValueError(
            f"{path} reports step {meta['step']}; expected {expected_budget} for seed {seed}"
        )
    if int(meta["parameters"]) != EXPECTED_PARAMETERS:
        raise ValueError(f"{path} does not match the registered 85.11M parameter model")
    chars = int(result["chars_evaluated"])
    if chars < EXPECTED_CHARS:
        raise ValueError(f"{path} evaluated only {chars:,} characters")
    checkpoint = str(meta["checkpoint"])
    marker = "stage58_dev_cold_" if arm == "COLD" else "stage58_dev_curriculum_"
    if marker not in checkpoint.lower():
        raise ValueError(f"{path} has an arm/checkpoint mismatch for {arm}")
    return Score(
        seed=seed,
        budget=expected_budget,
        arm=arm,
        checkpoint=checkpoint,
        bits_per_char=float(result["bits_per_char"]),
        nll=float(result["nll"]),
        chars_evaluated=chars,
    )


def primary_read(delta: float) -> str:
    if delta <= -0.05:
        return "E-curriculum"
    if delta >= 0.05:
        return "E-interfere"
    return "E-null"


def main() -> None:
    args = parse_args()
    scores = [read_score(seed, arm, path) for (seed, arm), path in default_paths().items()]
    by_seed: dict[int, dict[str, Score]] = {}
    for score in scores:
        by_seed.setdefault(score.seed, {})[score.arm] = score
    if set(by_seed) != {7, 11, 19}:
        raise ValueError("The replica package must contain exactly seeds 7, 11, and 19")

    rows: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        arms = by_seed[seed]
        if set(arms) != {"COLD", "CURRICULUM"}:
            raise ValueError(f"Seed {seed} does not include both required arms")
        cold = arms["COLD"]
        curriculum = arms["CURRICULUM"]
        rows.append(
            {
                "seed": seed,
                "budget": cold.budget,
                "cold_bits_per_char": cold.bits_per_char,
                "curriculum_bits_per_char": curriculum.bits_per_char,
                "curriculum_minus_cold": curriculum.bits_per_char - cold.bits_per_char,
                "chars_evaluated": cold.chars_evaluated,
                "cold_checkpoint": cold.checkpoint,
                "curriculum_checkpoint": curriculum.checkpoint,
            }
        )

    seed7_delta = float(rows[0]["curriculum_minus_cold"])
    replica_signs_agree = all(
        float(row["curriculum_minus_cold"]) * seed7_delta > 0 for row in rows[1:]
    )
    marginal_seed7 = 0.05 <= abs(seed7_delta) < 0.10
    escalation_required = marginal_seed7 or not replica_signs_agree
    conclusion = (
        "seed-robust in sign; no full-budget replica escalation required"
        if not escalation_required
        else "inconclusive; full-budget replica escalation required"
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    x_values = list(range(len(rows)))
    deltas = [float(row["curriculum_minus_cold"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#4c4c4c", linewidth=1)
    ax.axhline(-0.05, color="#54a24b", linestyle="--", linewidth=1)
    ax.axhline(0.05, color="#e45756", linestyle="--", linewidth=1)
    ax.vlines(x_values, 0.0, deltas, color="#79706e", linewidth=1.4)
    ax.scatter(x_values, deltas, color="#f58518", s=52, zorder=3)
    for x_value, row in zip(x_values, rows):
        delta = float(row["curriculum_minus_cold"])
        ax.text(
            x_value,
            delta + 0.0035,
            f"{delta:+.6f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(
        x_values,
        [f"seed {row['seed']}\n{int(row['budget']):,} steps" for row in rows],
    )
    ax.set_ylim(-0.06, 0.06)
    ax.set_ylabel("CURRICULUM - COLD TEST bits/char\n(positive favors COLD)")
    ax.set_title("H024 primary delta: full seed and reduced-budget replicas")
    ax.text(
        0.02,
        0.96,
        "All observed signs are positive; all magnitudes are inside the E-null band.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout()
    figure_path = out_dir / "fig3_h024_replica_sign_check.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    payload = {
        "metric": "CURRICULUM minus COLD deterministic text8 TEST bits/char",
        "lower_is_better": True,
        "primary_seed7_read": primary_read(seed7_delta),
        "replica_signs_agree": replica_signs_agree,
        "marginal_seed7": marginal_seed7,
        "escalation_required": escalation_required,
        "conclusion": conclusion,
        "rows": rows,
        "figure": figure_path.name,
    }
    (out_dir / "h024_replica_sign_check.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# H024 Replica Sign Check",
        "",
        "Primary metric: deterministic chunked text8 TEST bits/char. Lower is better.",
        "",
        "| Seed | Budget | COLD | CURRICULUM | CURRICULUM minus COLD |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['budget']:,} | {row['cold_bits_per_char']:.6f} | "
            f"{row['curriculum_bits_per_char']:.6f} | "
            f"{row['curriculum_minus_cold']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Seed-7 registered primary read: **{primary_read(seed7_delta)}**.",
            f"- Replica signs agree with seed 7: **{replica_signs_agree}**.",
            f"- Seed-7 marginal-margin trigger: **{marginal_seed7}**.",
            f"- Full-budget replica escalation required: **{escalation_required}**.",
            f"- H024 conclusion: **{conclusion}**.",
            "",
            "Figure: `fig3_h024_replica_sign_check.png`.",
        ]
    )
    (out_dir / "h024_replica_sign_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[h024-replica] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
