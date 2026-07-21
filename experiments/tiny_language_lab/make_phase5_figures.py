"""Generate the Phase 5 Stage 58 comparison figures from evaluation reports.

The generator consumes the deterministic text8 TEST reports and the
TinyStories checkpoint-series retention reports produced by Stage 58. It fails
closed when an expected report is missing or does not use the registered
evaluation conventions, then writes PNG figures, a Markdown summary, and the
JSON values plotted by those figures.

Usage from the repository root, after all three Stage 58 arms are evaluated:

  python .\\experiments\\tiny_language_lab\\make_phase5_figures.py
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
DEFAULT_FIG_DIR = REPO / "docs" / "figures" / "phase5"
EXPECTED_TEXT8_CHARS = 4_999_000
EXPECTED_RETENTION_CHARS = 1_499_904
EXPECTED_RETENTION_CORPUS = "tinystories_char_shards_500mb"
EXPECTED_PRIMARY_STEPS = 42_000
EXPECTED_PARAMETERS = 85_106_721

ARM_ORDER = ("COLD", "CURRICULUM", "MIXTURE")
ARM_COLORS = {
    "COLD": "#4c78a8",
    "CURRICULUM": "#f58518",
    "MIXTURE": "#54a24b",
}
ARM_CHECKPOINT_MARKERS = {
    "COLD": "stage58_dev_cold_",
    "CURRICULUM": "stage58_dev_curriculum_",
    "MIXTURE": "stage58_dev_mixture_",
}


@dataclass(frozen=True)
class Text8Result:
    arm: str
    checkpoint: str
    bits_per_char: float
    nll: float
    chars_evaluated: int
    step: int
    parameters: int = EXPECTED_PARAMETERS


@dataclass(frozen=True)
class RetentionPoint:
    arm: str
    checkpoint: str
    step: int
    bits_per_char: float
    nll: float
    chars_evaluated: int


def default_paths() -> dict[str, Path]:
    return {
        "cold_text8": RUNS / "stage58_dev_cold_85m_b42000_seed7_text8_test.json",
        "curriculum_text8": RUNS
        / "stage58_dev_curriculum_phase2_85m_b42000_seed7_text8_test.json",
        "mixture_text8": RUNS / "stage58_dev_mixture_85m_b42000_seed7_text8_test.json",
        "cold_retention": RUNS / "stage58_dev_cold_retention.json",
        "curriculum_retention": RUNS / "stage58_dev_curriculum_retention.json",
        "mixture_retention": RUNS / "stage58_dev_mixture_retention.json",
        # fig4 inputs: the Phase 5 arc and the matched-budget recipe contrast.
        "stage56_50k_seed7": RUNS / "stage56_broadchar_85m_b50000_seed7_text8_test.json",
        "stage56_20k_seed11": RUNS
        / "stage56_broadchar_85m_b20000_seed11_text8_test.json",
        "stage56_20k_seed19": RUNS
        / "stage56_broadchar_85m_b20000_seed19_text8_test.json",
        "stage58_cold_20k_seed11": RUNS
        / "stage58_dev_cold_85m_b20000_seed11_text8_test.json",
        "stage58_cold_20k_seed19": RUNS
        / "stage58_dev_cold_85m_b20000_seed19_text8_test.json",
        "phase4_figures_data": REPO / "docs" / "figures" / "phase4" / "figures_data.json",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Generate Stage 58 primary and retention comparison figures."
    )
    parser.add_argument("--cold-text8", type=Path, default=defaults["cold_text8"])
    parser.add_argument(
        "--curriculum-text8", type=Path, default=defaults["curriculum_text8"]
    )
    parser.add_argument("--mixture-text8", type=Path, default=defaults["mixture_text8"])
    parser.add_argument(
        "--cold-retention", type=Path, default=defaults["cold_retention"]
    )
    parser.add_argument(
        "--curriculum-retention", type=Path, default=defaults["curriculum_retention"]
    )
    parser.add_argument(
        "--mixture-retention", type=Path, default=defaults["mixture_retention"]
    )
    parser.add_argument(
        "--stage56-50k-seed7", type=Path, default=defaults["stage56_50k_seed7"]
    )
    parser.add_argument(
        "--stage56-20k-seed11", type=Path, default=defaults["stage56_20k_seed11"]
    )
    parser.add_argument(
        "--stage56-20k-seed19", type=Path, default=defaults["stage56_20k_seed19"]
    )
    parser.add_argument(
        "--stage58-cold-20k-seed11",
        type=Path,
        default=defaults["stage58_cold_20k_seed11"],
    )
    parser.add_argument(
        "--stage58-cold-20k-seed19",
        type=Path,
        default=defaults["stage58_cold_20k_seed19"],
    )
    parser.add_argument(
        "--phase4-figures-data", type=Path, default=defaults["phase4_figures_data"]
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required evaluation report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def require_arm_checkpoint(arm: str, checkpoint: str, report_path: Path) -> None:
    marker = ARM_CHECKPOINT_MARKERS[arm]
    if marker not in checkpoint.lower():
        raise ValueError(
            f"{report_path} labels a checkpoint as {arm}, but its path does not contain "
            f"the expected marker {marker!r}"
        )


def load_text8_result(arm: str, path: Path) -> Text8Result:
    payload = load_json(path)
    if payload.get("benchmark") != "text8" or payload.get("split") != "test":
        raise ValueError(f"{path} is not a deterministic text8 TEST report")
    if payload.get("method") != "chunked non-overlapping windows, context resets per window":
        raise ValueError(f"{path} does not use the registered chunked convention")
    models = payload.get("models")
    if not isinstance(models, dict) or len(models) != 1:
        raise ValueError(f"{path} must contain exactly one evaluated checkpoint")
    _, entry = next(iter(models.items()))
    if not isinstance(entry, dict):
        raise ValueError(f"{path} has an invalid model entry")
    meta = entry.get("meta")
    result = entry.get("result")
    if not isinstance(meta, dict) or not isinstance(result, dict):
        raise ValueError(f"{path} is missing model metadata or metric results")
    chars = int(result["chars_evaluated"])
    if chars < EXPECTED_TEXT8_CHARS:
        raise ValueError(
            f"{path} evaluated only {chars:,} text8 characters; expected at least "
            f"{EXPECTED_TEXT8_CHARS:,} for the full chunked TEST split"
        )
    checkpoint = str(meta["checkpoint"])
    require_arm_checkpoint(arm, checkpoint, path)
    step = int(meta["step"])
    if step != EXPECTED_PRIMARY_STEPS:
        raise ValueError(
            f"{path} reports step {step:,}; expected the {EXPECTED_PRIMARY_STEPS:,}-step "
            "Stage 58 final checkpoint"
        )
    parameters = int(meta["parameters"])
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(
            f"{path} reports {parameters:,} parameters; expected {EXPECTED_PARAMETERS:,}"
        )
    return Text8Result(
        arm=arm,
        checkpoint=checkpoint,
        bits_per_char=float(result["bits_per_char"]),
        nll=float(result["nll"]),
        chars_evaluated=chars,
        step=step,
    )


def load_retention_points(arm: str, path: Path) -> list[RetentionPoint]:
    payload = load_json(path)
    if payload.get("suite") != "stage58_tinystories_retention":
        raise ValueError(f"{path} is not a Stage 58 retention report")
    corpus = str(payload.get("corpus", "")).replace("\\", "/")
    if EXPECTED_RETENTION_CORPUS not in corpus:
        raise ValueError(
            f"{path} does not score the full Stage 58 TinyStories validation corpus"
        )
    if payload.get("method") != "chunked non-overlapping windows, context resets per window":
        raise ValueError(f"{path} does not use the registered chunked convention")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path} does not contain a checkpoint series")
    points: list[RetentionPoint] = []
    for row in models:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("result"), dict)
            or not isinstance(row.get("meta"), dict)
        ):
            raise ValueError(f"{path} contains an invalid retention row")
        result = row["result"]
        meta = row["meta"]
        checkpoint = str(row["checkpoint"])
        require_arm_checkpoint(arm, checkpoint, path)
        parameters = int(meta["parameters"])
        if parameters != EXPECTED_PARAMETERS:
            raise ValueError(
                f"{path} reports {parameters:,} parameters; expected {EXPECTED_PARAMETERS:,}"
            )
        chars = int(result["chars_evaluated"])
        if chars < EXPECTED_RETENTION_CHARS:
            raise ValueError(
                f"{path} evaluated only {chars:,} TinyStories characters; expected at "
                f"least {EXPECTED_RETENTION_CHARS:,} for the registered retention sample"
            )
        points.append(
            RetentionPoint(
                arm=arm,
                checkpoint=checkpoint,
                step=int(row["step"]),
                bits_per_char=float(result["bits_per_char"]),
                nll=float(result["nll"]),
                chars_evaluated=chars,
            )
        )
    return sorted(points, key=lambda point: (point.step, point.checkpoint))


def load_text8_point(path: Path, expected_step: int) -> Text8Result:
    """Load a text8 TEST score from any stage without the Stage 58 arm checks.

    Enforces the registered chunked TEST convention and the 85.11M parameter
    count, but accepts any checkpoint name and the caller's expected step, so
    Stage 56 (Recipe v1) and reduced-budget Stage 58 reports can feed fig4.
    """
    payload = load_json(path)
    if payload.get("benchmark") != "text8" or payload.get("split") != "test":
        raise ValueError(f"{path} is not a deterministic text8 TEST report")
    if payload.get("method") != "chunked non-overlapping windows, context resets per window":
        raise ValueError(f"{path} does not use the registered chunked convention")
    models = payload.get("models")
    if not isinstance(models, dict) or len(models) != 1:
        raise ValueError(f"{path} must contain exactly one evaluated checkpoint")
    _, entry = next(iter(models.items()))
    meta = entry.get("meta")
    result = entry.get("result")
    if not isinstance(meta, dict) or not isinstance(result, dict):
        raise ValueError(f"{path} is missing model metadata or metric results")
    chars = int(result["chars_evaluated"])
    if chars < EXPECTED_TEXT8_CHARS:
        raise ValueError(
            f"{path} evaluated only {chars:,} text8 characters; expected at least "
            f"{EXPECTED_TEXT8_CHARS:,} for the full chunked TEST split"
        )
    step = int(meta["step"])
    if step != expected_step:
        raise ValueError(f"{path} reports step {step:,}; expected {expected_step:,}")
    parameters = int(meta["parameters"])
    # Recipe v1 (Stage 56) used the 27-character broad vocabulary, so its head
    # and embedding are 9,222 parameters smaller than Recipe v2's 33-character
    # union vocabulary. Accept that 0.011 percent capacity difference but fail
    # on anything that is not the same 85.11M-class model.
    if abs(parameters - EXPECTED_PARAMETERS) / EXPECTED_PARAMETERS > 0.005:
        raise ValueError(
            f"{path} reports {parameters:,} parameters; expected within 0.5% of "
            f"{EXPECTED_PARAMETERS:,}"
        )
    return Text8Result(
        arm=path.stem,
        checkpoint=str(meta["checkpoint"]),
        bits_per_char=float(result["bits_per_char"]),
        nll=float(result["nll"]),
        chars_evaluated=chars,
        step=step,
        parameters=parameters,
    )


def load_flagship_anchor(path: Path) -> float:
    """Read the Phase 4 flagship zero-shot text8 anchor from its audit file."""
    payload = load_json(path)
    anchor = payload.get("text8_flagship_bits_per_char")
    if not isinstance(anchor, (int, float)):
        raise ValueError(
            f"{path} does not carry text8_flagship_bits_per_char; cannot anchor fig4"
        )
    return float(anchor)


def verdict_for_primary(delta: float) -> str:
    if delta <= -0.05:
        return "E-curriculum"
    if delta >= 0.05:
        return "E-interfere"
    return "E-null"


def plot_text8(results: list[Text8Result], out_dir: Path, delta: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    arms = [result.arm for result in results]
    values = [result.bits_per_char for result in results]
    bars = ax.bar(arms, values, color=[ARM_COLORS[arm] for arm in arms], width=0.62)
    lower = min(values) - 0.06
    upper = max(values) + 0.10
    ax.set_ylim(lower, upper)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    cold = results[0].bits_per_char
    ax.axhline(cold - 0.05, color="#54a24b", linestyle="--", linewidth=1)
    ax.axhline(cold + 0.05, color="#e45756", linestyle="--", linewidth=1)
    ax.text(2.42, cold - 0.046, "curriculum-help line", color="#54a24b", fontsize=7.5)
    ax.text(2.42, cold + 0.054, "interference line", color="#e45756", fontsize=7.5)
    ax.text(
        0.02,
        0.96,
        f"CURRICULUM - COLD = {delta:+.6f} bits/char",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set_ylabel("text8 TEST bits/char (lower is better)")
    ax.set_title("Stage 58 fixed-budget developmental comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_stage58_text8_primary.png", dpi=200)
    plt.close(fig)


def plot_retention(points_by_arm: dict[str, list[RetentionPoint]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARM_ORDER:
        points = points_by_arm[arm]
        ax.plot(
            [point.step for point in points],
            [point.bits_per_char for point in points],
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=ARM_COLORS[arm],
            label=arm,
        )
    ax.set_xlabel("global optimizer step")
    ax.set_ylabel("TinyStories validation bits/char (lower is better)")
    ax.set_title("Stage 58 retention and forgetting across checkpoint series")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_stage58_tinystories_retention.png", dpi=200)
    plt.close(fig)


def plot_arc_and_recipe(
    flagship_anchor: float,
    stage56_50k: Text8Result,
    stage58_cold_42k: Text8Result,
    recipe_pairs: dict[int, tuple[Text8Result, Text8Result]],
    out_dir: Path,
) -> None:
    """fig4 · left: the Phase 5 arc on one axis; right: matched 20k recipe pairs.

    The left panel is deliberately cross-budget and cross-model and says so in
    its labels: the flagship anchor is a 201.6M zero-shot score, the other two
    bars are 85.11M models trained on broad text under Recipe v1 and v2. The
    right panel is the controlled read: same seeds, same 20,000-step budget,
    Recipe v1 (Stage 56) against Recipe v2 (Stage 58 COLD).
    """
    fig, (ax_arc, ax_pair) = plt.subplots(1, 2, figsize=(10.8, 4.4))

    arc_labels = [
        "Phase 4 flagship\n201.6M zero-shot",
        "Stage 56 Recipe v1\n85.11M, 50k steps",
        "Stage 58 COLD Recipe v2\n85.11M, 42k steps",
    ]
    arc_values = [
        flagship_anchor,
        stage56_50k.bits_per_char,
        stage58_cold_42k.bits_per_char,
    ]
    arc_colors = ["#9d755d", "#72b7b2", "#4c78a8"]
    bars = ax_arc.bar(arc_labels, arc_values, color=arc_colors, width=0.62)
    for bar, value in zip(bars, arc_values):
        ax_arc.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_arc.set_ylim(0, max(arc_values) + 0.35)
    ax_arc.set_ylabel("text8 TEST bits/char (lower is better)")
    ax_arc.set_title("The Phase 5 arc (budgets differ; see labels)")
    ax_arc.tick_params(axis="x", labelsize=8)

    seeds = sorted(recipe_pairs)
    v1_values = [recipe_pairs[seed][0].bits_per_char for seed in seeds]
    v2_values = [recipe_pairs[seed][1].bits_per_char for seed in seeds]
    x_positions = range(len(seeds))
    for x, seed in zip(x_positions, seeds):
        v1 = recipe_pairs[seed][0].bits_per_char
        v2 = recipe_pairs[seed][1].bits_per_char
        ax_pair.plot([x, x], [v1, v2], color="#bab0ac", linewidth=1.2, zorder=1)
        ax_pair.annotate(
            f"{v2 - v1:+.6f}",
            xy=(x, (v1 + v2) / 2),
            xytext=(-8, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            ha="right",
        )
    ax_pair.scatter(
        list(x_positions), v1_values, color="#72b7b2", s=52, zorder=2, label="Stage 56 Recipe v1"
    )
    ax_pair.scatter(
        list(x_positions), v2_values, color="#4c78a8", s=52, zorder=2, label="Stage 58 COLD Recipe v2"
    )
    ax_pair.set_xticks(list(x_positions))
    ax_pair.set_xticklabels([f"seed {seed}" for seed in seeds])
    ax_pair.set_xlim(-0.5, len(seeds) + 0.4)
    ax_pair.set_ylabel("text8 TEST bits/char (lower is better)")
    ax_pair.set_title("Matched budget and seed: recipe effect at 20,000 steps")
    ax_pair.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_phase5_arc_and_recipe.png", dpi=200)
    plt.close(fig)


def result_to_dict(result: Text8Result) -> dict[str, object]:
    return {
        "checkpoint": result.checkpoint,
        "step": result.step,
        "bits_per_char": result.bits_per_char,
        "nll": result.nll,
        "chars_evaluated": result.chars_evaluated,
        "parameters": result.parameters,
    }


def point_to_dict(point: RetentionPoint) -> dict[str, object]:
    return {
        "checkpoint": point.checkpoint,
        "step": point.step,
        "bits_per_char": point.bits_per_char,
        "nll": point.nll,
        "chars_evaluated": point.chars_evaluated,
    }


def write_summary(
    path: Path,
    results: list[Text8Result],
    points_by_arm: dict[str, list[RetentionPoint]],
    delta: float,
    verdict: str,
) -> None:
    by_arm = {result.arm: result for result in results}
    mixture_vs_cold = by_arm["MIXTURE"].bits_per_char - by_arm["COLD"].bits_per_char
    mixture_vs_curriculum = (
        by_arm["MIXTURE"].bits_per_char - by_arm["CURRICULUM"].bits_per_char
    )
    lines = [
        "# Stage 58 Developmental Comparison",
        "",
        "Primary metric: deterministic chunked text8 TEST bits/char. Lower is better.",
        "",
        "| Arm | Global step | text8 TEST bits/char | TEST NLL | Characters evaluated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.arm} | {result.step:,} | {result.bits_per_char:.6f} | "
            f"{result.nll:.6f} | {result.chars_evaluated:,} |"
        )
    lines.extend(
        [
            "",
            f"- Primary delta, CURRICULUM minus COLD: `{delta:+.6f}` bits/char.",
            f"- Seed-7 primary read: **{verdict}**.",
            f"- Secondary delta, MIXTURE minus COLD: `{mixture_vs_cold:+.6f}` bits/char.",
            "- Secondary delta, MIXTURE minus CURRICULUM: "
            f"`{mixture_vs_curriculum:+.6f}` bits/char.",
            "",
            "This generator reports the seed-7 read only. The reduced-budget "
            "seed-11 and seed-19 sign replicas are generated separately by "
            "`make_h024_replica_figure.py` into `h024_replica_sign_check.json`; "
            "the combined three-seed verdict is recorded in ADR 0016.",
            "",
            "## Retention Series",
            "",
            "| Arm | Global step | TinyStories val bits/char | NLL | Characters evaluated |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        for point in points_by_arm[arm]:
            lines.append(
                f"| {arm} | {point.step:,} | {point.bits_per_char:.6f} | "
                f"{point.nll:.6f} | {point.chars_evaluated:,} |"
            )
    lines.extend(
        [
            "",
            "Figures: `fig1_stage58_text8_primary.png`, "
            "`fig2_stage58_tinystories_retention.png`, and "
            "`fig4_phase5_arc_and_recipe.png`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = [
        load_text8_result("COLD", args.cold_text8),
        load_text8_result("CURRICULUM", args.curriculum_text8),
        load_text8_result("MIXTURE", args.mixture_text8),
    ]
    points_by_arm = {
        "COLD": load_retention_points("COLD", args.cold_retention),
        "CURRICULUM": load_retention_points("CURRICULUM", args.curriculum_retention),
        "MIXTURE": load_retention_points("MIXTURE", args.mixture_retention),
    }
    delta = results[1].bits_per_char - results[0].bits_per_char
    verdict = verdict_for_primary(delta)
    stage56_50k = load_text8_point(args.stage56_50k_seed7, expected_step=50_000)
    recipe_pairs = {
        11: (
            load_text8_point(args.stage56_20k_seed11, expected_step=20_000),
            load_text8_point(args.stage58_cold_20k_seed11, expected_step=20_000),
        ),
        19: (
            load_text8_point(args.stage56_20k_seed19, expected_step=20_000),
            load_text8_point(args.stage58_cold_20k_seed19, expected_step=20_000),
        ),
    }
    flagship_anchor = load_flagship_anchor(args.phase4_figures_data)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_text8(results, out_dir, delta)
    plot_retention(points_by_arm, out_dir)
    plot_arc_and_recipe(flagship_anchor, stage56_50k, results[0], recipe_pairs, out_dir)
    data = {
        "primary_metric": "deterministic chunked text8 TEST bits/char",
        "text8_results": {result.arm: result_to_dict(result) for result in results},
        "primary_delta_curriculum_minus_cold": delta,
        "seed7_primary_read": verdict,
        "secondary_deltas": {
            "mixture_minus_cold": results[2].bits_per_char - results[0].bits_per_char,
            "mixture_minus_curriculum": results[2].bits_per_char - results[1].bits_per_char,
        },
        "retention_points": {
            arm: [point_to_dict(point) for point in points_by_arm[arm]]
            for arm in ARM_ORDER
        },
        "phase5_arc": {
            "phase4_flagship_zero_shot_bits_per_char": flagship_anchor,
            "phase4_flagship_source": str(args.phase4_figures_data),
            "stage56_recipe_v1_50k_seed7": result_to_dict(stage56_50k),
            "stage58_recipe_v2_cold_42k_seed7": result_to_dict(results[0]),
        },
        "recipe_matched_20k": {
            str(seed): {
                "stage56_recipe_v1": result_to_dict(pair[0]),
                "stage58_cold_recipe_v2": result_to_dict(pair[1]),
                "v2_minus_v1_bits_per_char": pair[1].bits_per_char
                - pair[0].bits_per_char,
            }
            for seed, pair in recipe_pairs.items()
        },
        "figures": [
            "fig1_stage58_text8_primary.png",
            "fig2_stage58_tinystories_retention.png",
            "fig4_phase5_arc_and_recipe.png",
        ],
    }
    (out_dir / "figures_data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(
        out_dir / "stage58_developmental_comparison.md",
        results,
        points_by_arm,
        delta,
        verdict,
    )
    print(f"[figures] wrote Stage 58 comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
