"""Generate the Phase 4 evaluation figures from real run artifacts.

Inputs: runs/phase4_stage55-flagship-cell*.log (training curves),
runs/stage52_crossover_*.jsonl (H019 matrix), and, when present,
runs/stage55_validation_suite.json and runs/stage55_text8_zero_shot.json.

Outputs: PNG figures under docs/figures/phase4/ plus a small
figures_data.json capturing every number that was plotted, so the report
can cite figures and raw values from one place.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\make_phase4_figures.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG2E = 1.4426950408889634
LAB_DIR = Path(__file__).resolve().parent
RUNS = LAB_DIR / "runs"
REPO = LAB_DIR.parent.parent
FIG_DIR = REPO / "docs" / "figures" / "phase4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

TRAIN_RE = re.compile(r"\[train\] step=(\d+)/\d+ batch_nll=([\d.]+)")
EVAL_RE = re.compile(r"\[eval\] step=(\d+)(?:/\d+)? train_nll=([\d.]+) val_nll=([\d.]+)")
SEED_RE = re.compile(r"\[matrix\].*seeds=(\d+)")


def parse_flagship_logs() -> dict[int, dict]:
    """Merge every flagship log leg into per-seed train/eval curves."""
    curves: dict[int, dict] = defaultdict(lambda: {"train": {}, "eval": {}})
    for log in sorted(RUNS.glob("phase4_stage55-flagship-cell*.log")):
        raw = log.read_bytes()
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8", errors="replace")
        seed_match = SEED_RE.search(text)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        for m in TRAIN_RE.finditer(text):
            curves[seed]["train"][int(m.group(1))] = float(m.group(2))
        for m in EVAL_RE.finditer(text):
            curves[seed]["eval"][int(m.group(1))] = float(m.group(3))
    return curves


def parse_stage52() -> dict:
    """Return {config: {params: {budget: mean val_nll}}} from the H019 jsonls."""
    rows: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seen_stems = set()
    for jf in sorted(RUNS.glob("stage52_crossover_*.jsonl")):
        stem = jf.stem
        base = stem.replace("_sharded", "")
        # Prefer the _sharded rerun of the prior arm when both exist.
        if not stem.endswith("_sharded") and (RUNS / f"{base}_sharded.jsonl").exists():
            continue
        if base in seen_stems:
            continue
        seen_stems.add(base)
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            config = row.get("comparison_name", "unknown")
            params = int(row.get("parameters", 0))
            budget = int(row.get("steps", 0))
            rows[config][params][budget].append(float(row["val_nll"]))
    out: dict = {}
    for config, per_params in rows.items():
        out[config] = {
            params: {b: sum(v) / len(v) for b, v in budgets.items()}
            for params, budgets in per_params.items()
        }
    return out


def fig1_learning_curves(curves: dict[int, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {7: "#1f77b4", 11: "#ff7f0e", 19: "#2ca02c"}
    for seed in sorted(curves):
        train = dict(sorted(curves[seed]["train"].items()))
        evals = dict(sorted(curves[seed]["eval"].items()))
        if train:
            ax.plot(list(train), list(train.values()), alpha=0.35, lw=0.8, color=colors.get(seed))
        if evals:
            steps = [s for s in evals if s > 0]
            ax.plot(
                steps,
                [evals[s] for s in steps],
                marker="o",
                ms=3.5,
                lw=1.6,
                color=colors.get(seed),
                label=f"seed {seed} val (sampled)",
            )
    ax.axhline(0.818608, color="gray", ls="--", lw=1, alpha=0.8)
    ax.text(500, 0.826, "Stage 51 · 25M · 5k steps (0.8186)", fontsize=7.5, color="gray")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation NLL (nats/char)")
    secax = ax.secondary_yaxis("right", functions=(lambda x: x * LOG2E, lambda x: x / LOG2E))
    secax.set_ylabel("bits/char")
    ax.set_title("Stage 55 flagship 201.6M · TinyStories char-level · training curves")
    ax.legend(fontsize=7.5)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_flagship_learning_curves.png")
    plt.close(fig)


def fig2_capacity_ladder(stage52: dict) -> dict:
    data = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    rf = stage52.get("random_full", {})
    budgets = [200, 500, 1000, 2000]
    cmap = plt.get_cmap("viridis")
    for i, budget in enumerate(budgets):
        xs, ys = [], []
        for params in sorted(rf):
            if budget in rf[params]:
                xs.append(params)
                ys.append(rf[params][budget] * LOG2E)
        if xs:
            ax.plot(xs, ys, marker="s", ms=4, color=cmap(i / 3.5), label=f"random_full · {budget} steps")
            data[f"random_full_b{budget}"] = dict(zip([str(x) for x in xs], ys))
    # Flagship and Stage 51 anchors (RESULTS.md, Stage 51 and Stage 55 closeouts)
    ax.scatter([201_609_249], [0.802730], marker="*", s=160, color="#d62728", zorder=5)
    ax.annotate("flagship 201.6M · 50k steps\n0.8027 bits/char", (201_609_249, 0.802730),
                textcoords="offset points", xytext=(-95, -18), fontsize=7.5, color="#d62728")
    ax.scatter([25_253_921], [1.181001], marker="D", s=40, color="#9467bd", zorder=5)
    ax.annotate("Stage 51 · 5k steps", (25_253_921, 1.181001),
                textcoords="offset points", xytext=(8, 4), fontsize=7.5, color="#9467bd")
    # Analytic prior floors (Stage 52 constant floor; Stage 54 gate A)
    ax.axhline(1.104 * LOG2E, color="#8c564b", ls=":", lw=1.2)
    ax.text(3.6e6, 1.104 * LOG2E + 0.012, "frozen order-4 count prior floor (1.593 bits)",
            fontsize=7.5, color="#8c564b")
    ax.axhline(0.984431 * LOG2E, color="#8c564b", ls="--", lw=1.2, alpha=0.7)
    ax.text(3.6e6, 0.984431 * LOG2E + 0.012, "sparse order-5 backoff prior floor (1.420 bits)",
            fontsize=7.5, color="#8c564b")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("validation bits/char")
    ax.set_title("Capacity ladder on TinyStories (char-level): gradient training vs analytic floors")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_capacity_ladder.png")
    plt.close(fig)
    return data


def fig3_crossover(stage52: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    rf = stage52.get("random_full", {})
    prior_cfg = stage52.get("count_prior_ng4_lora_r2", {})
    cmap = plt.get_cmap("plasma")
    sizes = sorted(rf)
    for i, params in enumerate(sizes):
        budgets = sorted(rf[params])
        ax.plot(budgets, [rf[params][b] for b in budgets], marker="o", ms=4,
                color=cmap(i / max(len(sizes) - 1, 1)),
                label=f"random_full {params/1e6:.1f}M")
    prior_vals = [v for per in prior_cfg.values() for v in per.values()]
    if prior_vals:
        floor = sum(prior_vals) / len(prior_vals)
        ax.axhline(floor, color="#8c564b", ls=":", lw=1.4)
        ax.text(210, floor + 0.006, f"frozen order-4 prior arm (constant ~{floor:.3f})",
                fontsize=7.5, color="#8c564b")
    ax.set_xscale("log")
    ax.set_xlabel("training steps (log scale)")
    ax.set_ylabel("validation NLL (nats/char)")
    ax.set_title("H019 crossover: one fixed analytic floor, four learning curves (Stage 52)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_h019_crossover.png")
    plt.close(fig)


def fig4_gpt1_comparison(text8_bits: float | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Panel A: scale comparison (log axis)
    ax = axes[0]
    cats = ["parameters", "training text\n(chars seen)", "context window\n(chars, approx)", "training\nGPU-count x days"]
    gpt1 = [117e6, 4.6e9, 512 * 4, 8 * 30]
    cass = [201.6e6, 204.8e6, 256, 1 * 2.2]
    x = range(len(cats))
    width = 0.38
    ax.bar([i - width / 2 for i in x], gpt1, width, label="GPT-1 (2018)", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], cass, width, label="Cassandra flagship (2026)", color="#dd8452")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=7.5)
    ax.set_title("Scale: GPT-1 vs Cassandra flagship")
    ax.legend(fontsize=7.5)
    for i, (g, c) in enumerate(zip(gpt1, cass)):
        ax.text(i - width / 2, g * 1.15, f"{g:,.0f}" if g < 1e6 else f"{g:.2g}", ha="center", fontsize=6.5)
        ax.text(i + width / 2, c * 1.15, f"{c:,.0f}" if c < 1e6 else f"{c:.2g}", ha="center", fontsize=6.5)

    # Panel B: quality anchors in bits/char, domains explicitly separated
    ax = axes[1]
    labels = [
        "GPT-1 117M\nBooksCorpus ppl 18.4\n(approx conversion)",
        "GPT-2 117M\ntext8 zero-shot",
        "GPT-2 1542M\ntext8 zero-shot",
        "Cassandra 201.6M\nTinyStories val\n(in-domain)",
    ]
    values = [math.log2(18.4) / 5.7, 1.17, 0.98, 0.802730]
    colors = ["#4c72b0", "#4c72b0", "#4c72b0", "#dd8452"]
    hatches = ["//", "", "", ""]
    if text8_bits is not None:
        labels.append("Cassandra 201.6M\ntext8 zero-shot\n(out-of-domain)")
        values.append(text8_bits)
        colors.append("#dd8452")
        hatches.append("//")
    bars = ax.bar(range(len(values)), values, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("bits/char (lower is better)")
    ax.set_title("Quality anchors, bits/char\n(hatched = converted or out-of-domain)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gpt1_comparison.png")
    plt.close(fig)


def fig5_efficiency() -> None:
    # (gpu_hours estimate, bits/char, label) from RESULTS.md closeouts.
    points = [
        (136.0 / 3600, 1.420, "order-5 analytic prior floor\n(no gradients)"),
        (695.6 / 3600, 1.181, "Stage 51 · 25.25M · 5k"),
        (8.4, 0.8787, "replica · 201.6M · 20k · seed 11"),
        (8.5, 0.8611, "replica · 201.6M · 20k · seed 19"),
        (26.4, 0.8027, "flagship · 201.6M · 50k · seed 7"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, marker="o", ms=5, lw=1.2, color="#dd8452")
    offsets = [(8, -14), (6, 6), (6, 8), (6, -16), (-150, 8)]
    for (x, y, label), off in zip(points, offsets):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=off, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("training cost (single RTX 4070 laptop GPU-hours, estimated)")
    ax.set_ylabel("validation bits/char")
    ax.set_title("Cost of quality on one laptop GPU (TinyStories char-level)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_efficiency_frontier.png")
    plt.close(fig)


def main() -> None:
    curves = parse_flagship_logs()
    stage52 = parse_stage52()

    text8_bits = None
    text8_json = RUNS / "stage55_text8_zero_shot.json"
    if text8_json.exists():
        payload = json.loads(text8_json.read_text(encoding="utf-8"))
        entry = payload.get("models", {}).get("flagship_200m_50k_seed7")
        if entry:
            text8_bits = float(entry["result"]["bits_per_char"])

    fig1_learning_curves(curves)
    ladder = fig2_capacity_ladder(stage52)
    fig3_crossover(stage52)
    fig4_gpt1_comparison(text8_bits)
    fig5_efficiency()

    data = {
        "flagship_eval_points": {
            str(seed): dict(sorted(curves[seed]["eval"].items())) for seed in sorted(curves)
        },
        "stage52_random_full_bits": ladder,
        "text8_flagship_bits_per_char": text8_bits,
        "figures": sorted(p.name for p in FIG_DIR.glob("fig*.png")),
    }
    (FIG_DIR / "figures_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[figures] wrote {len(data['figures'])} figures to {FIG_DIR}")
    for name in data["figures"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
