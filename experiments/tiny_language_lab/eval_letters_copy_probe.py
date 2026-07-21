from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cassandra_tiny_transformer import copy_answer_probe
from flagship_eval_lib import FINAL_CHECKPOINTS, free_model, load_model
from make_letters_copy_probe import (
    ANSWER_MARKER,
    DEFAULT_OUT as DEFAULT_PROBE,
    KEY_MARKER,
    PAYLOAD_ALPHABET,
    validate_corpus,
    write_probe,
)


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = LAB_DIR / "runs" / "phase5_behavior_letters_probe.json"
DEFAULT_SUMMARY = LAB_DIR / "runs" / "phase5_behavior_letters_probe.md"
CHANCE_ACCURACY = 1.0 / len(PAYLOAD_ALPHABET)
REOPEN_THRESHOLD = CHANCE_ACCURACY + 0.10


def choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return raw


def ensure_probe(path: Path, lines: int, seed: int) -> dict[str, object]:
    if not path.exists():
        return write_probe(path, lines, seed)
    text = path.read_text(encoding="utf-8")
    meta = validate_corpus(text)
    meta.update({"path": str(path)})
    return meta


def validate_codec(text: str, chars: list[str]) -> None:
    missing = sorted(set(text) - set(chars))
    if missing:
        raise ValueError(f"Probe text has chars outside checkpoint codec: {missing!r}")


def format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_summary(path: Path, payload: dict[str, object]) -> None:
    derived = payload["derived"]
    report = payload["copy_probe"]
    probe = payload["probe"]
    checkpoint = payload["checkpoint"]
    verdict = "REOPEN_BEHAVIOR_AXIS" if derived["reopen_behavior_axis"] else "KEEP_BEHAVIOR_AXIS_CLOSED"
    lines = [
        "# Phase 5 Behavior Probe: Letters-only copy",
        "",
        f"- checkpoint: `{checkpoint['path']}`",
        f"- probe: `{probe['path']}`",
        f"- cases scored: {report['copy_probe_cases']}",
        f"- key marker: `{report['copy_probe_key_marker']}`",
        f"- answer marker: `{report['copy_probe_marker']}`",
        f"- choice candidates: `{report['copy_probe_choice_candidates']}`",
        f"- flagship constrained-choice copy accuracy: {format_float(derived['choice_accuracy'])}",
        f"- chance baseline: {format_float(derived['chance_accuracy'])}",
        f"- reopen threshold: {format_float(derived['reopen_threshold'])}",
        f"- verdict: **{verdict}**",
        "",
        "Raw full-vocabulary argmax accuracy is also recorded for audit, but the phase gate uses constrained-choice accuracy because the payload alphabet has 16 possible answers.",
        "",
        "```json",
        json.dumps(
            {
                "choice_accuracy": derived["choice_accuracy"],
                "raw_accuracy": report["copy_probe_accuracy"],
                "choice_mrr": report["copy_probe_choice_mrr"],
                "choice_cases": report["copy_probe_choice_cases"],
                "nll": report["copy_probe_nll"],
                "seconds": payload["seconds"],
            },
            indent=2,
        ),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    device = choose_device(args.device)
    checkpoint = Path(args.checkpoint)
    probe_path = Path(args.probe)
    probe_meta = ensure_probe(probe_path, args.lines, args.seed)
    probe_text = probe_path.read_text(encoding="utf-8")

    started = time.perf_counter()
    model, codec, checkpoint_args, checkpoint_meta = load_model(checkpoint, device=device)
    try:
        validate_codec(probe_text, codec.chars)
        report = copy_answer_probe(
            model=model,
            val_text=probe_text,
            codec=codec,
            block_size=int(checkpoint_args["block_size"]),
            device=device,
            base_logits=None,
            residual_scale=1.0,
            recency_tau=0.0,
            recency_lambda=0.0,
            marker=ANSWER_MARKER,
            max_cases=args.max_cases,
            retrieval_template="none",
            retrieval_source="target",
            retrieval_corrupt="none",
            holdout_keys="",
            retrieval_memory=None,
            retrieval_memory_report=None,
            key_marker=KEY_MARKER,
        )
    finally:
        free_model(model)

    seconds = time.perf_counter() - started
    choice_accuracy = report["copy_probe_choice_accuracy"]
    derived = {
        "choice_accuracy": choice_accuracy,
        "chance_accuracy": CHANCE_ACCURACY,
        "reopen_threshold": REOPEN_THRESHOLD,
        "reopen_behavior_axis": bool(choice_accuracy is not None and choice_accuracy >= REOPEN_THRESHOLD),
    }
    payload = {
        "title": "Phase 5 letters-only zero-shot copy behavior probe",
        "seconds": seconds,
        "device": device,
        "checkpoint": {
            "path": str(checkpoint),
            "step": checkpoint_meta.get("step"),
            "formation_forward_passes": checkpoint_meta.get("formation_forward_passes"),
            "parameters": checkpoint_meta.get("parameters"),
            "block_size": checkpoint_args.get("block_size"),
            "n_layer": checkpoint_args.get("n_layer"),
            "n_head": checkpoint_args.get("n_head"),
            "n_embd": checkpoint_args.get("n_embd"),
        },
        "probe": probe_meta,
        "copy_probe": report,
        "derived": derived,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the Phase 4 flagship on the Phase 5 letters-only copy probe")
    parser.add_argument("--checkpoint", type=Path, default=FINAL_CHECKPOINTS["flagship_200m_50k_seed7"])
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--lines", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-cases", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    derived = payload["derived"]
    print(
        "choice_accuracy="
        f"{format_float(derived['choice_accuracy'])} "
        f"chance={format_float(derived['chance_accuracy'])} "
        f"threshold={format_float(derived['reopen_threshold'])} "
        f"reopen={derived['reopen_behavior_axis']}"
    )


if __name__ == "__main__":
    main()
