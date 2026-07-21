"""Stage 55 validation suite.

Three independent checks that the Phase 4 artifacts work properly:

1. High-precision deterministic validation NLL for each final checkpoint
   (the recorded numbers used 16 sampled batches, about 33k chars; this
   evaluates millions of deterministically spread chars).
2. ONNX parity: the exported Nsight artifact must produce the same logits
   as the PyTorch checkpoint it came from.
3. Reproducible text samples from every model for the human review packet.

Writes runs/stage55_validation_suite.md and .json.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\phase4_validate.py
  python .\\experiments\\tiny_language_lab\\phase4_validate.py --models flagship_200m_50k_seed7 --eval-chars 200000 --skip-parity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flagship_eval_lib import (
    FINAL_CHECKPOINTS,
    LAB_DIR,
    LOG2E,
    chunked_nll,
    free_model,
    load_model,
    load_val_ids,
    sample_text,
)

DEFAULT_ONNX = (
    LAB_DIR / "artifacts" / "phase4" / "nsight_dld" / "stage55_seed7_final_success_b1_s256.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 55 validation suite")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--eval-chars", type=int, default=8_000_000)
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(FINAL_CHECKPOINTS.keys()),
        default=sorted(FINAL_CHECKPOINTS.keys()),
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--sample-chars", type=int, default=400)
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=LAB_DIR / "runs" / "stage55_validation_suite",
    )
    return parser.parse_args()


def onnx_parity(onnx_path: Path, checkpoint_path: Path, ids: torch.Tensor) -> dict:
    import onnxruntime as ort

    model, _, _, _ = load_model(checkpoint_path, device="cpu")
    window = ids[:256].unsqueeze(0)
    with torch.no_grad():
        torch_logits, _ = model(window, base_logits=None)
    torch_logits = torch_logits.squeeze(0).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_logits = session.run(None, {input_name: window.numpy()})[0].squeeze(0)

    diff = abs(torch_logits - ort_logits)
    torch_top1 = torch_logits.argmax(axis=-1)
    ort_top1 = ort_logits.argmax(axis=-1)
    result = {
        "onnx": str(onnx_path),
        "checkpoint": str(checkpoint_path),
        "positions": int(torch_top1.shape[0]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement": float((torch_top1 == ort_top1).mean()),
    }
    free_model(model)
    return result


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    report: dict = {
        "suite": "stage55_validation",
        "device": args.device,
        "eval_chars_requested": args.eval_chars,
        "models": {},
    }

    val_ids = None
    prompt = "once upon a time "
    for name in args.models:
        path = FINAL_CHECKPOINTS[name]
        print(f"[validate] loading {name} from {path}")
        started = time.time()
        model, codec, margs, meta = load_model(path, device=args.device)
        if val_ids is None:
            print("[validate] building validation split ids (one-time)")
            val_ids = load_val_ids(codec)
        eval_started = time.time()
        stats = chunked_nll(
            model,
            val_ids,
            device=args.device,
            batch_windows=args.batch_windows,
            max_eval_chars=args.eval_chars,
        )
        samples = {}
        for temp, seed in ((0.8, 7), (1.0, 11)):
            samples[f"temp{temp}_seed{seed}"] = sample_text(
                model,
                codec,
                prompt,
                max_new_chars=args.sample_chars,
                temperature=temp,
                seed=seed,
                device=args.device,
            )
        report["models"][name] = {
            "meta": meta,
            "recorded_report_val_nll": None,
            "chunked_eval": stats,
            "eval_seconds": round(time.time() - eval_started, 1),
            "total_seconds": round(time.time() - started, 1),
            "samples": samples,
        }
        print(
            f"[validate] {name}: nll={stats['nll']:.6f} "
            f"bits/char={stats['bits_per_char']:.6f} "
            f"chars={stats['chars_evaluated']:,}"
        )
        free_model(model)

    if not args.skip_parity:
        flagship = FINAL_CHECKPOINTS["flagship_200m_50k_seed7"]
        if args.onnx.exists():
            print("[validate] running ONNX parity check (CPU)")
            if val_ids is None:
                _, codec, _, _ = load_model(flagship, device="cpu")
                val_ids = load_val_ids(codec)
            report["onnx_parity"] = onnx_parity(args.onnx, flagship, val_ids)
            print(f"[validate] parity: {report['onnx_parity']}")
        else:
            report["onnx_parity"] = {"error": f"missing ONNX at {args.onnx}"}

    json_path = args.out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Stage 55 Validation Suite", ""]
    lines.append(f"Device: {args.device} · requested eval chars: {args.eval_chars:,}")
    lines.append("")
    lines.append(
        "| Model | Params | Chunked val NLL | Bits/char | Chars evaluated | Eval seconds |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, entry in report["models"].items():
        stats = entry["chunked_eval"]
        lines.append(
            f"| {name} | {entry['meta']['parameters']:,} | {stats['nll']:.6f} "
            f"| {stats['bits_per_char']:.6f} | {stats['chars_evaluated']:,} "
            f"| {entry['eval_seconds']} |"
        )
    lines.append("")
    if "onnx_parity" in report:
        lines.append("## ONNX parity")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report["onnx_parity"], indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Samples")
    lines.append("")
    for name, entry in report["models"].items():
        lines.append(f"### {name}")
        lines.append("")
        for key, text in entry["samples"].items():
            lines.append(f"- `{key}`:")
            lines.append("")
            lines.append("```text")
            lines.append(text)
            lines.append("```")
            lines.append("")
    md_path = args.out_stem.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[validate] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
