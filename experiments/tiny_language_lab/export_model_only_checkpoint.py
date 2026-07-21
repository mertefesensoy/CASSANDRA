from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def convert_tensor(value: object, dtype: torch.dtype) -> object:
    if torch.is_tensor(value) and value.is_floating_point():
        return value.to(dtype=dtype)
    return value


def convert_state_dict(state: object, dtype: torch.dtype) -> object:
    if isinstance(state, dict):
        return {key: convert_tensor(value, dtype) for key, value in state.items()}
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model_only(checkpoint: Path, out: Path, dtype_name: str) -> dict[str, object]:
    dtype = DTYPES[dtype_name]
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("model_state") if "model_state" in ckpt else ckpt.get("model")
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {checkpoint}")

    payload = {
        "model_state": convert_state_dict(state, dtype),
        "optimizer_state": None,
        "step": ckpt.get("step"),
        "formation_forward_passes": ckpt.get("formation_forward_passes"),
        "loss_curve": ckpt.get("loss_curve"),
        "chars": ckpt.get("chars"),
        "args": ckpt.get("args"),
        "base_logits": convert_tensor(ckpt.get("base_logits"), dtype),
        "archive": {
            "model_only": True,
            "dtype": dtype_name,
            "source_checkpoint": str(checkpoint),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    report = {
        "source": str(checkpoint),
        "out": str(out),
        "dtype": dtype_name,
        "sha256": digest,
        "bytes": out.stat().st_size,
        "step": payload["step"],
        "formation_forward_passes": payload["formation_forward_passes"],
        "model_only": True,
        "optimizer_state": None,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an optimizer-free model-only checkpoint archive")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_model_only(args.checkpoint, args.out, args.dtype)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
