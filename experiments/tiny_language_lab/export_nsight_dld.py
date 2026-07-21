from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cassandra_tiny_transformer import (
    DEFAULT_RECENCY_LAMBDA,
    DEFAULT_RECENCY_TAU,
    SparseBackoffNgramPrior,
    TinyTransformer,
)


DEFAULT_CHECKPOINT = Path(
    "C:/cassandra_runs/stage55_flagship_checkpoints/"
    "stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt"
)
DEFAULT_OUT_DIR = Path(__file__).with_name("artifacts") / "phase4" / "nsight_dld"


class LogitsOnlyWrapper(nn.Module):
    def __init__(
        self,
        model: TinyTransformer,
        base_logits: torch.Tensor | None,
        residual_scale: float,
        recency_tau: float,
        recency_lambda: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.residual_scale = residual_scale
        self.recency_tau = recency_tau
        self.recency_lambda = recency_lambda
        if base_logits is not None:
            self.register_buffer("base_logits", base_logits)
        else:
            self.base_logits = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(
            input_ids,
            base_logits=self.base_logits,
            residual_scale=self.residual_scale,
            recency_tau=self.recency_tau,
            recency_lambda=self.recency_lambda,
        )
        return logits


def require_int(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    if value is None:
        raise ValueError(f"Checkpoint args are missing required field: {name}")
    return int(value)


def optional_float(args: dict[str, Any], name: str, default: float) -> float:
    value = args.get(name, default)
    return float(default if value is None else value)


def load_export_model(
    checkpoint_path: Path,
    device: str,
    seq_len_override: int | None,
) -> tuple[LogitsOnlyWrapper, dict[str, Any], int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint does not contain a usable args dictionary")
    chars = checkpoint.get("chars")
    if not isinstance(chars, list) or not chars:
        raise ValueError("Checkpoint does not contain a usable chars vocabulary")

    block_size = require_int(checkpoint_args, "block_size")
    seq_len = block_size if seq_len_override is None else int(seq_len_override)
    if seq_len < 1 or seq_len > block_size:
        raise ValueError(f"--seq-len must be between 1 and checkpoint block_size={block_size}")

    model = TinyTransformer(
        vocab_size=len(chars),
        block_size=block_size,
        n_layer=require_int(checkpoint_args, "n_layer"),
        n_head=require_int(checkpoint_args, "n_head"),
        n_embd=require_int(checkpoint_args, "n_embd"),
        dropout=optional_float(checkpoint_args, "dropout", 0.0),
        adapter_rank=int(checkpoint_args.get("adapter_rank", 0) or 0),
        lora_rank=int(checkpoint_args.get("lora_rank", 0) or 0),
        lora_alpha=optional_float(checkpoint_args, "lora_alpha", 1.0),
        lora_dropout=optional_float(checkpoint_args, "lora_dropout", 0.0),
        pos_encoding=str(checkpoint_args.get("pos_encoding", "learned")),
        activation_checkpoint=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    base_logits = checkpoint.get("base_logits")
    if isinstance(base_logits, SparseBackoffNgramPrior):
        raise ValueError("Sparse backoff priors are not supported for ONNX export")
    if isinstance(base_logits, torch.Tensor):
        base_logits = base_logits.to(device)
    elif base_logits is not None:
        raise ValueError(f"Unsupported base_logits payload for ONNX export: {type(base_logits)!r}")

    wrapper = LogitsOnlyWrapper(
        model=model,
        base_logits=base_logits,
        residual_scale=optional_float(checkpoint_args, "residual_scale", 1.0),
        recency_tau=optional_float(checkpoint_args, "recency_tau", DEFAULT_RECENCY_TAU),
        recency_lambda=optional_float(checkpoint_args, "recency_lambda", DEFAULT_RECENCY_LAMBDA),
    ).to(device)
    wrapper.eval()
    return wrapper, checkpoint, seq_len, len(chars)


def write_manifest(
    path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    checkpoint: dict[str, Any],
    seq_len: int,
    batch_size: int,
    opset: int,
) -> None:
    checkpoint_args = checkpoint.get("args", {})
    manifest = {
        "purpose": "NVIDIA Nsight Deep Learning Designer inference inspection export",
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "step": checkpoint.get("step"),
        "formation_forward_passes": checkpoint.get("formation_forward_passes"),
        "loss_curve": checkpoint.get("loss_curve", []),
        "vocab_size": len(checkpoint.get("chars", [])),
        "chars": checkpoint.get("chars", []),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "opset": opset,
        "model": {
            "block_size": checkpoint_args.get("block_size"),
            "n_layer": checkpoint_args.get("n_layer"),
            "n_head": checkpoint_args.get("n_head"),
            "n_embd": checkpoint_args.get("n_embd"),
            "pos_encoding": checkpoint_args.get("pos_encoding"),
            "dropout": checkpoint_args.get("dropout"),
            "residual_base": checkpoint_args.get("residual_base"),
            "train_scope": checkpoint_args.get("train_scope"),
            "optimizer": checkpoint_args.get("optimizer"),
        },
        "notes": [
            "This export is for inference graph inspection and profiling.",
            "It does not include optimizer state and should not be used to resume training.",
            "Open the .onnx file in NVIDIA Nsight Deep Learning Designer.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Cassandra TinyTransformer checkpoint for NVIDIA Nsight DL Designer."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--check-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    export_name = args.name.strip() or checkpoint_path.stem
    onnx_path = out_dir / f"{export_name}.onnx"
    manifest_path = out_dir / f"{export_name}.manifest.json"

    wrapper, checkpoint, seq_len, vocab_size = load_export_model(
        checkpoint_path=checkpoint_path,
        device=args.device,
        seq_len_override=args.seq_len,
    )
    dummy = torch.zeros((args.batch_size, seq_len), dtype=torch.long, device=args.device)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input_ids": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    if args.check_onnx:
        import onnx

        loaded = onnx.load(str(onnx_path))
        onnx.checker.check_model(loaded)

    write_manifest(
        path=manifest_path,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        checkpoint=checkpoint,
        seq_len=seq_len,
        batch_size=args.batch_size,
        opset=args.opset,
    )
    print(f"[nsight-dld] wrote {onnx_path}")
    print(f"[nsight-dld] wrote {manifest_path}")
    print(f"[nsight-dld] shape input_ids=({args.batch_size}, {seq_len}) logits=({args.batch_size}, {seq_len}, {vocab_size})")


if __name__ == "__main__":
    main()
