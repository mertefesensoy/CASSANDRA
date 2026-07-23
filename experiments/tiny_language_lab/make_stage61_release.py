"""Assemble a Hugging Face upload-ready package for the Stage 61 flagship.

Produces, into a staging directory, everything needed to publish the pure-broad
200M text8 flagship: the fp16 model-only weights (via the audited
export_model_only path), a config.json, the character codec, a manifest with
SHA-256s, and a round-trip check that the exported weights actually load and
generate. The prose artifacts (model card README, inference example, LICENSE,
NOTICE) are copied in separately by the caller.

This prepares the package only. It does NOT create any Hugging Face repo and
does NOT upload anything: publishing is the user's action, run with the user's
own token.

Usage (from repo root):
  python .\\experiments\\tiny_language_lab\\make_stage61_release.py `
    --checkpoint C:\\cassandra_runs\\stage61_pure_broad_200m_checkpoints\\stage61_pure_broad_200m_seed7_random_full_seed7.pt `
    --out-dir C:\\cassandra_runs\\stage61_release
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_file

from export_model_only_checkpoint import convert_state_dict, sha256_file
from flagship_eval_lib import free_model, load_model_from_safetensors, sample_text

EXPECTED_PARAMETERS = 201_609_249
WEIGHTS_NAME = "stage61_pure_broad_200m_text8_fp16.safetensors"

# The user's decisions, recorded 2026-07-23.
WEIGHTS_LICENSE = "Apache-2.0"
RECOMMENDED_INFERENCE = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_chars": 400,
    "note": (
        "top_p 0.9 removes the low-probability garbage tail; the model still "
        "drifts topically across long generations because its context window is "
        "256 characters (see the Cassandra H027 / ADR 0019 finding)."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 61 flagship release packager")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists() or args.checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing source checkpoint: {args.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(args.checkpoint)

    # Architecture, vocab, and weights from the source checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    ck_args = ckpt.get("args") or {}
    chars = ckpt.get("chars")
    if not isinstance(chars, list) or not chars:
        raise ValueError("Checkpoint has no usable chars vocab")
    step = ckpt.get("step")
    state = ckpt.get("model_state") if "model_state" in ckpt else ckpt.get("model")
    if state is None:
        raise ValueError("Checkpoint has no model state")

    # fp16, pickle-free safetensors weights. No weight tying in TinyTransformer
    # (token_embedding and lm_head are independent), so a contiguous clone is
    # enough to satisfy safetensors' no-shared-memory rule.
    fp16_state = {
        key: value.contiguous().clone()
        for key, value in convert_state_dict(state, torch.float16).items()
    }
    del ckpt, state
    weights_path = args.out_dir / WEIGHTS_NAME
    save_file(
        fp16_state,
        str(weights_path),
        metadata={
            "format": "pt",
            "model": "cassandra-200m-text8",
            "dtype": "fp16",
            "license": WEIGHTS_LICENSE,
        },
    )
    del fp16_state
    weights_sha = sha256_file(weights_path)
    (weights_path.with_suffix(weights_path.suffix + ".sha256")).write_text(
        f"{weights_sha}  {weights_path.name}\n", encoding="utf-8"
    )

    # config.json (architecture, training provenance, recommended inference).
    config = {
        "model_type": "cassandra_tiny_transformer",
        "architecture": "decoder-only character-level transformer",
        "parameters": EXPECTED_PARAMETERS,
        "n_layer": int(ck_args["n_layer"]),
        "n_head": int(ck_args["n_head"]),
        "n_embd": int(ck_args["n_embd"]),
        "block_size": int(ck_args["block_size"]),
        "vocab_size": len(chars),
        "pos_encoding": str(ck_args.get("pos_encoding") or "rope"),
        "dropout": float(ck_args.get("dropout") or 0.0),
        "weights_dtype": "fp16",
        "weights_format": "safetensors",
        "weights_file": WEIGHTS_NAME,
        "weights_sha256": weights_sha,
        "source_checkpoint_sha256_fp32": source_sha,
        "training": {
            "optimizer": "muon (hidden) + adamw (embeddings, head)",
            "precision_trained": "fp32",
            "steps": int(step),
            "lr_schedule": "cosine to lr_final_frac 0.1",
            "dataset": "text8 (Matt Mahoney, cleaned lowercase Wikipedia)",
            "dataset_lineage": "Wikipedia (CC BY-SA / GFDL)",
            "context_window_chars": int(ck_args["block_size"]),
        },
        "recommended_inference": RECOMMENDED_INFERENCE,
        "code": "https://github.com/mertefesensoy/CASSANDRA",
        "code_license": "Apache-2.0",
        "weights_license": WEIGHTS_LICENSE,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # codec.json: the 33-char alphabet in checkpoint order (index == token id).
    codec_path = args.out_dir / "codec.json"
    codec_path.write_text(
        json.dumps({"chars": chars, "vocab_size": len(chars)}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Round-trip: the safetensors weights must load pickle-free and generate.
    config_path = args.out_dir / "config.json"
    model, codec, _, meta = load_model_from_safetensors(
        weights_path, config_path, codec_path, device=args.device
    )
    try:
        if int(meta["parameters"]) != EXPECTED_PARAMETERS:
            raise ValueError("Round-trip parameter count mismatch")
        sample = sample_text(
            model, codec, prompt="the history of ",
            max_new_chars=200,
            temperature=RECOMMENDED_INFERENCE["temperature"],
            top_p=RECOMMENDED_INFERENCE["top_p"],
            seed=20260723, device=args.device,
        )
    finally:
        free_model(model)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model": "cassandra-200m-text8 (Stage 61 pure-broad flagship)",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256_fp32": source_sha,
        "weights_file": WEIGHTS_NAME,
        "weights_format": "safetensors",
        "weights_sha256_fp16": weights_sha,
        "weights_bytes_fp16": weights_path.stat().st_size,
        "parameters": EXPECTED_PARAMETERS,
        "step": int(step),
        "weights_license": WEIGHTS_LICENSE,
        "roundtrip_ok": True,
        "roundtrip_loader": "load_model_from_safetensors (pickle-free)",
        "roundtrip_sample_prefix": sample[:160],
        "files": ["config.json", "codec.json", WEIGHTS_NAME, f"{WEIGHTS_NAME}.sha256"],
    }
    (args.out_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
