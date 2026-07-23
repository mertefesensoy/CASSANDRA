"""Independently reproduce the published text8 bits/char for this model.

Anyone can run this on the RELEASED fp16 safetensors weights to verify the
model card's headline evaluation metric for themselves. It downloads text8
(~30 MB) if absent, then scores the standard test split (final 5,000,000
characters) with the deterministic chunked convention (non-overlapping
256-character windows, context reset per window) that the Cassandra project
uses for all closeout numbers.

Prerequisites (see the model card): the Cassandra repo code on the import path,
plus torch, numpy, and safetensors. Run from experiments/tiny_language_lab/
with the released weights, config.json, and codec.json beside it.

  python reproduce_text8_eval.py
"""

from flagship_eval_lib import chunked_nll, encode_fast, load_model_from_safetensors
from eval_text8 import ensure_text8

WEIGHTS = "stage61_pure_broad_200m_text8_fp16.safetensors"
DEVICE = "cuda"  # or "cpu"


def main() -> None:
    test_split = ensure_text8()[95_000_000:100_000_000]  # standard text8 test split
    model, codec, _config, meta = load_model_from_safetensors(
        WEIGHTS, "config.json", "codec.json", device=DEVICE
    )
    ids = encode_fast(test_split, codec)
    result = chunked_nll(model, ids, device=DEVICE, batch_windows=32)
    print(
        f"parameters      : {meta['parameters']:,}\n"
        f"chars evaluated : {result['chars_evaluated']:,}\n"
        f"text8 test NLL  : {result['nll']:.6f} nats\n"
        f"text8 bits/char : {result['bits_per_char']:.6f}"
    )


if __name__ == "__main__":
    main()
