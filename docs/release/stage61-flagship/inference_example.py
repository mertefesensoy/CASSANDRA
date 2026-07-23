"""Minimal inference example for the Cassandra 200M text8 flagship.

Prerequisites:
  1. Clone the Cassandra code (Apache-2.0):
       git clone https://github.com/mertefesensoy/CASSANDRA
  2. pip install torch numpy
  3. Download the released weights file
     (stage61_pure_broad_200m_text8_fp16.pt) beside this script, inside
     experiments/tiny_language_lab/ so `flagship_eval_lib` is importable.

Run:  python inference_example.py
"""

from flagship_eval_lib import load_model, sample_text

WEIGHTS = "stage61_pure_broad_200m_text8_fp16.pt"
DEVICE = "cuda"  # use "cpu" if you have no GPU (slower)

model, codec, args, meta = load_model(WEIGHTS, device=DEVICE)
print(f"loaded {meta['parameters']:,} parameters, block size {args['block_size']}")

for prompt in ("the history of ", "the united states ", "the city of "):
    text = sample_text(
        model,
        codec,
        prompt=prompt,
        max_new_chars=400,
        temperature=0.8,
        top_p=0.9,  # recommended: removes the low-probability garbage tail
        seed=7,
        device=DEVICE,
    )
    print(f"\n=== {prompt!r} ===\n{text}")
