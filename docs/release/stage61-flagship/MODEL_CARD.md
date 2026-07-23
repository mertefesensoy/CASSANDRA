---
license: apache-2.0
language:
- en
tags:
- text-generation
- character-level
- research
- cassandra
pipeline_tag: text-generation
model-index:
- name: cassandra-200m-text8
  results:
  - task:
      type: text-generation
      name: Character-level language modeling
    dataset:
      type: text8
      name: text8 (test split, final 5,000,000 characters)
    metrics:
    - type: bits_per_character
      value: 1.336061
      name: text8 test bits/char, released fp16 weights (lower is better)
      verified: false
    - type: cross_entropy
      value: 0.926087
      name: text8 test NLL in nats, released fp16 weights (lower is better)
      verified: false
---

# Cassandra 200M text8 (pure-broad character flagship)

A 201,609,249-parameter decoder-only **character-level** language model trained
from scratch on **text8** (cleaned lowercase Wikipedia) on a single laptop GPU
(RTX 4070, 8 GB). It is the Stage 61 flagship of the
[Cassandra](https://github.com/mertefesensoy/CASSANDRA) research project, which
studies how far useful language-model behavior can be formed under strict
hardware constraints, with every claim pre-registered and measured.

This is a research and education artifact, not a product. It writes fluent,
Wikipedia-flavored prose locally but drifts off topic across long generations
(see Limitations).

## Model details

| | |
| --- | --- |
| Parameters | 201,609,249 |
| Architecture | decoder-only transformer, 16 layers, 16 heads, width 1024 |
| Context window | 256 characters (about 45 words) |
| Vocabulary | 33 characters (text8 exercises 27: `a-z` and space) |
| Position encoding | RoPE |
| Optimizer | Muon (hidden matrices) + AdamW (embeddings, output head) |
| Training precision | fp32 (released weights are fp16) |
| Training budget | 50,000 steps, ~204.8M characters (about 2.28 epochs of the 90M-char text8 train split) |
| Seed | 7 (single seed) |

## Intended use

Local research, demonstration, and education on small-model training under
constrained hardware: learning-curve study, checkpoint evaluation, prompt
sampling, context-window and coherence experiments, and following the Cassandra
staged hypothesis record.

It is **not** intended for production assistance, factual question answering,
safety-critical decisions, medical/legal/financial advice, moderation, or any
claim of broad or reliable English competence.

## How to run

Requires the Cassandra repo code (Apache-2.0) for the model class and the
downloaded package (safetensors weights plus `config.json` and `codec.json`):

```bash
git clone https://github.com/mertefesensoy/CASSANDRA
pip install torch numpy safetensors
# put the .safetensors, config.json, and codec.json beside the code, then:
```

```python
from flagship_eval_lib import load_model_from_safetensors, sample_text

model, codec, config, meta = load_model_from_safetensors(
    "stage61_pure_broad_200m_text8_fp16.safetensors",
    "config.json", "codec.json", device="cuda",  # or "cpu"
)
print(sample_text(
    model, codec,
    prompt="the history of ",
    max_new_chars=400,
    temperature=0.8,
    top_p=0.9,            # recommended: removes the low-probability garbage tail
    seed=7, device="cuda",
))
```

Weights are shipped as `safetensors` (no pickle), so the file loads without
executing any code.

**Recommended inference: temperature 0.8 with top-p 0.9.** Nucleus truncation
removes a low-probability tail that otherwise emits rare garbage tokens. It does
not fix topic drift (that is a context-window limit, below). Very low
temperature makes the model repeat itself, so avoid greedy decoding.

## Evaluation

The headline metric is **bits per character on the standard text8 test split**
(the final 5,000,000 characters), a long-established character-language-modeling
benchmark, so the result is directly comparable to published models. Scoring is
deterministic: non-overlapping 256-character windows over the whole test split,
context reset per window (lower is better).

| Metric | Value |
| --- | ---: |
| **text8 test bits/char (released fp16 weights)** | **1.336061** |
| text8 test NLL (nats, released fp16 weights) | 0.926087 |
| text8 test bits/char (fp32 training checkpoint, project canonical) | 1.336059 |
| Reference: Cassandra 85M pure-broad model, same eval | 1.357318 |
| Reference: GPT-2 117M zero-shot text8 (Radford et al. 2019) | 1.17 |

The released fp16 weights reproduce the fp32 result to within `0.000002`
bits/char (fp16 rounding). The model beats the same-recipe 85M reference by
about `0.021` bits/char, roughly seven times the known between-seed noise floor
(`0.003035`). It does not match GPT-2, which used subword tokenization and far
more training; the `1.17` anchor is aspirational context, not a peer comparison.

### Independent verification

This number is self-reported but fully reproducible by anyone. The shipped
`reproduce_text8_eval.py` downloads text8, scores the standard test split on the
released fp16 weights with the deterministic method above, and prints
`1.336061`:

```bash
python reproduce_text8_eval.py   # -> text8 bits/char : 1.336061
```

There is no automated third-party evaluation service for a custom-architecture
character model (the Open LLM Leaderboard and Hugging Face inference-based evals
target instruction-tuned models loadable by the standard harness), so
reproducibility on a public benchmark is the honest form of external validation
here. A separate Cassandra probe (H027) measures long-range context use at
`+0.205` bits/char and is documented in the code repository.

## Limitations

- **Topic drift is the headline limitation.** Over long generations the model
  wanders off subject. This is a context-window limit, not a comprehension
  failure: the Cassandra context-utilization probe (H027 / ADR 0019) shows the
  model uses its context strongly to the edge of its 256-character window, so it
  drifts precisely when a generation runs past what it can still see. Use
  top-p 0.9 to remove garbage; expect drift regardless.
- **Character-level and small.** 200M parameters, laptop-trained, single seed.
  It reasons about no facts and grounds nothing; it produces plausible-looking
  but frequently false encyclopedic-style text.
- **No alignment.** No instruction tuning, no RLHF, no safety filtering. It can
  produce misleading, repetitive, or low-quality text and must not be presented
  as truthful or broadly knowledgeable.
- **Narrow input format.** Inputs are normalized to lowercase `a-z` and space
  (the text8 alphabet); other characters are folded to space.

## Training data

text8 (Matt Mahoney's cleaned 100M-character lowercase Wikipedia benchmark;
first 90M for training). The underlying text lineage is Wikipedia, licensed
CC BY-SA / GFDL. Training corpora are **not** distributed with this model;
Cassandra ships the preparation scripts and provenance metadata only. The
corpus is rebuildable from those scripts.

## License and attribution

- **Weights: Apache-2.0.** **Code: Apache-2.0**
  ([repo](https://github.com/mertefesensoy/CASSANDRA)).
- **Training-data lineage: Wikipedia (CC BY-SA / GFDL)**, disclosed here; no
  training text is redistributed.
- The Muon optimizer path is adapted from
  [KellerJordan/Muon](https://github.com/KellerJordan/Muon) (MIT); see the
  repository `NOTICE`.

## Citation

```
@software{cassandra_2026,
  title  = {Cassandra: a laptop-scale language-model formation lab},
  author = {Sensoy, Mert Efe},
  year   = {2026},
  url    = {https://github.com/mertefesensoy/CASSANDRA}
}
```
