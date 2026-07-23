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

All numbers are deterministic and regenerable from the Cassandra repo.

| Metric | Value | Regeneration |
| --- | ---: | --- |
| text8 TEST bits/char (chunked, full 5M-char split) | **1.336059** | `python eval_text8.py --split test --checkpoint <weights>` |
| Reference: 85M pure-broad model, same eval | 1.357318 | Cassandra Stage 58 COLD |
| Reference: GPT-2 117M zero-shot text8 (Radford et al. 2019) | 1.17 | published |
| Context-utilization, deep bucket (H027) | +0.205 bits/char | `python eval_context_utilization.py --checkpoint <weights> ...` |

The model beats the 85M same-recipe reference by 0.021 bits/char, about seven
times the known between-seed noise floor (0.003035). It does not match GPT-2,
which used subword tokenization and far more training; the 1.17 anchor is
aspirational context, not a peer comparison.

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
