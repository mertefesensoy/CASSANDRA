# Cassandra TinyStories Flagship 200M - Model Card Draft

Status: draft for release preparation, not a public release artifact yet.

This draft grows from `docs/phase4-flagship-evaluation-report.md` and the
Phase 5 evidence recorded on 2026-07-09. Every numeric claim below includes the
local command or artifact that regenerates or audits it.

## Model Details

- Model family: decoder-only character language model.
- Parameters: `201,609,249`.
- Architecture: 16 layers, 16 attention heads, embedding width 1024.
- Context window: 256 characters.
- Vocabulary: 33 characters, saved in checkpoint order.
- Position encoding: RoPE.
- Optimizer used for training: Muon for hidden matrices with auxiliary AdamW
  for embeddings and output head.
- Training precision: fp32.
- Initialization: random full-model initialization, no frozen prior and no
  external pretrained checkpoint.
- Primary checkpoint: Stage 55 seed 7, step 50,000.
- Current local artifact path:
  `experiments/tiny_language_lab/artifacts/phase4/checkpoints/stage55_flagship_200m_b50000_seed7/stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`.

## Intended Use

This model is intended for local research, demonstration, and education around
small language-model training under constrained hardware. Suitable uses include
studying learning curves, checkpoint evaluation, simple prompt sampling, and
the Cassandra project's staged hypothesis record.

It is not intended for production assistance, factual answering, safety-critical
decisions, medical/legal/financial advice, user-facing moderation, or claims of
broad English competence.

## Training Data

- Dataset: TinyStories, normalized to the Cassandra 33-character alphabet.
- Normalized corpus size: `494,094,421` characters.
- Train/validation split: `419,980,257` train characters and `74,114,164`
  validation characters.
- Training budget: 50,000 optimizer steps, `100,000` formation forward passes,
  about `204.8M` training characters, approximately `0.49` epoch.
- Dataset license due diligence: see `docs/phase5-licensing-notes.md`.

The release posture is scripts plus metadata, not generated corpus payloads.
Local corpus files are rebuildable and should not ship in the public repo.

## Evaluation Results

| Result | Value | Regeneration or audit path |
| --- | ---: | --- |
| Stage 55 sampled report validation NLL | 0.556410 | `experiments/tiny_language_lab/runs/stage55_flagship_200m_b50000_seed7.jsonl` |
| Stage 55 sampled report bits/char | 0.802730 | `experiments/tiny_language_lab/runs/stage55_flagship_200m_b50000_seed7.md` |
| High-precision chunked TinyStories val NLL | 0.563231 | `python .\experiments\tiny_language_lab\phase4_validate.py --models flagship_200m_50k_seed7` |
| High-precision chunked TinyStories bits/char | 0.812571 | `experiments/tiny_language_lab/runs/stage55_validation_suite.md` |
| Stage 51 25M reference chunked bits/char | 1.189778 | same validation suite |
| Zero-shot text8 TEST bits/char | 2.8817 | `python .\experiments\tiny_language_lab\eval_text8.py --models flagship_200m_50k_seed7` as recorded in the Phase 4 report |
| In-domain vs text8 specialization gap | 2.07 bits/char | `2.8817 - 0.812571`, recorded in `docs/phase4-flagship-evaluation-report.md` |
| Generation proxy score | 4.333 / 6 | `experiments/tiny_language_lab/runs/stage55_flagship_generation_quality.md` |
| Letters-only zero-shot copy choice accuracy | 0.060547 | `python .\experiments\tiny_language_lab\eval_letters_copy_probe.py --device auto --max-cases 1024` |

ONNX parity was verified for the Nsight DL Designer export. Re-run the
validation suite without `--skip-parity` to check it again:

```powershell
python .\experiments\tiny_language_lab\phase4_validate.py --models flagship_200m_50k_seed7
```

Recorded parity from the Phase 4 report: max absolute logit difference
`8.6e-6`, mean `1.0e-6`, top-1 agreement `100%` across all 256 positions.

## How to Run Locally

Interactive playground:

```powershell
python .\experiments\tiny_language_lab\playground.py
```

Validation suite:

```powershell
python .\experiments\tiny_language_lab\phase4_validate.py --models flagship_200m_50k_seed7
```

Phase 5 behavior probe:

```powershell
python .\experiments\tiny_language_lab\make_letters_copy_probe.py --lines 1024 --seed 20260709
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py --device auto --max-cases 1024
```

If local corpora have been pruned, regenerate the relevant corpus from its
script and metadata before running validation.

## Limitations

- The largest limitation is domain specialization. The model reaches
  `0.812571` bits/char on TinyStories validation but `2.8817` bits/char on
  zero-shot text8, a `2.07` bits/char gap.
- The model is character-level with only 256 characters of context. It cannot
  see an entire typical TinyStories story in one context window.
- The headline seed is one 50k run. Seeds 11 and 19 are 20k replicas, useful
  for recipe stability but not a three-seed 50k estimate.
- The generation proxy is deterministic but imperfect; the human A/B review is
  still pending and no coherence claim should be made until that review is
  folded into the card.
- The Phase 5 letters-only behavior probe was at chance, so the project does
  not claim general copy behavior from the flagship.
- The model was trained on synthetic children's stories and should be expected
  to imitate that narrow register.
- The release license and weight-release posture are not final until the user
  reviews `docs/phase5-licensing-notes.md` and chooses the license.

## Ethical and Safety Notes

- The model is small and narrow, but it can still generate misleading,
  repetitive, or low-quality text.
- It should not be presented as truthful or broadly knowledgeable.
- Dataset provenance and limitations should travel with any public weights.
- Public release should include the final license, NOTICE/CITATION files as
  needed, and the final A/B review status.
