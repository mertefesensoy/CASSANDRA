# Phase 4 Flagship Evaluation Report

Date: 2026-07-07 · prepared by Claude over Codex's Stage 51 to 55 artifacts.
Companion documents: `docs/phase4-flagship-midrun-report.md` (mid-run
retrospective and storage plan), `experiments/tiny_language_lab/RESULTS.md`
(Stage 55 closeout), ADR 0013 (phase scope).

Figures live in `docs/figures/phase4/` and are regenerated from raw run
artifacts by `experiments/tiny_language_lab/make_phase4_figures.py`. Every
number in this report traces to a run file, a checkpoint, or a cited paper.

## 1 · What Phase 4 achieved

Phase 4 turned the lab from a hypothesis machine into a model producer while
keeping the hypothesis discipline intact:

1. **Stage 53 (H020, KILL)** measured that the frozen order-4 analytic prior,
   helpful under tiny trainable surfaces, becomes a late-training handicap
   under full-body training (`+0.100` NLL at 2000 steps, surviving a
   lower-LR rerun at `+0.055`). This is a real scientific result: the
   prior's value is regime-dependent, and it decided the flagship's
   initialization by evidence instead of taste.
2. **Stage 54 (H021, CONFIRM)** built the sparse order-5 backoff prior and
   showed the analytic floor scales with corpus size (`-0.117` NLL versus
   order 4, all seeds), moving the 25.25M crossover from 1000 to 2000 steps.
   The Stage 35 "highest estimable order" ceiling is corpus-relative, as
   hypothesized.
3. **Stage 55 (flagship)** trained the largest model this laptop has ever
   produced: `201,609,249` parameters, character-level, 50,000 steps,
   reaching **sampled validation NLL `0.556410` (`0.8027` bits/char)**, with
   two 20k-step replicas (`0.6090`, `0.5969`) proving the recipe is not a
   single-seed fluke. Training survived a mid-run OneDrive checkpoint
   failure through the proven resume path, and the final artifact ships as
   both a resume-capable `.pt` and a **verified ONNX export**.

The evidence trail: 55 stages, 21 hypotheses, 13 ADRs, every negative
preserved.

## 2 · The artifact under evaluation

| Property | Value | Decided by |
| --- | --- | --- |
| Architecture | 16-layer decoder-only transformer, d=1024, 16 heads | D3a sizing gate (measured VRAM and wall-clock) |
| Parameters | 201,609,249 (all trainable, no frozen prior) | Stage 53 KILL verdict |
| Substrate | Character-level, vocab 33 | ADR 0011 D1 (inherited; see limitations) |
| Context | 256 chars | D3a sizing gate |
| Position encoding | RoPE | Stage 45 modern-baseline bridge |
| Optimizer | Muon, constant lr 0.01, batch 8 x grad-accum 2 | Stage 45; LR schedule gap noted in retrospective |
| Corpus | TinyStories, 494.09M chars (420M train / 74.1M val) | ADR 0011 D2 corpus rescale |
| Budget | 50k steps = 204.8M training chars (~0.49 epoch) | ADR 0013 D3 token target |
| Precision | fp32 (no autocast) | trainer default; retrospective item |
| Checkpoints | every 5000 steps with optimizer state, resume proven | ADR 0013 accepted-costs clause |

## 3 · Verification: does it work properly

Three independent checks, all run 2026-07-07 by
`experiments/tiny_language_lab/phase4_validate.py`:

1. **ONNX parity.** The Nsight export
   (`artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx`)
   was compared against the PyTorch checkpoint it came from on real
   validation text: **max logit difference `8.6e-6`, mean `1.0e-6`, top-1
   agreement `100%` across all 256 positions.** The exported graph is the
   model, exactly.
2. **High-precision re-evaluation.** The recorded closeout numbers used 16
   sampled batches (about 33k chars). The suite re-scored every final
   checkpoint on millions of deterministically spread validation chars
   (chunked non-overlapping windows, context reset per window). Results in
   Section 4; the recorded numbers are confirmed within sampling noise.
3. **Behavioral spot checks.** Reproducible samples from each checkpoint at
   two temperatures are stored in `runs/stage55_validation_suite.md` for the
   human review packet.

Also relevant, verified earlier: checkpoint resume reproduces an unbroken
run's eval exactly (Stage 55 D3a resume proof), and profiler measurements
show a stable `31 to 33 ms` fp32 forward pass at batch 1, sequence 256, peak
working set `1.22 GB` (Nsight profile,
`C:\cassandra_runs\nsight_reports\...nv-dld-report`).

## 4 · Results

High-precision chunked validation NLL (this report's suite; fills in the
official record):

| Model | Params | Steps | Chunked val NLL | Bits/char | Recorded (sampled-16) |
| --- | ---: | ---: | ---: | ---: | ---: |
| flagship seed 7 | 201.6M | 50,000 | 0.563231 | 0.812571 | 0.556410 |
| replica seed 11 | 201.6M | 20,000 | 0.592702 | 0.855089 | 0.609039 |
| replica seed 19 | 201.6M | 20,000 | 0.595205 | 0.858699 | 0.596864 |
| Stage 51 reference | 25.25M | 5,000 | 0.824691 | 1.189778 | 0.821911 (seed 7) |

Every recorded number is confirmed within sampling noise (8M chars per
model, deterministic coverage of the whole validation split). One new fact
falls out: under the tight eval the two 20k replicas are nearly identical
(`0.5927` vs `0.5952`), so the recorded 20k seed spread (`0.579` to
`0.609`) was mostly 16-batch eval noise, not true seed variance. The recipe
is more stable across seeds than the closeout table suggests.

Reading the figures:

- **Figure 1 (learning curves).** All three seeds track one curve family;
  the 20k-to-40k plateau in seed 7 resolved into a final dip (`0.5496` at
  45k), confirming the mid-run "drift" was sampled-eval noise, not
  divergence. The whole 200M family sits far below the best 25M result.
- **Figure 2 (capacity ladder).** Gradient training beats both analytic
  floors at every size once budgets pass the crossover; the flagship extends
  the ladder by an order of magnitude in parameters and 25x in budget.
- **Figure 3 (H019 crossover).** The Stage 52 physics in one image: a
  constant analytic floor (`~1.104` NLL) crossed by capacity-ordered
  learning curves. Capacity is monotonically helpful at every budget
  measured (16 of 16 cells).
- **Figure 5 (efficiency frontier).** Quality against single-GPU cost, from
  a 2-minute analytic prior (1.42 bits) to the 26-hour flagship (0.803
  bits). Each roughly 10x in compute has bought roughly 0.2 bits/char on
  this corpus so far.

## 5 · External anchor: zero-shot text8

To place the models against the published record, the suite scores them on
**text8** (Mahoney's 100M-char lowercase Wikipedia benchmark; standard test
split = final 5M chars). The lab alphabet is a strict superset of text8's
27 symbols, so no vocabulary surgery is involved; this is a genuine
zero-shot, out-of-domain evaluation of TinyStories-trained models.

| Model | text8 test bits/char (zero-shot) |
| --- | ---: |
| Cassandra flagship 201.6M | 2.8817 |
| Cassandra Stage 51 25.25M | 3.3118 |
| GPT-2 117M, zero-shot (Radford et al. 2019) | 1.17 |
| GPT-2 1542M, zero-shot (Radford et al. 2019) | 0.98 |
| Strong text8-trained models (e.g. Transformer-XL) | ~1.0 to 1.1 |

Three honest readings (full 5M-char test split, both models):

1. **The specialization gap is now a measured number.** In-domain 0.813
   bits/char against out-of-domain 2.882 is a 2.07 bits/char gap: most of
   the flagship's headline quality is TinyStories specialization, not
   general English modeling. This was expected (the corpus is a
   deliberately simple register and the model has never seen encyclopedic
   text) and is now quantified rather than assumed.
2. **Scale helps out-of-domain too.** The flagship beats the 25M reference
   by 0.43 bits/char on text that neither model ever saw (2.88 vs 3.31),
   so the capacity-monotone fact from Stage 52 extends, weakly, beyond the
   training domain.
3. **Against the published record**, GPT-2 117M's zero-shot 1.17 is the
   fair sibling comparison and it is far ahead, because WebText is broad
   where TinyStories is narrow. The gap (2.88 vs 1.17) measures what
   training-corpus breadth buys at similar parameter count, and it is the
   strongest argument for the Phase 5 substrate discussion (BPE, broader
   corpus) in the retrospective.

Chunked-evaluation caveat: our convention resets context at every 256-char
window and is slightly pessimistic versus sliding-window scoring used by
some published numbers.

## 6 · Comparison with GPT-1

The user asked where this work stands next to the first-generation GPT.
Facts from the primary source (Radford et al. 2018, section 4.1):

| Property | GPT-1 (2018) | Cassandra flagship (2026) |
| --- | --- | --- |
| Parameters | 117M | 201.6M |
| Architecture | 12-layer decoder-only, d=768, 12 heads | 16-layer decoder-only, d=1024, 16 heads |
| Tokenizer | BPE, 40,000 merges | Character-level, vocab 33 |
| Context window | 512 BPE tokens (~2,000 chars) | 256 chars |
| Position encoding | Learned embeddings | RoPE |
| Corpus | BooksCorpus, 7,000+ books (~1B tokens, broad register) | TinyStories, 494M chars (children's stories register) |
| Data seen in training | ~100 epochs of ~1B tokens | 204.8M chars (~0.49 epoch) |
| Optimizer | Adam, lr 2.5e-4, warmup + cosine anneal | Muon, constant 0.01 |
| Regularization | dropout 0.1, modified L2 | none (dropout 0) |
| Hardware | ~1 month on 8 GPUs (2018 class) | ~26 GPU-hours on one laptop RTX 4070 |
| Reported LM quality | token perplexity 18.4 (~0.74 bits/char at ~5.7 chars/word, approx conversion) | 0.803 bits/char on TinyStories val |

What the comparison legitimately says:

1. **Architecture lineage is direct.** The flagship IS a GPT-1-class
   decoder-only transformer with eight years of community refinements
   (RoPE instead of learned positions, Muon instead of Adam, activation
   checkpointing, and a modern eval protocol). Same species, newer organs.
2. **The compute story is the striking one.** GPT-1 needed a 2018 industry
   lab month on 8 GPUs. A functionally similar-size model now trains
   overnight on a consumer laptop, and the single biggest reason is not one
   invention but the accumulated recipe: better optimizers, better position
   encodings, memory tricks that let 200M parameters fit in 8 GB, and a
   corpus engineered to be learnable.
3. **What the comparison does NOT say.** GPT-1 modeled the full breadth of
   book English at 4x our effective context and transferred to 12 NLP
   benchmarks; the flagship models a deliberately simplified children's
   register and has no downstream-task story. The bits/char rows are on
   different corpora and the GPT-1 row passes through a chars-per-word
   conversion; treat them as anchors, not a leaderboard. On the shared
   yardstick that exists (zero-shot text8), GPT-2 117M's 1.17 against our
   2.88 shows exactly what corpus breadth buys at this parameter scale.

Figure 4 renders both panels of this table.

## 7 · Trying the models like a human user

`experiments/tiny_language_lab/playground.py` starts a local web UI
(gradio, loopback only):

```powershell
python .\experiments\tiny_language_lab\playground.py            # GPU
python .\experiments\tiny_language_lab\playground.py --device cpu
```

- **Generate tab**: pick any final checkpoint (flagship, both replicas,
  Stage 51 reference), type a prompt, tune temperature, top-k, and length.
  Prompts are normalized into the 33-char alphabet automatically.
- **Blind A/B tab**: one prompt, two randomly chosen models, unlabeled
  outputs, three vote buttons. Votes append to
  `runs/human_ab_votes.jsonl` with the identities recorded, and the reveal
  happens only after the vote. This is the ADR 0013 required human review,
  upgraded from impression to protocol; 20 to 30 votes across varied
  prompts gives a defensible flagship-versus-Stage-51 verdict, and the
  Stage 55 proxy-score anomaly (flagship proxy 4.33/6 versus Stage 51's
  5.67/6 despite far better NLL) is exactly the question it should settle.

## 8 · Limitations and threats to validity

1. In-domain quality is bounded by a 256-char context on a corpus whose
   stories run 750 to 1500 chars; no sample can exhibit whole-story
   structure. This is a design constraint, not a training failure.
2. The mixed-budget seed package (one 50k, two 20k) means the headline
   number is a single seed; the replicas bound seed variance at 20k
   (`0.579` to `0.609` sampled) but not at 50k.
3. Char-level bits/char on TinyStories is not comparable to word-level
   perplexities on broad corpora without conversion assumptions; every
   converted number in this report is marked as such.
4. The equal-compute control (an 85M model given the flagship's 16 hours)
   was never run; the D3a gate compared candidates at equal steps. The
   85M-at-matched-wall-clock arm remains the cheapest missing experiment.
5. The generation proxy rubric disagrees with NLL across model classes
   (Section 7); until the human A/B review lands, no coherence claim is
   made. This carries ADR 0011 and ADR 0013's rule forward.

## 9 · References

- Radford, Narasimhan, Salimans, Sutskever (2018). Improving Language
  Understanding by Generative Pre-Training. (GPT-1; specs and ppl from
  section 4.1.)
- Radford et al. (2019). Language Models are Unsupervised Multitask
  Learners. (GPT-2; zero-shot text8/enwik8 anchors, Table 3.)
- Eldan, Li (2023). TinyStories: How Small Can Language Models Be and
  Still Speak Coherent English? arXiv:2305.07759. (Corpus provenance.)
- Mahoney. text8 benchmark, mattmahoney.net/dc/textdata.
- Jordan et al. (2024). Muon optimizer, github.com/KellerJordan/Muon.
- `experiments/tiny_language_lab/RESULTS.md` Stages 51 to 55;
  ADR 0011 to 0013; `docs/phase4-intake.md`.
