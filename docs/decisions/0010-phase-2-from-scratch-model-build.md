# ADR 0010 · Phase 2 · From-Scratch Model Build on a Modded-nanoGPT Baseline

## Status

Proposed · Codex confirmation report produced at `docs/decisions/0010-confirmation-report.md`

## Context

Cassandra Phase 1 was an algorithm lab. It asked whether useful language-model behavior can be formed on consumer hardware by reducing brute-force gradient training through structure, reuse, compression, retrieval, search, or analytic initialization. Thirty-plus stages produced one load-bearing survivor and several clean kills.

Surviving finding · a frozen count-based n-gram prior used as an additive residual base, with a low-rank LoRA correction trained on top, is a bounded early-compute inductive-bias accelerator (ADR 0002). Its advantage is graded by the match between source order and prior order (ADR 0003), transfers to real natural text through order 4 (Stages 30 to 32), and decays as the full model catches up, on the order of 100 steps at current scale.

Retired findings · compact text-prefix external memory (ADR 0001) and data-side selection for the rank-2 residual (ADR 0004, consolidating Stages 11, 12, 33, 34). These branches are closed and out of scope for Phase 2.

Phase 2 changes the mode of work. Cassandra stops being only an algorithm lab and becomes a from-scratch model build that runs continuously while the algorithm research continues in parallel on the same harness. Two facts frame the decision.

Physical reality · the development machine is an RTX 4070 Laptop with 8 GB VRAM. VRAM, not raw throughput, is the binding constraint. A generally useful general-purpose model trained from scratch is not reachable at this scale. This is accepted, not contested. The build targets the honest reachable rung, and the code is written to scale up unchanged when hardware improves.

No-outsourcing principle · the goal is to build the model from scratch and own every weight, the full experience, with no dependence on a pretrained external model. This principle is recorded here so it is not relitigated later. From scratch means own weights, own training loop, own algorithm. It does not mean re-deriving standard architecture and optimizer components by hand. Adopting `modded-nanogpt` architecture and optimizer craft is analogous to using Adam or attention: a standard component, not a pretrained-model dependency. The model is trained from random initialization on Cassandra's own data with Cassandra's own frozen-prior module. No external checkpoint is ever loaded.

Field grounding · the intuition is literature-backed. Nguyen 2024 shows transformers learn n-gram rules progressively and that failing to learn low-count contexts is an optimization failure. Related capability results (`Can Transformers Learn n-gram Language Models`, arXiv 2410.03001) show transformers underperform plain counting on arbitrary n-gram distributions. Infini-gram shows n-gram structure still lowers neural perplexity but can harm open-ended generation. The constructive framing here, a frozen count prior as a residual base used to accelerate low-step training and characterized by a source-order and prior-order law, appears distinct from the analysis and inference-time-interpolation literature and is the publishable core.

## Decision

### D1 · Baseline

Adopt the `modded-nanogpt` architecture and optimizer stack as the Phase 2 training baseline, adapted and scaled down to the local hardware. In scope for adaptation: the Muon optimizer for hidden linear layers with AdamW for embeddings and the head, RoPE, and the current attention and normalization choices. This is an adaptation of ideas onto Cassandra's existing tiny transformer, not a fork-and-run of the upstream repository, which is tuned for an 8xH100, BPE, 124M, FineWeb setting. The adapted baseline replaces the plain gradient control as the reference against which the frozen-prior recipe is measured, under matched corpus, step budget, and hardware.

### D2 · Corpus

Move the primary corpus from `tiny_seed.txt` and normalized Tiny Shakespeare to TinyStories. This is the single highest-value change: it is the first corpus at which a small from-scratch model is expected to produce coherent, on-prompt English. Build a data pipeline that ingests locally staged or freshly downloaded TinyStories, encodes it, records the deterministic train/validation split, and can shard it for streaming under the VRAM budget.

### D3 · Tokenization · open sub-decision

Run TinyStories character-level first, to preserve continuity: the entire frozen-prior and order-law apparatus (`--residual-base count-ngram`, `--prior-order`) is defined over a character alphabet and ports unchanged. TinyStories is simple enough that character-level output is expected to be readable. Then introduce a small BPE vocabulary, 4k to 8k, as a separate ablation, which requires re-deriving the n-gram prior over BPE tokens. Do not change tokenization and corpus scale in the same run. The character-versus-BPE choice is recorded as open and will be resolved by measurement, not assumed now.

### D4 · Frozen prior role

The frozen n-gram prior rides along on the new baseline as the initialization and residual-base module. It is what makes this build Cassandra's own rather than a scaled nanoGPT clone. It is explicitly not claimed to be the lever that produces final model quality: ADR 0002 established that its advantage decays by roughly 100 steps, and a useful model needs far longer budgets where the benefit washes out. Its role is dual: a modest early-compute warm start for the build, and the object of the live research question in D5.

### D5 · Live research question · candidate `H019`

Register a new hypothesis to run in parallel with the build: the early-compute crossover budget, the step count at which full training catches the frozen-prior head start, moves as a function of model capacity and corpus complexity. Phase 1 located this crossover near 100 steps at char-level V=33 tiny scale. Phase 2 charts whether it shifts with model size and with the TinyStories and later corpora. This keeps the model build and the research on one harness and is the intended Phase 2 writeup.

### D6 · Evaluation

Add a generation-quality evaluation alongside the existing validation NLL and copy-accuracy metrics. At minimum: held-out prompt completion scored for coherence, grammaticality, and on-prompt relevance. Validation loss alone will not signal whether the model is producing usable text, which is the Phase 2 milestone.

### D7 · Hardware and scaling

Accept the 8 GB VRAM ceiling. Size the first build to the largest config that fits with gradient checkpointing and gradient accumulation, expected in the 1M to 30M parameter range at char-level. Write all data, training, and eval code to be hardware-agnostic so the same code runs on a 24 GB upgrade path, a used RTX 3090 being the reference value target, and on rented H100 or A100 time by the hour for occasional heavy runs. No architectural decision may hard-code the current VRAM budget.

## Consequences

### Positive

The build gives the real-data pipeline experience, get the data, sort it, tokenize it, train on it, evaluate it, that Phase 1 did not exercise. The frozen-prior work converts from a synthetic-corpus result into a clean ablation on real text against a strong, community-standard baseline. The no-outsourcing principle is preserved in full. The code scales up unchanged when hardware improves.

### Accepted costs

The final model at current hardware will be small and narrow, not generally useful. The frozen prior gives little to no benefit at the long budgets a usable model needs, so its value in Phase 2 is scientific and early-compute, not final-quality. Adopting the `modded-nanogpt` stack means a non-trivial adaptation effort onto the existing tiny transformer, scoped by the confirmation pass.

### Out of scope

External memory and retrieval interfaces (retired by ADR 0001). Data-side selection and curriculum filtering for the rank-2 residual (retired by ADR 0004). Loading any pretrained external checkpoint, ever, under the no-outsourcing principle.

## Open questions

1. Character-level versus BPE as the durable tokenization for the build (D3), to be resolved by ablation.
2. How much of the `modded-nanogpt` stack adapts cleanly onto the current tiny transformer at 8 GB, and at what effort (confirmation pass).
3. The largest model config that fits the VRAM budget at char-level (confirmation pass).
4. Whether the frozen-prior residual base composes with the Muon-optimized baseline without interaction effects that were absent under the plain gradient control.

## References

- `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`
- `docs/decisions/0003-graded-source-prior-order-law.md`
- `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`
- `docs/decisions/0004-retire-data-side-selection-rank2-residual.md`
- `experiments/tiny_language_lab/RESULTS.md`
- Nguyen 2024 · Understanding Transformers via N-gram Statistics · arXiv 2407.12034
- Liu et al. 2024 · Infini-gram · arXiv 2401.17377
- Karpathy llm.c and Jordan `modded-nanogpt` · NanoGPT speedrun baseline
- Eldan and Li 2023 · TinyStories
