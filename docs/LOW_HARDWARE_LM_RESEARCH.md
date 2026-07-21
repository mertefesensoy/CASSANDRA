# Cassandra Low-Hardware LM Research Program

Working date: 2026-06-15

## Core thesis

Most LLM progress is described as scale: more data, more parameters, more GPUs,
longer training. Cassandra should study the opposite pressure:

> How much language-model behavior can be formed when the training budget is
> treated as the scarce resource?

The practical target is not "build a frontier model from a laptop." The target is
to build a repeatable lab where very small models can be constructed, compared,
and improved using methods that reduce full brute-force training.

## Requirements from the goal

- Must run on a student laptop, ideally on CPU first and then on an RTX 4070.
- Must include normal training baselines so alternative methods have something honest to beat.
- Must include at least one non-backpropagation or parameter-by-parameter method.
- Must separate "language-model behavior" from "large language model" claims.
- Must record wall time, parameter count, validation loss, generated samples, and hardware notes.
- Must make failure informative. A method that stalls is still data.

## Research map

### 1. Tiny baselines

Start with models small enough to understand completely: bigrams, small recurrent
models, and tiny transformers. This gives Cassandra a control case before more
ambitious methods arrive.

Relevant anchors:

- nanoGPT shows a minimal GPT training loop and a practical tiny Shakespeare path.
- TinyStories showed that models under 10M parameters can produce coherent story-like text when the data distribution is intentionally simple.

### 2. Parameter-efficient adaptation

LoRA and QLoRA do not train a model from nothing. They freeze most of a pretrained
model and train small adapter matrices or quantized adapter paths. For Cassandra,
their deeper lesson is that the useful trainable surface can be much smaller than
the full parameter count.

Hypothesis: a Cassandra method may not need to alter all weights. It may need to
find the smallest trainable surface that changes behavior measurably.

*Sources & Prior Art:*
- **The Low-Rank Bottleneck and Strong Priors**: LoRA enforces a rigid inductive bias by restricting updates to a low-dimensional subspace. While this preserves the pre-trained "strong prior" and acts as a powerful regularizer, it imposes an adaptation ceiling. Stage 37 demonstrated that in a "prior-dominated" regime (e.g., using a high-order n-gram prior on natural text), adding a rank-2 to rank-4 residual adapter results in intrinsically tiny (or even negative) marginal gains. The adapter lacks the capacity to improve upon the strong prior and may introduce "intruder dimensions" that degrade performance.
- **NLL Divergence and Behavior-Forming PEFT**: Intrinsic metrics like NLL (perplexity) frequently diverge from downstream behavioral accuracy. Global NLL averages performance across all tokens (dominated by common syntax), masking a model's ability on critical reasoning tokens. Stage 38 proved this divergence: a rank-2 residual adapter failed to improve NLL over the frozen prior but successfully unlocked attention-mediated in-context copying, jumping from ~12.5% chance to ~32% accuracy. The tiny residual acts exclusively as a "behavior-forming surface" to wire up specific circuits, even when global probability distributions stagnate.
- **LoRA Rank Saturation and Intrinsic Dimension**: Increasing LoRA rank does not guarantee monotonic performance improvements. If the task's intrinsic dimension is low, excess rank can lead to "rank saturation," optimization instability, and overfitting, especially if hyperparameters (like alpha and learning rate) are not jointly scaled. Stage 39 confirmed this: doubling capacity from rank 2 to rank 4 on the copy task degraded mean accuracy from ~32% back down to ~27%, demonstrating that simple over-parameterization without stabilization fails.
- **Held-Out Key Generalization Failure in ICL**: Models fine-tuned for in-context learning can memorize input-output formatting for seen tokens while failing to generalize the underlying operation to held-out tokens. Stage 40 measured a free-vocabulary held-out collapse, and Stage 41 showed that candidate-restricted forced choice did not rescue it: both the rank-2 residual and the full-context control stayed at `0.000000` held-out choice accuracy. The result is not a clean cheap-surface-only reversal because the full control also collapses, but it does show that the current narrow identity-copy protocol does not induce held-out transfer at this budget.
- **Forced-Choice Evaluation and Surface Form Competition**: Generative metrics often suffer from "surface form competition," where models assign probability mass to synonyms or formatting artifacts, obscuring true capabilities. Forced-choice evaluation mitigates this by restricting the argmax to valid candidates, potentially revealing "hidden generalization." However, Cassandra Stage 41 demonstrated that the Stage 40 held-out failure was a structural circuit collapse, not an emission artifact. Even under restricted forced-choice evaluation, the held-out accuracy and Mean Reciprocal Rank (MRR) remained at the floor, proving the model fundamentally failed to route the unseen keys.
- **Induction Circuit Intrinsic Dimension and Capacity Walls**: Mechanistically, in-context learning requires "induction heads" (a QK matching circuit and an OV copying circuit). Forming these structures requires a minimal representational subspace (intrinsic dimension). Stage 42 demonstrated that when a task is made "memorization-proof" (random payload keys), a rank-2 residual adapter completely fails to form the circuit (`0.0638` accuracy vs `0.0625` chance), while a full-parameter model succeeds (`0.226`). This indicates a hard geometric capacity wall: the true intrinsic dimension of the generalized copy circuit is greater than rank 2. Furthermore, this stage exhibited NLL/behavior decoupling: the rank-2 adapter improved NLL by `0.062` over the floor without forming the behavior, mirroring Stage 38 where behavior formed without improving NLL.
- **Induction Heads and Zero-Shot Copy Failure in SLMs**: (Olsson et al., 2022) established that induction heads emerge suddenly during training and cause in-context learning. Stage 55 tested the 200M flagship on a letters-only zero-shot copy probe, scoring only 0.0605 against 0.0625 chance. This proves that despite broad structural capability, the flagship failed to form induction heads. It implies a structural generalization collapse on character-level data without an explicit copy curriculum.

### 3. Distillation and synthetic data

DeepSeek-R1 and its distilled releases are important because the small public
models inherit behavior from a larger reasoning model through generated data and
supervised fine-tuning. This is not "no training," but it moves part of the cost
from the student model into data construction.

Hypothesis: on laptop hardware, the best path to useful behavior may be:

1. narrow task,
2. generated or rule-verified examples,
3. small model,
4. strict evaluation,
5. only then larger architecture.

### 4. Rule rewards instead of preference mysticism

For math and code, rewards can be checked by a verifier: answer matches, tests
pass, format is valid. This matters because small experiments cannot afford
ambiguous reward models.

Hypothesis: Cassandra should prefer tasks where correctness is externally
checkable before attempting open-ended chat behavior.

### 5. Analytic and search-based formation

The most Cassandra-specific direction is parameter formation without ordinary
gradient descent:

- direct count-based construction,
- spectral or co-occurrence initialization,
- coordinate search,
- evolutionary mutation of small parameter blocks,
- retrieval tables that substitute memory for weights,
- hybrid methods that analytically initialize weights before limited gradient updates.

These methods will not beat transformers immediately. Their value is that they
test the exact question the project cares about: what part of training is
necessary, and what part is just expensive habit?

*Sources & Prior Art:*
- **MeZO: Fine-Tuning Language Models with Just Forward Passes (Malladi et al., 2023)**: Proves that zeroth-order optimization can match backpropagation on 30B+ parameter models by using random parameter perturbations. It relies on the "blessing of scale" where massive loss landscapes are smooth and redundant. Cassandra Stage 36 showed that this property does not scale *down* to tiny constrained adapters, where random perturbations are highly destructive.
- **Evolution Strategies (Salimans et al., 2017)**: Uses population-based mutation and evaluation instead of backpropagation. While successful for aligning large models on sparse rewards, Stage 36 demonstrated that ES fails to optimize sharp, low-capacity residual surfaces compared to AdamW.

### 6. Training Dynamics, Data Mixing, and Curriculums

Data mixing strategies control the composition of training corpora, while curriculum learning controls the pacing of data delivery (simple to complex).

Hypothesis: small models are highly sensitive to data composition and ordering. Static manual mixture heuristics or naive simple-to-complex curricula may induce catastrophic forgetting or sub-optimal optimization landscapes.

*Sources & Prior Art:*
- **Data Mixing Laws and DoReMi**: (Xie et al., 2023; Ye et al., 2024) prove that small proxy models can predict the optimal mixture weights for large models mathematically, surpassing manual heuristics. Stage 58 relies on a manual 25% TinyStories / 75% Text8 split. The next iteration should evaluate dynamic or proxy-derived data mixtures to close the specialization gap optimally.
- **Curriculum Learning from Simple to Complex**: (Bengio et al., 2009) proposed learning easy examples first. However, Stage 58 tests a hard domain shift (12.5k steps TinyStories -> 29.5k steps text8). The risk is catastrophic forgetting of the "simple" phase grammar.

### 7. Reinforcement Learning and the Emergence of Reasoning

Once a model has mastered basic grammar and recall, scaling up Supervised Fine-Tuning (SFT) yields diminishing returns on complex logic. To enable models to "think", the training paradigm must shift from imitation to search and reinforcement learning (RL).

Hypothesis: In laptop-scale models, next-word prediction cannot natively produce deep reasoning. We must freeze the base model and introduce a pure Reinforcement Learning reward on a scratchpad task to force the emergence of internal Chain-of-Thought and test-time compute scaling.

*Sources & Prior Art:*
- **OpenAI o1 and DeepSeek-R1**: (2024-2025) Prove that reasoning capabilities emerge when models are rewarded for the correctness of their final answer rather than the exact token sequence. This incentivizes the model to generate internal "thinking tokens" (scratchpad space) to verify and backtrack. This shift from train-time scaling to test-time compute scaling represents the next frontier, and Cassandra's Stage 9-12 Verifier-guided micro-reasoning is a deterministic precursor to this RL loop.

## Experimental ladder

### Stage 0 · Bigram parameter lab

Implemented in `experiments/tiny_language_lab`.

Methods:

- `count`: build logits from smoothed transition counts.
- `coordinate`: change one logit at a time and keep improvements.
- `gradient`: optimize the same logits with AdamW.

Why this stage matters:

- The full parameter matrix is inspectable.
- Parameter-by-parameter search is cheap enough to run.
- Count construction creates a real no-gradient baseline.
- Gradient training remains available as the control.

Completion condition:

- All three methods run.
- Each reports train loss, validation loss, bits per character, parameter count, and sample text.

### Stage 1 · Tiny transformer baseline

Implemented in `experiments/tiny_language_lab/cassandra_tiny_transformer.py`.

This adds a normal gradient-trained causal transformer with the same corpus
protocol and metrics as the bigram lab.

Target sizes:

- toy: 50K to 250K parameters,
- small: 1M to 5M parameters,
- stretch: 10M to 30M parameters on GPU.

Completion condition:

- A normal gradient-trained transformer produces reproducible loss curves.
- The bigram methods remain as cheap baselines.

Current status:

- The toy transformer is implemented.
- A 100-step CPU run is recorded in `experiments/tiny_language_lab/RESULTS.md`.
- The next useful comparison is a fixed-budget run against the Stage 0 bigram
  methods on the same corpus.

### Stage 2 · Analytic transformer initialization

Implemented as `--init count-bigram` in
`experiments/tiny_language_lab/cassandra_tiny_transformer.py`.

Use corpus statistics to initialize embeddings and output projections before
training. Compare against random initialization under the same step budget.

Test:

- Does the initialized model reach the same validation loss in fewer steps?
- Does it produce better samples under tiny budgets?
- Does the advantage disappear as training time increases?

Current status:

- The count-bigram initializer fits the smoothed transition logits into the
  transformer with max absolute error `0.00000453`.
- It beats random initialization at 0 and 20 steps.
- It loses the advantage by 50 steps under the current learning rate.
- Next hypothesis: keep the analytic count prior fixed and train a residual
  transformer path around it.

### Stage 3 · Count-prior residual transformer

Implemented as `--residual-base count-bigram` in
`experiments/tiny_language_lab/cassandra_tiny_transformer.py`.

Keep count-derived bigram logits as a frozen base model and train a transformer
only as a residual correction on top of that base.

Test:

- Does a protected analytic prior avoid the 50-step degradation seen in Stage 2?
- How many residual parameters are needed before context improves validation loss?
- Can the residual path be made adapter-like so only a small subset is trainable?

Current status:

- The frozen prior starts at train NLL `2.115222` and val NLL `2.502358`.
- Unconstrained residual training overfits quickly.
- Residual L2 or a lower learning rate gives a small 50-step validation
  improvement, reaching about `2.4897`.
- The method beats trainable count-seeded initialization at 50 steps, but not
  the best random transformer point on this tiny corpus.

### Stage 4 · Head-only and adapter-only residual updates

Implemented as `--train-scope head` and `--train-scope adapters` in
`experiments/tiny_language_lab/cassandra_tiny_transformer.py`.

Freeze most of the residual transformer and train only a small correction
surface, starting with the output head and then small adapter matrices.

Test:

- Can a smaller residual surface preserve the count prior better?
- Does head-only training match the best constrained residual run with far fewer
  trainable parameters?
- Does an adapter path improve over head-only without opening the full model?

Current status:

- Head-only trains 1885 parameters and is cheap but weak.
- Adapter rank 4 trains 2909 parameters and reaches val NLL `2.483014` at
  50 steps.
- Rank 4 adapters beat the best constrained full-residual run while training
  about 2.7% as many parameters as the full toy transformer.
- The best random transformer point on this tiny corpus still wins, so this
  needs larger-corpus and multi-seed replication before becoming a claim.

### Stage 5 · Larger corpus and multi-seed replication

Implemented with `make_synthetic_corpus.py` and `cassandra_compare.py`.

Generate a larger controlled corpus and compare methods across multiple random
seeds before treating tiny-corpus rankings as meaningful.

Test:

- Does the adapter result survive a larger corpus?
- Does the low-trainable-parameter method beat the random full transformer
  across seeds?
- Which cheap residual surface is strongest: head-only or adapters?

Current status:

- A deterministic structured corpus of about 25.6K characters is generated.
- Three seeds were run at 50 steps with sampled evaluation.
- Both frozen-count residual methods beat the random full transformer across
  all three seeds.
- Head-only is currently the best mean validation result, with 2600 trainable
  parameters.

### Stage 6 · LoRA-style training

Implemented as `--train-scope lora` in
`experiments/tiny_language_lab/cassandra_tiny_transformer.py`.

Freeze most of a tiny transformer and train only small injected matrices plus
the residual output head.

Test:

- How small can the trainable surface be before behavior stops improving?
- Does a low-rank update beat full training under a fixed time budget?

Current status:

- Rank 1 LoRA trains 4648 parameters and beats head-only and adapter rank 4 on
  the structured corpus.
- Rank 2 LoRA trains 6696 parameters and reaches mean val NLL `1.992162`
  across three seeds.
- The full random transformer trains 107304 parameters and reaches mean val NLL
  `2.078493` under the same 50-step budget.
- The best current Cassandra recipe is frozen count prior plus low-rank
  residual projection updates.

### Stage 7 · Long-context synthetic corpus

Implemented with `make_long_context_corpus.py` and `--copy-probe-marker`.

Add a controlled corpus where the next token depends on information more than
one character back, so a bigram count prior should be insufficient.

Test:

- Does the count-prior head still win when bigram statistics cannot solve the
  task?
- Does LoRA make better use of context than head-only correction?
- Does the random full transformer regain the lead when context is essential?

Current status:

- Full random training regains the validation-loss lead on the long-context
  corpus.
- Frozen-count LoRA still improves over head-only, but cannot match full
  training on general NLL.
- Copy accuracy remains near chance even when NLL improves, including a
  500-step diagnostic.
- This proves Cassandra needs task-aware probes and verifier-style training
  signals, not just next-character loss.

### Stage 8 · Task-aware copy training

Implemented with `--copy-train-marker` and `--copy-loss-weight`.

Train on the `answer=` positions directly, while keeping ordinary validation
NLL unchanged. The copy probe becomes the behavior metric rather than a side
report.

Test:

- Does adding answer-position loss improve copy accuracy without brute-force
  full-model training?
- Can LoRA or adapters solve the copy task with a small trainable surface?
- Can a rule reward or rejection-sampled dataset beat plain next-character
  training on the same corpus?

Current status:

- Moderate copy weighting at 200 steps improves full-model copy accuracy from
  about `0.13` to `0.21`, but does not yet help rank-2 LoRA accuracy.
- Stronger copy weighting at 500 steps improves both full training and
  frozen-prior rank-2 LoRA: mean copy accuracy reaches `0.320175` for full
  training and `0.293859` for LoRA across three seeds.
- The LoRA path uses only `6631` trainable parameters versus `111271` for full
  training, but pays a higher ordinary validation-NLL cost.
- This is Cassandra's first result where a small residual surface learns more
  of a held-out behavior when the training signal targets that behavior
  directly.

### Stage 9 · Verifier-guided micro-reasoning

First implemented for the copy corpus with `--copy-sampler answer|mixed`.

Use tiny tasks where correctness can be checked automatically, such as
key-answer copying, arithmetic, balanced parentheses, simple code formatting, or
unit-test-passing snippets. The first verifier checks whether the character
after `answer=` matches the earlier `key=` value, then uses verified targets to
shape sampling.

Test:

- Can rule rewards shape behavior with less data than next-token training?
- Can rejection-sampled synthetic data improve a tiny model?
- Can verified target sampling improve behavior without discarding ordinary
  next-character context?

Current status:

- Answer-only sampling is too narrow: it hurts ordinary validation NLL and does
  not improve copy accuracy at 200 steps.
- Mixed sampling works better. With 25 percent verified answer-anchored windows
  at 200 steps and weight `50`, full training reaches copy accuracy `0.25`
  versus `0.214912` for weighted random windows at similar validation NLL.
- At 500 steps and weight `200`, mixed sampling reaches copy accuracy
  `0.364035` for full training and `0.302632` for frozen-prior rank-2 LoRA.
- The gain is uneven: copy accuracy improves, but copy NLL and ordinary
  validation NLL do not always improve. The next method should use a verifier
  reward or correction loop rather than only reshaping which windows are seen.

### Stage 10 · Verifier choice-loss correction

Implemented with `--copy-choice-weight`.

At verified answer positions, add an auxiliary loss that asks the model to
choose among the valid key characters only. This is a small correction-style
objective: the verifier says which answer characters are valid choices, and the
model gets a sharper signal than whole-vocabulary next-character loss.

Test:

- Can a verifier-derived choice loss improve copy behavior more cheaply than
  stronger token weighting?
- Does the correction help small LoRA residual surfaces differently than full
  training?

Current status:

- At 200 steps, weight `50`, a gentle choice weight `0.25` improves
  weighted-random full-model copy accuracy from `0.214912` to `0.25`, and LoRA
  from `0.114035` to `0.131579`.
- A stronger choice weight `1.0` helps LoRA accuracy more at 200 steps
  (`0.140351`) but hurts the full model.
- Choice loss does not help the mixed sampler, and at the 500-step,
  weight-`200` setting it hurts copy accuracy while improving LoRA copy NLL.
- The lesson is useful but humbling: a hand-written verifier correction can
  help small surfaces under tight budgets, but it is not a general replacement
  for a better reward or correction loop.

### Stage 11 · Failed-case replay correction loop

Implemented with `--copy-sampler failed|failed_mixed` and `--copy-mine-every`.

The model periodically predicts the verified training copy cases. A verifier
keeps only failed cases, and the sampler replays those failed answer windows
until the next mining pass. The mixed version keeps ordinary random windows in
the batch and reserves only a fraction for failed-case replay.

Test:

- Does replaying actual model mistakes beat static verified sampling?
- Can a tiny LoRA residual surface benefit more from failed-case replay than a
  full model?

Current status:

- Failed-only replay is too narrow. At 200 steps, it raises LoRA copy accuracy
  from `0.114035` to `0.157895`, but validation NLL worsens from `1.584720` to
  `2.043136`; the full model gets worse.
- Mixed failed replay is less damaging but does not beat static mixed sampling.
  At 200 steps and 10 percent failed replay, LoRA reaches copy accuracy
  `0.140351`, while full training reaches only `0.144737`.
- At 500 steps and weight `200`, failed mixed replay reaches `0.289474` for
  LoRA and `0.201754` for full training, below the Stage 9 mixed-sampling
  results.
- The correction loop needs to synthesize better correction examples or rewards,
  not merely replay the same failed answer window more often.

### Stage 12 · Generated correction examples

Implemented with `--copy-sampler correction|correction_mixed`.

After mining failed copy cases, synthesize compact correction strings such as
`key=e answer=e` and mix them back into training. This tests whether a verifier
can create a distilled rule example instead of replaying the same long failed
window.

Test:

- Do generated correction examples transfer to held-out long-context copy
  prompts?
- Can a small fraction of synthetic corrections outperform heavier answer-token
  weighting?

Current status:

- Correction-only training is too distribution-shifted and fails badly.
- Gentle mixed corrections work. With 5 percent generated correction examples,
  copy weight `10`, and 500 steps, full training reaches copy accuracy
  `0.807018` with validation NLL `0.519950`.
- The fair same-weight baseline reaches copy accuracy `0.425438`, so the
  generated corrections are responsible for the jump.
- The LoRA residual path also improves over its same-weight baseline:
  `0.258772` versus `0.131579`, but it still trails the stronger Stage 8/9 LoRA
  recipes.
- This is the strongest evidence so far for Cassandra's central direction:
  a verifier-generated, tiny correction dataset can substitute for much more
  brute-force exposure on this controlled task.

### Stage 13 · Correction template shape

Implemented with `--copy-correction-template`.

Compare different generated correction artifacts:

- `compact`: `key=e answer=e`
- `focus`: `key=e noise=b answer=e`
- `prefix`: original failed line through the verified answer character
- `full`: original failed line

Test:

- Is the shortest distilled correction best, or does a more task-faithful
  correction transfer better?

Current status:

- `focus` corrections improve validation NLL versus compact corrections, but
  reduce full-model copy accuracy.
- `prefix` corrections are the strongest so far: at 500 steps, copy weight
  `10`, and 5 percent correction examples, full-model copy accuracy reaches
  `0.881579` with copy NLL `0.333680`.
- `prefix` also improves LoRA over compact corrections: copy accuracy rises
  from `0.258772` to `0.285088`, and copy NLL improves from `1.928378` to
  `1.767904`.
- The correction data should preserve just enough of the original task geometry
  to make the synthetic lesson transferable.

### Stage 14 · Retrieval-style probe context

Implemented with `--copy-probe-retrieval-template`.

At probe time, prepend a small external correction memory such as
`key=e answer=e` before the held-out copy prompt. This does not change training
or validation NLL; it tests whether useful behavior can move into an external
retrieval context instead of being stored entirely in weights.

Test:

- Does retrieval context improve copy behavior after correction-template
  training?
- Does retrieval context help a model that was not trained on generated
  correction examples?

Current status:

- Compact retrieval improves the prefix-correction-trained full model from
  `0.881579` to `0.960526` copy accuracy and lowers copy NLL from `0.333680`
  to `0.274268`.
- The same retrieval hint does not help the same-weight baseline without
  generated correction training: full-model accuracy moves from `0.425438` to
  `0.403509`.
- Retrieval hurts or is neutral for the small LoRA residual path in this setup.
- The lesson is that retrieval is not magic by itself; the model needs a small
  amount of training that teaches it how to use the retrieved correction.

### Stage 15 · Retrieval-use training

Implemented with `--copy-sampler retrieval`, `--copy-sampler retrieval_mixed`,
and `--copy-train-retrieval-template`.

Instead of adding retrieval only at probe time, this stage synthesizes training
windows that include a retrieved hint before the original copy prompt:

```text
key=e answer=e
case 0436 key=e noise=b even budget; amber signal; gentle update; answer=e
```

The goal is to train the interface to external memory directly, not merely test
whether a correction-trained model can exploit memory at inference.

Test:

- Can direct retrieval-use training beat the same-weight baseline without
  generated correction examples?
- Can it match or beat prefix-generated correction training plus probe-time
  retrieval?
- Does template phrasing matter for retrieval-use training?

Current status:

- Compact retrieval-use training improves the full same-weight baseline with
  compact retrieval from `0.403509` to `0.776316` copy accuracy, and lowers copy
  NLL from `1.543095` to `0.608794`.
- It does not match the best Stage 14 result: prefix correction training plus
  compact retrieval reaches `0.960526` copy accuracy and copy NLL `0.274268`.
- Among direct retrieval-use templates, compact is strongest for the full model:
  `0.776316` copy accuracy versus `0.592105` for focus and `0.723684` for
  prefix.
- The small LoRA residual path does not yet learn the retrieval interface:
  compact and focus both reach only `0.149123` copy accuracy, with prefix at
  `0.140351`.
- This suggests a split: retrieval can move facts out of weights, but the model
  still benefits from a teacher-like correction trace that explains the memory
  format before retrieval is asked to carry behavior.

### Stage 16 · Correction-retrieval curriculum

Implemented with `--copy-sampler correction_retrieval_mixed`.

This stage tests a direct combination of the two strongest signals so far:
prefix correction traces and compact retrieval-use examples. Each mixed batch
contains ordinary corpus windows plus both generated correction examples and
retrieval-use examples. With `--copy-sample-fraction 0.1` and batch size `16`,
the practical mix is one correction example, one retrieval-use example, and
fourteen ordinary corpus windows per batch.

Test:

- Does simultaneous correction plus retrieval-use training beat correction-only
  training with probe-time retrieval?
- Does the extra retrieval practice help the small LoRA residual path?
- Does adding both auxiliary sources create interference?

Current status:

- The full model gets worse: Stage 16 reaches `0.600877` copy accuracy and
  `0.978377` copy NLL, while Stage 14 prefix-correction training with compact
  retrieval reached `0.960526` copy accuracy and `0.274268` copy NLL.
- The LoRA path improves over Stage 14 compact retrieval and Stage 15 compact
  retrieval-use training: `0.302632` copy accuracy versus `0.241228` and
  `0.149123`, with copy NLL `1.780110`.
- Validation loss also shows the tradeoff. The full model worsens to
  `0.471090`; the LoRA path lands at `1.592112`, close to the Stage 14
  correction-trained value `1.609495`.
- The likely lesson is interference. For the full model, simultaneous
  correction and retrieval-use examples muddy the clean correction signal. For
  the constrained LoRA model, the extra retrieval practice may help because it
  has too little trainable surface to infer the interface from corrections
  alone.
- The next version should test staged curricula rather than simultaneous mixing:
  first train the correction interface, then introduce retrieval-use examples.

### Stage 17 · Staged correction-then-retrieval curriculum

Implemented with `--copy-sampler correction_then_retrieval_mixed` and
`--copy-curriculum-switch-fraction`.

This stage tests ordering instead of simultaneity. The first phase uses
prefix-correction mixed training. After the switch point, training changes to
compact retrieval-use mixed training. Two schedules were tested:

- `0.5` switch fraction: 250 correction steps, then 250 retrieval-use steps.
- `0.8` switch fraction: 400 correction steps, then 100 retrieval-use steps.

Test:

- Does a staged curriculum avoid the interference seen in Stage 16?
- Is the full model better served by a short retrieval adaptation phase after a
  longer correction-teacher phase?
- Does the LoRA path prefer simultaneous or staged signals?

Current status:

- Ordering helps the full model compared with simultaneous mixing. The `0.5`
  switch reaches `0.833333` copy accuracy, and the `0.8` switch reaches
  `0.881579`; Stage 16 simultaneous mixing reached only `0.600877`.
- The late switch is better than the even split for the full model, suggesting
  that the correction teacher should dominate the schedule.
- Staging still does not beat Stage 14's simpler recipe: prefix correction
  training plus compact retrieval at probe time remains strongest at `0.960526`
  copy accuracy and copy NLL `0.274268`.
- The LoRA path does not benefit from staging here. Both switch schedules reach
  `0.236842` copy accuracy, below Stage 16's simultaneous `0.302632`.
- The working lesson is now sharper: full models benefit from clean correction
  learning before retrieval, while very constrained residual models may need
  simultaneous pressure because they cannot preserve the correction interface
  through a phase switch.

### Stage 18 · Train-split memory retrieval probe

Implemented with `--copy-probe-retrieval-source memory`.

Stage 14 showed that compact retrieval context can help a correction-trained
full model, but its probe hint was derived from the held-out target answer.
Stage 18 replaces that with an honest lookup table built only from the training
split. The table maps each observed `key=` value to the majority verified
`answer=` value, then the held-out copy probe retrieves from that table.

Test:

- Does the Stage 14 retrieval result survive when the hint comes from train
  memory rather than the held-out target?
- Is retrieval still useless for the same-weight baseline without generated
  correction training?
- Does the report expose memory coverage and conflicts clearly enough for later
  external-memory experiments?

Current status:

- The train-split memory table has full coverage for this corpus: 8 entries,
  435 training observations, 0 conflicts, and 76 held-out probe hits in the
  smoke run.
- The correction-trained full model exactly preserves the Stage 14 compact
  retrieval result: `0.960526` copy accuracy and `0.274268` copy NLL.
- The same-weight baseline remains weak under honest memory retrieval:
  `0.403509` copy accuracy for the full model and `0.109649` for rank-2 LoRA.
- The LoRA correction-trained path also matches the earlier result: `0.241228`
  copy accuracy, so memory lookup does not fix the constrained surface.
- The result validates the measurement rather than adding a new capability.
  Retrieval was not secretly using held-out labels in Stage 14's reported
  effect, but this corpus is still too easy as a memory test because every
  validation key appears in training.
- The next memory experiment should stress generalization and retrieval
  failure: unseen keys, ambiguous key-answer memories, larger answer alphabets,
  missing memories, or retrieval from multiple candidate rows.

### Stage 19 · Memory-reliance corruption ablation

Implemented with `--copy-probe-retrieval-corrupt wrong-answer`.

Stage 18 made the retrieval source honest, but the copy task still has an
identity confound: the correct answer equals the key, and the compact hint
contains the answer character directly. Stage 19 keeps the retrieved memory
format valid but replaces the retrieved answer with a deterministic different
valid key character.

Test:

- Does a wrong retrieved answer hurt the correction-trained model?
- Is the Stage 14 to 18 retrieval gain genuine hint use or only a decorative
  format effect?
- Does the model collapse to chance when memory is wrong, or fall back on the
  local key in the held-out line?

Current status:

- The corrupted probe had full memory coverage: 8 entries, 76 hits, 0 misses,
  and 76 corrupted hints per run.
- The full model dropped from `0.960526` copy accuracy with correct memory to
  `0.824561` with corrupted memory. That is also below the no-hint Stage 13
  value of `0.881579`.
- The drop shows partial memory reliance: the model does attend to the retrieved
  answer, but it does not fully follow it, because accuracy remains far above
  chance.
- The LoRA path dropped from `0.241228` with correct memory to `0.166666` with
  corrupted memory, below its no-hint value of `0.285088`.
- This validates retrieval as a behaviorally relevant context signal, not merely
  a decorative prefix. It still does not prove useful external memory, because
  the local line already contains the answer-bearing key.
- The next memory stage should remove the identity shortcut by using held-out or
  non-identity key-to-answer mappings where the retrieved memory carries
  information the local prompt does not contain.

### Stage 20 · LoRA capacity sweep for curriculum interference

Implemented with rank-specific configs
`count_prior_lora_r4_corrretmix`, `count_prior_lora_r8_corrretmix`,
`count_prior_lora_r4_corrthenret`, and `count_prior_lora_r8_corrthenret`.

This stage tests Claude's capacity explanation for the Stage 17 split. Rank-2
LoRA preferred simultaneous correction plus retrieval training, while the full
model preferred staged correction then retrieval. The capacity hypothesis
predicts that raising LoRA rank should make the staged-minus-simultaneous gap
increase toward the full-model regime.

Test:

- Does the staged-minus-simultaneous gap rise monotonically as LoRA rank grows?
- Does rank 8 make staged training match or beat simultaneous training?
- Is trainable surface size sufficient to explain why rank-2 LoRA preferred
  simultaneous mixing?

Current status:

- Rank 2, already measured, has a gap of `-0.065790`.
- Rank 4 narrows the gap to `-0.013158`.
- Rank 8 falls back to `-0.061404`, with simultaneous training reaching
  `0.359649` copy accuracy and staged training reaching `0.298246`.
- The capacity-only explanation is not supported cleanly. More rank helps the
  simultaneous schedule at rank 8 more than it helps the staged schedule.
- The result suggests that phase ordering may need rehearsal, not just more
  capacity: keep a small correction fraction alive during the retrieval phase
  rather than switching completely.

### Stage 21 · Non-identity memory mapping probe

Implemented with `make_memory_mapping_corpus.py` and
`--copy-verify-mode key-answer`.

The earlier copy task was an identity task: `key=a` implied `answer=a`. Stage 21
uses a fixed non-identity mapping instead, for example `key=a answer=h` and
`key=b answer=e`. This removes the shortcut where a correct memory hint merely
repeats the key. The local line still contains the key, but it no longer
contains the answer directly unless the model has learned the mapping.

Test:

- Does compact train-split memory help when the answer is not identical to the
  key?
- Does corruption hurt more clearly when the memory carries answer information?
- Can prefix correction training alone transfer to a compact memory format on a
  non-identity task?

Current status:

- The generated corpus has 512 lines, 8 mapping entries, and 0 memory conflicts.
- The full model learns part of the mapping without a hint: `0.722944` copy
  accuracy.
- Correct compact memory hurts instead of helping: full-model accuracy drops to
  `0.450216`.
- Corrupted compact memory is slightly worse again at `0.398268`, which shows
  that the memory line is behaviorally active, but not in a useful way.
- The LoRA path remains weak across all three conditions: `0.194805` no hint,
  `0.177489` correct memory, and `0.203463` corrupted memory.
- The conclusion is negative but important. Stage 14 to 19's compact retrieval
  interface does not transfer to non-identity mappings just because the memory
  table is correct. External memory needs an interface curriculum when it must
  carry information rather than merely repeat an identity answer.
- The next stage should train directly on compact non-identity memory examples,
  then retest no-hint, correct-memory, and corrupted-memory probes.

### Stage 22 · Held-out external memory value test

Implemented from Claude Hypothesis 004 with a partitioned non-identity mapping
corpus, `retrieval_mixed` interface training on seen keys, full-corpus memory at
probe time, and split copy metrics for seen versus held-out keys.

Stage 22 asks the clean H4 question: can external memory supply a mapping that
the weights never saw? The generated corpus holds out keys `g h` from the train
split while keeping their answer symbols covered by seen keys through a
non-bijective mapping:

```text
a->h b->e c->g d->a e->c f->b g->h h->e
seen_keys: abcdef
holdout_keys: gh
```

The exact trainer split contained zero training examples for `key=g` and
`key=h`, while the full memory table contained all eight mappings.

Current status:

- Full model, held-out keys: no hint `0.194444`, correct memory `0.166667`,
  corrupted memory `0.166667`.
- Full model, seen keys: no hint `0.502564`, correct memory `0.502564`,
  corrupted memory `0.492308`.
- Full memory coverage was complete in the memory conditions: 8 entries, 512
  observations, 77 probe hits, 0 misses.
- The correct memory condition did not beat no hint on held-out keys and did not
  separate from corrupted memory.
- The seen-key interface gate was weak rather than strongly positive. Correct
  memory merely matched no hint, and corruption produced only a small drop.
- The LoRA path remains weak and does not show memory reliance: held-out
  accuracy was `0.083333` no hint, `0.138889` correct memory, and `0.166667`
  corrupted memory.

Conclusion:

Stage 22 does not support H4 under Cassandra's current compact retrieval
interface. The result should be treated as a local kill for this specific
fact-substitution mechanism, not as a proof that retrieval cannot ever
substitute for memorized facts. The external-memory branch should pause until a
new interface hypothesis or Gemini prior-art brief suggests a stronger training
signal. Stage 23 later measured the lower-value rank-2 rehearsal follow-up, and
Stage 24 later measured the higher-value corpus-complexity redirect from ADR
0001.

### Stage 23 · Rank-2 rehearsal for phase-switch forgetting

Implemented as the smaller alternative allowed by ADR 0001, after Stage 22
closed the compact external-memory branch. Stage 23 tests whether rank-2 staged
training failed because correction examples disappeared after the curriculum
switch.

The new sampler, `correction_then_retrieval_rehearsal_mixed`, keeps Stage 17's
correction phase and 0.8 switch point, then adds a small correction rehearsal
stream during the post-switch retrieval phase. The retrieval fraction remains
`0.1`.

Current status:

- Clean rank-2 staged baseline, Stage 17 late: `0.236842` copy accuracy.
- Rank-2 simultaneous baseline, Stage 16: `0.302632` copy accuracy.
- Stage 23 rehearsal at `0.05`: `0.245614` copy accuracy.
- Stage 23 rehearsal at `0.10`: `0.228070` copy accuracy.
- Rehearsal `0.05` gives a small `+0.008772` gain over clean staged training,
  but still trails simultaneous by `0.057018`.
- Rehearsal `0.10` is worse than clean staged training by `0.008772`.

Conclusion:

Small correction rehearsal does not rescue the rank-2 phase switch. The result
weakens the simple forgetting explanation: rank-2 staged training was not bad
only because correction examples disappeared after the switch. The copy-task
curriculum branch now has diminishing returns for the cheap residual surface.
Stage 24 then measured ADR 0001's redirect by characterizing the corpus regime
where the frozen count prior plus tiny residual surface beats full training.

### Stage 24 · Corpus-complexity regime for the cheap recipe

Implemented from Hypothesis 005 and the ADR 0001 redirect. Stage 24 asks where
the project's strongest positive result still holds: the frozen count prior plus
rank-2 LoRA beating a full random transformer on plain validation NLL under a
tight budget.

Codex added `make_complexity_corpus.py`, which mixes the structured corpus
generator and the long-context copy generator at long fractions `p = 0.00`,
`0.25`, `0.50`, `0.75`, and `1.00`. Each corpus has 512 lines and uses seed
`20260619`. The comparison matrix uses `--block-size 96`, sampled evaluation,
seeds `7 11 19`, and configs `random_full`, `count_prior_head`,
`count_prior_lora_r1`, and `count_prior_lora_r2` at both 50 and 100 steps.

Measured cheap-minus-full advantage, defined as `random_full` mean validation NLL
minus `count_prior_lora_r2` mean validation NLL:

| Long fraction | Count-bigram bits | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | -0.066921 | -0.438576 |
| 1.00 | 2.483352 | -0.162385 | -0.642078 |

Current status:

- At 50 steps, the cheap recipe wins through `p = 0.50` and loses at
  `p = 0.75` and `1.00`, so the crossover lies between `p = 0.50` and
  `p = 0.75`.
- At 100 steps, full training wins at every measured point, including the all
  structured generated corpus.
- The all-long corpus has the lowest count-bigram validation bits but the
  strongest full-model advantage, so count-bigram entropy alone does not explain
  the boundary. The generator's long-fraction knob is changing dependency type,
  not just local predictability.
- This supports Cassandra's cheap recipe as an early-budget accelerator in a
  bounded corpus regime, not as a universal replacement for full training.

Conclusion:

The core claim is now sharper and more honest: consumer hardware can expose a
real low-budget regime where analytic structure plus a tiny residual update beats
broader training, but that regime has a boundary and more steps can erase it.
Stage 25 then measured the time-budget surface at 10 and 25 steps. Gemini should
frame this branch against prior-strength, bias-variance, n-gram versus neural
language modeling, PEFT capacity work, and compute-scaling arguments before
Cassandra uses any public-facing novelty language.

### Stage 25 · Time-budget surface for the cheap recipe

Implemented from Hypothesis 006. Stage 25 completes the low-compute side of the
Stage 24 surface by running the same five corpus points at 10 and 25 steps, with
the same `--block-size 96`, sampled evaluation, seeds `7 11 19`, and configs
`random_full`, `count_prior_head`, `count_prior_lora_r1`, and
`count_prior_lora_r2`.

Measured cheap-minus-full advantage:

| Long fraction | Count-bigram bits | 10-step advantage | 25-step advantage | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.688454 | 0.247912 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.701242 | 0.278451 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.787087 | 0.299924 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | 0.856700 | 0.247387 | -0.066922 | -0.438575 |
| 1.00 | 2.483352 | 0.948164 | 0.241261 | -0.162385 | -0.642078 |

Current status:

- The surface-shape claim is supported. At every measured long fraction, the
  advantage decreases as the step budget increases: 10 > 25 > 50 > 100.
- The crossover contour moves as expected. It is above the measured range at 10
  and 25 steps, about `p = 0.635343` at 50 steps, and below the measured range by
  100 steps.
- The mechanism clause is partial. The full random model starts near
  `ln(vocab_size)`, but after 10 steps it is already about `0.89` to `1.09` NLL
  below that uniform floor. The cheap recipe remains close to the pure
  count-prior NLL, within about `0.001` to `0.008` NLL.

Conclusion:

Stage 25 gives Cassandra its cleanest current claim: the frozen count prior plus
tiny residual surface is an early-compute inductive-bias advantage. It wins when
the budget is tiny, loses its edge as compute grows, and is overtaken by full
training by 100 steps under this generator family. This supports a consolidating
ADR that fixes the scope of the core claim before opening another branch. Gemini
should compare the result to warm starts, learning-curve crossovers,
bias-variance tradeoffs, and the compute-scaling argument usually summarized by
the Bitter Lesson.

### Stage 26 · Order-matched analytic prior durability

Implemented from Hypothesis 007 and ADR 0002's new-method redirect. Stage 26
asks whether analytic priors are only early-compute head starts, or whether they
can be durable when the prior family matches the data-generating process.

Codex added `make_markov_corpus.py`, which generates pure seeded order-k Markov
character corpora and writes the source transition table plus entropy to a
metadata sidecar. Codex also added `--residual-base count-trigram`, a frozen
trigram residual base with adaptive backoff to bigram/unigram estimates for
sparse contexts and a bigram start row for position zero. The comparison keeps
the trainable surface fixed: `count_prior_tri_lora_r2` uses the same rank-2 LoRA
surface as `count_prior_lora_r2`.

Source metadata:

| Corpus | Source order | Vocab | Chars | Mean context entropy bits | Sampled source bits/char |
| --- | ---: | ---: | ---: | ---: | ---: |
| order2 | 2 | 16 | 40960 | 2.873596 | 2.886131 |
| order1 | 1 | 16 | 40960 | 2.840595 | 2.834059 |

Order-2 treatment:

| Steps | Bigram advantage | Trigram advantage |
| ---: | ---: | ---: |
| 10 | 0.031391 | 0.661099 |
| 25 | 0.017248 | 0.644896 |
| 50 | 0.010246 | 0.644227 |
| 100 | 0.008005 | 0.637544 |
| 200 | -0.000367 | 0.637384 |

Order-1 control:

| Steps | Bigram advantage | Trigram advantage |
| ---: | ---: | ---: |
| 50 | 0.042389 | 0.005914 |
| 100 | 0.029240 | -0.004205 |

Current status:

- H007 passes on the primary order-2 treatment. The matched trigram prior remains
  strongly ahead at 100 and 200 steps.
- The mismatched bigram prior on the order-2 source behaves like a small head
  start and is essentially tied with full training by 200 steps.
- The order-1 control supports the same diagnostic in reverse: the matched bigram
  prior stays positive, while the over-specified trigram prior is near tied or
  slightly negative.
- The Stage 24 `p = 0` bigram decay was therefore probably caused by
  supra-bigram structure in the old synthetic grammar, not by full training
  beating a correctly specified count model.

Conclusion:

Stage 26 qualifies ADR 0002. The frozen bigram recipe remains a bounded
early-compute accelerator on the previous synthetic family, but analytic priors
are not doomed to be only head starts. When the prior family matches the source,
the advantage can be durable. The next useful branch is a source-order versus
prior-order surface, followed later by natural text where the true source order is
unknown.

### Stage 27 · Source-order versus prior-order closeout

Stage 27 completed the immediate `V = 16` closeout opened by Stage 26. It reused
the Stage 26 order-2 rows and order-1 rows at 50 and 100 steps, then filled the
missing order-1 budgets at 10, 25, and 200 steps.

Matched priors stayed positive through all measured budgets:

| Steps | Order-1 source, bigram prior | Order-2 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.213916 | 0.661099 |
| 25 | 0.083939 | 0.644896 |
| 50 | 0.042389 | 0.644227 |
| 100 | 0.029240 | 0.637544 |
| 200 | 0.018835 | 0.637384 |

Mismatched priors did not have the same durability:

| Steps | Order-2 source, bigram prior | Order-1 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.031391 | 0.182281 |
| 25 | 0.017248 | 0.050448 |
| 50 | 0.010246 | 0.005914 |
| 100 | 0.008005 | -0.004205 |
| 200 | -0.000367 | -0.015480 |

The strongest local rule is now narrower and cleaner than the original cheap
recipe claim: prior order should match source order. Under-specification gives a
temporary head start, and over-specification can become harmful.

After this closeout, Claude added H008. That next phase is the fair order-3
surface: rerun at `V = 8`, add source order 3 and prior order 3, and report
sparsity and backoff diagnostics so order-3 failure is not confused with data
starvation. Stage 28 measured that model-order surface.

### Stage 28 · H008 full source-order versus prior-order surface

Stage 28 implemented H008. Codex added `--residual-base count-ngram`,
`--prior-order`, and the `count_prior_ng{1,2,3}_lora_r2` configs, then generated
pure Markov corpora at `V = 8` for source orders 1, 2, and 3. Each source was run
at budgets 10, 25, 50, 100, and 200 with seeds 7, 11, and 19.

At 200 steps, the advantage matrix was:

| Source order \ Prior order | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | 0.006804 | 0.002280 | -0.021482 |
| s = 2 | -0.096815 | 0.436673 | 0.405249 |
| s = 3 | -0.000504 | 0.106932 | 0.630598 |

The diagonal budget curves were:

| Steps | A(1,1) | A(2,2) | A(3,3) |
| ---: | ---: | ---: | ---: |
| 10 | 0.078849 | 0.555962 | 0.633965 |
| 25 | 0.031066 | 0.542886 | 0.632985 |
| 50 | 0.014747 | 0.539947 | 0.628784 |
| 100 | 0.012231 | 0.532621 | 0.628518 |
| 200 | 0.006804 | 0.436673 | 0.630598 |

The order-3 matched cell had full highest-order coverage and a strong advantage
at 200 steps, so the matched-prior result generalizes to order 3 under the
controlled `V = 8` setting. The strict H008 shape is partial: `A(3,2)=0.106932`
is still meaningfully positive, so an under-specified prior one order below the
source can retain useful lower-order structure. The measured rule is graded, not
binary: matching is best, severe under-specification fails, one-step
under-specification can still help, and over-specification is a sparsity/backoff
tradeoff.

### Stage 29 · Tiny-prose finite-order prior smoke

Stage 29 was Codex-derived exploratory evidence, not a Claude hypothesis. It used
`experiments/tiny_language_lab/corpus/tiny_seed.txt`, a 1,129-character
human-written project-prose seed, to check whether `count-ngram` immediately
collapses outside pure Markov sources.

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage |
| ---: | ---: | ---: | ---: |
| 10 | 0.143111 | 0.275569 | 0.284888 |
| 50 | 0.031407 | 0.196536 | 0.234387 |
| 100 | 0.217754 | 0.350333 | 0.399935 |

The result is positive but weak. Higher-order finite priors beat full random
training at all measured budgets, but the corpus is too small for a
natural-language claim and the sampled validation estimate is noisy. This
supports drafting a real natural-text hypothesis after Claude reviews ADR 0003
and Gemini compares the result against n-gram smoothing and model-order
selection.

### Stage 30 · Natural-text finite-order prior transfer

Stage 30 implemented Claude H009 after Gemini note 05 on n-gram order selection
and bias-variance. Codex downloaded Tiny Shakespeare, normalized it to a
33-character lowercase alphabet, and produced
`experiments/tiny_language_lab/corpus/natural_text_seed.txt` with `1,100,721`
characters. The split was deterministic prefix train (`935,612` chars) and
suffix validation (`165,109` chars).

The smoother was recursive add-alpha interpolation with `count_alpha=0.1` and
`ngram_backoff=10`, using context mass `count/(count+backoff)`. This is a strong
backoff-style control, not Katz or Kneser-Ney discounting.

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage |
| ---: | ---: | ---: | ---: |
| 10 | 0.359362 | 0.764853 | 0.990828 |
| 25 | 0.132707 | 0.541385 | 0.769688 |
| 50 | 0.066731 | 0.471524 | 0.701298 |
| 100 | 0.033787 | 0.438188 | 0.672109 |
| 200 | 0.004999 | 0.404669 | 0.637470 |
| 500 | -0.266069 | 0.110081 | 0.340641 |

Coverage:

| Prior order | Table coverage | Validation hit coverage |
| ---: | ---: | ---: |
| 1 | 1.000000 | 1.000000 |
| 2 | 0.645546 | 0.999927 |
| 3 | 0.193700 | 0.995997 |

This is the first positive non-synthetic evidence for the analytic-prior thread:
finite-order frozen priors can still accelerate early training on natural text,
and orders 2 and 3 remain positive through 500 steps. The sharper H009 sweet-spot
law is not closed. The measured curve is monotone increasing through order 3, so
the descending limb remains unmeasured. Stage 31 below resolves the order-4
branch. The remaining decision is a harsher vocabulary or split, or an explicit
neural plus n-gram interpolation comparison before drafting a natural-text ADR.

### Stage 31 · Order-4 natural-text extension

Stage 31 tested H009's optional order-4 cell after Stage 30 left the descending
limb unmeasured. Codex generalized `count-ngram` to `--prior-order 4` and added
`count_prior_ng4_lora_r2`. On the same normalized Tiny Shakespeare corpus, the
order-4 frozen base has `42,802,056` logits and remained feasible on CPU.

Advantage versus `random_full`, reusing Stage 30 rows for orders 1 to 3:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage | ng4 advantage |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.359362 | 0.764853 | 0.990828 | 1.108415 |
| 25 | 0.132707 | 0.541385 | 0.769688 | 0.890557 |
| 50 | 0.066731 | 0.471524 | 0.701298 | 0.823300 |
| 100 | 0.033787 | 0.438188 | 0.672109 | 0.792813 |
| 200 | 0.004999 | 0.404669 | 0.637470 | 0.757900 |
| 500 | -0.266069 | 0.110081 | 0.340641 | 0.463572 |

Order 4 is best at every measured budget, so the sweet-spot falloff is still not
located. The coverage diagnostic explains why: order-4 table coverage is sparse
(`0.029713`), but validation hit coverage remains high (`0.961867`). This means
the current deterministic prefix/suffix split is still friendly to high-order
local statistics. Stage 32 tested one such natural-text validity gate.

### Stage 32 · Cross-domain natural-text validity gate

Stage 32 implemented Claude H009b on a cross-domain validation split. Codex kept
the Stage 30 Shakespeare train prefix (`935,612` chars) and replaced the
validation suffix with normalized Cassandra project prose (`165,109` chars) in
the same `V=33` alphabet. This is a real held-out character-domain gate, but not
the preferred second public-domain author, so it remains a source-choice caveat.

Validation-hit coverage dropped but stayed high:

| Prior order | Table coverage | Stage 31 hit coverage | Stage 32 hit coverage |
| ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 0.645546 | 0.999927 | 0.989043 |
| 3 | 0.193700 | 0.995997 | 0.964350 |
| 4 | 0.029713 | 0.961867 | 0.889803 |

Advantage versus `random_full` on the Stage 32 split:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage | ng4 advantage |
| ---: | ---: | ---: | ---: | ---: |
| 100 | -0.034461 | +0.183530 | +0.362316 | +0.460628 |
| 200 | -0.048219 | +0.164621 | +0.339183 | +0.441501 |
| 500 | -0.203702 | +0.016767 | +0.199130 | +0.291191 |

The humped sweet-spot prediction is not supported on this implemented split:
order 4 remains best at every decision budget. The result locally kills the idea
that this cross-domain gate would reveal a moderate-order peak, while preserving
the caveat that order-3 and order-4 hit coverage did not collapse enough to rule
out every harsher or more genre-matched validation source. The measured local
state after Stage 32 is that order 4 remains the strongest natural-text finite
prior, with a prior-art and source-choice caveat for Gemini.

### Stage 33 · Mixed prior-loss curriculum filter

Stage 33 implemented Claude H010 after Gemini note 13 framed the method as
loss-based data selection and hard example mining. Codex added a plain language
model batch sampler that scores every legal training window by mean per-token
NLL under the frozen order-2 prior, keeps the top 10 percent as a high-loss pool,
and draws a fixed fraction of each batch from that pool. The baseline stayed the
same `count_prior_ng2_lora_r2` model with uniform sampling.

The target was the uniform 200-step mean validation NLL on normalized Tiny
Shakespeare, `2.040189`:

| Arm | Earliest measured budget at or below target | 500-step mean NLL |
| --- | ---: | ---: |
| uniform | 100 | 2.050701 |
| filter `f=0.25` | 200 | 2.051195 |
| filter `f=0.50` | not reached | 2.052930 |
| filter `f=1.00` | not reached | 2.065845 |

H010 is killed on its primary metric. The mixed filters did not reach the target
faster than uniform, and the pure high-loss negative control was worse at every
budget. This supports the lab's earlier warning from failed-case replay: raw
hardness can select irreducible noise, rare formatting, or awkward local spans
instead of useful residual structure. The result is not a broad claim against
reducible-loss data pruning. It is a local negative for a fixed frozen-prior-NLL
hard-example sampler on the order-2 prior plus rank-2 residual.

### Stage 34 · Dynamic reducible-loss curriculum filter

Stage 34 implemented Claude H011, the final data-side selection attempt for the
order-2 frozen-prior plus rank-2 residual. Instead of using static frozen-prior
NLL, Codex re-scored a deterministic 4096-window pool every 25 steps under the
live prior-plus-residual model, selected top-decile current-loss windows with a
positive smoothed loss delta, and mixed them into batches at `f=0.25` and
`f=0.50`.

The target was again the uniform 200-step mean validation NLL, `2.040189`:

| Arm | Earliest measured budget at or below target | 500-step mean NLL |
| --- | ---: | ---: |
| uniform | 100 | 2.050701 |
| dynamic `f=0.50` | not reached | 2.060916 |
| dynamic `f=0.25` | not reached | 2.053026 |

H011 is killed. Neither dynamic arm reached the target faster than uniform, and
neither reached it by 500 steps. The dynamic arms also spent extra wall-clock on
pool re-scoring. Together with Stages 11, 12, and 33, this retires data-side
curriculum selection for the frozen-prior rank-2 residual. The local explanation
is capacity and data-order insensitivity, not simply a bad static loss proxy.
Later work should pivot to a richer frozen base, a model-side long-range
primitive, a different residual-capacity question, retrieval-interface redesign,
or non-gradient residual formation.

### Stage 35 · Frozen recency base

Stage 35 implemented Claude H012, the first model-side frozen-base follow-up to
ADR 0004. Codex interpolated the order-2 count prior with an analytic
exponential-recency character cache, using `tau=96` and `lambda=0.25`, then kept
the same rank-2 LoRA residual on top. The diagnostic arm was the existing order-3
count prior.

Mean validation NLL:

| Steps | count ng2 | ng2 + recency | recency delta | count ng3 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 2.045084 | 2.130612 | +0.085528 | 1.815311 |
| 100 | 2.040163 | 2.120414 | +0.080251 | 1.806242 |
| 200 | 2.040189 | 2.112827 | +0.072638 | 1.807388 |
| 500 | 2.050701 | 2.112697 | +0.061996 | 1.820142 |

H012 is killed. The simple character-recency interpolation is worse than the
order-2 count-only baseline at every budget and slower because it is computed per
block position. The order-3 count prior remains far stronger, so this result says
that the tested recency cache is a bad frozen base for this ladder, not that all
model-side frozen long-range primitives are exhausted. Gemini note 08 connects the
failure to cache language-model prior art: word or token caches exploit topical
burstiness, while a character cache is structure-blind and injects noise. If the
model-side branch continues, the next primitive should preserve order, for example
a frozen SSM kernel, a one-dimensional convolutional kernel, or an n-gram cache.

### Stage 36 · Non-gradient residual formation

Stage 36 implemented Claude H013 and tested whether the small residual on top of
the frozen count prior can be formed without backpropagation. The structured
corpus was deliberately chosen because the frozen floor is strong but the rank-2
AdamW residual still improves it.

On CUDA, with seeds `7 11 19`, the frozen-prior floor had mean validation NLL
`2.018509`, the rank-2 AdamW target reached `2.000801`, full rank-2 ES reached
`2.060750`, and rank-1 coordinate search reached `2.011522`. ES therefore failed
the immediate no-backprop claim, while coordinate search gave only weak reduced
surface feasibility. This bounds the current Cassandra recipe: it reduces gradient
training through the frozen prior and small residual, but Stage 36 does not show
that the residual itself can be formed without backprop.

The key confound is fixed-objective overfitting. ES lowered its fixed search loss
but worsened validation; a four-search-batch seed-7 diagnostic reduced the harm
but still missed the floor while using `3212` formation forward passes. Gemini
should frame any future discussion against known evolution-strategy,
neuroevolution, coordinate-search, and zeroth-order optimization literature.

### Stage 37 · Residual marginal-value gate

Stage 37 implemented the ADR 0005 redirect. It measured whether the residual's
small Stage 36 value was specific to the structured corpus or intrinsic to the
current frozen-prior recipe. The gate used the floor-to-target validation NLL gap,
`*_floor` minus AdamW target, across natural text at 200 and 500 steps for orders
2 to 4 and a structured rank 1/2/4 sweep.

The gate closed. Natural-text gaps were mixed or negative despite the recipe's
large advantage over `random_full`, showing the finite-order prior itself carries
the win. The structured rank sweep widened the residual gap with capacity, but
only from `+0.010810` at rank 1 to `+0.023058` at rank 4. No tested regime reached
the `0.05` NLL reopening line, and the largest stable gap stayed below `0.03`.

Roadmap impact: formation-side optimizer work still has no large local target.
The evidence points away from further residual-formation NLL mechanics in these
cheap regimes and toward either the prior itself or the behavior axis, where NLL
and task behavior already diverged in the copy-probe stages. Claude owns that
roadmap decision.

### Stage 38 · Behavior residual marginal-value gate

Stage 38 implemented Claude H014 and tested the Stage 37 prior-dominance law on
copy behavior rather than validation NLL. Codex added a frozen-prior floor for the
weighted copy arm, `count_prior_lora_r2_copyw_floor`, then compared that floor
against `count_prior_lora_r2_copyw`, `count_prior_lora_r2_copymix`, and
`random_full_copymix` on the long-context copy corpus.

The behavior gate confirmed. The floor copied at `0.118421`, near `1 / 8 = 0.125`
chance. The rank-2 residual arms copied at `0.320176` (`copyw`) and `0.307017`
(`copymix`) mean accuracy, and both beat the floor by more than `0.10` on every
seed. Plain validation NLL did not carry the same signal: `copymix` had worse
mean validation NLL than the floor while still forming behavior above the floor.

Roadmap impact: the formation-side closure from Stage 37 is specific to NLL. The
behavior axis is open, and the residual is the behavior-forming surface on this
controlled copy task. This does not reopen gradient-free or data-selection work
for NLL. It points future behavior hypotheses toward sampler, rank, verifier, and
interface questions, with Claude owning the roadmap decision and Gemini owning the
induction-head and in-context-learning prior-art placement.

### Stage 39 · Behavior rank sweep

Stage 39 implemented ADR 0006's provisional rank-sweep handoff. The sweep kept the
Stage 38 corpus and copy-weighted protocol fixed, added rank-1 and rank-4 mirrors
of the copy-weighted frozen-prior residual, and measured copy-probe accuracy with
validation NLL tracked alongside.

All trained ranks stayed above the frozen floor, so the behavior-forming residual
claim survives. The rank story is not monotone: rank 1 reaches `0.250000`, rank 2
reaches `0.320176`, and rank 4 reaches `0.271930` mean copy accuracy. Rank 4 does
not beat rank 1 with stable sign and is worse than rank 2 on two seeds.

Roadmap impact: the behavior branch should not simply scale LoRA rank next. Rank 2
is enough to form the strongest measured copy behavior under this fixed signal,
and extra rank through 4 does not produce a stable gain. The next behavior
hypothesis should test sampler signal, verifier signal, harder generalization, or
optimization stability rather than capacity alone.

### Stage 40 · Held-out-key copy generalization

Stage 40 implemented H015 by adding `--holdout-keys` to the identity-copy
long-context corpus generator. The held-out corpus used keys `g h`, kept them out
of training `key=...answer=` rows, and left them in the character vocabulary
through filler text. The validation split had 97 seen and 18 held-out copy cases
per seed.

The result is a held-out collapse, not a clean generalizing copy circuit. The
rank-2 residual tied the frozen floor at `0.000000` held-out accuracy on all
three seeds. It also did not satisfy the clean memorization reversal clause,
because its mean seen-key gain over the floor was only `+0.027491`, far below the
`+0.10` seen-formation threshold. The full context control also scored
`0.000000` held-out accuracy on every seed, though it improved seen-key copy
accuracy.

Roadmap impact: ADR 0006 should be scoped to seen-key identity copy at the Stage
38 budget rather than reversed outright. The behavior axis remains open, but the
next hypothesis needs either a larger held-out split, a stronger sampler or
verifier signal, non-identity mapping, or more full-control budget before claiming
generalization.

### Stage 41 · Forced-choice held-out copy circuit

Stage 41 implemented H016 by adding a forced-choice readout to the copy probe. The
model was unchanged; the readout restricted answer-position logits to the
validation key alphabet `abcdefgh` and reported correct-key rank through MRR. The
matrix reused the Stage 40 held-out corpus, arms, seeds, block size, and `500`
step CUDA protocol.

The forced-choice artifact hypothesis did not confirm. The rank-2 residual stayed
at `0.000000` held-out choice accuracy on all seeds, tied with the floor and below
`1 / 8` chance. Its held-out MRR was only `+0.002645` above the floor on the mean.
The result also does not cleanly fire the memorization reversal clause because Arm
B's seen choice accuracy was only `0.202749`, below the `+0.10` seen-power line.
The full model is the key caveat: it reached `0.364261` mean seen choice accuracy,
with seed `19` at `0.670103`, but still scored `0.000000` held-out choice accuracy
and floor-level held-out MRR.

Roadmap impact: the simple explanation that Stage 40 only failed because the
free-vocabulary output head suppressed held-out keys is not supported. The current
protocol still looks like a task or budget failure for held-out identity-copy
transfer, not a cheap-surface-only failure. The next behavior hypothesis should
strengthen formation or increase task diversity before treating held-out
generalization as settled.

### Stage 42 - Memorization-proof copy probe

Stage 42 replaced the invalid held-out-token generalization probe with a valid
random-payload copy probe. The payload token is drawn per line from a fully seen
16-symbol alphabet, so every answer token is emittable and there is no fixed
mapping or line-index shortcut to memorize.

The result is a local reversal for the current cheap rank-2 residual recipe. The
frozen floor was near chance (`0.043478` versus `1 / 16 = 0.062500`), the cheap
rank-2 residual was also at chance (`0.063768`), and the full control cleared the
learnability gate (`0.226087` mean copy accuracy, positive on every seed). The
cheap residual still improved validation NLL over its floor, so the dual-axis rule
remains important: NLL movement did not imply general copy behavior.

Research implication: the behavior branch should not claim that the current
rank-2 PEFT residual forms a general in-context copy or induction circuit. It
forms seen-key identity copy under Stage 38, but the valid random-token copy probe
points toward needing a stronger behavior-forming surface, signal, budget, or
interface for general copy. Gemini should compare this directly with random-token
copying and induction-head work before any outside claim.
### Stage 43 - Minimal surface for general copy

Stage 43 implemented H018 as a trainable-surface ladder on the Stage 42
random-payload corpus. The ladder compared the failing rank-2 frozen-prior
residual, rank 8 and rank 16 LoRA with alpha matched to rank, a full-body
trainable transformer under the frozen count-bigram base, and the no-prior full
model ceiling.

The result fires the registered KILL line for this frozen-prior family at the
Stage 42 budget. Rank 8 (`0.049275`) and rank 16 (`0.049276`) did not beat chance
(`0.062500`) or rank 2 (`0.063768`) on copy accuracy. The full-body-on-frozen-base
arm also stayed at `0.043478` copy accuracy on every seed, while the no-prior full
model cleared chance on every seed and reached `0.226087` mean copy accuracy.
Validation NLL and behavior split again: rank 16 had the best frozen-prior NLL
(`1.642052`) while still failing the behavior probe.

Research implication: the Stage 38 seen-content behavior result should not be
upgraded to a general in-context copy claim for the current frozen-prior family.
The Stage 43 diagnostic points toward frozen-base interference under this protocol,
not just insufficient LoRA rank. Future behavior work should change the base,
interface, retrieval mechanism, attention prior, or budget before claiming a small
frozen-prior surface forms a general induction-like copy circuit.
### Stage 44 - Phase 2 TinyStories bridge

Stage 44 starts ADR 0010's Phase 2 model-build branch. Codex downloaded a bounded
official TinyStories train slice, normalized it into a `10,000,001` character
corpus with `V = 33`, added visible PowerShell launch scripts, and vectorized the
order-n count-prior builder so TinyStories-scale count priors are practical on the
RTX 4070 laptop.

The first bridge matrices compare a 4-layer, 4-head, 256-dimensional
character-level full random transformer with order-3 and order-4 frozen n-gram
priors plus rank-2 LoRA. At 500 steps across seeds `7 11 19`,
`count_prior_ng4_lora_r2` reaches `1.139715` mean validation NLL, compared with
`2.352297` for `random_full`, while training only `41,249` parameters instead of
`3,209,249`. The generation samples from `once upon a time ` are still rough, but
the frozen-prior arms produce recognizable story-like fragments, while the random
full baseline remains mostly character noise.

Research implication: ADR 0010 is now executable. The Phase 1 frozen-prior
early-compute advantage transfers to a TinyStories-scale character corpus slice,
and order 4 remains better than order 3 in this bridge setting. This is not a
public TinyStories benchmark and not yet the modded-nanoGPT baseline. The next
Phase 2 engineering questions at this point are RoPE, Muon, gradient
accumulation, activation checkpointing, streaming shard training, and later BPE
tokenization. Stage 45 addresses the first four.
### Stage 45 - Phase 2 modern TinyStories baseline

Stage 45 implements the first modern-baseline pass from ADR 0010. Codex added
RoPE, gradient accumulation, activation checkpointing, and a single-device Muon
optimizer path to the local trainer, then exposed the same controls through the
comparison harness and visible PowerShell launcher.

The `modern500` run used the Stage 44 TinyStories character corpus, CUDA, seeds
`7 11 19`, `--block-size 128`, `--batch-size 8`, `--grad-accum-steps 2`,
`--n-layer 4`, `--n-head 4`, `--n-embd 256`, `--pos-encoding rope`,
`--activation-checkpoint`, `--optimizer muon`, and prompt `once upon a time `.
At 500 steps, `random_full` reached `1.144942` mean validation NLL, a sharp
improvement over the Stage 44 bridge baseline at `2.352297`. The modern
`count_prior_ng4_lora_r2` arm reached `1.102748`, retaining a smaller early
advantage of `0.042194` NLL while training `41,249` parameters instead of
`3,176,481`.

Research implication: the train-from-scratch Phase 2 path is now credible on the
RTX 4070 laptop for the character-level TinyStories rung. The frozen prior still
helps under the stronger baseline, but the advantage is much smaller than in the
plain AdamW bridge. The next useful Phase 2 questions are no longer RoPE or Muon
plumbing; they are streaming shard consumption, a BPE ablation with n-gram priors
defined over BPE tokens, longer-budget crossover measurement, and formal
generation-quality scoring.
### Stage 46 - TinyStories generation-quality score sheet

Stage 46 implements ADR 0010's minimum generation-quality scoring loop for saved
prompt completions. Codex added `score_generation_samples.py`, which reads JSONL
run artifacts and writes a Markdown sheet with deterministic proxy scores for
coherence, grammaticality, and on-prompt relevance. The proxy uses prompt-prefix
adherence, story-cue words, corpus-word ratio, sentence punctuation, repetition,
and bad-marker checks. It is a local trend signal, not a human evaluation or
public benchmark.

On the Stage 45 modern b500 samples, the order-4 frozen-prior arm scored
`5.667/6` mean total, while the modern random-full arm scored `3.000/6`. The
split matches the qualitative readout: the random model now forms word and
sentence fragments but still emits artifacts, while the prior samples have more
TinyStories-like vocabulary and cadence at this budget.

Research implication: Phase 2 now tracks both NLL and a reproducible generation
proxy. Human review is still required before making external quality claims, but
the local train-from-scratch loop can now compare prompt completions across runs
without relying only on ad hoc sample reading.
### Stage 47 - TinyStories shard-consumption smoke

Stage 47 implements shard consumption for the current plain language-model
training path. The trainer now accepts `--train-shard-dir` and samples random
character windows from the TinyStories `train_*.txt` shard files, while using a
bounded prefix for train-loss reporting and the normal validation split for
validation loss. The compare harness and visible launcher expose the same path
through `stream-smoke`.

The visible smoke consumed five train shards, reported `train_chars =
8,500,000`, `train_eval_chars = 200,000`, and completed a 20-step RoPE/Muon CUDA
run at `2.216325` validation NLL. The scope is intentionally narrow: copy-aware
training, curriculum filtering, and frozen-prior residual bases still require
the full train tensor because they build marker positions, prior tables, or
loss-ranked windows.

Research implication: the character-level Phase 2 loop now has a shard-backed
training-batch path for the no-prior baseline. This is enough to remove
"trainer cannot consume shards" as a blocker, while leaving future work to make
the Cassandra frozen-prior ablation itself shard-native.
### Stage 48 - Modern TinyStories 1000-step crossover

Stage 48 measures the first Phase 2 crossover interval under the modern
character baseline. The visible `modern1000` matrix doubled Stage 45's budget
from 500 to 1000 steps while keeping the same TinyStories character corpus,
model size, RoPE, Muon, activation checkpointing, gradient accumulation, seeds,
and two arms: `random_full` and `count_prior_ng4_lora_r2`.

The NLL crossover fires. At 500 steps, Stage 45 had `count_prior_ng4_lora_r2`
ahead by `0.042194` mean validation NLL. At 1000 steps, `random_full` reaches
`1.052559` mean validation NLL versus `1.123161` for the prior arm. The
prior-minus-full gap is positive on every seed: `+0.071768`, `+0.079357`, and
`+0.060680`.

The generation proxy does not fully agree with NLL: it still favors the prior arm
at `4.667/6` versus `2.667/6`, partly because two random-full samples emit
`endoftext` artifacts. This reinforces the Stage 46 rule that validation loss
and sample quality must both be tracked.

Research implication: the bounded early-compute accelerator story survives the
stronger Phase 2 baseline. The order-4 prior gives a small 500-step head start,
but the full modern model passes it by 1000 steps. The current crossover interval
is therefore between 500 and 1000 steps for this corpus, model size, optimizer,
and character-level tokenization.
### Stage 49 - TinyStories BPE smoke

Stage 49 implements the first BPE feasibility path without relying on an
external tokenizer package. Codex added `make_bpe_corpus.py`, which trains a
small local BPE tokenizer and writes each BPE token ID as a Unicode private-use
codepoint so the current trainer can treat each BPE token as one symbol. Codex
also added `decode_bpe_samples.py` to turn private-use generated samples back
into readable text.

The first BPE artifact used vocab size `256`, trained merges on `500,000`
characters, encoded `1,000,000` TinyStories characters into `446,694` BPE tokens,
and achieved `2.238669` source characters per BPE token. The visible BPE smoke
ran `random_full` and `count_prior_lora_r2`, where `count_prior_lora_r2` is a
frozen BPE-token bigram prior plus rank-2 LoRA residual. At 20 steps, the prior
arm reached `3.405228` validation NLL versus `3.847587` for the random full
model.

Research implication: BPE tokenization and a frozen n-gram prior over BPE tokens
are now mechanically live in the Cassandra harness. This is only a smoke: the
durable BPE decision needs larger vocab/corpus settings, a cleaner prompt
encoding path, multi-seed runs, and comparison against the character baseline on
matched training budgets.
### Stage 50 - BPE 500-step multi-seed matrix

Stage 50 gives the first multi-seed BPE decision surface. It keeps the v256 BPE
artifact from Stage 49 and compares `random_full` against `count_prior_lora_r2`,
the BPE-token bigram prior plus rank-2 LoRA residual, across seeds `7 11 19` for
500 steps.

The result is a strong reversal of the 20-step BPE smoke ordering. Full BPE
training reaches `2.404960` mean validation NLL, while the BPE-token bigram prior
arm reaches `3.344760`. With the BPE artifact's `2.238669` source chars/token,
that is approximately `1.549860` bits/source-char for full BPE versus `2.155508`
for the prior arm. The prior-minus-full NLL gap is positive on every seed:
`+0.987558`, `+0.910987`, and `+0.920855`.

Research implication: this small BPE-token bigram prior is not the right
Cassandra default. Character-level TinyStories remains the Phase 2 baseline for
the current laptop-scale work. BPE stays live as an engineering branch, but the
next BPE attempt should use a larger corpus/vocab and likely a higher-order BPE
prior or a pure full-BPE baseline comparison rather than the v256 bigram-prior
setup.
### GPU transition validity audit

After Stage 35, Codex switched real comparison matrices to CUDA on the laptop
RTX 4070. The audit is recorded in `docs/gpu-transition-validity-audit.md`.
The key result is that exact zero-step full evaluation agrees between CPU and
CUDA to displayed precision, but sampled runs are not bitwise comparable across
devices because the trainer uses `torch.Generator(device=device)`.

This does not resurrect the killed hypotheses. Large NLL and behavior kills
remain local evidence, especially H012's recency failure and H009b's implemented
cross-domain sweet-spot failure. The switch does retire CPU wall-clock as a
future planning baseline. Any old claim that depends on seconds, or on a tiny
sampled NLL margin below about `0.003`, should be rerun as a CUDA result before
Claude treats it as a hard GPU-era stop.

## First hypotheses

H1: Count-based initialization will beat random initialization under very small
training budgets, even if gradient training eventually catches up.

H2: Literal parameter-by-parameter search will improve tiny models, but will
stall quickly unless it is seeded by structure.

H3: Narrow data distributions create the appearance of general language skill
far earlier than broad web text.

H4: Retrieval might substitute for memorized facts, allowing model parameters to
focus on syntax, control flow, and local reasoning. Stage 22 did not support
this under the current compact held-out-memory interface, so future H4 work
needs a redesigned interface or a stronger outside baseline.

H5: The most useful laptop-scale path is probably hybrid: analytic seed,
small gradient update, retrieval, and verifier-filtered data.

## Metrics

Every experiment should record:

- method name,
- parameter count,
- trainable parameter count,
- corpus path and size,
- train/validation split,
- training or construction steps,
- wall time,
- device,
- train negative log likelihood,
- validation negative log likelihood,
- validation bits per character,
- generated samples from fixed prompts.

## Source anchors

- TinyStories paper: https://arxiv.org/abs/2305.07759
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- Parameter-efficient fine-tuning survey: https://arxiv.org/abs/2303.15647
- nanoGPT repository: https://github.com/karpathy/nanoGPT
- minGPT repository: https://github.com/karpathy/minGPT
- BitNet repository: https://github.com/microsoft/BitNet
- Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models: https://arxiv.org/abs/2601.14290
- Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors: https://arxiv.org/abs/2601.15625
- UR²: Unify RAG and Reasoning through Reinforcement Learning: https://arxiv.org/abs/2508.06165
- Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM): https://arxiv.org/abs/1911.00172
- Improving language models by retrieving from trillions of tokens (RETRO): https://arxiv.org/abs/2112.04426
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection: https://arxiv.org/abs/2310.11511
- Lost in the Middle: How Language Models Use Long Contexts: https://arxiv.org/abs/2307.03172
- Corrective Retrieval Augmented Generation (CRAG): https://arxiv.org/abs/2401.15884
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?: https://arxiv.org/abs/2202.12837
- Larger language models do in-context learning differently: https://arxiv.org/abs/2303.03846
- Can In-context Learning Really Generalize to Out-of-distribution Tasks?: https://arxiv.org/abs/2410.09695
- Investigating Catastrophic Forgetting in Parameter-Efficient Fine-Tuning (Sequential Learning limitations): Various arXiv
- The Trade-off between Analytic Priors and Neural Expressivity (Corpus Complexity): Various arXiv
- The Bitter Lesson (Rich Sutton, 2019): http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Trade-offs between Inductive Bias and Compute Scaling in Language Models: Various arXiv
- Phase Transitions and Grokking in Neural Networks: Various arXiv
- Rectified Scaling Laws and Early Training Dynamics: Various arXiv
- Inductive Bias Alignment and Data Generating Functions: Various arXiv
- Misspecified Priors in Deep Learning: Various arXiv
- N-Gram Backoff and Kneser-Ney Smoothing in Language Models: Various arXiv
- Hierarchical Priors and Robustness under Misspecification: Various arXiv
- N-Gram Order Selection and the Bias-Variance Tradeoff: Various arXiv
- Neural-Ngram Interpolation and Hybrid Language Models: Various arXiv
- Domain Shift and Phrase Reuse in Language Models: Various arXiv
- Loss-Based Data Selection and Hard Example Mining: Various arXiv
- Reducible Holdout Loss Selection for Data Pruning: Various arXiv
- Static Proxies vs Iterative Data Scoring: Various arXiv
- Capacity Bottlenecks in LoRA Data Pruning (PRILoRA, LoPrune): Various arXiv
- Cache Language Models and Continuous Cache: Various literature (Kuhn & De Mori, 1990)
- Dr. Fırat Akba: Machine Learning, Feature Selection, and Time Series Analysis: Independent Research
