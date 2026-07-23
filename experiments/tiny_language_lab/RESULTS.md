# Tiny Language Lab Results

Date: 2026-06-15

Device: CPU

Corpus: `experiments/tiny_language_lab/corpus/tiny_seed.txt`

## Verified Runs

| Experiment | Steps | Parameters | Trainable | Changed | Train NLL | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bigram count construction | 0 | 841 | 0 | 841 | 2.110719 | 2.430600 | 3.506614 | 0.0311 |
| Bigram coordinate search, count init | 100 | 841 | 841 | 74 | 2.109115 | 2.425266 | 3.498919 | 0.0928 |
| Tiny transformer, gradient | 20 | 105885 | 105885 | 105885 | 2.357911 | 2.580663 | 3.723110 | 2.0098 |
| Tiny transformer, gradient | 100 | 105885 | 105885 | 105885 | 1.763346 | 2.583703 | 3.727496 | 3.2442 |

## Transformer Loss Curve

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --steps 100 --eval-interval 50 --log-every 50 --max-new-tokens 100 --prompt "cassandra "
```

Result:

| Step | Train NLL | Val NLL |
| ---: | ---: | ---: |
| 0 | 3.379681 | 3.377009 |
| 50 | 2.086340 | 2.465410 |
| 100 | 1.763346 | 2.583703 |

## Interpretation

The bigram count baseline is still stronger on validation loss than the current
tiny transformer after 100 steps. That is not a bad result. It says the corpus is
small enough that a direct transition table is a serious baseline, while the
105K-parameter transformer starts to memorize the training split quickly.

The first Cassandra-style positive signal remains the count-seeded coordinate
run: only 74 accepted one-parameter updates slightly improved the analytic
count baseline.

Next comparison:

1. Add count-seeded transformer initialization.
2. Compare random-init transformer vs count-seeded transformer under 20, 50,
   and 100 training steps.
3. Repeat on a larger but still laptop-friendly corpus before drawing broader
   conclusions.

## Stage 2 · Count-Seeded Transformer Initialization

Date: 2026-06-16

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --init random --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 60 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --init count-bigram --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 60 --prompt "cassandra "
```

The count-bigram initializer embeds the smoothed bigram transition table into
the transformer at construction time:

- transformer blocks are zeroed,
- token embeddings encode character identity,
- position embeddings are zeroed,
- the output head is solved by least squares so the initial transformer logits
  match count logits.

Fit check:

| Init | Seeded parameters | Count fit max abs error |
| --- | ---: | ---: |
| count-bigram | 5789 | 0.00000453 |

Fixed-budget comparison:

| Init | Steps | Parameters | Train NLL | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0 | 105885 | 3.379681 | 3.377009 | 4.871995 | 2.0368 |
| count-bigram | 0 | 105885 | 2.115222 | 2.502358 | 3.610139 | 2.1025 |
| random | 20 | 105885 | 2.357911 | 2.580663 | 3.723110 | 4.5579 |
| count-bigram | 20 | 105885 | 2.115614 | 2.541046 | 3.665955 | 4.5743 |
| random | 50 | 105885 | 2.086340 | 2.465410 | 3.556835 | 5.4986 |
| count-bigram | 50 | 105885 | 2.094859 | 2.519280 | 3.634553 | 5.5303 |
| random | 100 | 105885 | 1.763346 | 2.583703 | 3.727496 | 6.3772 |
| count-bigram | 100 | 105885 | 2.085894 | 2.596955 | 3.746615 | 6.5056 |

Interpretation:

The count-bigram initializer proves that corpus-derived structure can be placed
inside the transformer before gradient training. At step 0 it is far better than
random initialization, and it keeps a small validation advantage at 20 steps.
By 50 steps, the random model has caught up and slightly passed it. By 100 steps,
both models show overfitting on this tiny corpus.

This does not invalidate the analytic-initialization idea. It says the current
implementation treats the count prior as just another trainable surface, so the
optimizer can overwrite it. The next Cassandra method should protect the
analytic prior and train only a residual path, or use a much smaller learning
rate for seeded components.

## Stage 3 · Frozen Count Prior Plus Residual Transformer

Date: 2026-06-16

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 60 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --residual-l2 1.0 --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 40 --prompt "cassandra "
```

This mode keeps the count-derived bigram logits outside the trainable model and
adds transformer logits as a residual correction. With the default
`--zero-residual-head`, step 0 exactly matches the frozen count prior.

Fixed-prior check:

| Mode | Frozen prior parameters | Initial train NLL | Initial val NLL | Val bits/char |
| --- | ---: | ---: | ---: | ---: |
| residual-base count-bigram | 841 | 2.115222 | 2.502358 | 3.610139 |

Residual comparison:

| Mode | LR | Residual L2 | Steps | Train NLL | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen count prior | 0.0030 | 0.0 | 20 | 2.081700 | 2.525854 | 3.644038 | 3.9106 |
| frozen count prior | 0.0030 | 0.0 | 50 | 2.017663 | 2.538745 | 3.662635 | 4.6844 |
| frozen count prior | 0.0030 | 0.0 | 100 | 1.865589 | 2.623859 | 3.785429 | 5.4101 |
| frozen count prior | 0.0030 | 0.1 | 50 | 2.008916 | 2.491961 | 3.595140 | 3.7710 |
| frozen count prior | 0.0030 | 1.0 | 50 | 2.036422 | 2.489763 | 3.591969 | 3.6979 |
| frozen count prior | 0.0030 | 1.0 | 100 | 1.918852 | 2.503229 | 3.611395 | 6.1303 |
| frozen count prior | 0.0003 | 0.0 | 50 | 2.079945 | 2.489744 | 3.591942 | 3.7684 |
| frozen count prior | 0.0003 | 0.0 | 100 | 2.000249 | 2.511748 | 3.623687 | 6.0983 |
| frozen count prior | 0.0003 | 1.0 | 100 | 2.019838 | 2.502919 | 3.610949 | 5.5066 |
| frozen count prior | 0.0030 | 10.0 | 50 | 2.102793 | 2.502861 | 3.610865 | 4.3717 |

Interpretation:

Freezing the count prior works mechanically: the model starts as the analytic
bigram model and the prior cannot be overwritten. Unconstrained residual
training still overfits quickly. Small residual corrections are better:
`--residual-l2 1.0` or `--lr 0.0003` improves validation at 50 steps
(`2.4897-2.4898`) compared with the frozen prior's `2.5024`, while keeping the
analytic prior mostly intact.

This is not yet better than the best random transformer point in this tiny
corpus (`2.4654` at 50 steps), but it is better than the trainable count-seeded
transformer at 50 steps (`2.5193`). The useful lesson is that preserving
analytic structure needs both a frozen base and a constrained residual path.

Next comparison:

1. Freeze more of the residual transformer and train only the output head.
2. Add adapter-only residual layers so the trainable surface is much smaller
   than 105885 parameters.
3. Repeat the same comparison on a larger corpus before trusting the ranking.

## Stage 4 · Head-Only and Adapter-Only Residual Updates

Date: 2026-06-16

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope head --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 40 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope adapters --adapter-rank 4 --steps 50 --eval-interval 50 --log-every 50 --max-new-tokens 30 --prompt "cassandra "
```

This stage freezes most or all of the residual transformer. The frozen count
prior remains outside the trainable model. The residual path can train:

- `head`: only the output head,
- `adapters`: tiny bottleneck adapters plus the output head.

Comparison:

| Train scope | Adapter rank | Trainable params | LR | Residual L2 | Steps | Train NLL | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 0 | 1885 | 0.0030 | 0.0 | 50 | 2.092857 | 2.506742 | 3.616464 | 4.4721 |
| head | 0 | 1885 | 0.0030 | 0.0 | 100 | 2.082192 | 2.529383 | 3.649128 | 5.5650 |
| head | 0 | 1885 | 0.0030 | 1.0 | 50 | 2.100210 | 2.497952 | 3.603782 | 4.2864 |
| head | 0 | 1885 | 0.0003 | 0.0 | 50 | 2.108765 | 2.498686 | 3.604842 | 3.3991 |
| adapters | 2 | 2397 | 0.0030 | 0.0 | 50 | 2.091274 | 2.498518 | 3.604599 | 3.2810 |
| adapters | 4 | 2909 | 0.0030 | 0.0 | 50 | 2.088377 | 2.483014 | 3.582231 | 3.9851 |
| adapters | 4 | 2909 | 0.0030 | 0.0 | 100 | 2.072756 | 2.510333 | 3.621644 | 4.6374 |
| adapters | 4 | 2909 | 0.0030 | 1.0 | 50 | 2.098551 | 2.484041 | 3.583713 | 3.3104 |
| adapters | 8 | 3933 | 0.0030 | 0.0 | 50 | 2.085689 | 2.486849 | 3.587765 | 5.2476 |
| adapters | 8 | 3933 | 0.0030 | 0.0 | 100 | 2.066942 | 2.516864 | 3.631067 | 5.9134 |
| adapters | 8 | 3933 | 0.0030 | 1.0 | 50 | 2.096931 | 2.490349 | 3.592814 | 5.2697 |
| adapters | 8 | 3933 | 0.0003 | 0.0 | 50 | 2.107669 | 2.497402 | 3.602989 | 3.9870 |
| adapters | 16 | 5981 | 0.0030 | 0.0 | 50 | 2.088344 | 2.490219 | 3.592627 | 3.5193 |

Interpretation:

Head-only training is extremely cheap, but weak. It barely improves the frozen
prior unless regularized or slowed down, and it overfits by 100 steps.

Adapter-only residual updates are the best Cassandra-style result so far in
this branch. Rank 4 adapters plus the residual head train only 2909 parameters
and reach validation NLL `2.483014` at 50 steps. That is better than:

- the frozen prior alone: `2.502358`,
- the trainable count-bigram initialization at 50 steps: `2.519280`,
- the best constrained full residual run: about `2.4897`.

It still does not beat the best random transformer run on this tiny corpus
(`2.465410` at 50 steps), but it gets closer while training only about 2.7% as
many parameters as the full 105885-parameter transformer.

Next comparison:

1. Repeat adapter rank 4 on a larger corpus and multiple seeds.
2. Add LoRA-style low-rank updates to attention and MLP projections.
3. Compare LoRA-style updates against adapters at the same trainable-parameter
   budget.

## Stage 5 · Larger Corpus and Multi-Seed Replication

Date: 2026-06-16

Artifacts:

- corpus generator: `experiments/tiny_language_lab/make_synthetic_corpus.py`
- comparison runner: `experiments/tiny_language_lab/cassandra_compare.py`
- generated corpus: `experiments/tiny_language_lab/corpus/structured_seed.txt`
- generated run summary: `experiments/tiny_language_lab/runs/stage5_replication.md`
- generated JSONL: `experiments/tiny_language_lab/runs/stage5_replication.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_synthetic_corpus.py --lines 320 --seed 20260616
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19
```

The structured corpus has about 25.6K characters. It is synthetic and controlled,
not natural web text. Its purpose is to test whether the tiny-corpus ranking
survives a larger repeated grammar and multiple random seeds.

Replication summary:

| Config | Seeds | Trainable params | Mean val NLL | Min val NLL | Max val NLL | Mean bits/char | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_adapter_r4 | 3 | 3624 | 2.012933 | 2.005843 | 2.023098 | 2.904049 | 1.2719 |
| count_prior_head | 3 | 2600 | 2.009231 | 1.998740 | 2.019545 | 2.898708 | 0.9426 |
| random_full | 3 | 107304 | 2.078493 | 2.061794 | 2.109998 | 2.998631 | 1.4410 |

Raw runs:

| Config | Seed | Trainable params | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_full | 7 | 107304 | 2.109998 | 3.044084 | 2.1379 |
| count_prior_adapter_r4 | 7 | 3624 | 2.023098 | 2.918713 | 1.2632 |
| count_prior_head | 7 | 2600 | 2.019545 | 2.913588 | 1.0378 |
| random_full | 11 | 107304 | 2.063686 | 2.977270 | 1.1751 |
| count_prior_adapter_r4 | 11 | 3624 | 2.005843 | 2.893820 | 1.3072 |
| count_prior_head | 11 | 2600 | 1.998740 | 2.883572 | 0.8988 |
| random_full | 19 | 107304 | 2.061794 | 2.974540 | 1.0100 |
| count_prior_adapter_r4 | 19 | 3624 | 2.009859 | 2.899614 | 1.2452 |
| count_prior_head | 19 | 2600 | 2.009409 | 2.898964 | 0.8911 |

Interpretation:

This is the strongest Cassandra result so far. On the larger structured corpus,
both frozen-count methods beat the full random transformer across all three
seeds while training a tiny fraction of the parameters.

The surprise is that head-only beats rank 4 adapters here. That means the
frozen bigram prior is doing most of the work on this controlled corpus, and a
small output correction is enough to improve it. Adapter rank 4 is close behind
and may matter more on corpora where context dependence is stronger.

This result is still narrow:

- the corpus is synthetic,
- evaluation is sampled rather than full-split,
- only three seeds were used,
- the task remains character-level language modeling.

But it is no longer just a one-run curiosity. The current best low-hardware
method is: construct a count prior, freeze it, and train a very small residual
surface.

Next comparison:

1. Add LoRA-style low-rank updates to the frozen residual path.
2. Compare head-only, adapters, and LoRA at similar trainable-parameter budgets.
3. Add a corpus with stronger long-context dependencies where bigram counts
   should be insufficient.

## Stage 6 · LoRA-Style Low-Rank Residual Updates

Date: 2026-06-16

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
- comparison runner: `experiments/tiny_language_lab/cassandra_compare.py`
- generated run summary: `experiments/tiny_language_lab/runs/stage6_lora.md`
- generated JSONL: `experiments/tiny_language_lab/runs/stage6_lora.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage6_lora.jsonl --summary .\experiments\tiny_language_lab\runs\stage6_lora.md --title "Stage 6 LoRA Summary"
```

LoRA mode keeps the count prior frozen and trains:

- low-rank matrices attached to attention and MLP projections,
- the residual output head.

Replication summary:

| Config | Seeds | Trainable params | Mean val NLL | Min val NLL | Max val NLL | Mean bits/char | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_adapter_r4 | 3 | 3624 | 2.012933 | 2.005843 | 2.023098 | 2.904049 | 1.1855 |
| count_prior_head | 3 | 2600 | 2.009231 | 1.998740 | 2.019545 | 2.898708 | 1.0382 |
| count_prior_lora_r1 | 3 | 4648 | 2.000880 | 1.987741 | 2.015528 | 2.886659 | 1.5565 |
| count_prior_lora_r2 | 3 | 6696 | 1.992162 | 1.978828 | 2.002047 | 2.874083 | 1.4961 |
| random_full | 3 | 107304 | 2.078493 | 2.061794 | 2.109998 | 2.998631 | 1.6003 |

Raw runs:

| Config | Seed | Trainable params | Val NLL | Val bits/char | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_full | 7 | 107304 | 2.109998 | 3.044084 | 2.7468 |
| count_prior_head | 7 | 2600 | 2.019545 | 2.913588 | 1.1442 |
| count_prior_adapter_r4 | 7 | 3624 | 2.023098 | 2.918713 | 1.2445 |
| count_prior_lora_r1 | 7 | 4648 | 2.015528 | 2.907792 | 1.5639 |
| count_prior_lora_r2 | 7 | 6696 | 2.002047 | 2.888343 | 1.5075 |
| random_full | 11 | 107304 | 2.063686 | 2.977270 | 0.9718 |
| count_prior_head | 11 | 2600 | 1.998740 | 2.883572 | 0.9926 |
| count_prior_adapter_r4 | 11 | 3624 | 2.005843 | 2.893820 | 1.1767 |
| count_prior_lora_r1 | 11 | 4648 | 1.987741 | 2.867704 | 1.5727 |
| count_prior_lora_r2 | 11 | 6696 | 1.978828 | 2.854846 | 1.5067 |
| random_full | 19 | 107304 | 2.061794 | 2.974540 | 1.0822 |
| count_prior_head | 19 | 2600 | 2.009409 | 2.898964 | 0.9778 |
| count_prior_adapter_r4 | 19 | 3624 | 2.009859 | 2.899614 | 1.1352 |
| count_prior_lora_r1 | 19 | 4648 | 1.999370 | 2.884482 | 1.5328 |
| count_prior_lora_r2 | 19 | 6696 | 1.995612 | 2.879059 | 1.4740 |

Interpretation:

LoRA is the strongest low-trainable-parameter method so far on the structured
corpus. Rank 1 LoRA beats head-only and adapters across the three-seed mean.
Rank 2 LoRA is stronger again, reaching mean validation NLL `1.992162` with
6696 trainable parameters. That is about 6.2% of the full random transformer's
trainable parameter count.

The current best Cassandra recipe is now:

1. construct a count-based prior,
2. freeze it outside the trainable transformer,
3. train a small residual surface,
4. prefer low-rank projection updates when the budget allows more than a head.

This is still not a claim about general LLM training. It is a real local result
for controlled character-level modeling: structured construction plus small
updates can beat full random training under the same step budget.

Next comparison:

1. Add a corpus with long-context dependencies that bigram counts cannot solve.
2. Test whether LoRA still beats head-only when context matters.
3. Add full-split evaluation for medium corpora or increase sampled eval batches
   for tighter confidence intervals.

## Stage 7 · Long-Context Copy Probe

Date: 2026-06-16

Artifacts:

- corpus generator: `experiments/tiny_language_lab/make_long_context_corpus.py`
- generated corpus: `experiments/tiny_language_lab/corpus/long_context_seed.txt`
- copy probe implementation: `--copy-probe-marker` in `cassandra_tiny_transformer.py`
- generated run summary: `experiments/tiny_language_lab/runs/stage7_long_context.md`
- generated 500-step diagnostic: `experiments/tiny_language_lab/runs/stage7_long_context_500.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --lines 512 --seed 20260617
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage7_long_context.jsonl --summary .\experiments\tiny_language_lab\runs\stage7_long_context.md --title "Stage 7 Long-Context Summary"
```

Each line contains a key earlier in the line and an answer later in the line:

```text
case 0000 key=a noise=f brisk method; direct measure; amber signal; answer=a
```

The probe asks the trained model to predict the character after `answer=`. With
eight possible keys, chance accuracy is about `0.125`.

Three-seed comparison at 100 steps:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_adapter_r4 | 3 | 3559 | 1.706257 | 2.461609 | 0.135965 | 2.113758 | 2.6143 |
| count_prior_head | 3 | 2535 | 1.711219 | 2.468768 | 0.114035 | 2.085737 | 2.0414 |
| count_prior_lora_r2 | 3 | 6631 | 1.559663 | 2.250118 | 0.131579 | 2.114940 | 3.4210 |
| random_full | 3 | 111271 | 0.896153 | 1.292876 | 0.131579 | 2.144677 | 3.0284 |

500-step diagnostic, seed 7:

| Config | Trainable params | Val NLL | Val bits/char | Copy accuracy | Copy NLL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_full | 111271 | 0.301169 | 0.434495 | 0.131579 | 2.125726 | 11.2567 |
| count_prior_lora_r2 | 6631 | 1.219263 | 1.759025 | 0.105263 | 2.170438 | 10.7469 |

Interpretation:

The long-context corpus breaks the previous ranking. The full random transformer
wins strongly on validation NLL, while the frozen-count methods fall behind.
That is expected: a bigram prior helps local syntax, but it cannot solve a
specific key-copy dependency.

The more important result is the copy probe. Even when validation NLL improves
dramatically, copy accuracy stays near chance. The model is learning the line
format much faster than it is learning the actual long-context copy rule.

This is a valuable failure. It shows why Cassandra needs task-specific probes
and verifier-style objectives, not just next-character validation loss. The next
step should train against the copy positions directly, or use a rule reward /
rejection loop that gives credit only when `answer=` matches the earlier `key=`.

## Stage 8 · Task-Aware Copy-Position Training

Date: 2026-06-16

Artifacts:

- copy training implementation: `--copy-train-marker` and `--copy-loss-weight`
  in `cassandra_tiny_transformer.py`
- comparison configs: `random_full_copyw` and
  `count_prior_lora_r2_copyw` in `cassandra_compare.py`
- moderate-weight run summary:
  `experiments/tiny_language_lab/runs/stage8_copy_training.md`
- high-pressure run summary:
  `experiments/tiny_language_lab/runs/stage8_copy_weight200.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --seeds 7 11 19 --configs random_full count_prior_lora_r2 random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage8_copy_training.jsonl --summary .\experiments\tiny_language_lab\runs\stage8_copy_training.md --title "Stage 8 Copy Training Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage8_copy_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage8_copy_weight200.md --title "Stage 8 Copy Weight 200 Summary"
```

The training loss now gives extra weight to the character immediately after
`answer=` in the training split. Validation NLL is still ordinary
next-character loss, so the copy probe remains the behavior-specific metric.

Moderate copy weighting, 200 steps, three seeds:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2 | 3 | 6631 | 1.464593 | 2.112961 | 0.144737 | 2.136991 | 5.1388 |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.584720 | 2.286268 | 0.114035 | 2.099739 | 5.2249 |
| random_full | 3 | 111271 | 0.395584 | 0.570708 | 0.131579 | 2.101714 | 4.4545 |
| random_full_copyw | 3 | 111271 | 0.506129 | 0.730190 | 0.214912 | 1.913073 | 4.1467 |

High-pressure copy weighting, 500 steps, three seeds:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.710059 | 2.467094 | 0.293859 | 1.744836 | 12.0439 |
| random_full_copyw | 3 | 111271 | 0.510496 | 0.736490 | 0.320175 | 1.657034 | 10.7698 |

Interpretation:

Task-aware copy weighting changes the behavior metric in a way plain
next-character training did not. In Stage 7, even a 500-step diagnostic stayed
near chance on copy accuracy. Here, targeting the `answer=` positions lifts copy
accuracy well above chance, including for frozen-prior rank-2 LoRA.

The result is not free. Ordinary validation NLL gets worse because the model is
spending capacity on the rare answer positions instead of only optimizing the
whole-line format. That tradeoff is exactly the point of this stage: Cassandra
needs to track task behavior separately from broad next-token loss.

The most important local result is that a small trainable residual surface can
move a held-out behavior when the objective gives direct credit for that
behavior. The next step should replace hand-weighted labels with verifier-guided
data selection or rewards, so the model receives credit for matching
`answer=` to the earlier `key=` rather than merely upweighting that character
position.

## Stage 9 · Verifier-Guided Copy Sampling

Date: 2026-06-16

Artifacts:

- verifier target discovery: `find_verified_copy_targets` in
  `cassandra_tiny_transformer.py`
- copy samplers: `--copy-sampler answer` and `--copy-sampler mixed`
- comparison configs: `random_full_copys`, `random_full_copymix`,
  `count_prior_lora_r2_copys`, and `count_prior_lora_r2_copymix`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage9_answer_sampler.md`,
  `experiments/tiny_language_lab/runs/stage9_mixed_sampler_frac025.md`, and
  `experiments/tiny_language_lab/runs/stage9_mixed_sampler_weight200.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw random_full_copys count_prior_lora_r2_copys --out .\experiments\tiny_language_lab\runs\stage9_answer_sampler.jsonl --summary .\experiments\tiny_language_lab\runs\stage9_answer_sampler.md --title "Stage 9 Answer Sampler Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs random_full_copymix count_prior_lora_r2_copymix --out .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_frac025.jsonl --summary .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_frac025.md --title "Stage 9 Mixed Sampler Fraction 0.25"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs random_full_copymix count_prior_lora_r2_copymix --out .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.md --title "Stage 9 Mixed Sampler Weight 200"
```

The verifier scans each training line and accepts an answer target only when the
character after `answer=` matches the earlier character after `key=`. The
`answer` sampler trains only on answer-anchored windows. The `mixed` sampler
keeps ordinary random windows but reserves a fraction of each batch for verified
answer-anchored windows.

Answer-only sampling, 200 steps, weight `50`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copys | 3 | 6631 | 2.324794 | 3.353969 | 0.114035 | 2.176376 | 5.1417 |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.584720 | 2.286268 | 0.114035 | 2.099739 | 5.1165 |
| random_full_copys | 3 | 111271 | 3.469097 | 5.004849 | 0.127193 | 2.587891 | 4.5005 |
| random_full_copyw | 3 | 111271 | 0.506129 | 0.730190 | 0.214912 | 1.913073 | 4.7654 |

Mixed sampling, 25 percent verified windows, 200 steps, weight `50`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copymix | 3 | 6631 | 1.569256 | 2.263958 | 0.122807 | 2.082357 | 6.7236 |
| random_full_copymix | 3 | 111271 | 0.510210 | 0.736078 | 0.250000 | 1.932733 | 5.7576 |

Mixed sampling, 25 percent verified windows, 500 steps, weight `200`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copymix | 3 | 6631 | 1.712748 | 2.470973 | 0.302632 | 1.756351 | 12.3590 |
| random_full_copymix | 3 | 111271 | 0.688730 | 0.993628 | 0.364035 | 1.687795 | 11.1143 |

Interpretation:

The verifier-guided sampler is useful only when it stays mixed with ordinary
context. Answer-only sampling is too narrow: it damages validation NLL and does
not improve copy accuracy. A 25 percent mixed sampler is better. At 200 steps it
improves full-model copy accuracy from the Stage 8 weighted-random result
`0.214912` to `0.25` at nearly identical validation NLL. It also gives the LoRA
path a small copy-NLL and accuracy improvement.

At the stronger 500-step setting, mixed sampling improves copy accuracy over
Stage 8 for both models: full training rises from `0.320175` to `0.364035`, and
frozen-prior rank-2 LoRA rises from `0.293859` to `0.302632`. The tradeoff is
that validation NLL and copy NLL are not uniformly better.

This is a good Cassandra-shaped result: a verifier can make training more
behavior-aware, but sampling alone is a blunt tool. The next step should make
the verifier produce corrections or rewards, so the model is updated because it
matched the rule, not just because verified windows were oversampled.

## Stage 10 · Verifier Choice-Loss Correction

Date: 2026-06-16

Artifacts:

- choice candidate construction: `build_choice_tensors` in
  `cassandra_tiny_transformer.py`
- choice correction knob: `--copy-choice-weight`
- comparison configs: `random_full_copyw_choice`,
  `random_full_copymix_choice`, `count_prior_lora_r2_copyw_choice`, and
  `count_prior_lora_r2_copymix_choice`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage10_choice_loss.md`,
  `experiments/tiny_language_lab/runs/stage10_choice_loss_weight025.md`,
  `experiments/tiny_language_lab/runs/stage10_mixed_choice_loss.md`, and
  `experiments/tiny_language_lab/runs/stage10_choice_loss_weight200.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-choice-weight 1 --seeds 7 11 19 --configs random_full_copyw random_full_copyw_choice count_prior_lora_r2_copyw count_prior_lora_r2_copyw_choice --out .\experiments\tiny_language_lab\runs\stage10_choice_loss.jsonl --summary .\experiments\tiny_language_lab\runs\stage10_choice_loss.md --title "Stage 10 Choice Loss Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-choice-weight 0.25 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs random_full_copyw_choice count_prior_lora_r2_copyw_choice random_full_copymix_choice count_prior_lora_r2_copymix_choice --out .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.jsonl --summary .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.md --title "Stage 10 Choice Loss Weight 0.25"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-choice-weight 0.25 --seeds 7 11 19 --configs random_full_copyw_choice count_prior_lora_r2_copyw_choice --out .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight200.md --title "Stage 10 Choice Loss Weight 200"
```

The choice loss activates only at answer positions. It restricts the logits to
the verified answer candidates, `a` through `h`, and adds a small auxiliary
cross-entropy over that candidate set.

Weighted-random copy training, 200 steps, choice weight `1.0`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.584720 | 2.286268 | 0.114035 | 2.099739 | 5.1932 |
| count_prior_lora_r2_copyw_choice | 3 | 6631 | 1.701722 | 2.455067 | 0.140351 | 2.073026 | 5.6215 |
| random_full_copyw | 3 | 111271 | 0.506129 | 0.730190 | 0.214912 | 1.913073 | 4.7498 |
| random_full_copyw_choice | 3 | 111271 | 0.630309 | 0.909343 | 0.153509 | 2.082941 | 4.7563 |

Gentler choice correction, 200 steps, choice weight `0.25`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copymix_choice | 3 | 6631 | 1.626435 | 2.346450 | 0.114035 | 2.075933 | 6.3154 |
| count_prior_lora_r2_copyw_choice | 3 | 6631 | 1.659330 | 2.393906 | 0.131579 | 2.089639 | 5.5834 |
| random_full_copymix_choice | 3 | 111271 | 0.605788 | 0.873967 | 0.245614 | 2.013108 | 4.8443 |
| random_full_copyw_choice | 3 | 111271 | 0.580049 | 0.836833 | 0.250000 | 1.913565 | 5.0518 |

High-pressure setting, 500 steps, copy weight `200`, choice weight `0.25`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyw_choice | 3 | 6631 | 1.723149 | 2.485979 | 0.254386 | 1.703366 | 10.9973 |
| random_full_copyw_choice | 3 | 111271 | 0.496521 | 0.716328 | 0.201754 | 1.808739 | 10.6578 |

Interpretation:

Choice loss is a sharper verifier-derived signal, but it is not automatically
better. At 200 steps, it helps the smaller frozen-prior LoRA surface under
weighted-random training: accuracy rises from `0.114035` to `0.131579` with a
gentle weight and to `0.140351` with weight `1.0`. The full model only benefits
from the gentler weighted-random setting, where accuracy reaches `0.25`; the
stronger choice loss hurts it.

The mixed sampler does not combine well with choice loss at these settings.
At the 500-step, high-copy-pressure setting, choice loss improves LoRA copy NLL
from the Stage 8 value `1.744836` to `1.703366`, but accuracy falls from
`0.293859` to `0.254386`. For the full model, both copy accuracy and copy NLL
are worse than Stage 8.

This stage is still progress. It shows that verifier-derived corrections can
help a small trainable surface under tight budgets, but a simple auxiliary
choice loss is too blunt. The next Cassandra step should try an actual
generate-check-correct loop: sample an answer, verify whether it matches the
key, then create a focused correction example only for failed cases.

## Stage 11 · Failed-Case Replay Correction Loop

Date: 2026-06-16

Artifacts:

- failed-case miner: `mine_failed_copy_starts` in
  `cassandra_tiny_transformer.py`
- failed replay samplers: `--copy-sampler failed` and
  `--copy-sampler failed_mixed`
- mining interval knob: `--copy-mine-every`
- comparison configs: `random_full_copyfail`,
  `random_full_copyfailmix`, `count_prior_lora_r2_copyfail`, and
  `count_prior_lora_r2_copyfailmix`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage11_failed_replay.md`,
  `experiments/tiny_language_lab/runs/stage11_failed_mixed_replay.md`,
  `experiments/tiny_language_lab/runs/stage11_failed_mixed_replay_frac010.md`,
  and `experiments/tiny_language_lab/runs/stage11_failed_mixed_replay_weight200.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-mine-every 50 --seeds 7 11 19 --configs random_full_copyw random_full_copyfail count_prior_lora_r2_copyw count_prior_lora_r2_copyfail --out .\experiments\tiny_language_lab\runs\stage11_failed_replay.jsonl --summary .\experiments\tiny_language_lab\runs\stage11_failed_replay.md --title "Stage 11 Failed Replay Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-sample-fraction 0.1 --copy-mine-every 50 --seeds 7 11 19 --configs random_full_copyfailmix count_prior_lora_r2_copyfailmix --out .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_frac010.jsonl --summary .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_frac010.md --title "Stage 11 Failed Mixed Replay Fraction 0.10"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.1 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copyfailmix count_prior_lora_r2_copyfailmix --out .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.md --title "Stage 11 Failed Mixed Replay Weight 200"
```

The failed replay loop periodically asks the current model to predict each
verified training copy case. Failed predictions are converted back into
answer-anchored training windows. In `failed` mode, every copy-focused example
comes from current failures. In `failed_mixed` mode, only a fraction of each
batch comes from failed cases and the rest remains ordinary random context.

Failed-only replay, 200 steps, weight `50`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyfail | 3 | 6631 | 2.043136 | 2.947621 | 0.157895 | 2.142648 | 5.9741 |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.584720 | 2.286268 | 0.114035 | 2.099739 | 5.0913 |
| random_full_copyfail | 3 | 111271 | 3.301010 | 4.762351 | 0.135965 | 2.309369 | 5.1754 |
| random_full_copyw | 3 | 111271 | 0.506129 | 0.730190 | 0.214912 | 1.913073 | 4.5178 |

Mixed failed replay, 10 percent failed windows, 200 steps, weight `50`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyfailmix | 3 | 6631 | 1.699782 | 2.452266 | 0.140351 | 2.101084 | 6.1313 |
| random_full_copyfailmix | 3 | 111271 | 0.749703 | 1.081593 | 0.144737 | 2.090700 | 5.7877 |

Mixed failed replay, 10 percent failed windows, 500 steps, weight `200`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyfailmix | 3 | 6631 | 1.717675 | 2.478081 | 0.289474 | 1.875184 | 12.2681 |
| random_full_copyfailmix | 3 | 111271 | 1.005360 | 1.450429 | 0.201754 | 1.918011 | 11.6014 |

Interpretation:

Failed-case replay is a more literal correction loop than Stages 9 and 10, but
the simple version is not better. Failed-only replay gives the LoRA residual a
small copy-accuracy gain at 200 steps, but the validation loss cost is large.
For the full model it is worse than weighted random training.

Mixed failed replay is less destructive, but it still trails static mixed
verified sampling. At 500 steps and copy weight `200`, the Stage 9 mixed sampler
reached copy accuracy `0.364035` for full training and `0.302632` for LoRA.
Failed mixed replay reaches only `0.201754` and `0.289474`.

The lesson is precise: replaying mistakes is not enough when the replay example
is just the same answer window again. The next version should generate a
correction artifact for failed cases, such as a short supervised string that
states the rule in-place (`key=e ... answer=e`) or a contrastive pair showing
the wrong prediction and the verified correction.

## Stage 12 · Generated Correction Examples

Date: 2026-06-16

Artifacts:

- correction text builder: `build_correction_data` in
  `cassandra_tiny_transformer.py`
- correction samplers: `--copy-sampler correction` and
  `--copy-sampler correction_mixed`
- comparison configs: `random_full_copycorr`,
  `random_full_copycorrmix`, `count_prior_lora_r2_copycorr`, and
  `count_prior_lora_r2_copycorrmix`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage12_correction_examples.md`,
  `experiments/tiny_language_lab/runs/stage12_correction_examples_gentle.md`,
  `experiments/tiny_language_lab/runs/stage12_weight10_baseline_500.md`, and
  `experiments/tiny_language_lab/runs/stage12_correction_examples_gentle_500.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-sample-fraction 0.25 --copy-mine-every 50 --seeds 7 11 19 --configs random_full_copyw random_full_copycorr random_full_copycorrmix count_prior_lora_r2_copyw count_prior_lora_r2_copycorr count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage12_correction_examples.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_correction_examples.md --title "Stage 12 Correction Examples Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 50 --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle.md --title "Stage 12 Gentle Correction Examples"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage12_weight10_baseline_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_weight10_baseline_500.md --title "Stage 12 Weight 10 Baseline 500"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.md --title "Stage 12 Gentle Correction Examples 500"
```

The generated correction examples are compact strings like:

```text
key=e answer=e
```

They are synthesized from failed verified train cases. The correction-only
sampler trains entirely on those strings. The mixed sampler reserves a fraction
of each batch for correction strings and keeps the rest as ordinary corpus
windows.

Initial correction examples, 200 steps, copy weight `50`, fraction `0.25`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copycorr | 3 | 6631 | 3.064680 | 4.421398 | 0.122807 | 3.560395 | 5.8551 |
| count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.753224 | 2.529368 | 0.127193 | 2.084865 | 5.8379 |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.584720 | 2.286268 | 0.114035 | 2.099739 | 5.2798 |
| random_full_copycorr | 3 | 111271 | 6.002450 | 8.659704 | 0.083333 | 5.207959 | 5.1089 |
| random_full_copycorrmix | 3 | 111271 | 1.900144 | 2.741329 | 0.114035 | 2.105173 | 5.0450 |
| random_full_copyw | 3 | 111271 | 0.506129 | 0.730190 | 0.214912 | 1.913073 | 4.6190 |

Gentle corrections, 200 steps, copy weight `10`, fraction `0.05`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.647392 | 2.376684 | 0.149123 | 2.089268 | 6.0867 |
| random_full_copycorrmix | 3 | 111271 | 0.584307 | 0.842977 | 0.192982 | 2.051816 | 5.5223 |

Same-weight baseline, 500 steps, copy weight `10`, no correction examples:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.292021 | 1.863991 | 0.131579 | 2.112194 | 10.4607 |
| random_full_copyw | 3 | 111271 | 0.351587 | 0.507233 | 0.425438 | 1.357607 | 10.2118 |

Gentle corrections, 500 steps, copy weight `10`, fraction `0.05`:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.549951 | 2.236106 | 0.258772 | 1.928378 | 11.8932 |
| random_full_copycorrmix | 3 | 111271 | 0.519950 | 0.750129 | 0.807018 | 0.586816 | 11.2056 |

Interpretation:

This is the first strongly positive generate-check-correct result. The naive
correction-only setting is bad because it is too far from the long-context line
distribution. A gentle mixed dose is very different. With only 5 percent
generated correction examples and a copy weight of `10`, the full model reaches
mean copy accuracy `0.807018` at 500 steps. The fair same-weight baseline,
without correction examples, reaches `0.425438`.

The LoRA path also benefits from generated corrections: at 500 steps it rises
from `0.131579` to `0.258772` under the same copy weight. It still trails the
best Stage 8 and Stage 9 LoRA recipes, but the generated correction data clearly
changes behavior.

The Cassandra lesson is important: a verifier-generated correction artifact can
teach the rule more efficiently than replaying failed windows. The next step is
to make corrections more structurally faithful to the original task, for example
by generating compact long-context traces rather than only `key=x answer=x`
strings.

## Stage 13 · Correction Template Shape

Date: 2026-06-16

Artifacts:

- correction template switch: `--copy-correction-template`
- template implementation: `correction_line` in
  `cassandra_tiny_transformer.py`
- supported templates: `compact`, `focus`, `prefix`, and `full`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage13_focus_corrections_500.md` and
  `experiments/tiny_language_lab/runs/stage13_prefix_corrections_500.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template focus --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage13_focus_corrections_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage13_focus_corrections_500.md --title "Stage 13 Focus Corrections 500"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.md --title "Stage 13 Prefix Corrections 500"
```

Templates:

```text
compact: key=e answer=e
focus:   key=e noise=b answer=e
prefix:  case 0436 key=e noise=b even budget; amber signal; gentle update; answer=e
```

All runs use 500 steps, copy weight `10`, 5 percent correction examples, and
three seeds. The compact numbers are the Stage 12 gentle 500-step result.

| Template | Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| compact | random_full_copycorrmix | 3 | 111271 | 0.519950 | 0.807018 | 0.586816 |
| focus | random_full_copycorrmix | 3 | 111271 | 0.427344 | 0.697369 | 0.777646 |
| prefix | random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.881579 | 0.333680 |
| compact | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.549951 | 0.258772 | 1.928378 |
| focus | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.591405 | 0.250000 | 1.807520 |
| prefix | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 0.285088 | 1.767904 |

Interpretation:

Correction shape matters. The compact template was already strong, but the
prefix template is better on the behavior metric. For the full model, mean copy
accuracy rises from `0.807018` to `0.881579`, and copy NLL improves from
`0.586816` to `0.333680`. For the frozen-prior LoRA residual path, accuracy
rises from `0.258772` to `0.285088`, and copy NLL improves from `1.928378` to
`1.767904`.

The focus template is informative too: adding `noise=` helps ordinary
validation NLL for the full model, but it loses copy accuracy. The best
correction is not merely the shortest or the most fluent; it preserves enough of
the original task geometry to transfer.

This strengthens Cassandra's working hypothesis: on a controlled task, verifier
logic can create small synthetic correction data that teaches behavior more
efficiently than broad next-token exposure. The next experiment should try
retrieval-style context injection so the model can use an external correction
memory rather than storing every rule adaptation in weights.

## Stage 14 · Retrieval-Style Probe Context

Date: 2026-06-16

Artifacts:

- probe-time retrieval context: `--copy-probe-retrieval-template`
- retrieval builder: `copy_probe_retrieval_context` in
  `cassandra_tiny_transformer.py`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage14_retrieval_probe_compact.md`,
  `experiments/tiny_language_lab/runs/stage14_retrieval_probe_focus.md`,
  `experiments/tiny_language_lab/runs/stage14_retrieval_probe_prefix.md`, and
  `experiments/tiny_language_lab/runs/stage14_retrieval_weight10_baseline.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.md --title "Stage 14 Retrieval Probe Compact"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template focus --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_focus.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_focus.md --title "Stage 14 Retrieval Probe Focus"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template prefix --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_prefix.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_prefix.md --title "Stage 14 Retrieval Probe Prefix"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage14_retrieval_weight10_baseline.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_weight10_baseline.md --title "Stage 14 Retrieval Weight 10 Baseline"
```

Retrieval context is inserted only during the copy probe. It does not affect
training or ordinary validation NLL. Example compact retrieval:

```text
key=e answer=e
case 0436 key=e noise=b even budget; amber signal; gentle update; answer=
```

Prefix-correction-trained model, retrieval templates:

| Retrieval | Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| none | random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.881579 | 0.333680 |
| compact | random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.960526 | 0.274268 |
| focus | random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.798246 | 1.022270 |
| prefix | random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.714912 | 1.208356 |
| none | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 0.285088 | 1.767904 |
| compact | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 0.241228 | 1.805327 |
| focus | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 0.276316 | 1.755235 |
| prefix | count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 0.271930 | 1.863075 |

Same-weight baseline without generated corrections:

| Retrieval | Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| none | random_full_copyw | 3 | 111271 | 0.351587 | 0.425438 | 1.357607 |
| compact | random_full_copyw | 3 | 111271 | 0.351587 | 0.403509 | 1.543095 |
| none | count_prior_lora_r2_copyw | 3 | 6631 | 1.292021 | 0.131579 | 2.112194 |
| compact | count_prior_lora_r2_copyw | 3 | 6631 | 1.292021 | 0.109649 | 2.113377 |

Interpretation:

Retrieval works when the model has been prepared to use the hint. The
prefix-correction-trained full model improves from `0.881579` to `0.960526`
copy accuracy with compact retrieval, and copy NLL improves from `0.333680` to
`0.274268`. Longer retrieval strings are worse under the 96-character context
limit, likely because they compete with the held-out prompt for context space.

Retrieval does not help the same-weight model trained without generated
correction examples. In that baseline, compact retrieval slightly lowers copy
accuracy for both full training and LoRA. The small LoRA residual path also does
not benefit from retrieval in the prefix-correction setup.

The useful conclusion is narrow but important: external correction memory can
substitute for some behavior stored in weights, but only after the model has
learned how to use that memory format. The next step should train a tiny model
explicitly on retrieval-use tasks, then test whether increasingly much behavior
can be moved from weights into generated or retrieved context.

## Stage 15 · Retrieval-Use Training

Date: 2026-06-16

Artifacts:

- retrieval training builder: `build_retrieval_data` in
  `cassandra_tiny_transformer.py`
- retrieval example formatter: `retrieval_training_line`
- retrieval samplers: `--copy-sampler retrieval` and
  `--copy-sampler retrieval_mixed`
- retrieval training template switch: `--copy-train-retrieval-template`
- comparison configs: `random_full_retmix` and
  `count_prior_lora_r2_retmix`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage15_retrieval_use_training.md`,
  `experiments/tiny_language_lab/runs/stage15_retrieval_use_training_focus.md`,
  and
  `experiments/tiny_language_lab/runs/stage15_retrieval_use_training_prefix.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.jsonl --summary .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.md --title "Stage 15 Retrieval Use Training"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template focus --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template focus --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training_focus.jsonl --summary .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training_focus.md --title "Stage 15 Retrieval Use Training Focus"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template prefix --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template prefix --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training_prefix.jsonl --summary .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training_prefix.md --title "Stage 15 Retrieval Use Training Prefix"
```

Retrieval-use examples prepend an external memory hint to the original copy
prompt and train through the answer:

```text
key=e answer=e
case 0436 key=e noise=b even budget; amber signal; gentle update; answer=e
```

All runs use 500 steps, copy weight `10`, 5 percent retrieval examples, and
three seeds.

| Train/probe retrieval | Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| compact | random_full_retmix | 3 | 111271 | 0.347947 | 0.776316 | 0.608794 |
| focus | random_full_retmix | 3 | 111271 | 0.340703 | 0.592105 | 0.962518 |
| prefix | random_full_retmix | 3 | 111271 | 0.336505 | 0.723684 | 0.825652 |
| compact | count_prior_lora_r2_retmix | 3 | 6631 | 1.276510 | 0.149123 | 2.113718 |
| focus | count_prior_lora_r2_retmix | 3 | 6631 | 1.296860 | 0.149123 | 2.089798 |
| prefix | count_prior_lora_r2_retmix | 3 | 6631 | 1.300680 | 0.140351 | 2.121482 |

Comparison to key baselines:

| Method | Config | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: |
| same-weight baseline, compact retrieval probe | random_full_copyw | 0.351587 | 0.403509 | 1.543095 |
| direct compact retrieval-use training | random_full_retmix | 0.347947 | 0.776316 | 0.608794 |
| prefix correction training, no retrieval probe | random_full_copycorrmix | 0.435122 | 0.881579 | 0.333680 |
| prefix correction training, compact retrieval probe | random_full_copycorrmix | 0.435122 | 0.960526 | 0.274268 |

Interpretation:

Direct retrieval-use training works, but it is not yet the best route. The full
model learns the compact retrieval interface well enough to nearly double the
same-weight compact-retrieval baseline: copy accuracy moves from `0.403509` to
`0.776316`, and copy NLL drops from `1.543095` to `0.608794`.

However, generated prefix correction traces remain stronger. Stage 14's
prefix-correction-trained model with compact retrieval reaches `0.960526` copy
accuracy and copy NLL `0.274268`. The likely reason is that the correction trace
teaches the task geometry first, while raw retrieval-use examples ask the model
to learn both the external-memory interface and the answer behavior at once.

Template shape matters differently than it did for correction examples. For
direct retrieval-use training, compact is the best behavior template for the
full model. Focus and prefix have slightly better validation NLL, but worse copy
accuracy. The small LoRA residual path still does not learn this interface; it
stays near chance even when retrieval examples are included in training.

The next experiment should combine the two useful signals: generated correction
traces that teach the interface, then retrieval-use examples that move more of
the answer-bearing content into external memory.

## Stage 16 · Correction-Retrieval Curriculum

Date: 2026-06-16

Artifacts:

- dual auxiliary batch mixer: `get_dual_aux_mixed_batch` in
  `cassandra_tiny_transformer.py`
- combined sampler: `--copy-sampler correction_retrieval_mixed`
- comparison configs: `random_full_corrretmix` and
  `count_prior_lora_r2_corrretmix`
- generated summary:
  `experiments/tiny_language_lab/runs/stage16_correction_retrieval_curriculum.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrretmix count_prior_lora_r2_corrretmix --out .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.jsonl --summary .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.md --title "Stage 16 Correction Retrieval Curriculum"
```

This stage mixes three sources in one batch:

- ordinary corpus windows,
- generated prefix correction examples,
- compact retrieval-use examples.

With batch size `16` and `--copy-sample-fraction 0.1`, the implementation uses
one correction example and one retrieval-use example per batch, with the rest
coming from ordinary corpus windows.

Three-seed result:

| Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_full_corrretmix | 3 | 111271 | 0.471090 | 0.600877 | 0.978377 |
| count_prior_lora_r2_corrretmix | 3 | 6631 | 1.592112 | 0.302632 | 1.780110 |

Comparison to the relevant previous stages:

| Method | Config | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: |
| Stage 14 prefix corrections plus compact retrieval probe | random_full_copycorrmix | 0.435122 | 0.960526 | 0.274268 |
| Stage 15 compact retrieval-use training | random_full_retmix | 0.347947 | 0.776316 | 0.608794 |
| Stage 16 simultaneous correction plus retrieval training | random_full_corrretmix | 0.471090 | 0.600877 | 0.978377 |
| Stage 14 prefix corrections plus compact retrieval probe | count_prior_lora_r2_copycorrmix | 1.609495 | 0.241228 | 1.805327 |
| Stage 15 compact retrieval-use training | count_prior_lora_r2_retmix | 1.276510 | 0.149123 | 2.113718 |
| Stage 16 simultaneous correction plus retrieval training | count_prior_lora_r2_corrretmix | 1.592112 | 0.302632 | 1.780110 |

Interpretation:

The simple combined curriculum is not a win for the full model. It is worse
than both parent strategies on copy behavior: `0.600877` accuracy versus
`0.960526` for correction training plus retrieval at probe time, and `0.776316`
for direct compact retrieval-use training. The likely cause is interference:
mixing correction traces and retrieval-use traces simultaneously weakens the
clean prefix-correction signal that worked in Stage 14.

The LoRA path is more interesting. Stage 16 is the best LoRA copy-accuracy
result in this retrieval branch: `0.302632`, compared with `0.241228` for Stage
14 compact retrieval and `0.149123` for Stage 15 compact retrieval-use training.
It still remains weak, but the combined signal helps the constrained residual
path more than either signal alone.

The next experiment should test ordering rather than simultaneity: train on
prefix correction traces first, then switch to compact retrieval-use examples,
or alternate phases with separate weights. Cassandra's current evidence says
that external memory is useful, but the way the model is taught to read that
memory matters as much as the memory itself.

## Stage 17 · Staged Correction-Then-Retrieval Curriculum

Date: 2026-06-16

Artifacts:

- staged sampler: `--copy-sampler correction_then_retrieval_mixed`
- curriculum switch: `--copy-curriculum-switch-fraction`
- comparison configs: `random_full_corrthenret` and
  `count_prior_lora_r2_corrthenret`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage17_staged_correction_then_retrieval.md`
  and
  `experiments/tiny_language_lab/runs/stage17_staged_correction_then_retrieval_late.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.5 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.md --title "Stage 17 Staged Correction Then Retrieval"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.md --title "Stage 17 Staged Correction Then Retrieval Late Switch"
```

The staged sampler uses correction-mixed training until the switch step, then
retrieval-mixed training for the rest of the run. With 500 total steps:

- switch fraction `0.5` switches at step `250`,
- switch fraction `0.8` switches at step `400`.

Three-seed result:

| Switch fraction | Config | Seeds | Trainable params | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | random_full_corrthenret | 3 | 111271 | 0.472298 | 0.833333 | 0.480206 |
| 0.8 | random_full_corrthenret | 3 | 111271 | 0.464284 | 0.881579 | 0.432282 |
| 0.5 | count_prior_lora_r2_corrthenret | 3 | 6631 | 1.569861 | 0.236842 | 1.760320 |
| 0.8 | count_prior_lora_r2_corrthenret | 3 | 6631 | 1.572889 | 0.236842 | 1.798999 |

Comparison to nearby stages:

| Method | Config | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: |
| Stage 14 prefix corrections plus compact retrieval probe | random_full_copycorrmix | 0.435122 | 0.960526 | 0.274268 |
| Stage 15 compact retrieval-use training | random_full_retmix | 0.347947 | 0.776316 | 0.608794 |
| Stage 16 simultaneous correction plus retrieval training | random_full_corrretmix | 0.471090 | 0.600877 | 0.978377 |
| Stage 17 staged, 0.5 switch | random_full_corrthenret | 0.472298 | 0.833333 | 0.480206 |
| Stage 17 staged, 0.8 switch | random_full_corrthenret | 0.464284 | 0.881579 | 0.432282 |
| Stage 14 prefix corrections plus compact retrieval probe | count_prior_lora_r2_copycorrmix | 1.609495 | 0.241228 | 1.805327 |
| Stage 15 compact retrieval-use training | count_prior_lora_r2_retmix | 1.276510 | 0.149123 | 2.113718 |
| Stage 16 simultaneous correction plus retrieval training | count_prior_lora_r2_corrretmix | 1.592112 | 0.302632 | 1.780110 |
| Stage 17 staged, 0.5 switch | count_prior_lora_r2_corrthenret | 1.569861 | 0.236842 | 1.760320 |
| Stage 17 staged, 0.8 switch | count_prior_lora_r2_corrthenret | 1.572889 | 0.236842 | 1.798999 |

Interpretation:

Ordering matters. For the full model, staging is much better than simultaneous
mixing: copy accuracy rises from Stage 16's `0.600877` to `0.833333` with an
even switch and `0.881579` with a late switch. The late switch is better,
suggesting the model should spend most of its budget learning the correction
interface before retrieval-use examples are introduced.

But staging does not beat the simpler Stage 14 recipe. Prefix correction
training with compact retrieval only at probe time remains best for the full
model: `0.960526` copy accuracy and `0.274268` copy NLL. The staged retrieval
phase appears to trade away some of the clean correction behavior rather than
adding a new capability under this 500-step budget.

The LoRA result points the other way. The small residual path did best under
Stage 16 simultaneous mixing (`0.302632` copy accuracy). Both staged schedules
fall to `0.236842`. A plausible explanation is that LoRA has too little
trainable surface to preserve a correction interface across a phase switch; it
needs both pressures at once, even though that hurts the full model.

The next step should move beyond synthetic retrieval strings toward a real
external-memory table: keep the strongest correction-trained model behavior, but
make the retrieved context come from an indexed memory lookup rather than a
hand-constructed probe hint.

## Stage 18 · Train-Split Memory Retrieval Probe

Date: 2026-06-16

Artifacts:

- memory source switch: `--copy-probe-retrieval-source target|memory`
- train-split memory builder: `build_copy_memory` in
  `cassandra_tiny_transformer.py`
- memory-aware probe reporting:
  `copy_probe_retrieval_source`, `copy_probe_retrieval_memory_entries`,
  `copy_probe_retrieval_memory_observations`,
  `copy_probe_retrieval_memory_conflicts`, `copy_probe_retrieval_hits`, and
  `copy_probe_retrieval_misses`
- comparison summaries:
  `experiments/tiny_language_lab/runs/stage18_memory_retrieval_probe.md` and
  `experiments/tiny_language_lab/runs/stage18_memory_retrieval_baseline.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.md --title "Stage 18 Memory Retrieval Probe"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.md --title "Stage 18 Memory Retrieval Baseline"
```

What changed:

Stage 14's retrieval probe constructed the compact hint from the held-out target
answer. That was useful as an interface test, but too generous as a memory
claim. Stage 18 adds `--copy-probe-retrieval-source memory`, which builds a
key-answer table from the training split only. At probe time the held-out line
contributes the key, and the answer hint must come from the table.

In the smoke run, the memory source reported:

| Memory entries | Training observations | Conflicts | Probe hits | Probe misses |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 435 | 0 | 76 | 0 |

Correction-trained model with train-split memory retrieval:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_full_copycorrmix | 3 | 111271 | 0.435122 | 0.627748 | 0.960526 | 0.274268 |
| count_prior_lora_r2_copycorrmix | 3 | 6631 | 1.609495 | 2.322011 | 0.241228 | 1.805327 |

Same-weight baseline with train-split memory retrieval:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_full_copyw | 3 | 111271 | 0.351587 | 0.507233 | 0.403509 | 1.543095 |
| count_prior_lora_r2_copyw | 3 | 6631 | 1.292021 | 1.863991 | 0.109649 | 2.113377 |

Comparison to Stage 14:

| Method | Config | Retrieval source | Mean val NLL | Mean copy accuracy | Mean copy NLL |
| --- | --- | --- | ---: | ---: | ---: |
| Stage 14 prefix corrections plus compact retrieval probe | random_full_copycorrmix | held-out target | 0.435122 | 0.960526 | 0.274268 |
| Stage 18 prefix corrections plus compact retrieval probe | random_full_copycorrmix | train-split memory | 0.435122 | 0.960526 | 0.274268 |
| Stage 14 same-weight compact retrieval baseline | random_full_copyw | held-out target | 0.351587 | 0.403509 | 1.543095 |
| Stage 18 same-weight compact retrieval baseline | random_full_copyw | train-split memory | 0.351587 | 0.403509 | 1.543095 |

Interpretation:

Stage 18 validates the earlier retrieval measurement. The strongest Stage 14
full-model result survives unchanged when retrieval comes from training memory
instead of the held-out answer: copy accuracy remains `0.960526`, and copy NLL
remains `0.274268`.

The same-weight baseline also remains weak, so the conclusion is unchanged:
external memory is useful only after correction training teaches the model how
to use the memory format. Retrieval alone does not solve the task.

This does not yet prove robust external memory. The current corpus has only
eight possible keys, and every validation key is present in the training split.
So Stage 18 is a hygiene and measurement stage, not a generalization stage. The
next memory experiment should make the lookup harder: unseen keys, ambiguous
memories, larger answer alphabets, missing entries, or multiple retrieved
candidates.

## Stage 19 · Memory Corruption Ablation

Date: 2026-06-16

Artifacts:

- probe corruption knob: `--copy-probe-retrieval-corrupt none|wrong-answer`
- corruption helper: `corrupt_copy_probe_hint` in
  `cassandra_tiny_transformer.py`
- corrupted-hint report field: `copy_probe_retrieval_corrupted`
- generated summary:
  `experiments/tiny_language_lab/runs/stage19_memory_corruption_ablation.md`
- generated JSONL:
  `experiments/tiny_language_lab/runs/stage19_memory_corruption_ablation.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.jsonl --summary .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.md --title "Stage 19 Memory Corruption Ablation"
```

What changed:

The probe still retrieves from the Stage 18 train-split memory table, but after
lookup the retrieved answer is replaced with a deterministic different valid key
character. Example: a correct memory hint `key=e answer=e` becomes
`key=f answer=f`, while the held-out prompt still asks for `answer=e`.

Memory coverage check:

| Retrieval source | Corruption | Memory entries | Probe hits | Probe misses | Corrupted hints |
| --- | --- | ---: | ---: | ---: | ---: |
| memory | wrong-answer | 8 | 76 | 0 | 76 |

Three-way comparison:

| Probe condition | Config | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no hint, Stage 13 | random_full_copycorrmix | 111271 | 0.435122 | 0.627748 | 0.881579 | 0.644737 | 1.000000 | 0.333680 |
| correct memory, Stage 18 | random_full_copycorrmix | 111271 | 0.435122 | 0.627748 | 0.960526 | 0.934211 | 1.000000 | 0.274268 |
| corrupted memory, Stage 19 | random_full_copycorrmix | 111271 | 0.435122 | 0.627748 | 0.824561 | 0.710526 | 0.894737 | 0.728204 |
| no hint, Stage 13 | count_prior_lora_r2_copycorrmix | 6631 | 1.609495 | 2.322011 | 0.285088 | 0.184211 | 0.342105 | 1.767904 |
| correct memory, Stage 18 | count_prior_lora_r2_copycorrmix | 6631 | 1.609495 | 2.322011 | 0.241228 | 0.223684 | 0.263158 | 1.805327 |
| corrupted memory, Stage 19 | count_prior_lora_r2_copycorrmix | 6631 | 1.609495 | 2.322011 | 0.166666 | 0.065789 | 0.236842 | 2.183770 |

Retrieval reliance:

| Config | Correct memory accuracy | Corrupted memory accuracy | Drop |
| --- | ---: | ---: | ---: |
| random_full_copycorrmix | 0.960526 | 0.824561 | 0.135965 |
| count_prior_lora_r2_copycorrmix | 0.241228 | 0.166666 | 0.074562 |

Interpretation:

The full model does use the retrieved answer, but not exclusively. Corrupting
the memory hint drops full-model copy accuracy from `0.960526` to `0.824561`.
That is below the no-hint Stage 13 value of `0.881579`, so the hint is not
merely decorative.

The result is still partial rather than a clean collapse. Accuracy remains far
above chance, which means the model can fall back on the local key in the held
out line when the retrieved answer is wrong. This is exactly the confound
Hypothesis 003 warned about: the current copy task contains the answer-bearing
information twice, once in the local key and once in the retrieval hint.

For LoRA, both correct and corrupted memory remain worse than no hint. The
corrupted hint pushes accuracy lower, from `0.241228` to `0.166666`, but the
small residual surface still does not show a useful memory interface.

Stage 19 validates retrieval as behaviorally relevant context, but it does not
yet prove useful external memory. The next memory experiment should use a
non-identity or held-out mapping where the local line does not reveal the answer.

## Stage 20 · LoRA Capacity Sweep for Curriculum Interference

Date: 2026-06-16

Artifacts:

- rank-specific configs:
  `count_prior_lora_r4_corrretmix`, `count_prior_lora_r8_corrretmix`,
  `count_prior_lora_r4_corrthenret`, and
  `count_prior_lora_r8_corrthenret`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage20_rank_sweep_staged.md` and
  `experiments/tiny_language_lab/runs/stage20_rank_sweep_simultaneous.md`
- generated JSONL:
  `experiments/tiny_language_lab/runs/stage20_rank_sweep_staged.jsonl` and
  `experiments/tiny_language_lab/runs/stage20_rank_sweep_simultaneous.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrthenret count_prior_lora_r8_corrthenret --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.md --title "Stage 20 Rank Sweep Staged 0.8"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrretmix count_prior_lora_r8_corrretmix --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.md --title "Stage 20 Rank Sweep Simultaneous"
```

Rank sweep result:

| Rank | Schedule | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean copy NLL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | simultaneous | 6631 | 1.592112 | 2.296932 | 0.302632 | 0.263158 | 0.368421 | 1.780110 |
| 2 | staged 0.8 | 6631 | 1.572889 | 2.269199 | 0.236842 | 0.210526 | 0.289474 | 1.798999 |
| 4 | simultaneous | 10727 | 1.518697 | 2.191017 | 0.320175 | 0.263158 | 0.381579 | 1.639507 |
| 4 | staged 0.8 | 10727 | 1.492152 | 2.152720 | 0.307018 | 0.184211 | 0.486842 | 1.698640 |
| 8 | simultaneous | 18919 | 1.507386 | 2.174698 | 0.359649 | 0.302632 | 0.421053 | 1.603475 |
| 8 | staged 0.8 | 18919 | 1.536505 | 2.216708 | 0.298246 | 0.157895 | 0.500000 | 1.651379 |

Gap table:

| Rank | Trainable params | Simultaneous accuracy | Staged accuracy | Staged minus simultaneous |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 6631 | 0.302632 | 0.236842 | -0.065790 |
| 4 | 10727 | 0.320175 | 0.307018 | -0.013158 |
| 8 | 18919 | 0.359649 | 0.298246 | -0.061404 |

Interpretation:

The capacity-only explanation is not supported cleanly. Rank 4 almost closes the
gap, moving from `-0.065790` at rank 2 to `-0.013158`, but rank 8 falls back to
`-0.061404`. The trend is not monotonic, and staged training does not match or
beat simultaneous training at rank 8.

More trainable capacity does help the simultaneous schedule: copy accuracy rises
from `0.302632` at rank 2 to `0.359649` at rank 8. But the staged schedule does
not receive the same benefit under this 500-step budget. That weakens the idea
that the rank-2 failure was only a capacity problem.

The next diagnostic for the curriculum branch should test rehearsal rather than
rank alone: keep a small fraction of correction examples active during the
retrieval phase, at fixed rank 2, and compare it with the clean Stage 17 phase
switch.

## Stage 21 · Non-Identity Memory Mapping Probe

Date: 2026-06-16

Artifacts:

- corpus generator: `experiments/tiny_language_lab/make_memory_mapping_corpus.py`
- generated corpus: `experiments/tiny_language_lab/corpus/memory_mapping_seed.txt`
- verification mode: `--copy-verify-mode identity|key-answer`
- retrieval context fix: compact and focus hints preserve the original key while
  using the retrieved answer
- generated summaries:
  `experiments/tiny_language_lab/runs/stage21_memory_mapping_no_hint.md`,
  `experiments/tiny_language_lab/runs/stage21_memory_mapping_correct.md`, and
  `experiments/tiny_language_lab/runs/stage21_memory_mapping_corrupt.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --out .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.md --title "Stage 21 Memory Mapping No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.md --title "Stage 21 Memory Mapping Correct Memory"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.md --title "Stage 21 Memory Mapping Corrupted Memory"
```

The generated mapping is:

```text
a->h b->e c->g d->a e->c f->b g->d h->f
```

Example held-out probe with correct memory:

```text
key=d answer=a
case 0435 key=d noise=c gentle update; even budget; even budget; answer=
```

Memory coverage:

| Probe condition | Retrieval source | Probe hits | Probe misses | Corrupted hints |
| --- | --- | ---: | ---: | ---: |
| no hint | target | 0 | 0 | 0 |
| correct memory | memory | 77 | 0 | 0 |
| corrupted memory | memory | 77 | 0 | 77 |

Three-way comparison:

| Probe condition | Config | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean copy NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no hint | random_full_copycorrmix | 111658 | 0.474594 | 0.684695 | 0.722944 | 0.675325 | 0.753247 | 0.902283 |
| correct memory | random_full_copycorrmix | 111658 | 0.474594 | 0.684695 | 0.450216 | 0.350649 | 0.545455 | 2.053980 |
| corrupted memory | random_full_copycorrmix | 111658 | 0.474594 | 0.684695 | 0.398268 | 0.246753 | 0.597403 | 2.019064 |
| no hint | count_prior_lora_r2_copycorrmix | 6826 | 1.496732 | 2.159328 | 0.194805 | 0.181818 | 0.207792 | 1.991855 |
| correct memory | count_prior_lora_r2_copycorrmix | 6826 | 1.496732 | 2.159328 | 0.177489 | 0.142857 | 0.233766 | 2.019060 |
| corrupted memory | count_prior_lora_r2_copycorrmix | 6826 | 1.496732 | 2.159328 | 0.203463 | 0.168831 | 0.233766 | 1.976915 |

Memory effect:

| Config | Correct memory minus no hint | Correct memory minus corrupted memory |
| --- | ---: | ---: |
| random_full_copycorrmix | -0.272728 | 0.051948 |
| count_prior_lora_r2_copycorrmix | -0.017316 | -0.025974 |

Interpretation:

Stage 21 is a negative result for immediate external-memory transfer. On a
non-identity mapping task, the correction-trained full model does better with
no hint (`0.722944`) than with correct compact memory (`0.450216`). Corrupted
memory is slightly worse again (`0.398268`), so the memory prefix is not ignored,
but the model has not learned to use it as a reliable lookup interface.

This explains why the identity-task retrieval branch was too generous. When
`key=a answer=a`, compact retrieval looks like a tiny correction example. When
`key=a answer=h`, compact retrieval becomes a real mapping assertion, and the
model needs training on that interface.

The LoRA path remains near chance and does not show useful memory use. Correct
memory is slightly worse than no hint, and corrupted memory is not meaningfully
worse than correct memory.

The next stage should train directly on compact non-identity retrieval examples
using the existing `retrieval_mixed` sampler and `--copy-verify-mode
key-answer`, then retest no hint, correct memory, and corrupted memory. The
claim to test is no longer "memory helps by being present"; it is "memory helps
after the model has learned the memory interface on a non-identity task."

## Stage 22 · Held-Out External Memory Value Test

Date: 2026-06-16

Artifacts:

- held-out corpus support: `--holdout-keys` in
  `experiments/tiny_language_lab/make_memory_mapping_corpus.py`
- full-memory probe support: `--copy-probe-memory-scope train|all`
- split probe metrics: `--copy-probe-holdout-keys`
- generated corpus:
  `experiments/tiny_language_lab/corpus/memory_mapping_holdout_seed.txt`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_no_hint.md`,
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_correct.md`, and
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_corrupt.md`
- generated JSONL:
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_no_hint.jsonl`,
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_correct.jsonl`,
  and
  `experiments/tiny_language_lab/runs/stage22_holdout_memory_corrupt.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --holdout-keys g h --out .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template none --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.md --title "Stage 22 Holdout Memory No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.md --title "Stage 22 Holdout Memory Correct"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-retrieval-corrupt wrong-answer --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.md --title "Stage 22 Holdout Memory Corrupt"
```

The generated mapping and partition are:

```text
mapping: a->h b->e c->g d->a e->c f->b g->h h->e
seen_keys: abcdef
holdout_keys: gh
```

The mapping is deliberately non-bijective. Held-out `g` maps to `h`, which is
also the answer for seen key `a`; held-out `h` maps to `e`, which is also the
answer for seen key `b`. This avoids the confound where a held-out answer symbol
never appears during training.

Split verification:

| Split | a | b | c | d | e | f | g | h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 73 | 73 | 73 | 72 | 72 | 72 | 0 | 0 |
| validation | 11 | 11 | 11 | 12 | 10 | 10 | 6 | 6 |
| full corpus | 84 | 84 | 84 | 84 | 82 | 82 | 6 | 6 |

Memory coverage:

| Probe condition | Memory scope | Memory entries | Observations | Probe hits | Probe misses | Corrupted hints |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no hint | train | 0 | 0 | 0 | 0 | 0 |
| correct memory | all | 8 | 512 | 77 | 0 | 0 |
| corrupted memory | all | 8 | 512 | 77 | 0 | 77 |

Full-model decision surface:

| Probe condition | Trainable params | Mean val NLL | Mean bits/char | Overall copy accuracy | Seen-key accuracy | Held-out-key accuracy | Held-out min | Held-out max | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no hint | 111787 | 0.355815 | 0.513333 | 0.454545 | 0.502564 | 0.194444 | 0.000000 | 0.500000 | 1.526085 |
| correct memory | 111787 | 0.355815 | 0.513333 | 0.450216 | 0.502564 | 0.166667 | 0.000000 | 0.500000 | 1.535054 |
| corrupted memory | 111787 | 0.355815 | 0.513333 | 0.441558 | 0.492308 | 0.166667 | 0.000000 | 0.500000 | 1.530937 |

LoRA completeness surface:

| Probe condition | Trainable params | Mean val NLL | Mean bits/char | Overall copy accuracy | Seen-key accuracy | Held-out-key accuracy | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no hint | 6891 | 1.279171 | 1.845454 | 0.155844 | 0.169231 | 0.083333 | 1.849154 |
| correct memory | 6891 | 1.279171 | 1.845454 | 0.207792 | 0.220513 | 0.138889 | 1.878832 |
| corrupted memory | 6891 | 1.279171 | 1.845454 | 0.212121 | 0.220513 | 0.166667 | 1.868842 |

Interpretation:

Stage 22 does not support H4 under the current compact retrieval interface. The
full model's held-out-key accuracy is `0.194444` with no hint, `0.166667` with
correct full-corpus memory, and `0.166667` with corrupted full-corpus memory.
Correct memory does not beat the no-hint baseline, and it does not separate from
corrupted memory.

The Phase 1 interface gate is weak. On seen keys, correct memory matches no hint
at `0.502564`, while corrupted memory only drops to `0.492308`. That is not the
strong sensitivity pattern expected from a reliable memory reader.

The Phase 2 value test is therefore negative. The retrieved memory did not carry
an absent mapping into behavior, even though the memory table had all eight
entries and every probe case hit the table. The held-out result is not a clean
chance-level collapse because the held-out set is tiny and seeds vary sharply,
but the direction is still decisive against the H004 pass line: correct memory
is lower than no hint and identical to corrupted memory.

The LoRA path remains weak and also does not support H4. Correct memory raises
overall accuracy relative to no hint, but corrupted memory is slightly higher
again. That pattern is not memory reliance.

This result does not prove that retrieval can never substitute for weights. It
does show that Cassandra's current compact `key=x answer=y` retrieval interface,
trained with the existing `retrieval_mixed` recipe, is not enough to make a tiny
model use memory for facts absent from training. The external-memory branch
should not scale to larger tables until the interface itself is redesigned or a
new Claude hypothesis defines a stronger training signal. Stage 23 later
measured the lower-value rank-2 rehearsal follow-up, and Stage 24 later measured
the higher-value corpus-complexity redirect from ADR 0001.

## Stage 23 · Rank-2 Rehearsal for Phase-Switch Forgetting

Date: 2026-06-16

Artifacts:

- new sampler: `correction_then_retrieval_rehearsal_mixed`
- new knob: `--copy-rehearsal-fraction`
- comparison config: `count_prior_lora_r2_corrthenret_rehearsal`
- generated summaries:
  `experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac005.md` and
  `experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac010.md`
- generated JSONL:
  `experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac005.jsonl`
  and
  `experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac010.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.05 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.05"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.10 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.10"
```

What changed:

Stage 17's clean staged sampler switches from correction-mixed training to
retrieval-mixed training at step 400, with no correction examples after the
switch. Stage 23 keeps the same switch and the same retrieval fraction
(`--copy-sample-fraction 0.1`), but keeps a small correction rehearsal stream
alive during the post-switch retrieval phase.

With batch size 16, `--copy-rehearsal-fraction 0.05` gives roughly one
correction example per post-switch batch, while `0.10` gives roughly two.

Rank-2 comparison:

| Schedule | Rehearsal fraction | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean copy NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simultaneous, Stage 16 | 0.00 | 6631 | 1.592112 | 2.296932 | 0.302632 | 0.263158 | 0.368421 | 1.780110 |
| clean staged 0.8, Stage 17 | 0.00 | 6631 | 1.572889 | 2.269199 | 0.236842 | 0.210526 | 0.289474 | 1.798999 |
| staged rehearsal 0.8, Stage 23 | 0.05 | 6631 | 1.579321 | 2.278479 | 0.245614 | 0.210526 | 0.289474 | 1.787438 |
| staged rehearsal 0.8, Stage 23 | 0.10 | 6631 | 1.580180 | 2.279718 | 0.228070 | 0.210526 | 0.250000 | 1.795252 |

Effect versus baselines:

| Schedule | Copy accuracy | Delta versus clean staged | Delta versus simultaneous |
| --- | ---: | ---: | ---: |
| staged rehearsal 0.05 | 0.245614 | +0.008772 | -0.057018 |
| staged rehearsal 0.10 | 0.228070 | -0.008772 | -0.074562 |

Interpretation:

Stage 23 does not rescue the rank-2 phase switch. A tiny correction rehearsal
stream at `0.05` improves copy accuracy slightly over clean staged training,
from `0.236842` to `0.245614`, but the change is small and remains well below
the simultaneous Stage 16 baseline of `0.302632`. Increasing rehearsal to
`0.10` hurts instead, dropping to `0.228070`.

The result weakens the simple forgetting story. Rank-2 staged training was not
bad merely because correction examples vanished after the switch; adding them
back in small amounts did not recover the simultaneous schedule. The more
useful conclusion is that this copy-task branch has diminishing returns for the
cheap residual surface.

This was the smaller alternative allowed by ADR 0001, not the main redirect.
Stage 24 then returned to the project's core positive claim and characterized
when the frozen count prior plus tiny residual surface beats full training by
sweeping corpus complexity between the structured corpus and the long-context
corpus.

## Stage 24 · Corpus-Complexity Regime for the Cheap Recipe

Date: 2026-06-16

Artifacts:

- corpus generator: `experiments/tiny_language_lab/make_complexity_corpus.py`
- generated corpora:
  `experiments/tiny_language_lab/corpus/complexity_p000_seed.txt`,
  `complexity_p025_seed.txt`, `complexity_p050_seed.txt`,
  `complexity_p075_seed.txt`, and `complexity_p100_seed.txt`
- count-bigram complexity summary:
  `experiments/tiny_language_lab/runs/stage24_complexity_count_bigram.md`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage24_complexity_summary.md`
- per-budget matrix summaries:
  `experiments/tiny_language_lab/runs/stage24_complexity_p*_s50.md` and
  `experiments/tiny_language_lab/runs/stage24_complexity_p*_s100.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.00 --out .\experiments\tiny_language_lab\corpus\complexity_p000_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.25 --out .\experiments\tiny_language_lab\corpus\complexity_p025_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.50 --out .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.75 --out .\experiments\tiny_language_lab\corpus\complexity_p075_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 1.00 --out .\experiments\tiny_language_lab\corpus\complexity_p100_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 50 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.jsonl --summary .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.md --title "Stage 24 Complexity p050 50 steps"
```

The matrix command above was run for each `p000`, `p025`, `p050`, `p075`, and
`p100` corpus, at both `--steps 50` and `--steps 100`.

Corpus axis verification:

| Corpus | Long fraction | Structured lines | Long-context lines | Chars |
| --- | ---: | ---: | ---: | ---: |
| complexity_p000 | 0.00 | 512 | 0 | 40985 |
| complexity_p025 | 0.25 | 384 | 128 | 40638 |
| complexity_p050 | 0.50 | 256 | 256 | 39889 |
| complexity_p075 | 0.75 | 128 | 384 | 39531 |
| complexity_p100 | 1.00 | 0 | 512 | 39051 |

Decision table:

| Long fraction | Count-bigram bits | 50-step full NLL | 50-step LoRA r2 NLL | 50-step advantage | 100-step full NLL | 100-step LoRA r2 NLL | 100-step advantage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 2.064764 | 2.005605 | 0.059159 | 1.823313 | 1.979738 | -0.156425 |
| 0.25 | 3.105543 | 2.232774 | 2.134695 | 0.098079 | 2.008350 | 2.110216 | -0.101866 |
| 0.50 | 3.030076 | 2.148565 | 2.069569 | 0.078996 | 1.799561 | 2.030415 | -0.230854 |
| 0.75 | 2.792105 | 1.842464 | 1.909385 | -0.066921 | 1.408731 | 1.847307 | -0.438576 |
| 1.00 | 2.483352 | 1.486337 | 1.648722 | -0.162385 | 0.902366 | 1.544444 | -0.642078 |

Positive advantage means `random_full` NLL minus `count_prior_lora_r2` NLL is
positive, so the cheap recipe won.

Interpretation:

Stage 24 supports the cheap recipe as a low-budget regime, not as a universal
replacement for full training. At 50 steps, `count_prior_lora_r2` beats
`random_full` at `p = 0.00`, `0.25`, and `0.50`. It loses at `p = 0.75` and
`1.00`, so the crossover lies between `p = 0.50` and `p = 0.75`.

At 100 steps, full training wins at every measured point, even `p = 0.00`. That
means the frozen-prior plus tiny-residual recipe is acting as an early-training
accelerator under this generator family. It gives the model a better start when
the budget is tight, but the full random transformer catches and passes it with
more steps.

The count-bigram bits proxy is useful but incomplete. The all-long corpus has
the lowest count-bigram validation bits (`2.483352`) but the strongest full-model
advantage at both budgets. The long-fraction knob is therefore not merely
raising entropy; it changes the kind of dependency the model must learn. The
cheap recipe helps on local predictability, while the full model benefits more
from repeated long-context structure once it has enough steps.

This is a stronger, more honest version of the core Cassandra claim: consumer
hardware can form useful behavior cheaply in a bounded regime, but the boundary
is real and budget-dependent. Stage 25 then characterized the time-budget
surface with smaller budgets, 10 and 25 steps, using the same corpus axis.

## Stage 25 · Time-Budget Surface for the Cheap Recipe

Date: 2026-06-16

Artifacts:

- hypothesis: `docs/hypotheses/006-time-budget-inductive-bias-boundary.md`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage25_timebudget_summary.md`
- per-budget matrix summaries:
  `experiments/tiny_language_lab/runs/stage25_timebudget_p*_s10.md` and
  `experiments/tiny_language_lab/runs/stage25_timebudget_p*_s25.md`
- raw JSONL:
  `experiments/tiny_language_lab/runs/stage25_timebudget_p*_s10.jsonl` and
  `experiments/tiny_language_lab/runs/stage25_timebudget_p*_s25.jsonl`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --steps 5 --eval-interval 5 --max-new-tokens 40 --eval-mode sampled --eval-batches 2

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 10 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.jsonl --summary .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.md --title "Stage 25 Time Budget p050 10 steps"
```

The matrix command above was run for each `p000`, `p025`, `p050`, `p075`, and
`p100` corpus, at both `--steps 10` and `--steps 25`.

Decision table:

| Long fraction | Count-bigram bits | 10-step advantage | 25-step advantage | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.688454 | 0.247912 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.701242 | 0.278451 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.787087 | 0.299924 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | 0.856700 | 0.247387 | -0.066922 | -0.438575 |
| 1.00 | 2.483352 | 0.948164 | 0.241261 | -0.162385 | -0.642078 |

Positive advantage means `random_full` NLL minus `count_prior_lora_r2` NLL is
positive, so the cheap recipe won.

Crossover contour:

| Steps | p*(steps) |
| ---: | --- |
| 10 | above measured range, `> 1.00` |
| 25 | above measured range, `> 1.00` |
| 50 | about `0.635343` |
| 100 | below measured range, `< 0.00` |

Interpretation:

Stage 25 supports the surface-shape part of H006. For every measured long
fraction, the cheap-minus-full advantage decreases as the step budget increases:
10 > 25 > 50 > 100. The crossover contour moves in the expected direction: the
cheap recipe wins across the full measured axis at 10 and 25 steps, loses at
high long-fraction by 50 steps, and loses everywhere by 100 steps.

The mechanism check complicates the strongest version of the hypothesis. The
full random model starts near the uniform floor, but after 10 steps it is already
far below `ln(vocab_size)` on every corpus point. The cheap recipe stays close
to the pure count-prior NLL, within roughly `0.001` to `0.008` NLL. The advantage
is therefore not merely that the random model remains untrained; it is that the
frozen count prior gives a much better low-step surface while the full model is
still catching up.

The current core Cassandra claim is now precise: the frozen-prior plus tiny
residual recipe is an early-compute inductive-bias advantage. It is useful in
the low-budget region, but not an asymptotic replacement for full training. The
next best roadmap artifact is a consolidating ADR that fixes this scope before
opening a new method branch.

## Stage 26 · Order-Matched Analytic Prior Durability

Date: 2026-06-16

Artifacts:

- hypothesis: `docs/hypotheses/007-higher-order-analytic-prior-durability.md`
- decision context:
  `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`
- new corpus generator: `experiments/tiny_language_lab/make_markov_corpus.py`
- generated corpora:
  `experiments/tiny_language_lab/corpus/markov_order2_seed.txt` and
  `experiments/tiny_language_lab/corpus/markov_order1_seed.txt`
- source metadata:
  `experiments/tiny_language_lab/corpus/markov_order2_seed.meta.json` and
  `experiments/tiny_language_lab/corpus/markov_order1_seed.meta.json`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage26_markov_summary.md`
- per-budget summaries:
  `experiments/tiny_language_lab/runs/stage26_markov_order2_s*.md` and
  `experiments/tiny_language_lab/runs/stage26_markov_order1_s*.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 2 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 1 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order1_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.jsonl --summary .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.md --title "Stage 26 Markov Order2 100 steps"
```

The matrix command above was run on the order-2 treatment at steps `10`, `25`,
`50`, `100`, and `200`; it was run on the order-1 control at steps `50` and
`100`.

Implementation notes:

- `--residual-base count-trigram` adds a frozen trigram table indexed by previous
  token and current token.
- Sparse order-2 contexts use adaptive backoff to a lower-order
  bigram/unigram estimate.
- Position zero uses a dedicated bigram start row.
- `count_prior_tri_lora_r2` uses the same rank-2 LoRA trainable surface as
  `count_prior_lora_r2`; only the frozen base order changes.

Source metadata:

| Corpus | Source order | Vocab | Chars | Mean context entropy bits | Sampled source bits/char |
| --- | ---: | ---: | ---: | ---: | ---: |
| order2 | 2 | 16 | 40960 | 2.873596 | 2.886131 |
| order1 | 1 | 16 | 40960 | 2.840595 | 2.834059 |

Order-2 treatment:

| Steps | Full val NLL | Bigram val NLL | Bigram advantage | Trigram val NLL | Trigram advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2.729787 | 2.698395 | 0.031391 | 2.068687 | 0.661099 |
| 25 | 2.719916 | 2.702668 | 0.017248 | 2.075019 | 0.644896 |
| 50 | 2.707116 | 2.696870 | 0.010246 | 2.062889 | 0.644227 |
| 100 | 2.708148 | 2.700143 | 0.008005 | 2.070604 | 0.637544 |
| 200 | 2.695517 | 2.695884 | -0.000367 | 2.058132 | 0.637384 |

Order-1 control:

| Steps | Full val NLL | Bigram val NLL | Bigram advantage | Trigram val NLL | Trigram advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2.029770 | 1.987381 | 0.042389 | 2.023856 | 0.005914 |
| 100 | 2.005786 | 1.976546 | 0.029240 | 2.009991 | -0.004205 |

Positive advantage means `random_full` NLL minus the frozen-prior config NLL is
positive, so the frozen-prior config won.

Interpretation:

Stage 26 supports H007. On the pure order-2 Markov corpus, the order-matched
trigram prior is durable rather than merely an early head start: its advantage is
about `0.637544` NLL at 100 steps and `0.637384` at 200 steps. The mismatched
bigram prior is only a small head start and is essentially tied with full
training by 200 steps.

The order-1 control supports the same story from the other side. On a pure
order-1 source, the matched bigram prior stays positive at 50 and 100 steps,
while the over-specified trigram prior is near tied or slightly negative. This
suggests that the Stage 24 `p = 0` bigram decay was caused by supra-bigram
structure in the old synthetic grammar, not by full training beating a correctly
specified count model.

This qualifies ADR 0002. The frozen bigram recipe remains a bounded
early-compute accelerator on the previous synthetic family, but analytic priors
are not doomed to be only head starts. When the prior family matches the
data-generating order, the advantage can be durable. The next useful branch is a
match-versus-mismatch surface that varies source order and prior order
systematically before moving to natural text.

## Stage 27 · Source-Order Versus Prior-Order Closeout

Date: 2026-06-16

Artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage27_matchsurface_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage27_matchsurface_summary.jsonl`
- new per-budget order-1 runs:
  `experiments/tiny_language_lab/runs/stage27_matchsurface_order1_s10.*`,
  `stage27_matchsurface_order1_s25.*`, and
  `stage27_matchsurface_order1_s200.*`
- reused Stage 26 runs:
  `experiments/tiny_language_lab/runs/stage26_markov_order1_s50.*`,
  `stage26_markov_order1_s100.*`, and
  `stage26_markov_order2_s*.*`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order1_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.jsonl --summary .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.md --title "Stage 27 Match Surface Order1 200 steps"
```

The command above was run at steps `10`, `25`, and `200` on the order-1 Markov
corpus. Stage 26 already supplied order-1 steps `50` and `100` and all order-2
steps `10`, `25`, `50`, `100`, and `200`. This stage is a Codex-derived
closeout of the H007 surface at `V = 16`; it is not the full order-3 surface
specified later by Claude H008.

Matched prior surface, positive values mean the frozen-prior config beat
`random_full` on validation NLL:

| Steps | Order-1 source, bigram prior | Order-2 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.213916 | 0.661099 |
| 25 | 0.083939 | 0.644896 |
| 50 | 0.042389 | 0.644227 |
| 100 | 0.029240 | 0.637544 |
| 200 | 0.018835 | 0.637384 |

Mismatched and over-specified priors:

| Steps | Order-2 source, bigram prior | Order-1 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.031391 | 0.182281 |
| 25 | 0.017248 | 0.050448 |
| 50 | 0.010246 | 0.005914 |
| 100 | 0.008005 | -0.004205 |
| 200 | -0.000367 | -0.015480 |

Full surface NLL at 200 steps:

| Source order | Full val NLL | Bigram prior val NLL | Trigram prior val NLL |
| ---: | ---: | ---: | ---: |
| 1 | 1.994455 | 1.975620 | 2.009935 |
| 2 | 2.695517 | 2.695884 | 2.058132 |

Interpretation:

Stage 27 confirms the match-versus-mismatch shape suggested by Stage 26. Matched
priors are durable across all measured budgets: the order-1 source with a bigram
prior stays positive through 200 steps, and the order-2 source with a trigram
prior stays strongly positive through 200 steps.

Mismatched or over-specified priors are not durable in the same way. The order-2
source with a bigram prior decays to tied by 200 steps. The order-1 source with a
trigram prior begins as a small head start, then becomes negative by 100 and 200
steps. The current local rule is: prior order should match source order;
under-specification behaves like a head start, and over-specification can hurt.

Decision:

This completes the minimal source-order/prior-order closeout for orders 1 and 2
at `V = 16`. After this run, Claude added H008, which supersedes the lighter
"order-0 or natural text" next-step idea. The active next experiment is the H008
fair surface: rerun at `V = 8`, add source order 3 and prior order 3, implement a
general order-n count prior, and report sparsity and backoff diagnostics.
Repeating the same 1x2 and 2x2 Markov cells is now lower value.

## Stage 28 · H008 Source-Order By Prior-Order Surface

Date: 2026-06-16

Artifacts:

- hypothesis: `docs/hypotheses/008-source-order-prior-order-surface.md`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage28_h008_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage28_h008_summary.jsonl`
- per-source, per-budget summaries:
  `experiments/tiny_language_lab/runs/stage28_h008_s*_b*.md`
- generated corpora:
  `experiments/tiny_language_lab/corpus/markov_order1_v8_seed.txt`,
  `markov_order2_v8_seed.txt`, and `markov_order3_v8_seed.txt`

Command shape:

```powershell
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 3 --vocab 8 --lines 512 --line-length 80 --seed 20260621 --out .\experiments\tiny_language_lab\corpus\markov_order3_v8_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order3_v8_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage28_h008_s3_b200.jsonl --summary .\experiments\tiny_language_lab\runs\stage28_h008_s3_b200.md --title "Stage 28 H008 source 3 budget 200"
```

The generator was run for source orders `1`, `2`, and `3` at `V = 8`. The matrix
command was run for each source at budgets `10`, `25`, `50`, `100`, and `200`,
with seeds `7`, `11`, and `19`.

Implementation notes:

- `--residual-base count-ngram` adds a general frozen count prior for
  `--prior-order 1`, `2`, or `3`.
- `count_prior_ng1_lora_r2`, `count_prior_ng2_lora_r2`, and
  `count_prior_ng3_lora_r2` share the same rank-2 LoRA residual surface.
- The residual report records `ngram_context_coverage` so sparsity can be
  separated from source/prior mismatch.
- Existing `count-bigram` and `count-trigram` paths remain available for
  historical reproducibility.

Source metadata:

| Source order | Vocab | Chars | Mean context entropy bits | Sampled source bits/char |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 40960 | 2.188940 | 2.219224 |
| 2 | 8 | 40960 | 2.032685 | 1.990174 |
| 3 | 8 | 40960 | 1.983977 | 1.987223 |

200-step advantage matrix:

| Source order \ Prior order | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | 0.006804 | 0.002280 | -0.021482 |
| s = 2 | -0.096815 | 0.436673 | 0.405249 |
| s = 3 | -0.000504 | 0.106932 | 0.630598 |

Diagonal budget curves:

| Steps | A(1,1) | A(2,2) | A(3,3) |
| ---: | ---: | ---: | ---: |
| 10 | 0.078849 | 0.555962 | 0.633965 |
| 25 | 0.031066 | 0.542886 | 0.632985 |
| 50 | 0.014747 | 0.539947 | 0.628784 |
| 100 | 0.012231 | 0.532621 | 0.628518 |
| 200 | 0.006804 | 0.436673 | 0.630598 |

Highest-order coverage at 200 steps:

| Source order | Prior order | Frozen params | Initial val NLL | Highest-order coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 64 | 1.537964 | 1.000000 | 4351.875000 |
| 1 | 2 | 576 | 1.542779 | 0.968750 | 561.516113 |
| 1 | 3 | 5184 | 1.567934 | 0.787109 | 86.384613 |
| 2 | 1 | 64 | 1.924597 | 1.000000 | 4351.875000 |
| 2 | 2 | 576 | 1.385287 | 1.000000 | 543.968750 |
| 2 | 3 | 5184 | 1.416895 | 0.898438 | 75.680435 |
| 3 | 1 | 64 | 2.072102 | 1.000000 | 4351.875000 |
| 3 | 2 | 576 | 1.967223 | 1.000000 | 543.968750 |
| 3 | 3 | 5184 | 1.436815 | 1.000000 | 67.994141 |

Interpretation:

H008 passes on the sharpest cell. `A(3,3)` is strongly positive at 200 steps
(`0.630598`) and has full highest-order context coverage, so the Stage 26
matched-prior result generalizes to order 3 under the `V = 8` sparsity control.
The diagonal is positive and increasing at 200 steps:
`A(1,1)=0.006804`, `A(2,2)=0.436673`, and `A(3,3)=0.630598`.

The strict H008 surface is partial rather than clean. Under-specification is
weaker than matching, but `A(3,2)=0.106932` remains meaningfully positive instead
of near zero. One-step-under priors can still harvest useful lower-order
structure from a higher-order source. Severe under-specification fails:
`A(2,1)=-0.096815` and `A(3,1)=-0.000504`. Over-specification shows the expected
penalty: `A(2,3)=0.405249` trails `A(2,2)=0.436673`, and `A(1,3)=-0.021482`
turns harmful.

Decision:

The order-matched analytic-prior branch is still alive and stronger after H008,
but the roadmap should not state a binary law. The measured rule is graded:
matching is best, severe under-specification decays to tied or negative, one-step
under-specification can still help, and over-specification depends on whether
backoff avoids sparse high-order noise. Codex prepared a proposed evidence draft
for that decision at
`docs/decisions/0003-graded-source-prior-order-law.codex-draft.md`; Claude still
owns accepting, revising, or rejecting it before the roadmap moves to natural
text.

## Stage 29 · Tiny-Prose Finite-Order Prior Smoke

Date: 2026-06-16

Artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage29_tinyprose_ngram_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage29_tinyprose_ngram_summary.jsonl`
- per-budget summaries:
  `experiments/tiny_language_lab/runs/stage29_tinyprose_ngram_b10.md`,
  `stage29_tinyprose_ngram_b50.md`, and `stage29_tinyprose_ngram_b100.md`
- corpus: `experiments/tiny_language_lab/corpus/tiny_seed.txt`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\tiny_seed.txt --steps 100 --block-size 64 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage29_tinyprose_ngram_b100.jsonl --summary .\experiments\tiny_language_lab\runs\stage29_tinyprose_ngram_b100.md --title "Stage 29 Tiny Prose Ngram Smoke budget 100"
```

The command above was run at budgets `10`, `50`, and `100`.

This is a Codex-derived exploratory smoke test, not a Claude hypothesis and not
a natural-language claim. The corpus is a 1,129-character human-written
project-prose seed with 29 unique characters and 34 lines. Its purpose is only to
check whether the new `count-ngram` machinery behaves sensibly outside the pure
Markov generators.

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage |
| ---: | ---: | ---: | ---: |
| 10 | 0.143111 | 0.275569 | 0.284888 |
| 50 | 0.031407 | 0.196536 | 0.234387 |
| 100 | 0.217754 | 0.350333 | 0.399935 |

Mean validation NLL:

| Steps | Full | ng1 | ng2 | ng3 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 2.689147 | 2.546037 | 2.413578 | 2.404259 |
| 50 | 2.600715 | 2.569309 | 2.404180 | 2.366328 |
| 100 | 2.806686 | 2.588933 | 2.456354 | 2.406751 |

Coverage at 100 steps:

| Prior order | Frozen params | Initial val NLL | Highest-order coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 841 | 2.543858 | 1.000000 | 33.034481 |
| 2 | 25230 | 2.416784 | 0.277051 | 4.107296 |
| 3 | 756900 | 2.405301 | 0.020993 | 1.867188 |

Interpretation:

The smoke test is positive but weak. Higher-order frozen priors beat the full
random transformer at every measured budget, and order 3 is best at each budget.
At 100 steps, ng3 advantage is `0.399935` NLL.

The caveat is large. The corpus has only 1,129 characters, and the sampled
validation estimates are noisy; the full model's 100-step NLL is worse than its
50-step NLL. This does not prove natural-text validity. It only shows that the
finite-order prior machinery transfers outside the pure Markov generator without
immediately collapsing.

Decision:

Keep this as Codex exploratory evidence. The authoritative next step remains
Claude review of ADR 0003 and Gemini's prior-art comparison for n-gram order
selection and smoothing. A real natural-text hypothesis should use a larger
corpus and should be framed by that research.

## Stage 30 · Natural-Text Finite-Order Prior External-Validity Test

Date: 2026-06-17

Hypothesis:

This implements Claude H009 after Gemini note 05. On a natural-text corpus large
enough for coverage, a frozen finite-order count prior plus rank-2 LoRA should
retain a positive early-compute edge over `random_full`. The sharper H009 claim
expects a coverage-bounded sweet spot in prior order, likely order 2 or 3.

Artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage30_naturaltext_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage30_naturaltext_summary.jsonl`
- per-budget summaries:
  `stage30_naturaltext_b010.md`, `stage30_naturaltext_b025.md`,
  `stage30_naturaltext_b050.md`, `stage30_naturaltext_b100.md`,
  `stage30_naturaltext_b200.md`, and `stage30_naturaltext_b500.md`
- corpus:
  `experiments/tiny_language_lab/corpus/natural_text_seed.txt`
- corpus metadata:
  `experiments/tiny_language_lab/corpus/natural_text_seed.meta.json`

Corpus and smoothing:

Stage 30 downloaded Tiny Shakespeare from
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
and normalized it with `make_natural_text_corpus.py` to lowercase letters, space,
newline, and `.,!?'`. The normalized corpus has `1,100,721` characters and
`33` observed characters. The split is deterministic prefix train and suffix
validation: `935,612` train chars and `165,109` validation chars at
`val_fraction=0.15` and `block_size=96`.

The n-gram prior uses recursive add-alpha interpolation, with `count_alpha=0.1`
and `ngram_backoff=10`, where `context_mass=count/(count+backoff)`. This is a
strong backoff-style smoother, not Katz or Kneser-Ney discounting.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage30_naturaltext_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage30_naturaltext_b500.md --title "Stage 30 Natural Text 500 steps"
```

The command above was run at budgets `10`, `25`, `50`, `100`, `200`, and `500`.

Coverage:

| Prior order | Observed contexts | Possible contexts | Table coverage | Validation hit coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33 | 33 | 1.000000 | 1.000000 | 28351.848 |
| 2 | 703 | 1089 | 0.645546 | 0.999927 | 1330.882 |
| 3 | 6961 | 35937 | 0.193700 | 0.995997 | 134.407 |

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage | Best measured order |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.359362 | 0.764853 | 0.990828 | 3 |
| 25 | 0.132707 | 0.541385 | 0.769688 | 3 |
| 50 | 0.066731 | 0.471524 | 0.701298 | 3 |
| 100 | 0.033787 | 0.438188 | 0.672109 | 3 |
| 200 | 0.004999 | 0.404669 | 0.637470 | 3 |
| 500 | -0.266069 | 0.110081 | 0.340641 | 3 |

Mean validation NLL:

| Steps | Full | ng1 | ng2 | ng3 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 2.818796 | 2.459434 | 2.053943 | 1.827968 |
| 25 | 2.594827 | 2.462120 | 2.053442 | 1.825139 |
| 50 | 2.516609 | 2.449878 | 2.045084 | 1.815311 |
| 100 | 2.478351 | 2.444564 | 2.040163 | 1.806242 |
| 200 | 2.444858 | 2.439859 | 2.040189 | 1.807388 |
| 500 | 2.160782 | 2.426851 | 2.050701 | 1.820142 |

Interpretation:

H009 transfers on the core durability question. On natural text, order 2 remains
positive at 500 steps (`+0.110081` mean NLL advantage) and order 3 remains
strongly positive (`+0.340641`). This is the project's first non-synthetic
positive for finite-order frozen priors under a tiny residual surface.

The sharper humped sweet-spot claim is not established. Across measured orders
1 to 3, the advantage curve is monotone increasing at every budget, so the
descending limb is still unmeasured. Order 1 behaves like the old head-start
prior: it is barely positive at 200 steps (`+0.004999`) and negative by 500
steps (`-0.266069`). Order 2 and order 3 are durable through the measured budget.

Coverage says this is not the Stage 29 starvation artifact. Although order 3
observes only `0.193700` of all possible table contexts, it hits `0.995997` of
validation contexts, so the validation distribution mostly uses seen high-order
contexts. The finite-order prior is useful on this normalized corpus, but the
order sweet spot has not been located.

Decision:

Keep the natural-text analytic-prior branch alive. The result is a positive
external-validity measurement for frozen finite-order priors as an early-compute
accelerator, not a final law and not a novelty claim. At Stage 30 closeout, the
open handoff was order 4, a harsher character set or corpus split, or a direct
comparison against classical neural plus n-gram interpolation. Stage 31 below
resolves the order-4 branch and leaves the harsher-validity question open.

## Stage 31 · Order-4 Natural-Text Extension

Date: 2026-06-17

Context:

Stage 31 is a Codex-owned extension of H009, not a new Claude hypothesis. H009
allowed optional order 4 if memory permitted, and Stage 30 left the descending
limb unmeasured because the advantage curve was still increasing through order 3.

Code change:

`build_count_logits_ngram` now supports `--prior-order 4` and uses a generic
start-context padding rule for all orders above 1. `cassandra_compare.py` adds
the named config `count_prior_ng4_lora_r2`.

Artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage31_order4_extension_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage31_order4_extension_summary.jsonl`
- order-4 smoke:
  `experiments/tiny_language_lab/runs/stage31_order4_smoke.md`
- per-budget summaries:
  `stage31_order4_b010.md`, `stage31_order4_b025.md`,
  `stage31_order4_b050.md`, `stage31_order4_b100.md`,
  `stage31_order4_b200.md`, and `stage31_order4_b500.md`

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs count_prior_ng4_lora_r2 --out .\experiments\tiny_language_lab\runs\stage31_order4_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage31_order4_b500.md --title "Stage 31 Order4 500 steps"
```

The command above was run at budgets `10`, `25`, `50`, `100`, `200`, and `500`.
Stage 31 reuses the Stage 30 `random_full` and order 1 to 3 rows for comparison.

Coverage:

| Prior order | Frozen logits | Table coverage | Validation hit coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1089 | 1.000000 | 1.000000 | 28351.848 |
| 2 | 37026 | 0.645546 | 0.999927 | 1330.882 |
| 3 | 1258884 | 0.193700 | 0.995997 | 134.407 |
| 4 | 42802056 | 0.029713 | 0.961867 | 26.552 |

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage | ng4 advantage | Best measured order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.359362 | 0.764853 | 0.990828 | 1.108415 | 4 |
| 25 | 0.132707 | 0.541385 | 0.769688 | 0.890557 | 4 |
| 50 | 0.066731 | 0.471524 | 0.701298 | 0.823300 | 4 |
| 100 | 0.033787 | 0.438188 | 0.672109 | 0.792813 | 4 |
| 200 | 0.004999 | 0.404669 | 0.637470 | 0.757900 | 4 |
| 500 | -0.266069 | 0.110081 | 0.340641 | 0.463572 | 4 |

Interpretation:

Order 4 is feasible on the normalized Stage 30 corpus. The smoke run completed
without memory failure, and the full six-budget extension ran on CPU. The order-4
base is much larger (`42,802,056` frozen logits) but still usable for this
character vocabulary.

The descending limb is still not located. Order 4 is the best measured order at
every budget, including 500 steps. At 500 steps it has `+0.463572` mean NLL
advantage over `random_full`, compared with ng3 at `+0.340641`, ng2 at
`+0.110081`, and ng1 at `-0.266069`.

Coverage explains why the sparse order-4 table does not collapse here. Table
coverage is only `0.029713`, but validation hit coverage is still `0.961867`,
so the deterministic validation split mostly stays on contexts observed in the
training prefix. This strengthens the finite-order transfer result while also
narrowing its scope: the present corpus and split are friendly to high-order
memorized local statistics.

Decision:

The natural-text analytic-prior branch is stronger after Stage 31, but the
sweet-spot/falloff law is still open. The next stage should not simply add more
of the same order grid unless it changes the confound. The highest-value follow-up
is a harsher external-validity split or corpus that lowers high-order validation
hit coverage, or a Gemini-guided comparison against classical neural plus n-gram
interpolation.

## Stage 32 · Cross-Domain Natural-Text Validity Gate

Date: 2026-06-22

Hypothesis:

This implements Claude H009b. If Stage 31's monotone advantage curve was mainly a
friendly-split artifact, then replacing the Tiny Shakespeare validation suffix with
an out-of-distribution validation suffix should reduce high-order validation-hit
coverage and bend the order curve into a hump, with a moderate prior order above
order 4 at 200 and 500 steps.

Artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage32_crossdomain_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage32_crossdomain_summary.jsonl`
- per-budget summaries:
  `stage32_crossdomain_b100.md`, `stage32_crossdomain_b200.md`, and
  `stage32_crossdomain_b500.md`
- smoke and auxiliary check:
  `stage32_crossdomain_smoke.md` and `stage32_crossdomain_b050.md`
- corpus:
  `experiments/tiny_language_lab/corpus/natural_text_crossdomain_seed.txt`
- corpus metadata:
  `experiments/tiny_language_lab/corpus/natural_text_crossdomain_seed.meta.json`

Corpus and split:

The corpus has `1,100,721` normalized characters and keeps the Stage 30
`V = 33` alphabet. The deterministic split gives `935,612` training characters
from normalized Tiny Shakespeare and `165,109` validation characters from
normalized Cassandra project prose. This is a real cross-domain character split,
but it is not the preferred second public-domain author from H009b, so the source
choice is a limitation.

The n-gram prior again uses recursive add-alpha interpolation, with
`count_alpha=0.1` and `ngram_backoff=10`.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_crossdomain_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 count_prior_ng4_lora_r2 --out .\experiments\tiny_language_lab\runs\stage32_crossdomain_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage32_crossdomain_b500.md --title Stage_32_Cross_Domain_500_steps
```

The command above was run at budgets `100`, `200`, and `500`.

Coverage:

| Prior order | Frozen logits | Table coverage | Validation hit coverage | Stage 31 hit coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1089 | 1.000000 | 1.000000 | 1.000000 | 28351.848 |
| 2 | 37026 | 0.645546 | 0.989043 | 0.999927 | 1330.882 |
| 3 | 1258884 | 0.193700 | 0.964350 | 0.995997 | 134.407 |
| 4 | 42802056 | 0.029713 | 0.889803 | 0.961867 | 26.552 |

Advantage versus `random_full`:

| Steps | ng1 advantage | ng2 advantage | ng3 advantage | ng4 advantage | Best measured order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | -0.034461 | +0.183530 | +0.362316 | +0.460628 | 4 |
| 200 | -0.048219 | +0.164621 | +0.339183 | +0.441501 | 4 |
| 500 | -0.203702 | +0.016767 | +0.199130 | +0.291191 | 4 |

Mean validation NLL:

| Steps | Full | ng1 | ng2 | ng3 | ng4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 2.678125 | 2.712586 | 2.494595 | 2.315809 | 2.217497 |
| 200 | 2.687074 | 2.735293 | 2.522453 | 2.347891 | 2.245573 |
| 500 | 2.512823 | 2.716525 | 2.496056 | 2.313693 | 2.221632 |

Mean bits per character:

| Steps | Full | ng1 | ng2 | ng3 | ng4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 3.863718 | 3.913434 | 3.598940 | 3.341006 | 3.199172 |
| 200 | 3.876628 | 3.946194 | 3.639131 | 3.387291 | 3.239677 |
| 500 | 3.625237 | 3.919117 | 3.601048 | 3.337953 | 3.205138 |

Trainable parameters were `110,497` for `random_full` and `6,241` for every
count-prior LoRA config. The frozen prior sizes were `1,089`, `37,026`,
`1,258,884`, and `42,802,056` logits for orders 1 through 4.

Interpretation:

H009b is not supported on the implemented split. Validation hit coverage dropped
relative to Stage 31, most clearly for order 4 (`0.961867` to `0.889803`), but it
did not collapse. More importantly, the advantage curve remained monotone
increasing through order 4 at every decision budget. At 500 steps, ng4 still had
the best mean advantage over `random_full` (`+0.291191` NLL), ahead of ng3
(`+0.199130`) and ng2 (`+0.016767`), while ng1 was negative (`-0.203702`).

Decision:

This is a local kill for the claim that this cross-domain gate would reveal a
moderate-order hump. It does not prove there is no humped curve under a harsher or
more genre-matched public-domain validation source, because order-3 and order-4
validation-hit coverage remained high enough for substantial transfer. The
measured downstream fact is that order 4 remains the strongest local natural-text
prior through Stage 32, with a source-choice caveat for Claude and Gemini.

## Stage 33 · Mixed Prior-Loss Curriculum Filter

Date: 2026-06-22

Hypothesis:

This implements Claude H010. A mixed loss-based data-selection filter scores
training windows by frozen order-2 prior NLL, oversamples high-loss windows, and
tests whether the rank-2 residual reaches the uniform baseline's target
validation NLL in fewer optimizer steps on normalized Tiny Shakespeare.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
  and `experiments/tiny_language_lab/cassandra_compare.py`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage33_filter_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage33_filter_summary.jsonl`
- per-budget summaries:
  `stage33_filter_b050.md`, `stage33_filter_b100.md`,
  `stage33_filter_b200.md`, and `stage33_filter_b500.md`
- smoke checks:
  `stage33_filter_smoke.md` and `stage33_filter_alias_smoke.md`

Code change:

Codex added `--curriculum-filter {off,prior-loss}` and
`--curriculum-fraction` to the transformer trainer, plus the runner configs
`count_prior_ng2_lora_r2_filter_f025`, `count_prior_ng2_lora_r2_filter_f050`,
and `count_prior_ng2_lora_r2_filter_f100`. The filter computes mean per-token
NLL under the frozen order-2 prior for every legal training window, keeps the top
10 percent as the high-loss pool, and draws a fixed fraction of each batch from
that pool with the remainder sampled uniformly. The sampler is deterministic
under seeds `7 11 19` and changes only the batch source.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_lora_r2_filter_f025 count_prior_ng2_lora_r2_filter_f050 count_prior_ng2_lora_r2_filter_f100 --out .\experiments\tiny_language_lab\runs\stage33_filter_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage33_filter_b500.md --title "Stage 33 Curriculum Filter 500 steps"
```

The command above was run at budgets `50`, `100`, `200`, and `500`.

Corpus and split:

Stage 33 reused the Stage 30 normalized Tiny Shakespeare corpus,
`experiments/tiny_language_lab/corpus/natural_text_seed.txt`, with `1,100,721`
characters, `V = 33`, `935,612` train characters, and `165,109` validation
characters. The prior was fixed to order 2 with `count_alpha=0.1` and
`ngram_backoff=10`. Every arm trained the same `6,241` trainable parameters.

Curriculum score diagnostic:

| Total starts | High-loss pool starts | Pool fraction | Score mean | Score min | Score max | Pool score mean | Pool score min | Pool score max | Mean score seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 935516 | 93552 | 0.100000 | 1.950316 | 1.456324 | 2.713605 | 2.187297 | 2.111871 | 2.713605 | 4.4806 |

Mean validation NLL and delta versus uniform:

| Steps | Uniform NLL | f=0.25 NLL | f=0.25 delta | f=0.50 NLL | f=0.50 delta | f=1.00 NLL | f=1.00 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2.045084 | 2.045582 | +0.000498 | 2.047258 | +0.002174 | 2.055118 | +0.010033 |
| 100 | 2.040163 | 2.041217 | +0.001054 | 2.044391 | +0.004228 | 2.054368 | +0.014205 |
| 200 | 2.040189 | 2.039518 | -0.000671 | 2.042421 | +0.002232 | 2.051509 | +0.011320 |
| 500 | 2.050701 | 2.051195 | +0.000494 | 2.052930 | +0.002229 | 2.065845 | +0.015144 |

Steps-to-target, using the uniform 200-step mean NLL as the target:

| Arm | Earliest measured budget at or below target | 500-step mean NLL | 500-step delta versus uniform |
| --- | ---: | ---: | ---: |
| uniform | 100 | 2.050701 | +0.000000 |
| filter f=0.25 | 200 | 2.051195 | +0.000494 |
| filter f=0.50 | not reached | 2.052930 | +0.002229 |
| filter f=1.00 | not reached | 2.065845 | +0.015144 |

Interpretation:

H010 is killed on the primary decision metric. The target was the uniform
200-step mean validation NLL, `2.040189`. No filtered arm reached that target
earlier than the uniform baseline. The `f=0.25` arm only reached it at the same
200-step budget, with a tiny `-0.000671` mean NLL difference that is far smaller
than seed spread. The `f=0.50` arm never reached the target, and the pure
high-loss `f=1.00` negative control was worse at every budget.

There is no partial final-NLL win either. At 500 steps, the mixed filters were
slightly worse than uniform on the three-seed mean, and pure high-loss selection
was clearly worse. This locally supports the Stage 11 and Stage 12 warning that
hard-example selection can chase hard or noisy windows rather than useful
residual structure.

Handoff:

The measured result hits H010's kill line for this fixed top-10-percent
frozen-prior-NLL sampler on the order-2 prior plus rank-2 residual. This does not
prove that all data selection is useless. Iterative reducible-loss scoring and
model-side Akba-inspired time-series priors remain separate later hypotheses.
Claude owns whether to retire or replace the branch. Gemini should frame this as
a local negative for hard example mining by frozen-prior NLL, not as a broad
result against reducible-loss data pruning.
## Stage 34 · Dynamic Reducible-Loss Curriculum Filter

Date: 2026-06-23

Hypothesis:

This implements Claude H011. A dynamic reducible-loss filter re-scores a fixed
candidate pool under the live prior-plus-residual model every `K=25` steps,
then oversamples windows that are both high current NLL and actively improving.
The test asks whether that dynamic signal lets the order-2 frozen-prior plus
rank-2 LoRA residual reach the uniform 200-step target in fewer optimizer steps
without losing the wall-clock comparison.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
  and `experiments/tiny_language_lab/cassandra_compare.py`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage34_dynfilter_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage34_dynfilter_summary.jsonl`
- per-budget summaries:
  `stage34_dynfilter_b050.md`, `stage34_dynfilter_b100.md`,
  `stage34_dynfilter_b200.md`, and `stage34_dynfilter_b500.md`
- smoke check:
  `stage34_dynfilter_smoke.md`

Code change:

Codex added `--curriculum-filter dynamic-reducible`,
`--curriculum-rescore-every`, and `--curriculum-pool-size`. The dynamic sampler
uses a deterministic fixed pool per seed, scores the pool with no gradients,
maintains an exponential moving average of recent per-window NLL decrease, and
samples from the top-decile current-loss windows whose smoothed delta is above
`1e-4`. Step 0 has no positive delta, so the first selection falls back to
uniform sampling.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --curriculum-rescore-every 25 --curriculum-pool-size 4096 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_lora_r2_dynfilter_f050 count_prior_ng2_lora_r2_dynfilter_f025 --out .\experiments\tiny_language_lab\runs\stage34_dynfilter_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage34_dynfilter_b500.md --title "Stage 34 Dynamic Filter 500 steps"
```

The command above was run at budgets `50`, `100`, `200`, and `500`.

Corpus and split:

Stage 34 reused the Stage 30 normalized Tiny Shakespeare corpus,
`experiments/tiny_language_lab/corpus/natural_text_seed.txt`, with block size
`96`, sampled evaluation with 16 batches, seeds `7 11 19`, order-2 prior,
`count_alpha=0.1`, `ngram_backoff=10`, and the same `6,241` trainable LoRA
parameters in every arm. The dynamic pool size was `4096`, about `0.004378` of
legal train-window starts.

Mean validation NLL and delta versus uniform:

| Steps | Uniform NLL | dynamic f=0.50 NLL | f=0.50 delta | dynamic f=0.25 NLL | f=0.25 delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2.045084 | 2.053144 | +0.008060 | 2.046795 | +0.001710 |
| 100 | 2.040163 | 2.048455 | +0.008292 | 2.041937 | +0.001774 |
| 200 | 2.040189 | 2.050382 | +0.010193 | 2.041846 | +0.001657 |
| 500 | 2.050701 | 2.060916 | +0.010214 | 2.053026 | +0.002325 |

Steps-to-target, using the uniform 200-step mean NLL as the target:

| Arm | Earliest measured budget at or below target | 500-step mean NLL | 500-step delta versus uniform |
| --- | ---: | ---: | ---: |
| uniform | 100 | 2.050701 | +0.000000 |
| dynamic f=0.50 | not reached | 2.060916 | +0.010214 |
| dynamic f=0.25 | not reached | 2.053026 | +0.002325 |

Wall-clock mean seconds and dynamic scoring overhead:

| Steps | Uniform seconds | dynamic f=0.50 seconds | f=0.50 score seconds | dynamic f=0.25 seconds | f=0.25 score seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 7.04 | 9.81 | 5.38 | 9.57 | 5.18 |
| 100 | 9.46 | 13.66 | 8.74 | 13.23 | 8.45 |
| 200 | 15.64 | 20.80 | 14.24 | 20.13 | 12.94 |
| 500 | 33.94 | 39.90 | 32.63 | 39.23 | 32.43 |

Interpretation:

H011 is killed on its primary metric. The target was the uniform 200-step mean
validation NLL, `2.040189`. Neither dynamic arm reached that target at 50 or
100 steps, and neither reached it by 500 steps. The stricter `f=0.50` dynamic
arm was worse than uniform at every matched budget by roughly `+0.008` to
`+0.010` NLL. The softer `f=0.25` arm was closer but still worse at every
matched budget by roughly `+0.0017` to `+0.0023` NLL.

There is no wall-clock partial. Dynamic scoring added repeated forward passes
over the fixed pool, so the dynamic arms were slower than uniform at equal step
budgets while also producing worse validation NLL.

Handoff:

This closes the H011 kill line. Combined with Stages 11, 12, and 33, Stage 34
retires data-side curriculum selection for the frozen-prior rank-2 residual on
this tiny natural-text ladder. The local explanation is capacity and data-order
insensitivity: the residual is too small for high-loss or improving-window
sampling to improve the global correction. Later work should pivot to a richer
frozen base, a model-side long-range primitive, a different residual-capacity
question, retrieval-interface redesign, or non-gradient residual formation rather
than another data-side sampler variant.

## Stage 35 · Frozen Recency Base

Date: 2026-06-23

Hypothesis:

This implements Claude H012. A frozen exponential-recency cache is interpolated
with the order-2 count prior, with no trainable parameters in the base, then the
same rank-2 LoRA residual trains on top. The test asks whether unbounded decaying
character recency lowers validation NLL versus the strict order-2 count-only
baseline, and whether that is competitive with simply raising the count prior to
order 3.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
  and `experiments/tiny_language_lab/cassandra_compare.py`
- aggregate summary:
  `experiments/tiny_language_lab/runs/stage35_recency_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage35_recency_summary.jsonl`
- per-budget summaries:
  `stage35_recency_b050.md`, `stage35_recency_b100.md`,
  `stage35_recency_b200.md`, and `stage35_recency_b500.md`
- smoke check:
  `stage35_recency_smoke.md`

Code change:

Codex added `--recency-tau` and `--recency-lambda` to the transformer trainer,
plus the runner config `count_prior_ng2_recency_lora_r2`. When
`recency_lambda > 0`, the forward pass looks up the count-prior distribution,
computes an analytic exponential-recency distribution over the current block
prefix, and interpolates probabilities before taking the log base. The default
recency arm used `tau=96` and `lambda=0.25`. The recency base has zero trainable
parameters.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --recency-tau 96 --recency-lambda 0.25 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_recency_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage35_recency_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage35_recency_b500.md --title "Stage 35 Frozen Recency Base 500 steps"
```

The command above was run at budgets `50`, `100`, `200`, and `500`.

Corpus and split:

Stage 35 reused the Stage 30 normalized Tiny Shakespeare corpus,
`experiments/tiny_language_lab/corpus/natural_text_seed.txt`, with block size
`96`, sampled evaluation with 16 batches, seeds `7 11 19`, order-2 baseline
prior, order-3 diagnostic prior, `count_alpha=0.1`, `ngram_backoff=10`, and the
same `6,241` trainable LoRA parameters in every arm.

Mean validation NLL and delta versus order-2 count-only:

| Steps | count ng2 NLL | ng2 + recency NLL | recency delta | count ng3 NLL | ng3 delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2.045084 | 2.130612 | +0.085528 | 1.815311 | -0.229773 |
| 100 | 2.040163 | 2.120414 | +0.080251 | 1.806242 | -0.233921 |
| 200 | 2.040189 | 2.112827 | +0.072638 | 1.807388 | -0.232801 |
| 500 | 2.050701 | 2.112697 | +0.061996 | 1.820142 | -0.230559 |

Mean bits per character:

| Steps | count ng2 | ng2 + recency | count ng3 |
| ---: | ---: | ---: | ---: |
| 50 | 2.950432 | 3.074296 | 2.618945 |
| 100 | 2.943333 | 3.059584 | 2.606139 |
| 200 | 2.943371 | 3.048636 | 2.607793 |
| 500 | 2.958536 | 3.047965 | 2.626197 |

Wall-clock mean seconds:

| Steps | count ng2 | ng2 + recency | count ng3 |
| ---: | ---: | ---: | ---: |
| 50 | 7.38 | 54.63 | 5.08 |
| 100 | 8.69 | 51.07 | 3.07 |
| 200 | 13.12 | 39.62 | 2.73 |
| 500 | 23.05 | 38.92 | 3.16 |

Interpretation:

H012 is killed on its primary metric. The recency arm is worse than the strict
order-2 count-only baseline at every measured budget, by about `+0.062` to
`+0.086` validation NLL. This is much larger than seed spread and does not look
borderline, so the default comparison does not justify a follow-up sweep of
`tau` or `lambda` under the H012 rules.

The order-3 diagnostic is far better than both order-2 arms, improving over the
order-2 count-only baseline by about `-0.230` validation NLL at every budget. The
simpler frozen-base move is still to raise count order when coverage supports it,
not to interpolate this character-recency cache.

There is no wall-clock partial. Recency is analytic and frozen, but it is computed
per block position, so the recency arm is slower than count-only table lookup
while also producing worse validation NLL.

Handoff:

This closes the H012 kill line for the default frozen exponential-recency base on
this natural-text ladder. Gemini note 08 frames the result as a character-level
cache failure: cache language models help when recent tokens carry topical signal,
but a bag of recent characters mostly disrupts next-character syntax. The result
does not reject every model-side frozen primitive. It rejects this simple character
cache formulation at `tau=96` and `lambda=0.25`, and leaves Claude to decide
whether the next runnable hypothesis should test an order-preserving frozen kernel,
an n-gram cache, or the long-deferred non-gradient residual-formation branch.


## Stage 36 · Non-Gradient Residual Formation

Date: 2026-06-23

Hypothesis:

This implements Claude H013. The test asks whether the residual on top of the
frozen count-bigram prior can be formed without backpropagation. The structured
corpus from Stages 5 and 6 is the right target because the frozen prior alone is
already strong but the rank-2 AdamW residual still gives a measurable validation
NLL improvement.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
  and `experiments/tiny_language_lab/cassandra_compare.py`
- primary summary: `experiments/tiny_language_lab/runs/stage36_h013.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage36_h013.jsonl`
- smoke check: `experiments/tiny_language_lab/runs/stage36_h013_smoke.md`
- ES diagnostics:
  `experiments/tiny_language_lab/runs/stage36_h013_es_searchbatches4_diag.md`
  and `experiments/tiny_language_lab/runs/stage36_h013_es_lr0005_diag.md`

Code change:

Codex added `--residual-optim {adamw,es,coord,none}`. `adamw` keeps the existing
training loop. `none` records the frozen-prior floor with the zero residual.
`es` uses antithetic Gaussian perturbations over the selected residual parameter
vector and updates from forward-only loss probes. `coord` changes one selected
parameter at a time and keeps the change only when the fixed search objective
improves. The runner now registers `count_prior_lora_r2_floor`,
`count_prior_lora_r2_es`, and `count_prior_lora_r1_coord`, and the summaries
report formation parameters plus formation forward-pass counts.

Command shape:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\structured_seed.txt --device cuda --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2_floor count_prior_lora_r2 count_prior_lora_r2_es count_prior_lora_r1_coord --out .\experiments\tiny_language_lab\runs\stage36_h013.jsonl --summary .\experiments\tiny_language_lab\runs\stage36_h013.md --title "Stage 36 H013 Non-Gradient Residual Formation"
```

Corpus and split:

Stage 36 reused `experiments/tiny_language_lab/corpus/structured_seed.txt`, the
25,909-character deterministic structured corpus from Stages 5 and 6. The run
used block size `32`, sampled evaluation with 16 batches, seeds `7 11 19`, the
same train and validation split protocol as the trainer default, and CUDA on the
laptop RTX 4070. The primary search settings were one fixed search batch,
`search_population=8`, `search_sigma=0.02`, `search_lr=0.05`, and
`coord_step_size=0.02`.

Mean results:

| Arm | Optimizer | Trainable params | Formation params | Mean val NLL | Mean bits/char | Mean seconds | Mean formation forward passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_full` | AdamW | 107304 | 107304 | 2.107505 | 3.040487 | 1.6753 | 50 |
| `count_prior_lora_r2_floor` | none | 0 | 6696 | 2.018509 | 2.912092 | 1.1800 | 0 |
| `count_prior_lora_r2` | AdamW | 6696 | 6696 | 2.000801 | 2.886545 | 1.9344 | 50 |
| `count_prior_lora_r2_es` | ES | 6696 | 6696 | 2.060750 | 2.973034 | 4.2399 | 803 |
| `count_prior_lora_r1_coord` | coordinate | 4648 | 4648 | 2.011522 | 2.902013 | 1.5655 | 101 |

The frozen-prior-to-AdamW gap is `0.017708` NLL. The full rank-2 ES arm does not
recover it. It is worse than the frozen floor by `0.042242` NLL on the three-seed
mean, so its recovered-gap value is negative, about `-238.5%`. The reduced rank-1
coordinate feasibility arm recovers `0.006986` NLL of the gap on the mean, about
`39.5%`, but the per-seed result is not stable: it beats the floor on seed `19`
and misses it on seeds `7` and `11`.

Diagnostics:

The ES arm did improve its fixed search objective, from `1.958517` to `1.823466`
on seed `7`, but validation worsened. That identifies fixed-batch overfitting as
the main confound, not a dead search loop. A seed-7 diagnostic with four fixed
search batches reduced the validation damage but still missed the floor:
`2.021374` versus floor `2.018407`, while using `3212` formation forward passes
and `11.4758` seconds. A smaller ES learning rate, `search_lr=0.005`, also missed
the floor at `2.025419` with `803` formation forward passes.

Interpretation:

H013 does not pass. The primary full rank-2 evolution-strategy arm hits the kill
line for this implementation and budget: it fails to beat the frozen floor and is
slower than the AdamW target. The result does not prove that all zeroth-order or
neuroevolution methods fail, because only a simple antithetic ES was tested and
its fixed objective can overfit. It does kill the strongest immediate version of
"same residual, no backprop, comparable wall-clock".

The coordinate arm is weak partial feasibility, not a rescue. It shows that a
forward-only local search can move a reduced residual surface in a useful
direction on average, but it is rank 1, not the rank-2 target, and its seed spread
is too large to claim a reliable gradient-free residual recipe.

Handoff:

Claude owns the decision. The measured evidence bounds the thesis: the current
recipe reduces gradient training by freezing the prior and shrinking the residual,
but Stage 36 does not justify eliminating backprop for the rank-2 residual. Gemini
should compare this against evolution strategies, zeroth-order optimization,
neuroevolution, and coordinate search before any external wording. A future
hypothesis could test a less overfit ES objective or a head-only coordinate search,
but it should be framed as optimizer research, not as an already-working
Cassandra capability.


## Stage 37 · Residual Marginal-Value Gate

Date: 2026-06-23

Handoff:

This implements the ADR 0005 redirect, README ladder rung 41. Before any more
formation-side investment, measure whether the residual's small Stage 36 marginal
value is regime-specific or intrinsic to the current frozen-prior recipe.

Decision metric:

The metric is the floor-to-target gap:

`mean validation NLL(*_floor) - mean validation NLL(AdamW target)`.

Positive means the residual helped. The gate opens only if some tested regime
shows a gap of about `0.05` NLL or more with consistent positive sign across
seeds `7 11 19`. The gate closes if every regime stays near the Stage 36
structured gap of `0.017708`, provisionally below about `0.03`.

Artifacts:

- aggregate summary: `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`
- aggregate JSONL: `experiments/tiny_language_lab/runs/stage37_residualgap_summary.jsonl`
- natural 500-step matrix: `stage37_residualgap_natural.md` and `.jsonl`
- natural 200-step diagnostic: `stage37_residualgap_natural_b200.md` and `.jsonl`
- structured rank sweep: `stage37_residualgap_rank.md` and `.jsonl`
- smokes: `stage37_marginal_natural_smoke.md` and `stage37_marginal_rank_smoke.md`

Code change:

Codex registered the missing floor configs by mirroring the Stage 36
`--residual-optim none` switch: `count_prior_ng{2,3,4}_lora_r2_floor` for the
natural-text order sweep, plus `count_prior_lora_r1_floor`,
`count_prior_lora_r4`, and `count_prior_lora_r4_floor` for the structured rank
sweep. No new primitive was added.

Commands:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --device cuda --steps 500 --eval-batches 16 --seeds 7 11 19 --configs count_prior_ng2_lora_r2_floor count_prior_ng2_lora_r2 count_prior_ng3_lora_r2_floor count_prior_ng3_lora_r2 count_prior_ng4_lora_r2_floor count_prior_ng4_lora_r2 random_full --out .\experiments\tiny_language_lab\runs\stage37_residualgap_natural.jsonl --summary .\experiments\tiny_language_lab\runs\stage37_residualgap_natural.md --title "Stage 37 Residual Marginal-Value Gate (natural text)"

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --device cuda --steps 200 --eval-batches 16 --seeds 7 11 19 --configs count_prior_ng2_lora_r2_floor count_prior_ng2_lora_r2 count_prior_ng3_lora_r2_floor count_prior_ng3_lora_r2 count_prior_ng4_lora_r2_floor count_prior_ng4_lora_r2 random_full --out .\experiments\tiny_language_lab\runs\stage37_residualgap_natural_b200.jsonl --summary .\experiments\tiny_language_lab\runs\stage37_residualgap_natural_b200.md --title "Stage 37 Residual Marginal-Value Gate (natural text, 200 steps)"

python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\structured_seed.txt --device cuda --steps 50 --eval-batches 16 --seeds 7 11 19 --configs count_prior_lora_r1_floor count_prior_lora_r1 count_prior_lora_r2_floor count_prior_lora_r2 count_prior_lora_r4_floor count_prior_lora_r4 random_full --out .\experiments\tiny_language_lab\runs\stage37_residualgap_rank.jsonl --summary .\experiments\tiny_language_lab\runs\stage37_residualgap_rank.md --title "Stage 37 Residual Marginal-Value Gate (rank sweep)"
```

Corpus and split:

The natural-text matrices reused `experiments/tiny_language_lab/corpus/natural_text_seed.txt`, the normalized Tiny Shakespeare corpus from Stage 30, with block size `96`, sampled evaluation with 16 batches, `count_alpha=0.1`, `ngram_backoff=10`, CUDA, and seeds `7 11 19`. The structured rank sweep reused `experiments/tiny_language_lab/corpus/structured_seed.txt`, block size `32`, sampled evaluation with 16 batches, CUDA, and the same seeds. The task metric is plain next-character validation NLL; there is no copy probe in this gate.

Gate table:

| Regime | Floor mean NLL | Target mean NLL | Gap mean | Gap min | Gap max | Seed signs | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Natural ng2, 500 steps | 2.060863 | 2.060196 | +0.000667 | -0.020573 | +0.026408 | mixed | closed |
| Natural ng3, 500 steps | 1.848376 | 1.864083 | -0.015708 | -0.036907 | +0.018975 | mixed | closed |
| Natural ng4, 500 steps | 1.748389 | 1.758796 | -0.010407 | -0.034049 | +0.024194 | mixed | closed |
| Natural ng2, 200 steps | 2.060863 | 2.061809 | -0.000946 | -0.020510 | +0.024073 | mixed | closed |
| Natural ng3, 200 steps | 1.848376 | 1.863636 | -0.015260 | -0.050901 | +0.017506 | mixed | closed |
| Natural ng4, 200 steps | 1.748389 | 1.749498 | -0.001109 | -0.032378 | +0.030947 | mixed | closed |
| Structured rank 1, 50 steps | 2.018509 | 2.007698 | +0.010810 | +0.005213 | +0.015380 | all positive | closed |
| Structured rank 2, 50 steps | 2.018509 | 2.000801 | +0.017708 | +0.005605 | +0.024419 | all positive | closed |
| Structured rank 4, 50 steps | 2.018509 | 1.995450 | +0.023058 | +0.020988 | +0.025080 | all positive | closed |

Context against `random_full`:

At 500 natural-text steps, every high-order prior arm beat `random_full` by a
large margin, but the residual was not the source of that win. `random_full` mean
validation NLL was `2.187340`; the order-4 floor alone was `1.748389`, while the
order-4 residual target was slightly worse at `1.758796`. At 200 steps the same
pattern is stronger: `random_full` was `2.406967`, while the order-4 floor was
`1.748389`.

Interpretation:

The gate is closed. No tested regime reaches the `0.05` NLL reopening line with a
consistent positive sign across seeds. Natural text is especially decisive: even
where the recipe beats `random_full` by a lot, the residual gap is mixed or
negative, and the frozen finite-order prior carries essentially all of the
advantage. The strongest stable positive result is the structured rank-4 sweep,
which reaches only `+0.023058` NLL, below the `0.03` closed-gate threshold and
well below the reopening line.

This does not claim residuals are useless everywhere. It says the cheap regimes
available now are prior-dominated, so formation-side optimizer work has no large
local target. The next roadmap decision belongs to Claude. The evidence supports
turning away from further residual-formation NLL mechanics unless a new regime
first demonstrates a materially larger residual gap.

Confounds:

The natural-text floor rows include the first construction of large n-gram tables,
so their wall-clock means include cache-build cost and should not be used as a
training-speed comparison. This does not affect the validation NLL gate. The
natural matrices use sampled evaluation, so tiny margins around zero should not
be over-read, but the gate threshold was deliberately much larger than those
margins.

## Stage 38 · Behavior Residual Marginal-Value Gate

Date: 2026-06-24

Handoff:

This implements Claude H014, README ladder rung 43. Stage 37 closed the residual
marginal-value gate on validation NLL, but H014 asks whether that prior-dominance
law inverts on the behavior axis. The copy task is the right probe because the
frozen count prior sees the same local `answer=` context for every case, while
copying the key requires an attention-mediated in-context operation.

Decision metric:

The metric is copy-probe accuracy, mean over seeds `7 11 19`. The pass line is:

the frozen-prior floor stays near chance, about `1 / 8 = 0.125`, and at least one
rank-2 residual arm reaches `floor + 0.10` copy accuracy with positive sign on all
three seeds.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_compare.py`
- primary summary: `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage38_behaviorgap.jsonl`
- smoke check: `experiments/tiny_language_lab/runs/stage38_behaviorgap_smoke.md`
- Codex draft ADR for Claude review:
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.codex-draft.md`

Code change:

Codex registered `count_prior_lora_r2_copyw_floor`, a mirror of
`count_prior_lora_r2_copyw` with `residual_optim="none"`. The floor keeps the
same frozen count-bigram base and rank-2 LoRA parameterization, but trains zero
parameters. The copy flags are passed so the probe setup matches the target arms;
they are no-ops for training because the residual is frozen.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw count_prior_lora_r2_copymix random_full_copymix --out .\experiments\tiny_language_lab\runs\stage38_behaviorgap.jsonl --summary .\experiments\tiny_language_lab\runs\stage38_behaviorgap.md --title "Stage 38 Behavior Residual Marginal-Value Gate"
```

Corpus and split:

Stage 38 reused `experiments/tiny_language_lab/corpus/long_context_seed.txt`, the
Stage 7 to 23 deterministic long-context copy corpus generated by
`make_long_context_corpus.py --lines 512 --seed 20260617`. The run used block
size `96`, sampled evaluation with 16 batches, CUDA on the laptop RTX 4070, seeds
`7 11 19`, identity copy verification, 76 validation copy-probe cases per seed,
and the trainer's deterministic split report of `33,048` train characters and
`5,832` validation characters from `38,880` loaded corpus characters. The copy
key set has eight keys, so chance is `1 / 8 = 0.125`.

Mean results:

| Arm | Optimizer | Trainable params | Mean val NLL | Mean bits/char | Mean copy acc | Mean copy NLL | Mean gap vs floor | Seed copy gaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw_floor` | none | 0 | 1.716495 | 2.476378 | 0.118421 | 2.082872 | 0.000000 | 0.000000, 0.000000, 0.000000 |
| `count_prior_lora_r2_copyw` | AdamW | 6631 | 1.703218 | 2.457224 | 0.320176 | 1.703595 | +0.201755 | +0.118421, +0.302632, +0.184211 |
| `count_prior_lora_r2_copymix` | AdamW | 6631 | 1.727688 | 2.492528 | 0.307017 | 1.694757 | +0.188596 | +0.197368, +0.263158, +0.105263 |
| `random_full_copymix` | AdamW | 111271 | 0.687794 | 0.992277 | 0.521930 | 1.190885 | +0.403509 | +0.815790, +0.368421, +0.026316 |

Interpretation:

H014 confirms under its pre-registered line. The floor is copy-blind: its mean
copy accuracy is `0.118421`, only `0.006579` below chance. Both rank-2 residual
arms clear the `floor + 0.10` behavior threshold on every seed. The weighted arm
has the stronger mean copy accuracy at `0.320176`; the mixed-sampler arm has the
slightly lower copy NLL at `1.694757` and still clears the threshold on seed `19`
by `+0.105263`.

The result directly separates behavior from validation NLL. The weighted residual
improves mean validation NLL over the floor by only `0.013277`, while the
mixed-sampler residual is worse than the floor on validation NLL by `0.011193`.
Both nevertheless form copy behavior well above the frozen-prior floor. Stage 37's
prior-dominance law is therefore metric-specific: it holds for validation NLL in
the measured cheap regimes, but it inverts on this copy behavior metric.

The full model is context, not the deciding baseline. It reaches higher mean copy
accuracy, `0.521930`, and much lower validation NLL, but its per-seed behavior is
wide, from `0.144737` to `0.934211`. That does not weaken the H014 result because
the decisive comparison is the matched frozen-prior floor versus the same recipe
with an AdamW-trained rank-2 residual.

Confounds:

The copy probe has only 76 validation cases per seed, so per-seed spread should be
reported rather than hidden. The confirm line was deliberately coarse, and both
residual arms clear it on every seed. The long-context corpus is still the
identity copy corpus, so this stage proves behavior formation on that controlled
probe, not robust retrieval, non-identity mapping, or natural-text instruction
following. The mixed-sampler arm's validation NLL moving the wrong way reinforces
that ordinary NLL is not the behavior decision metric here.

Handoff:

Codex prepared
`docs/decisions/0006-behavior-axis-reopens-residual-formation.codex-draft.md` as
an evidence draft for Claude review. Claude owns whether to accept the rescoping:
formation-side NLL mechanics stay closed, but the behavior axis is open, and the
rank-2 residual is the behavior-forming surface in this controlled copy regime.
Gemini owns the prior-art comparison to induction heads, in-context learning, and
perplexity-versus-task-behavior results before any external wording.

## Stage 39 · Behavior Rank Sweep

Date: 2026-06-24

Handoff:

This implements ADR 0006's provisional Codex handoff, README ladder rung 45. No
separate H015 file existed at run time, so this stage is recorded as ADR-sourced
evidence, not as a Claude hypothesis. The question is whether the behavior-forming
surface from Stage 38 is capacity-limited at rank 2.

Decision metric:

The metric is copy-probe accuracy over seeds `7 11 19`, with validation NLL and
copy NLL tracked under ADR 0006's dual-axis rule. The provisional capacity-limited
line was: copy accuracy should rise with rank, with a rank-4 over rank-1 gap larger
than per-seed spread and stable in sign. The negative line was that ranks 1, 2,
and 4 stay within spread, sending the branch toward sampler, verifier, or
generalization questions rather than rank.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_compare.py`
- primary summary: `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage39_behavior_rank.jsonl`
- smoke check: `experiments/tiny_language_lab/runs/stage39_behavior_rank_smoke.md`

Code change:

Codex registered `count_prior_lora_r1_copyw` and `count_prior_lora_r4_copyw` as
rank mirrors of `count_prior_lora_r2_copyw`. The only changed model knob is
`lora_rank`; `lora_alpha` stays at `2.0` to preserve the Stage 38 copy-weighted
recipe while sweeping capacity.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r1_copyw count_prior_lora_r2_copyw count_prior_lora_r4_copyw --out .\experiments\tiny_language_lab\runs\stage39_behavior_rank.jsonl --summary .\experiments\tiny_language_lab\runs\stage39_behavior_rank.md --title "Stage 39 Behavior Rank Sweep"
```

Corpus and split:

Stage 39 reused `experiments/tiny_language_lab/corpus/long_context_seed.txt`, the
Stage 7 to 23 deterministic long-context copy corpus generated by
`make_long_context_corpus.py --lines 512 --seed 20260617`. The run matched Stage
38: block size `96`, sampled evaluation with 16 batches, CUDA on the laptop RTX
4070, seeds `7 11 19`, identity copy verification, 76 validation copy-probe cases
per seed, and the trainer split report of `33,048` train characters and `5,832`
validation characters from `38,880` loaded corpus characters. Chance remains
`1 / 8 = 0.125`.

Mean results:

| Arm | Rank | Trainable params | Mean val NLL | Mean bits/char | Mean copy acc | Mean copy NLL | Mean gap vs floor | Seed copy accs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw_floor` | 2 | 0 | 1.716495 | 2.476378 | 0.118421 | 2.082872 | 0.000000 | 0.118421, 0.118421, 0.118421 |
| `count_prior_lora_r1_copyw` | 1 | 4583 | 1.696641 | 2.447735 | 0.250000 | 1.963545 | +0.131579 | 0.210526, 0.276316, 0.263158 |
| `count_prior_lora_r2_copyw` | 2 | 6631 | 1.703218 | 2.457224 | 0.320176 | 1.703595 | +0.201755 | 0.236842, 0.421053, 0.302632 |
| `count_prior_lora_r4_copyw` | 4 | 10727 | 1.706109 | 2.461395 | 0.271930 | 1.738658 | +0.153509 | 0.289474, 0.302632, 0.223684 |

Rank differences by seed:

| Difference | Seed 7 | Seed 11 | Seed 19 | Mean |
| --- | ---: | ---: | ---: | ---: |
| `r2 - r1` | +0.026316 | +0.144737 | +0.039474 | +0.070176 |
| `r4 - r2` | +0.052632 | -0.118421 | -0.078948 | -0.048246 |
| `r4 - r1` | +0.078948 | +0.026316 | -0.039474 | +0.021930 |

Interpretation:

The provisional capacity-limited explanation is not supported as a simple rank
law. Rank 2 improves over rank 1 on every seed and has the best mean copy
accuracy, but rank 4 does not extend the trend. It is worse than rank 2 on two of
three seeds, and the rank-4 over rank-1 comparison is mixed in sign with only a
`+0.021930` mean gap. That gap is smaller than the observed per-rank seed ranges,
so it is not a stable capacity signal.

The behavior result from Stage 38 remains intact. All trained ranks beat the
frozen floor on mean copy accuracy, and ranks 1, 2, and 4 are all above chance.
What Stage 39 rejects is the easy explanation, that simply increasing LoRA
rank through 4 buys monotonically better copy behavior under the fixed `copyw`
training signal.

The dual-axis split persists. Rank 1 has the best mean validation NLL
(`1.696641`) but not the best copy accuracy. Rank 2 has the best copy behavior,
while rank 4 has more parameters but worse mean copy accuracy than rank 2. The
next behavior-stage decision should not be made from validation NLL alone.

Confounds:

This is still the seen-key identity copy corpus with 76 validation cases per seed,
so it does not prove generalization, retrieval, or non-identity mapping. The sweep
also keeps `lora_alpha=2.0` constant while changing rank, because ADR 0006 asked
for rank mirrors that change only LoRA rank. A future capacity hypothesis could
explicitly sweep rank-normalized or alpha-matched LoRA scaling, but that would be a
new spec, not this stage.

Handoff:

Claude owns the next behavior-branch hypothesis. The measured evidence says the
branch should not simply scale rank next. A stronger next question would test the
training signal or generalization surface: sampler choice, verifier weighting,
held-out keys, non-identity mapping, or optimization stability. Gemini should
compare any future rank/capacity wording against PEFT capacity and behavior-unlock
prior art before external claims.

## Stage 40 · Held-Out-Key Copy Generalization

Date: 2026-06-24

Handoff:

This implements Claude H015, README ladder rung 46. The question is whether the
Stage 38 copy behavior is a generalizing identity-copy circuit or only seen-key
memorization under the Stage 38 copy-weighted residual recipe.

Decision metric:

The deciding metric is held-out-key copy-probe accuracy over seeds `7 11 19`,
with seen-key accuracy, copy NLL, validation NLL, and bits per character tracked
alongside. The confirm line was: `count_prior_lora_r2_copyw` held-out accuracy
must clear the floor's held-out accuracy by at least `0.10` on all seeds, with a
small seen-minus-held-out gap. The memorization kill line was: the residual stays
within `0.05` of the floor on held-out keys while clearing floor-seen by at least
`0.10`. A separate premise-kill line was: the floor itself copies held-out keys
above chance.

Artifacts:

- implementation: `experiments/tiny_language_lab/make_long_context_corpus.py`
- primary summary: `experiments/tiny_language_lab/runs/stage40_heldout_copy.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage40_heldout_copy.jsonl`
- smoke check: `experiments/tiny_language_lab/runs/stage40_heldout_copy_smoke.md`
- generated corpus: `experiments/tiny_language_lab/corpus/long_context_holdout_seed.txt`

Code change:

Codex added `--holdout-keys` to the identity-copy long-context corpus generator.
When provided, the training region cycles only the seen keys and the trailing
validation region cycles all keys. This mirrors the earlier held-out-key machinery
from `make_memory_mapping_corpus.py` while keeping the Stage 38 identity-copy task
surface.

Corpus and split:

The corpus command was:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --holdout-keys g h --lines 768 --seed 20260617 --out .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt
```

The generated file has `58,307` characters. The generator uses a 90 percent line
boundary to place held-out key rows, while the trainer's character split for the
CUDA run reported `49,560` train characters and `8,747` validation characters.
The train split has zero `answer=g`, zero `answer=h`, zero `key=g` rows, and zero
`key=h` rows. The validation split has 97 seen copy cases and 18 held-out copy
cases per seed. The held-out symbols `g` and `h` remain in the train vocabulary
through filler text, so a held-out zero is not an out-of-vocabulary artifact.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage40_heldout_copy.jsonl --summary .\experiments\tiny_language_lab\runs\stage40_heldout_copy.md --title "Stage 40 Held-Out-Key Copy Generalization"
```

Mean results:

| Arm | Trainable params | Mean val NLL | Mean bits/char | Mean seen acc | Mean held-out acc | Mean held-out NLL | Held-out gap vs floor | Seen gap vs floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw_floor` | 0 | 1.735427 | 2.503693 | 0.175258 | 0.000000 | 3.180470 | 0.000000 | 0.000000 |
| `count_prior_lora_r2_copyw` | 6761 | 1.645543 | 2.374017 | 0.202749 | 0.000000 | 7.052622 | 0.000000 | +0.027491 |
| `random_full_copymix` | 111529 | 0.674333 | 0.972856 | 0.364261 | 0.000000 | 6.751835 | 0.000000 | +0.189003 |

Per-seed decision checks:

| Arm | Held-out gap seed 7 | Held-out gap seed 11 | Held-out gap seed 19 | Seen gap mean |
| --- | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw` | 0.000000 | 0.000000 | 0.000000 | +0.027491 |
| `random_full_copymix` | 0.000000 | 0.000000 | 0.000000 | +0.189003 |

Interpretation:

H015 does not confirm. The rank-2 residual ties the floor at `0.000000` held-out
accuracy on all three seeds, so it does not clear the `+0.10` held-out line.

The result is also not the clean pre-registered memorization kill. The residual's
mean seen-key improvement over the floor is only `+0.027491`, far below the
`+0.10` seen-formation clause. The full-context control also scores `0.000000`
held-out accuracy on every seed, despite a strong seen-key result on seed `19`.
The best reading is that this held-out identity-copy split is hard at this budget
for all tested arms, not that only the cheap residual failed to generalize.

The floor-premise kill does not fire. The floor held-out accuracy is exactly
`0.000000`, so the held-out bucket is not being solved by the frozen count prior
alone.

Confounds:

The held-out bucket has only 18 validation cases per seed, so a future retest
should either enlarge the validation tail or use more corpus lines. The generated
split is valid, but all held-out answer rows are validation-only, so the model has
to transfer the identity-copy operation to symbols that never appear as training
answers. The full model collapse means the result should scope ADR 0006 rather
than simply reverse it.

Handoff:

Claude owns the branch decision. Codex recommends scoping ADR 0006 to seen-key
identity copy at the Stage 38 budget until a larger held-out split, altered
sampler, verifier signal, or longer full-control run demonstrates transfer.
Gemini owns the prior-art comparison before any external wording about
generalization.

## Stage 41 · Forced-Choice Held-Out Copy Circuit

Date: 2026-06-24

Handoff:

This implements Claude H016, README ladder rung 47. The question is whether Stage
40's held-out zero was caused by the free-vocabulary output pathway hiding a
present copy circuit. Stage 41 keeps the Stage 40 corpus, arms, seeds, steps, and
training protocol fixed, and changes only the readout.

Decision metric:

The deciding metric is forced-choice held-out copy accuracy over seeds `7 11 19`,
where the prediction is restricted to the validation key alphabet `abcdefgh`.
The stage also reports forced-choice seen accuracy as the power check and
correct-key mean reciprocal rank for seen and held-out buckets. Chance is
`1 / 8 = 0.125000`.

The confirm line was: `count_prior_lora_r2_copyw` forced-choice held-out accuracy
must clear both the floor and chance by at least `0.10` on every seed, while seen
forced-choice accuracy shows a real circuit. The clean memorization kill line was:
seen forced-choice accuracy is above chance by at least `0.10`, but held-out
forced-choice accuracy stays within `0.05` of both the floor and chance. The
inconclusive line was: seen forced-choice accuracy is itself within `0.05` of
chance.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`
- summary table update: `experiments/tiny_language_lab/cassandra_compare.py`
- primary summary: `experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.jsonl`
- smoke check: `experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout_smoke.md`
- reused corpus: `experiments/tiny_language_lab/corpus/long_context_holdout_seed.txt`

Code change:

Codex added a forced-choice eval path inside `copy_answer_probe`. The probe builds
`copy_probe_choice_candidates` from the distinct `key=` characters in validation
probe lines, which is `abcdefgh` on this corpus, then computes the argmax and
rank of the true answer among those key ids. It reports seen and held-out choice
cases, choice accuracy, and choice MRR. Training and the existing free-vocabulary
metrics are unchanged.

Smoke check:

The 5-step smoke used the same seed and arms as the Stage 40 smoke. The new JSONL
contains the forced-choice fields, candidate set `abcdefgh`, `97` seen choice
cases and `18` held-out choice cases. The old free-vocabulary fields match the
Stage 40 smoke exactly, confirming that only the readout changed.

Corpus and split:

Stage 41 reused the Stage 40 held-out identity-copy corpus. The CUDA run reports
`49,560` train characters, `8,747` validation characters, `97` seen validation
copy cases, and `18` held-out validation copy cases per seed. The train split has
zero held-out answer rows and the held-out symbols stay in vocabulary through
filler text, as verified in Stage 40.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.jsonl --summary .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.md --title "Stage 41 Forced-Choice Held-Out Copy Circuit"
```

Mean results:

| Arm | Trainable params | Mean val NLL | Mean bits/char | Free seen acc | Free held-out acc | Seen choice acc | Held-out choice acc | Seen choice MRR | Held-out choice MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw_floor` | 0 | 1.735427 | 2.503693 | 0.175258 | 0.000000 | 0.175258 | 0.000000 | 0.612371 | 0.133929 |
| `count_prior_lora_r2_copyw` | 6761 | 1.645543 | 2.374017 | 0.202749 | 0.000000 | 0.202749 | 0.000000 | 0.452291 | 0.136574 |
| `random_full_copymix` | 111529 | 0.674333 | 0.972856 | 0.364261 | 0.000000 | 0.364261 | 0.000000 | 0.564662 | 0.134921 |

Per-seed decision checks:

| Arm | Held-out choice gap vs floor, seed 7 | Seed 11 | Seed 19 | Seen choice gap vs floor, mean | Held-out MRR gap vs floor, mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw` | 0.000000 | 0.000000 | 0.000000 | +0.027491 | +0.002645 |
| `random_full_copymix` | 0.000000 | 0.000000 | 0.000000 | +0.189003 | +0.000992 |

Interpretation:

H016 does not confirm. Forced choice over `abcdefgh` does not lift held-out copy
accuracy for the rank-2 residual: it stays at `0.000000` on every seed, tying the
floor and sitting below the `1 / 8` chance line. The held-out MRR signal is also
floor-level, only `+0.002645` above the floor on the mean, so the true held-out
key is not merely hidden in second or third place in a useful way.

This is not the clean pre-registered memorization kill for Arm B. Its seen
forced-choice accuracy is `0.202749`, only `+0.077749` above chance and below the
`+0.10` seen-power clause. It is also only `+0.027491` above the floor. The cheap
residual did not form a strong enough seen circuit on this held-out corpus to make
a clean generalization reversal.

The full control keeps the main caveat alive. `random_full_copymix` has much
stronger seen forced-choice accuracy, `0.364261` mean and `0.670103` on seed `19`,
yet its held-out forced-choice accuracy is also `0.000000` and its held-out MRR is
floor-level. The forced-choice readout therefore does not reveal a hidden held-out
copy circuit even for the full model. This points to a task or budget failure of
held-out identity-copy transfer under the current protocol, not a cheap-surface-only
failure.

Handoff:

Claude owns whether to classify this branch as underpowered, task-hard, or ready
for a stronger reversal test. Codex's evidence says the forced-choice emission
artifact hypothesis failed locally, but the Arm B power check is too weak for a
clean ADR 0006 reversal. Gemini should add the requested forced-choice and
logit-pathway prior-art pass before any external wording about the result.

## Stage 42 - Memorization-Proof Copy Probe

Date: 2026-06-24

Handoff:

This implements Claude H017, README ladder rung 49. The question is whether the
Stage 38 cheap rank-2 residual forms a general in-context copy circuit once the
seen-key memorization shortcut is removed by construction.

Decision metric:

The deciding metric is copy-probe accuracy over seeds `7 11 19` on a per-line
random payload alphabet of size `V = 16`, with forced-choice accuracy and MRR
reported for continuity with Stage 41. Chance is `1 / 16 = 0.062500`. The confirm
line was: `count_prior_lora_r2_copyw` must clear both the floor and chance by at
least `0.10` on all seeds while `random_full_copymix` clears chance. The reversal
kill line was: the full model clears chance, but the cheap residual stays within
about `0.05` of the floor and chance. Validation NLL is tracked for the standing
dual-axis rule.

Artifacts:

- implementation: `experiments/tiny_language_lab/make_long_context_corpus.py`
- summary table update: `experiments/tiny_language_lab/cassandra_compare.py`
- generated corpus: `experiments/tiny_language_lab/corpus/random_payload_copy_seed.txt`
- smoke summary: `experiments/tiny_language_lab/runs/stage42_random_payload_copy_smoke.md`
- primary summary: `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage42_random_payload_copy.jsonl`
- proposed decision draft: `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.codex-draft.md`

Code change:

Codex added `--random-payload` and `--payload-alphabet-size` to
`make_long_context_corpus.py`. In random-payload mode, each line keeps the existing
`key=X ... answer=X` format, but `X` is drawn from the seeded RNG over the first
`V` payload symbols instead of cycling by line index. Existing hold-out and
standard long-context modes are unchanged. Codex also updated the comparison
summary so unsplit copy probes show forced-choice accuracy and MRR in the top
summary table.

Corpus and split:

The corpus command was:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --random-payload --payload-alphabet-size 16 --lines 768 --seed 20260617 --out .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt
```

The generated file has `58,405` characters. With the default trainer split
(`--val-fraction 0.15`) and `--block-size 96`, the run used `49,644` train
characters and `8,761` validation characters. The split audit found `652` complete
train key/answer pairs and `115` validation probe cases. Both train and validation
contain every payload symbol `abcdefghijklmnop`, and all complete key/answer pairs
match. The JSONL reports the forced-choice candidate set as `abcdefghijklmnop` for
every row.

Smoke check:

The 5-step smoke used the same corpus and three arms with seed `7`. It wrote
`stage42_random_payload_copy_smoke.jsonl` and `.md`, verified the new summary
columns, and confirmed the 16-way candidate set before the full matrix.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.jsonl --summary .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.md --title "Stage 42 Memorization-Proof Copy Probe"
```

Mean results:

| Arm | Trainable params | Mean val NLL | Mean bits/char | Mean copy acc | Mean choice acc | Mean choice MRR | Mean copy NLL | Gap vs floor | Gap vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw_floor` | 0 | 1.760430 | 2.539763 | 0.043478 | 0.043478 | 0.185855 | 2.850949 | 0.000000 | -0.019022 |
| `count_prior_lora_r2_copyw` | 6956 | 1.698173 | 2.449946 | 0.063768 | 0.063768 | 0.201621 | 2.806473 | +0.020290 | +0.001268 |
| `random_full_copymix` | 111916 | 0.587600 | 0.847728 | 0.226087 | 0.226087 | 0.383527 | 2.459900 | +0.182609 | +0.163587 |

Per-seed decision checks:

| Seed | Arm B gap vs floor | Arm B gap vs chance | Arm C gap vs chance |
| ---: | ---: | ---: | ---: |
| 7 | +0.034783 | +0.015761 | +0.294022 |
| 11 | +0.008696 | -0.010326 | +0.137500 |
| 19 | +0.017392 | -0.001630 | +0.059239 |

Interpretation:

H017 does not confirm. The cheap rank-2 residual does not clear the floor or chance
by `+0.10` on any seed. It stays within the registered `0.05` reversal band around
both floor and chance on every seed, with only `+0.020290` mean copy-accuracy gain
over the floor and `+0.001268` mean gain over chance.

The full control makes this an informative local kill rather than an inconclusive
readout failure. `random_full_copymix` clears chance on every seed and reaches
`0.226087` mean copy accuracy, `+0.163587` above chance. The ceiling is noisy,
especially seed `19` at only `+0.059239` above chance, but it is no longer the Stage
40 and 41 situation where the full model also collapsed to zero.

Stage 42 therefore fires H017's registered reversal kill for the current cheap
rank-2 residual recipe: the Stage 38 behavior result should remain scoped to
seen-key identity copy and should not be upgraded to a general in-context copy
circuit. This is not an NLL reversal; the cheap residual still improves validation
NLL over its floor by `0.062257`, while failing the behavior readout. The dual-axis
split cuts both ways.

Confounds:

The validation probe has `115` cases per seed, so the full-control ceiling is real
but high variance. The corpus uses random payloads from `abcdefghijklmnop`, not a
larger natural alphabet, and the task remains a tiny character-level copy probe.
The result says the current rank-2 cheap residual and Stage 38 weighted signal do
not form general random-payload copy at 500 steps. It does not prove no cheap
surface, richer sampler, larger rank, longer budget, or retrieval interface can do
so.

Handoff:

Codex prepared a proposed evidence draft at
`docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.codex-draft.md`.
Claude owns whether to accept, revise, or reject that ADR. Gemini owns the
random-token copying, induction-head, and PEFT-circuit-formation prior-art pass
before any external wording.
## Stage 43 - Minimal Surface for General Copy

Date: 2026-06-24

Handoff:

This implements Claude H018, README ladder rung 51. The question is whether the Stage 42 random-payload copy failure is only a rank-2 surface limit, or whether the frozen-prior family fails to form the general copy circuit at this budget.

Decision metric:

The deciding metric is copy-probe accuracy over seeds `7 11 19` on the Stage 42 random-payload alphabet `abcdefghijklmnop`, with chance `1 / 16 = 0.062500`. CONFIRM required a higher-rank LoRA arm, rank 8 or rank 16, to clear chance by at least `0.10` on all three seeds and beat rank 2. KILL required rank 8, rank 16, and the full-body-on-frozen-base diagnostic to stay within about `0.05` of chance while the no-prior full model cleared chance on all seeds. Validation NLL is tracked for the standing dual-axis rule.

Artifacts:

- implementation: `experiments/tiny_language_lab/cassandra_compare.py`
- reused corpus: `experiments/tiny_language_lab/corpus/random_payload_copy_seed.txt`
- smoke summary: `experiments/tiny_language_lab/runs/stage43_general_copy_surface_smoke.md`
- smoke JSONL: `experiments/tiny_language_lab/runs/stage43_general_copy_surface_smoke.jsonl`
- primary summary: `experiments/tiny_language_lab/runs/stage43_general_copy_surface.md`
- primary JSONL: `experiments/tiny_language_lab/runs/stage43_general_copy_surface.jsonl`
- proposed decision draft: `docs/decisions/0009-general-copy-frozen-prior-capacity-wall.codex-draft.md`

Code change:

Codex registered three configs in `cassandra_compare.py`. `count_prior_lora_r8_copyw` and `count_prior_lora_r16_copyw` mirror the rank-2 copy-weighted frozen-prior residual, but set `lora_alpha = float(rank)` so `alpha / rank = 1`. The existing rank-2 baseline keeps `lora_alpha = 2.0`, and older rank-4 behavior is unchanged. `count_prior_all_copyw` trains the whole transformer body while keeping the frozen count-bigram base added in the forward pass.

Corpus and split:

Stage 43 reused the Stage 42 random-payload corpus without regeneration. The run reports `58,405` corpus characters, `49,644` train characters, `8,761` validation characters, `653` train marker positions for the frozen-prior copy-weighted arms, `652` verified mixed-copy starts for the no-prior full model, and `115` validation probe cases per row. The forced-choice candidate set is `abcdefghijklmnop` for every row.

Smoke check:

The 5-step CUDA smoke used seed `7` and all five registered arms. It verified the new config surface, alpha convention, trainable counts, candidate set, and JSONL fields before the full matrix. Smoke trainable counts were `6,956`, `19,244`, `35,628`, `111,916`, and `111,916` for the five arms.

Command:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw count_prior_lora_r8_copyw count_prior_lora_r16_copyw count_prior_all_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.jsonl --summary .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.md --title "Stage 43 General-Copy Surface Ladder"
```

Mean results:

| Arm | Trainable params | LoRA rank | LoRA alpha | Mean val NLL | Mean bits/char | Mean copy acc | Mean choice MRR | Mean copy NLL | Gap vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw` | 6,956 | 2 | 2.0 | 1.698173 | 2.449946 | 0.063768 | 0.201621 | 2.806473 | +0.001268 |
| `count_prior_lora_r8_copyw` | 19,244 | 8 | 8.0 | 1.696027 | 2.446849 | 0.049275 | 0.193762 | 2.818236 | -0.013225 |
| `count_prior_lora_r16_copyw` | 35,628 | 16 | 16.0 | 1.642052 | 2.368980 | 0.049276 | 0.200312 | 2.836268 | -0.013224 |
| `count_prior_all_copyw` | 111,916 | 0 | 1.0 | 1.772588 | 2.557304 | 0.043478 | 0.183403 | 2.814110 | -0.019022 |
| `random_full_copymix` | 111,916 | 0 | 1.0 | 0.587600 | 0.847728 | 0.226087 | 0.383527 | 2.459900 | +0.163587 |

Per-seed decision checks:

| Seed | Rank 8 gap vs chance | Rank 16 gap vs chance | Full frozen-base gap vs chance | No-prior full gap vs chance | Rank 8 gap vs rank 2 | Rank 16 gap vs rank 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -0.001630 | -0.027717 | -0.019022 | +0.294022 | -0.017391 | -0.043478 |
| 11 | -0.019022 | -0.010326 | -0.019022 | +0.137500 | -0.008696 | +0.000000 |
| 19 | -0.019022 | -0.001630 | -0.019022 | +0.059239 | -0.017392 | +0.000000 |

Interpretation:

H018 does not confirm. Increasing LoRA rank from 2 to 8 or 16, with alpha matched to rank, did not lift the frozen-prior residual above chance and did not beat the rank-2 baseline. The rank trend is flat to negative on behavior: rank 2 `0.063768`, rank 8 `0.049275`, rank 16 `0.049276`.

The registered KILL line fires. Arms B, C, and D all stayed within about `0.05` of chance on every seed, while Arm E cleared chance on all three seeds. The diagnostic is stronger than a pure LoRA-capacity story: `count_prior_all_copyw` trained the whole body with the same trainable count as `random_full_copymix`, but failed under the frozen count base. Arm D against Arm E therefore points toward frozen-base interference under this protocol, not merely insufficient LoRA rank.

The dual-axis split remains important. Rank 16 has the best frozen-prior validation NLL (`1.642052`) but chance-level copy behavior, while the no-prior full model has both the best NLL and the only real general-copy signal. Stage 43 says the current frozen-prior family does not form the general random-payload copy circuit at this budget; it does not prove that longer training, a different analytic base, retrieval, or a trainable attention prior would fail.

Handoff:

Codex prepared a proposed evidence draft at `docs/decisions/0009-general-copy-frozen-prior-capacity-wall.codex-draft.md`. Claude owns whether to accept, revise, or reject that ADR and whether to open a longer-budget or different-prior follow-up. Gemini owns the prior-art comparison for LoRA rank, induction-circuit formation, and frozen-base interference before any external wording.
## Stage 44 - Phase 2 TinyStories Bridge

Date: 2026-07-01

Handoff:

This is the first Phase 2 execution pass from ADR 0010. The goal was to leave the
Phase 1 algorithm lab and begin a from-scratch TinyStories-scale model build
without loading any pretrained checkpoint. Codex treated this as a bridge
baseline, not yet the full modded-nanoGPT baseline, because RoPE, Muon, gradient
accumulation, activation checkpointing, and streaming shard training are still
pending at the time of Stage 44.

Code change:

Codex added `download_tinystories.py`, `make_tinystories_corpus.py`, and
`run_phase2_visible.ps1`. The downloader pulls an official TinyStories text file
with retry support and an optional byte cap. The corpus builder normalizes local
TinyStories `.txt`, `.jsonl`, or `.json` files into the existing character-level
alphabet, records metadata, and writes optional train/validation shards. The
visible runner opens a normal PowerShell window so the user can watch `[matrix]`,
`[run]`, `[train]`, `[eval]`, and `[done]` progress live while the same output is
saved to `runs/*.log`.

Codex also vectorized `build_count_logits_ngram` with `torch.bincount`, replacing
per-token Python tensor updates. A small deterministic equivalence check matched
the old loop exactly for prior orders 1 through 4. This was required for
TinyStories scale: the old CUDA update loop timed out before producing useful
smoke artifacts.

Corpus and split:

Codex downloaded the first `50,000,000` bytes of the official
`TinyStories-train.txt` file from Hugging Face and prepared a bounded
character-level corpus:

- raw download: `experiments/tiny_language_lab/corpus/tinystories_raw/TinyStories-train.head50mb.txt`
- normalized corpus: `experiments/tiny_language_lab/corpus/tinystories_char_seed.txt`
- metadata: `experiments/tiny_language_lab/corpus/tinystories_char_seed.meta.json`
- shards: `experiments/tiny_language_lab/corpus/tinystories_char_shards/`

The normalized corpus has `10,000,001` characters, `V = 33`, `8,500,000` train
characters, and `1,500,001` validation characters under the recorded
`--block-size 256` split. The trainer runs used `--block-size 128` and the same
default `--val-fraction 0.15`, so the same prefix/suffix split size applies.

Smoke checks:

Two visible CUDA smokes passed on the actual TinyStories corpus:

- `phase2_tinystories_smoke_fast.md`: `random_full`, seed `7`, 5 steps.
- `phase2_tinystories_smoke_prior.md`: `count_prior_ng3_lora_r2`, seed `7`, 5 steps.

Both runs wrote JSONL, Markdown summaries, and visible PowerShell logs.

Primary commands:

The visible b100 and b500 bridge matrices were launched through:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bridge100 -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bridge500 -KeepOpen"
```

Both matrices used CUDA, seeds `7 11 19`, `--block-size 128`, `--batch-size 8`,
`--n-layer 4`, `--n-head 4`, `--n-embd 256`, sampled evaluation with
`--eval-batches 16`, and configs `random_full`, `count_prior_ng3_lora_r2`, and
`count_prior_ng4_lora_r2`. The b500 run used prompt `once upon a time ` for
generation samples.

Artifacts:

- b100 summary: `experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b100.md`
- b100 JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b100.jsonl`
- b500 summary: `experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b500.md`
- b500 JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b500.jsonl`
- b500 visible log: `experiments/tiny_language_lab/runs/phase2_bridge500_20260701_160314.log`

Mean b500 results:

| Config | Trainable params | Mean val NLL | Mean bits/char | Mean seconds |
| --- | ---: | ---: | ---: | ---: |
| `random_full` | 3,209,249 | 2.352297 | 3.393648 | 9.8022 |
| `count_prior_ng3_lora_r2` | 41,249 | 1.335694 | 1.927000 | 12.3874 |
| `count_prior_ng4_lora_r2` | 41,249 | 1.139715 | 1.644261 | 12.2654 |

The order-4 frozen prior plus rank-2 LoRA beats the full random baseline by
`1.212582` mean validation NLL at 500 steps while training about `1.29%` as many
parameters. The b100 result showed the same ordering: `random_full` mean NLL
`2.482619`, ng3 `1.327625`, and ng4 `1.133369`.

Generation samples:

The b500 `random_full` sample from `once upon a time ` was still mostly
character-level noise with fragments of word structure. The frozen-prior samples
were rough but story-like at the character level. The order-4 sample began:

```text
once upon a time to her all the sing very day on, i will was a story, es? anyone her favorite throught it is happy and the friends in the batterflies and sees a big and billy!
```

Interpretation:

Stage 44 is a successful Phase 2 bridge result. It proves the repository can now
download and prepare a real TinyStories corpus slice, launch visible CUDA training
from a normal terminal window, run a larger from-scratch character model than the
Phase 1 tiny lab, and reproduce the Phase 1 frozen-prior early-compute advantage
on a TinyStories-scale corpus slice. It also confirms that order 4 remains better
than order 3 under this bridge protocol.

The result does not prove that Cassandra has reached the TinyStories coherence
milestone. Generation is more story-like than the random baseline but still
unpolished. At Stage 44, ADR 0010's full baseline stack was also not implemented:
modded-nanoGPT architecture adaptation, RoPE, Muon, gradient accumulation,
activation checkpointing, streaming training, and BPE ablation remained open.
Stage 45 below addresses RoPE, Muon, gradient accumulation, and activation
checkpointing.

Handoff:

Claude can treat ADR 0010 as executable rather than speculative: the first bridge
stage passed. The next implementation stage should choose between extending the
current character bridge to longer budgets or implementing the modded-nanoGPT
baseline pieces, with RoPE plus gradient accumulation as the most immediate
hardware-relevant upgrades. Stage 45 below takes the modern-baseline path.
Gemini should frame Stage 44 as a local TinyStories character-level bridge
result, not a public TinyStories benchmark.
## Stage 45 - Phase 2 Modern TinyStories Baseline

Date: 2026-07-01

Handoff:

Stage 45 implements the first ADR 0010 modern-baseline pass on top of the Stage
44 TinyStories character corpus. It keeps the run from scratch and local: no
pretrained checkpoint is loaded. The goal was to adapt the practical
modded-nanoGPT-style pieces into Cassandra's own trainer, then prove they run in
the same visible terminal workflow.

Code change:

Codex added RoPE positional encoding, gradient accumulation, activation
checkpointing, and a single-device Muon optimizer path. Muon is applied to
hidden matrix parameters, while AdamW remains available for embeddings, heads,
biases, and non-matrix auxiliary parameters. The compare harness now exposes the
same controls: `--grad-accum-steps`, `--pos-encoding`, `--activation-checkpoint`,
`--optimizer`, Adam beta/epsilon overrides, and Muon learning-rate, momentum, and
Newton-Schulz step controls.

The visible PowerShell launcher now has two additional modes:

- `modern-smoke`: a 20-step CUDA smoke on `random_full` and
  `count_prior_ng4_lora_r2`.
- `modern500`: a 500-step, three-seed CUDA matrix using RoPE, Muon, activation
  checkpointing, and gradient accumulation.

Smoke and verification:

The visible `modern-smoke` run passed before the full matrix. It verified CUDA,
RoPE, checkpointing, Muon parameter grouping, gradient accumulation, JSONL
fields, and generation samples. Python AST checks passed for the changed
training and data scripts, and the PowerShell launcher parsed successfully.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode modern500 -KeepOpen"
```

The run used the Stage 44 corpus, CUDA, seeds `7 11 19`, `--block-size 128`,
`--batch-size 8`, `--grad-accum-steps 2`, `--n-layer 4`, `--n-head 4`,
`--n-embd 256`, `--pos-encoding rope`, `--activation-checkpoint`,
`--optimizer muon`, `--muon-lr 0.01`, sampled evaluation with
`--eval-batches 16`, and prompt `once upon a time `.

Artifacts:

- modern smoke summary: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_smoke.md`
- modern smoke JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_smoke.jsonl`
- modern b500 summary: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`
- modern b500 JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.jsonl`
- modern b500 visible log: `experiments/tiny_language_lab/runs/phase2_modern500_20260701_163301.log`

Mean b500 results:

| Config | Optimizer | Position | Effective batch | Trainable params | Mean val NLL | Mean bits/char | Mean seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `random_full` | Muon | RoPE | 16 | 3,176,481 | 1.144942 | 1.651803 | 39.8650 |
| `count_prior_ng4_lora_r2` | Muon | RoPE | 16 | 41,249 | 1.102748 | 1.590929 | 55.0514 |

Compared with the Stage 44 AdamW/learned-position bridge, the modern
`random_full` baseline improves from `2.352297` to `1.144942` mean validation NLL
at the same 500-step budget. The modern `count_prior_ng4_lora_r2` arm improves
from `1.139715` to `1.102748`. The prior arm still leads by `0.042194` mean NLL
at 500 steps while training about `1.30%` as many trainable parameters as the
full modern baseline.

Generation samples:

The modern random-full sample is no longer pure character noise. Seed `7` begins:

```text
once upon a time there was a place. so many boat. my in his did not log on the pasket and liked the askleprous.
```

The modern order-4 prior sample remains rough but has stronger story cadence:

```text
once upon a time towery. she tree.
ben said, of being new that was thanked fridges of funny was the asked
```

Interpretation:

Stage 45 is a successful Phase 2 modern-baseline result. Cassandra now has a
visible from-scratch TinyStories character training path with the main
hardware-relevant baseline pieces from ADR 0010: RoPE, Muon, gradient
accumulation, and activation checkpointing. The full random baseline becomes
credible on TinyStories after 500 steps, and the frozen order-4 prior still
retains a smaller early-compute advantage under the stronger optimizer and
positional encoding.

The result does not close all Phase 2 work. Streaming shard consumption and BPE
tokenization with BPE-space n-gram priors remain open. Stage 45 does prove that
the train-from-scratch baseline is no longer blocked on the missing
modded-nanoGPT-style engineering pieces. Stage 46 below adds the first formal
generation-quality score sheet.

## Stage 46 - TinyStories Generation-Quality Score Sheet

Date: 2026-07-01

Handoff:

Stage 46 addresses ADR 0010 D6 for the current Phase 2 character baseline by
adding a repeatable generation-quality score sheet over saved run samples. This
is not a human literary judgment and should not be used as a public benchmark.
It is a deterministic local proxy that lets Cassandra track whether prompt
completions are moving from character noise toward story-like text.

Code change:

Codex added `score_generation_samples.py`. The script reads one or more JSONL
run artifacts, extracts `sample` and `used_prompt`, builds a corpus-word
vocabulary from the TinyStories character corpus, and writes a Markdown score
sheet. It scores three ADR 0010 dimensions from `0` to `2`:

- coherence proxy: enough words, sentence punctuation, story-cue density,
  known-word ratio, and no obvious bad marker.
- grammar proxy: corpus-word ratio, word-shape sanity, punctuation, and low
  repeated-character noise.
- relevance proxy: prompt prefix adherence plus story-cue presence.

Primary command:

```powershell
python .\experiments\tiny_language_lab\score_generation_samples.py --runs .\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500.jsonl --out .\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500_generation_quality.md --title "Stage 45 TinyStories Modern Generation Quality"
```

Artifact:

- generation-quality sheet: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500_generation_quality.md`

Mean proxy scores:

| Config | Rows | Mean total | Mean coherence | Mean grammar | Mean relevance | Mean val NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_full` | 3 | 3.000 | 0.667 | 0.667 | 1.667 | 1.144942 |
| `count_prior_ng4_lora_r2` | 3 | 5.667 | 1.667 | 2.000 | 2.000 | 1.102748 |

The proxy score agrees with the qualitative readout: the modern random baseline
has learned word and sentence fragments, but one seed emits an `endoftext`
artifact and the samples remain unstable. The order-4 frozen prior samples are
still rough, but have higher corpus-word ratios, more story cues, and fewer
detected artifacts at this budget.

Interpretation:

ADR 0010's minimum generation-quality scoring requirement is now implemented for
saved prompt completions. The score is intentionally conservative as evidence:
it is useful for local trend tracking and for deciding which runs deserve human
review, not for claiming TinyStories benchmark quality.

## Stage 47 - TinyStories Shard-Consumption Smoke

Date: 2026-07-01

Handoff:

Stage 47 addresses the ADR 0010 streaming-shard requirement for the current
plain language-model training path. The corpus builder already wrote train and
validation shards in Stage 44; Stage 47 adds a trainer path that samples plain LM
batches from `train_*.txt` shard files instead of always drawing training windows
from the in-memory train split.

Code change:

Codex added a `ShardedBatchSampler` to `cassandra_tiny_transformer.py`, plus
`--train-shard-dir` and `--stream-train-eval-chars`. The sampler loads the shard
files with normalized text newlines, samples random character windows across the
train shards, and reports the shard files used in the run JSONL. The train-loss
readout uses a bounded shard prefix controlled by `--stream-train-eval-chars`,
while validation still uses the deterministic validation split from
`--corpus`.

The compare harness exposes the same flags, and `run_phase2_visible.ps1` adds a
visible `stream-smoke` mode.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode stream-smoke -KeepOpen"
```

Artifacts:

- stream smoke summary: `experiments/tiny_language_lab/runs/phase2_tinystories_stream_smoke.md`
- stream smoke JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_stream_smoke.jsonl`
- stream smoke visible log: `experiments/tiny_language_lab/runs/phase2_stream-smoke_20260701_170748.log`

Smoke result:

The visible smoke ran `random_full` with RoPE, Muon, activation checkpointing,
and gradient accumulation for 20 steps. It consumed five train shard files:
`train_00000.txt` through `train_00004.txt`. The JSONL reports
`train_chars = 8,500,000`, `train_eval_chars = 200,000`, and `val_chars =
1,500,001`. The run completed at `2.216325` validation NLL and `3.197480`
bits/char.

Interpretation:

The current Phase 2 trainer can now consume TinyStories train shards for visible
plain-LM training. This closes the minimum shard-consumption requirement for the
character baseline. The implementation is deliberately scoped: shard training is
currently blocked for copy-aware training, curriculum filtering, and frozen-prior
residual bases, because those modes still require full train-split tensors for
marker positions, prior tables, or window scoring.

## Stage 48 - Modern TinyStories 1000-Step Crossover

Date: 2026-07-01

Handoff:

Stage 48 measures ADR 0010's live crossover question on the Stage 45 modern
character baseline. Stage 45 showed that the order-4 frozen prior still had a
small NLL lead at 500 steps under RoPE and Muon. Stage 48 doubles the budget to
1000 steps to test whether the full model catches and passes that head start.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode modern1000 -KeepOpen"
```

The run used the Stage 44 corpus, CUDA, seeds `7 11 19`, `--block-size 128`,
`--batch-size 8`, `--grad-accum-steps 2`, `--n-layer 4`, `--n-head 4`,
`--n-embd 256`, `--pos-encoding rope`, `--activation-checkpoint`,
`--optimizer muon`, `--muon-lr 0.01`, sampled evaluation with
`--eval-batches 16`, and prompt `once upon a time `.

Artifacts:

- b1000 summary: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000.md`
- b1000 JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000.jsonl`
- b1000 generation score: `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000_generation_quality.md`
- b1000 visible log: `experiments/tiny_language_lab/runs/phase2_modern1000_20260701_171143.log`

Mean b1000 NLL results:

| Config | Trainable params | Mean val NLL | Mean bits/char | Mean seconds |
| --- | ---: | ---: | ---: | ---: |
| `random_full` | 3,176,481 | 1.052559 | 1.518522 | 68.7135 |
| `count_prior_ng4_lora_r2` | 41,249 | 1.123161 | 1.620378 | 100.5078 |

Per-seed NLL gaps, prior minus full:

| Seed | `random_full` | `count_prior_ng4_lora_r2` | Prior minus full |
| ---: | ---: | ---: | ---: |
| 7 | 1.038267 | 1.110035 | +0.071768 |
| 11 | 1.055469 | 1.134826 | +0.079357 |
| 19 | 1.063941 | 1.124621 | +0.060680 |

The 1000-step matrix therefore crosses decisively on NLL: the full model beats
the frozen-prior arm on every seed, with mean advantage `0.070602` NLL. This
fits ADR 0002's bounded-head-start framing. The Stage 45 prior advantage at 500
steps was `0.042194` NLL; by 1000 steps the sign is reversed.

Generation score:

| Config | Rows | Mean total | Mean coherence | Mean grammar | Mean relevance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_full` | 3 | 2.667 | 0.667 | 0.667 | 1.333 |
| `count_prior_ng4_lora_r2` | 3 | 4.667 | 1.000 | 1.667 | 2.000 |

The dual-axis readout matters. NLL favors the full model at 1000 steps, but the
deterministic generation proxy still favors the prior arm because two
`random_full` seeds emitted `endoftext` artifacts and repeated prompt-like
starts. This is not a claim that the prior is a better long-budget model. It is
a warning that sample quality and validation NLL are not identical at this
character-level stage.

Interpretation:

Stage 48 provides the first Phase 2 crossover measurement: under the modern
character baseline, the order-4 frozen prior leads at 500 steps and loses by
1000 steps. The crossover therefore lies between 500 and 1000 steps for this
corpus, model size, optimizer, and evaluation protocol. Future Phase 2 work can
refine that interval, but the broad bounded-accelerator story survives the
stronger baseline.

## Stage 49 - TinyStories BPE Smoke

Date: 2026-07-01

Handoff:

Stage 49 implements the first ADR 0010 BPE ablation path without adding an
external tokenizer dependency. The aim is not to claim a strong BPE model yet.
It is to prove that Cassandra can tokenize TinyStories into BPE tokens, train the
existing from-scratch transformer over that token stream, and attach a frozen
n-gram prior over BPE tokens.

Code change:

Codex added `make_bpe_corpus.py`, which trains a small local BPE tokenizer on a
bounded source slice and writes each BPE token ID as one Unicode private-use
codepoint. This lets the current trainer treat each BPE token as one character
without changing the tensor model. The script writes metadata containing the
vocabulary, merge list, private-use base codepoint, source length, encoded token
count, and compression ratio.

Codex also added `decode_bpe_samples.py`, which decodes private-use sample
strings from run JSONL back into readable text using the BPE metadata.

BPE corpus command:

```powershell
python .\experiments\tiny_language_lab\make_bpe_corpus.py --source .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt --out .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.txt --metadata-out .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.meta.json --vocab-size 256 --train-chars 500000 --max-chars 1000000
```

The BPE artifact has requested vocab `256`, observed run vocab `253`, `1,000,000`
source characters, `446,694` encoded BPE tokens, and `2.238669` source
characters per BPE token.

Visible smoke command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bpe-smoke -KeepOpen"
```

Artifacts:

- BPE corpus: `experiments/tiny_language_lab/corpus/tinystories_bpe_v256_seed.txt`
- BPE metadata: `experiments/tiny_language_lab/corpus/tinystories_bpe_v256_seed.meta.json`
- BPE smoke summary: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_smoke.md`
- BPE smoke JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_smoke.jsonl`
- decoded BPE samples: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_smoke_decoded_samples.md`
- BPE visible log: `experiments/tiny_language_lab/runs/phase2_bpe-smoke_20260701_172910.log`

Smoke result:

| Config | Token vocab | Trainable params | Mean val NLL | Mean bits/token | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_full` | 253 | 3,289,341 | 3.847587 | 5.550894 | 5.2993 |
| `count_prior_lora_r2` | 253 | 97,789 | 3.405228 | 4.912705 | 11.3317 |

The `count_prior_lora_r2` arm is a frozen BPE-token bigram prior plus rank-2
LoRA residual, so this is the first local n-gram-prior-over-BPE-token smoke.
Decoded samples are still rough. The prior sample begins:

```text
one day, the ble bit arink today the grany. annecame. she loves ondod joe caretix abule, look?
once upon a time, thankest rushave decather.
```

Interpretation:

Stage 49 closes the minimum BPE feasibility gate: BPE tokenization, BPE-token
training, decoded BPE samples, and a frozen n-gram prior over BPE tokens all run
locally and visibly. This is intentionally a smoke, not the durable BPE decision.
The next BPE work should scale the corpus and vocab, choose a better prompt
encoding path, and run a multi-seed matrix before comparing BPE against the
character-level baseline.

## Stage 50 - BPE 500-Step Multi-Seed Matrix

Date: 2026-07-01

Handoff:

Stage 50 turns the Stage 49 BPE smoke into a multi-seed BPE decision surface. It
keeps the same bounded BPE artifact, because the immediate question is whether
the newly live BPE-token bigram prior has the same kind of useful 500-step
head-start that the character order-4 prior showed in Stage 45.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bpe500 -KeepOpen"
```

The run used the BPE v256 corpus, CUDA, seeds `7 11 19`, `--block-size 128`,
`--batch-size 8`, `--grad-accum-steps 2`, `--n-layer 4`, `--n-head 4`,
`--n-embd 256`, `--pos-encoding rope`, `--activation-checkpoint`,
`--optimizer muon`, and configs `random_full` and `count_prior_lora_r2`.
In this BPE corpus, `count_prior_lora_r2` is a frozen BPE-token bigram prior plus
rank-2 LoRA residual.

Artifacts:

- bpe500 summary: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_b500.md`
- bpe500 JSONL: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_b500.jsonl`
- bpe500 decoded samples: `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_b500_decoded_samples.md`
- bpe500 visible log: `experiments/tiny_language_lab/runs/phase2_bpe500_20260701_173929.log`

Mean bpe500 results:

| Config | Trainable params | Mean val NLL | Mean bits/token | Approx bits/source char | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_full` | 3,289,341 | 2.404960 | 3.469624 | 1.549860 | 37.2783 |
| `count_prior_lora_r2` | 97,789 | 3.344760 | 4.825469 | 2.155508 | 58.0354 |

Per-seed NLL gaps, prior minus full:

| Seed | `random_full` | `count_prior_lora_r2` | Prior minus full |
| ---: | ---: | ---: | ---: |
| 7 | 2.367540 | 3.355098 | +0.987558 |
| 11 | 2.406623 | 3.317610 | +0.910987 |
| 19 | 2.440717 | 3.361572 | +0.920855 |

Interpretation:

The BPE-token bigram prior does not reproduce the character order-4 prior's
500-step advantage. It is useful at step 0 and in the 20-step smoke, but by 500
steps the full BPE model wins decisively on every seed. This is a real BPE
ablation result, not just a smoke: the BPE tokenizer, BPE-token model, decoded
samples, and BPE-token prior all run across three seeds.

Decision:

Keep the character-level TinyStories baseline as the Phase 2 default for the
current laptop-scale model. Keep BPE as a live but secondary branch. The next BPE
branch should not spend more time on this small v256 bigram-prior setup; it
should either use a larger corpus/vocab and compare full BPE against full char,
or re-derive a higher-order BPE prior before asking for a prior advantage.

## Stage 51 - Phase 3 Coherence Checkpoint

Date: 2026-07-02

Handoff:

Stage 51 implements ADR 0011 D2 and D3. The goal was to fix the
params-to-data mismatch before training the Phase 3 coherence checkpoint, then
train a 25.25M-parameter `random_full` character-level TinyStories model from
random initialization. No pretrained checkpoint was loaded.

Gate A, corpus rescale:

Codex downloaded a `500,000,000` byte TinyStories slice, ten times the previous
50 MB cap, then rebuilt `tinystories_char_seed.txt` with the existing
character-level alphabet and deterministic prefix/suffix split. The resulting
corpus has `494,094,421` normalized characters, `419,980,257` train characters,
`74,114,164` validation characters, and `V = 33`. The corpus is large enough
that Stage 51 and Stage 52 use the Stage 47 shard-streaming path for
`random_full` training.

The original one-shot corpus builder hit a Python `MemoryError` on the 500 MB
raw file, so Codex changed `make_tinystories_corpus.py` to normalize text in a
streaming path while preserving the old normalizer's behavior. A small
equivalence check against the old normalizer passed.

Gate B, 85M confirmation:

The 85.11M `random_full` confirmation smoke ran for 200 steps on the rescaled
corpus with RoPE, Muon, activation checkpointing, gradient accumulation, and
`--eval-mode sampled`. It completed with `1,683.4297 MiB` peak CUDA allocation,
`75.1112` seconds wall-clock, and `1.307224` validation NLL. The 85M point was
therefore kept in Stage 52.

Gate C, 25M timing and budget:

The 25.25M timing pass ran 500 steps and completed in `84.6193` seconds, about
`0.169` seconds per step including load and sampled evaluation, with
`948.2817 MiB` peak CUDA allocation. Codex chose the Stage 51 budget as `5000`
steps, the top of ADR 0011's requested `2000` to `5000` range, because the
measured wall-clock was still practical and the checkpoint deliverable benefits
from the strongest local quality pass.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage51"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode score51"
```

The Stage 51 training command used CUDA, seeds `7 11 19`, `--steps 5000`,
`--block-size 128`, `--batch-size 8`, `--grad-accum-steps 2`,
`--n-layer 8`, `--n-head 8`, `--n-embd 512`, `--pos-encoding rope`,
`--activation-checkpoint`, `--optimizer muon`, `--muon-lr 0.01`,
`--eval-mode sampled`, `--eval-batches 16`, `--train-shard-dir
.\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb`, and
`random_full`.

Artifacts:

- corpus: `experiments/tiny_language_lab/corpus/tinystories_char_seed.txt`
- metadata: `experiments/tiny_language_lab/corpus/tinystories_char_seed.meta.json`
- shards: `experiments/tiny_language_lab/corpus/tinystories_char_shards_500mb/`
- Gate B JSONL: `experiments/tiny_language_lab/runs/stage51_gateB_85m_smoke.jsonl`
- Gate C JSONL: `experiments/tiny_language_lab/runs/stage51_gateC_25m_timing.jsonl`
- Stage 51 JSONL: `experiments/tiny_language_lab/runs/stage51_coherence_25m_b5000.jsonl`
- Stage 51 summary: `experiments/tiny_language_lab/runs/stage51_coherence_25m_b5000.md`
- checkpoints: `experiments/tiny_language_lab/runs/stage51_checkpoints/`
- generation score sheet: `experiments/tiny_language_lab/runs/stage51_coherence_25m_b5000_generation_quality.md`
- visible log: `experiments/tiny_language_lab/runs/phase3_stage51_20260701_223937.log`

Stage 51 results:

| Seed | Val NLL | Bits/char | Seconds | Peak CUDA MiB |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.821911 | 1.185767 | 682.5285 | 948.2817 |
| 11 | 0.810016 | 1.168606 | 706.9184 | 952.2817 |
| 19 | 0.823896 | 1.188631 | 697.4736 | 952.2817 |

Mean result:

| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean seconds | Mean generation score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_full` | 3 | 25,253,921 | 0.818608 | 1.181001 | 695.6402 | 5.667/6 |

Interpretation:

Stage 51 produces a saved 25.25M checkpoint family with substantially lower
sampled validation NLL than the Phase 2 3.2M baselines. The deterministic
generation proxy score is high, but it is still a proxy. Per ADR 0011 D3, Codex
does not claim genuine coherence until Mert performs human review of the saved
samples.

Human review (2026-07-06):

Mert reviewed the saved samples and PASSED them. His read: the samples were
"really cool", they "made sense", and there were "a lot of little stories".
The ADR 0011 D3 gate is cleared. The claim this supports is genuine local
coherence at the TinyStories micro-story level: fluent, sensible,
story-shaped text within the block-128 window, consistent with the 5.667/6
proxy score. It is not a claim of long-range narrative structure beyond that
window. The review also calibrates the Stage 55 flagship bar (ADR 0013 open
question 3): the flagship sits at 0.835 bits/char by step 30,000 versus
Stage 51's 1.181, so its own required human review should expect samples
that read clearly better than these.

## Stage 52 - H019 Crossover Scaling Matrix

Date: 2026-07-02

Handoff:

Stage 52 implements ADR 0011 D4 and Hypothesis 019. It holds the rescaled
TinyStories character corpus and modern recipe fixed, then tests whether the
step budget where `random_full` catches `count_prior_ng4_lora_r2` changes with
model capacity.

Code change:

The first Stage 52 prior pass OOMed on every `count_prior_ng4_lora_r2` cell
because order-4 prior construction still tried to operate through the full
rescaled train tensor. Codex kept those failed JSONLs as evidence, then added a
shard-native order-4 count-prior builder and a disk cache for the prior tensor.
A small equivalence check against the original tensor builder produced
`max_abs_diff = 0.0`. The corrected prior cells were rerun with `_sharded`
artifact names and no errors.

Primary command:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage52-matrix"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage52-prior-sharded"
```

The full Stage 52 matrix used sizes `3m`, `10m`, `25m`, and `85m`, budgets
`200`, `500`, `1000`, and `2000`, arms `random_full` and
`count_prior_ng4_lora_r2`, seeds `7 11 19`, CUDA, RoPE, Muon, activation
checkpointing, gradient accumulation, `--eval-mode sampled`, and
`--eval-batches 16`.

Artifacts:

- aggregate summary: `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
- full-model cells: `experiments/tiny_language_lab/runs/stage52_crossover_*_random_full.jsonl`
- corrected prior cells: `experiments/tiny_language_lab/runs/stage52_crossover_*_count_prior_ng4_lora_r2_sharded.jsonl`
- failed first prior pass: `experiments/tiny_language_lab/runs/stage52_crossover_*_count_prior_ng4_lora_r2.jsonl`
- visible matrix log: `experiments/tiny_language_lab/runs/phase3_stage52-matrix_20260701_231748.log`
- visible corrected-prior log: `experiments/tiny_language_lab/runs/phase3_stage52-prior-sharded_20260702_035628.log`
- shared prior cache: `experiments/tiny_language_lab/runs/stage52_prior_cache/count_ngram_561fefe48ed103be.pt`

Interruption audit:

After the laptop move, Codex audited active processes, JSONL validity, row
counts, statuses, requested step counts, logs, checkpoints, and the shared prior
cache. No experiment process was still running. Every Stage 52 decision cell has
three successful rows, all requested `--eval-mode sampled`, and all rows reached
their requested `steps` and `formation_steps`. The original non-sharded prior
pass has `48` expected OOM/error rows and is not used for the decision table.
Two corrected-prior rows have inflated wall-clock due to pause or sleep during
the visible run: 25.25M at 1000 steps, seed `11`, `3156.4317` seconds, and
85.11M at 2000 steps, seed `19`, `15858.9964` seconds. The logs show both rows
continued to final train step, sampled eval, JSONL write, markdown write, and
exit code `0`; treat those `seconds` fields as throughput confounds, not failed
training rows.

Mean validation NLL:

| Size | Budget | `random_full` | `count_prior_ng4_lora_r2` | Prior minus full | Mean winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 3.2M | 200 | 1.450852 | 1.111521 | -0.339331 | prior |
| 3.2M | 500 | 1.165709 | 1.103676 | -0.062033 | prior |
| 3.2M | 1000 | 1.052545 | 1.107266 | +0.054721 | full |
| 3.2M | 2000 | 0.969006 | 1.111348 | +0.142342 | full |
| 10.67M | 200 | 1.393438 | 1.112065 | -0.281373 | prior |
| 10.67M | 500 | 1.124503 | 1.103436 | -0.021067 | prior |
| 10.67M | 1000 | 1.016949 | 1.107198 | +0.090249 | full |
| 10.67M | 2000 | 0.933909 | 1.112423 | +0.178514 | full |
| 25.25M | 200 | 1.353240 | 1.113529 | -0.239711 | prior |
| 25.25M | 500 | 1.106377 | 1.104469 | -0.001908 | prior |
| 25.25M | 1000 | 0.999363 | 1.108225 | +0.108862 | full |
| 25.25M | 2000 | 0.922147 | 1.112237 | +0.190090 | full |
| 85.11M | 200 | 1.311739 | 1.115967 | -0.195772 | prior |
| 85.11M | 500 | 1.090837 | 1.104971 | +0.014134 | full |
| 85.11M | 1000 | 0.982652 | 1.109477 | +0.126825 | full |
| 85.11M | 2000 | 0.918379 | 1.113583 | +0.195204 | full |

Crossovers:

| Size | Crossover budget | Per-seed note |
| --- | ---: | --- |
| 3.2M | 1000 | `0/3` full wins at 200 and 500, `3/3` full wins at 1000 and 2000 |
| 10.67M | 1000 | `0/3` full wins at 200 and 500, `3/3` full wins at 1000 and 2000 |
| 25.25M | 1000 | `1/3` full wins at 500, `3/3` full wins at 1000 and 2000 |
| 85.11M | 500 | `0/3` full wins at 200, `3/3` full wins at 500, 1000, and 2000 |

Interpretation:

Stage 52 is `GRADED` under H019. The crossover is flat at `1000` steps for
3.2M, 10.67M, and 25.25M, then moves earlier to `500` steps at 85.11M. This is
not clean E1, because the first three sizes do not strictly decrease. It is not
E2, because the largest model crosses earlier. It is not E3, because the 85M
point leaves the same discrete crossover rung as the smaller sizes. The local
result supports a capacity effect at the top end, but it is not enough to claim
a smooth monotonic scaling law.

## Stage 53 - H020 Frozen Prior Free-Accelerator Test

Date: 2026-07-02

Handoff:

Stage 53 implements ADR 0013 D1 and Hypothesis 020. It tests the missing cell
from Stage 52: a 25.25M transformer with the frozen order-4 count n-gram prior
AND the full body trainable. The question was whether the prior is a free
accelerator under full training, a transient head start, or a late-training
handicap.

Code change:

Codex added the `count_prior_ng4_all` config in `cassandra_compare.py`:
`--residual-base count-ngram --prior-order 4 --train-scope all`, with no LoRA
keys. The copy-arm `lora_alpha` gotcha does not apply to this config. Codex also
added `run_phase4_visible.ps1` so Stage 53 and its LR sensitivity rerun launch
through the ADR 0012 visible-terminal protocol and always carry the mandatory
Stage 52 prior cache path.

Confirm-first audit:

- The runner forwards the Stage 52 launch shape into per-config args, including
  model size, block and batch shape, gradient accumulation, RoPE, activation
  checkpointing, Muon options, sampled eval, train shards, and prior cache dir.
- The prior cache is consulted only when `--prior-cache-dir` is passed. Stage 53
  rows report `prior_cache_status=hit` for
  `experiments\tiny_language_lab\runs\stage52_prior_cache\count_ngram_561fefe48ed103be.pt`.
- `base_logits` is a separate frozen tensor passed into forward, not a model
  parameter. With `train_scope=all`, the model reports `25,253,921` trainable
  parameters plus `42,802,056` frozen prior logits.
- The 10-step CUDA smoke showed identical step-0 sampled validation NLL for
  `count_prior_ng4_lora_r2` and `count_prior_ng4_all`: `1.103458`.
- Checkpoint save exists through the Stage 51 checkpoint path, but resume is not
  implemented yet. The resume proof remains a required Stage 55 item.

Primary command shape:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage53"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage53-cell -Budget 2000 -MuonLr 0.005"
```

The main matrix used CUDA, seeds `7 11 19`, budgets `200`, `500`, `1000`, and
`2000`, `--block-size 128`, `--batch-size 8`, `--grad-accum-steps 2`,
`--n-layer 8`, `--n-head 8`, `--n-embd 512`, `--pos-encoding rope`,
`--activation-checkpoint`, `--optimizer muon`, `--muon-lr 0.01`,
`--eval-mode sampled`, `--eval-batches 16`, `--train-shard-dir
.\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb`, and
`--prior-cache-dir .\experiments\tiny_language_lab\runs\stage52_prior_cache`.

Artifacts:

- summary: `experiments/tiny_language_lab/runs/stage53_h020_free_accelerator_summary.md`
- smoke: `experiments/tiny_language_lab/runs/stage53_smoke_prior_all_10step.jsonl`
  and `.md`
- main cells: `experiments/tiny_language_lab/runs/stage53_prior_all_25m_b{200,500,1000,2000}.jsonl`
  and `.md`
- LR sensitivity: `experiments/tiny_language_lab/runs/stage53_prior_all_25m_b2000_muonlr005.jsonl`
  and `.md`
- interrupted partial evidence preserved as
  `stage53_prior_all_25m_b2000_interrupted_partial.jsonl`, `.md`, and
  `phase4_stage53_20260702_214906_interrupted_partial.log`

Stage 53 results:

| Budget | `count_prior_ng4_all` mean val NLL | Mean bits/char | Mean seconds | Stage 52 `random_full` mean | Mean delta | Paired deltas by seed |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 200 | 1.101573 | 1.589234 | 46.7 | 1.353240 | -0.251667 | `7:-0.254035`, `11:-0.244824`, `19:-0.256142` |
| 500 | 1.081726 | 1.560600 | 88.8 | 1.106377 | -0.024651 | `7:-0.027513`, `11:-0.032568`, `19:-0.013873` |
| 1000 | 1.071762 | 1.546225 | 174.9 | 0.999363 | +0.072398 | `7:+0.078042`, `11:+0.064813`, `19:+0.074340` |
| 2000 | 1.022622 | 1.475331 | 472.4 | 0.922147 | +0.100475 | `7:+0.101006`, `11:+0.106925`, `19:+0.093493` |

LR sensitivity:

H020 requires one `2000`-step `--muon-lr 0.005` rerun before recording
E-interfere. Lower LR improved the prior-full arm, but did not remove the late
gap:

| Cell | Mean val NLL | Mean bits/char | Stage 52 `random_full` mean | Mean delta | Paired deltas by seed |
| --- | ---: | ---: | ---: | ---: | --- |
| `count_prior_ng4_all`, `--muon-lr 0.005`, 2000 steps | 0.976702 | 1.409083 | 0.922147 | +0.054555 | `7:+0.051790`, `11:+0.058232`, `19:+0.053643` |

Decision:

Stage 53 is **KILL E-interfere** under H020. The frozen prior is a strong early
accelerator at 200 steps and still positive at 500 steps, but by 1000 and 2000
steps all three paired deltas are worse than `random_full` by more than `0.01`.
The required lower-LR rerun shrank the 2000-step gap but all three paired deltas
remained positive by about `0.05`.

Interpretation:

This result does not say the order-4 prior is useless. It says the current
full-body Muon recipe does not get the prior for free: the additive frozen base
helps early, then becomes a late-training handicap relative to random
initialization. Per H020 and ADR 0013, the Stage 55 flagship should use random
initialization unless Claude explicitly schedules a longer-budget or retuned
prior follow-up before the flagship recipe is frozen. This stage remains a local
NLL result, not a coherence or sample-quality claim.

## Stage 54 - H021 Prior-Order Floor Scaling

Date: 2026-07-02

Handoff:

Stage 54 implements ADR 0013 D2 and Hypothesis 021. Stage 35 said the best
analytic prior is the highest estimable count order. The 494M-character
TinyStories corpus reopens that ceiling because order-5 contexts are now much
better estimated than they were on the earlier small corpus. The stage first
gates feasibility, then runs the 25.25M crossover column only if the gate
passes.

Code change:

Codex added a shard-native sparse order-5 backoff prior in
`cassandra_tiny_transformer.py`. Dense prior construction remains capped at
order 4. Order 5 is available only through the sparse path: keep sorted int64
5-character context keys whose total count is at least `--prior5-min-count`
(default `10`), store fp16 `ctx_logits` rows as `log(alpha + count)`, and fall
back to the dense order-4 prior for misses or early positions. Codex also added
`count_prior_ng5_lora_r2` and `count_prior_ng5_lora_r2_floor` to
`cassandra_compare.py`, plus Stage 54 modes in `run_phase4_visible.ps1`.

Verification:

- AST parse passed for `cassandra_tiny_transformer.py` and
  `cassandra_compare.py`.
- Small split-shard equivalence check with `prior5_min_count=1` passed:
  `max_abs_diff=0.0` for stored sparse rows.
- Tiny CPU CLI smoke passed paired order-4/order-5 floor configs through shard
  loading, cache writing, generation, JSONL writing, and summary writing.
- The visible Stage 54 smoke, gate, and Phase B cells all completed with the
  expected row counts and no error rows.

Primary command shape:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage54-smoke
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage54-gateA
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage54-phaseB-cell -Budget 1000
```

The Phase B cell command was repeated for budgets `200`, `500`, `1000`, and
`2000`.

Artifacts:

- summary: `experiments/tiny_language_lab/runs/stage54_h021_prior_order_floor_scaling_summary.md`
- smoke: `experiments/tiny_language_lab/runs/stage54_smoke_ng5_20step.jsonl`
  and `.md`
- gate:
  `experiments/tiny_language_lab/runs/stage54_gateA_floor_pair.jsonl` and `.md`
- Phase B:
  `experiments/tiny_language_lab/runs/stage54_ng5_25m_b{200,500,1000,2000}.jsonl`
  and `.md`
- sparse prior cache:
  `experiments\tiny_language_lab\runs\stage52_prior_cache\count_ngram_e9b2816129609620.pt`

Sparse prior diagnostics:

| Metric | Value |
| --- | ---: |
| Build seconds | 135.9158 |
| Total tensor bytes | 182,837,102 |
| Cache file bytes | 182,844,894 |
| Observed order-5 contexts | 289,818 |
| Kept order-5 contexts | 157,147 |
| Validation backoff rate | 0.001016 |
| Peak CUDA memory in 20-step smoke | 929.2324 MiB |

Phase A gate:

| Seed | Order-4 floor | Order-5 floor | Delta |
| ---: | ---: | ---: | ---: |
| 7 | 1.098857 | 0.980171 | -0.118686 |
| 11 | 1.112710 | 0.997405 | -0.115305 |
| 19 | 1.094673 | 0.975717 | -0.118956 |
| Mean | 1.102080 | 0.984431 | -0.117649 |

The gate passes: the mean paired delta is below the `-0.02` line and all three
paired deltas are negative.

Phase B crossover:

| Budget | Order-5 prior mean | Stage 52 `random_full` mean | Delta vs random | Stage 52 order-4 prior mean | Delta vs order 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.995437 | 1.353240 | -0.357803 | 1.113529 | -0.118092 |
| 500 | 0.988338 | 1.106377 | -0.118039 | 1.104469 | -0.116131 |
| 1000 | 0.989131 | 0.999363 | -0.010233 | 1.108225 | -0.119094 |
| 2000 | 0.998215 | 0.922147 | +0.076068 | 1.112237 | -0.114022 |

Decision:

Stage 54 is **CONFIRM** under H021. The sparse order-5 prior beats order 4 by
about `0.116` to `0.119` NLL and moves the 25.25M discrete crossover from
1000 steps to 2000 steps. This confirms the corpus-relative interpretation of
the Stage 35 ceiling.

Interpretation:

Stage 54 does not overturn Stage 53. The Stage 55 flagship still points to
random initialization because H020 killed the "free prior under full-body
training" claim. Stage 54 improves the low-budget tiny-surface recipe: on the
494M-character TinyStories corpus, a frozen analytic prior plus small residual
should use the sparse order-5 backoff prior rather than the order-4 prior. This
stage remains a local NLL result, not a coherence or sample-quality claim.

## Stage 55 - D3a Flagship Sizing And Resume Gate

Date: 2026-07-03 local launcher timestamp

Handoff:

Stage 55 implements ADR 0013 D3. Stage 53 selected random initialization for
the flagship because H020 fired KILL E-interfere. Stage 54 confirmed the sparse
order-5 prior for low-budget tiny-surface work, but does not change the
full-body flagship initialization.

Code change:

Codex added periodic checkpoint and resume support to
`cassandra_tiny_transformer.py` and forwarded it through `cassandra_compare.py`.
The new flags are `--checkpoint-dir`, `--checkpoint-every`, and
`--resume-from`. Checkpoints store model state, optimizer state, generator
state, current step, formation forward-pass count, loss curve, args, chars, and
the base prior if present. The Stage 55 flagship path has no base prior.

Verification:

- AST parse passed for `cassandra_tiny_transformer.py` and
  `cassandra_compare.py`.
- A fast CPU checkpoint smoke saved at step 2 and resumed to step 4 with
  `resume_loaded=True`, `resume_step=2`, and `formation_steps=4`.
- Visible CUDA sizing and resume rows completed with expected row counts and no
  error rows.
- The `stage55-flagship-cell` visible launcher dry-run emitted the expected
  200M random-init command.

Primary command shape:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-size-200m-b256
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-resume-unbroken
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-resume-interrupted
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-resume-resumed
```

Artifacts:

- gate report:
  `experiments/tiny_language_lab/runs/stage55_d3a_sizing_resume_summary.md`
- sizing:
  `experiments/tiny_language_lab/runs/stage55_size_{85m_b128,85m_b256,200m_b128,200m_b256}.jsonl`
  and `.md`
- resume:
  `experiments/tiny_language_lab/runs/stage55_resume_unbroken_400.jsonl`,
  `experiments/tiny_language_lab/runs/stage55_resume_interrupted_400.jsonl`,
  `experiments/tiny_language_lab/runs/stage55_resume_resumed_400.jsonl`, and
  their `.md` summaries
- resume checkpoints:
  `experiments\tiny_language_lab\runs\stage55_resume_checkpoints`

Sizing gate:

All rows used CUDA, seed `7`, 200 steps, TinyStories train shards, sampled eval
with `--eval-batches 16`, RoPE, Muon `--muon-lr 0.01`, activation
checkpointing, batch `8`, grad accumulation `2`, random init, and no frozen
prior.

| Candidate | Trainable params | Frozen params | Block | Seconds | Seconds/step | Peak CUDA MiB | Val NLL | Bits/char |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 85M | 85,106,721 | 0 | 128 | 67.1839 | 0.335920 | 1,683.4297 | 1.307224 | 1.885925 |
| 85M | 85,106,721 | 0 | 256 | 104.9980 | 0.524990 | 1,799.7617 | 1.214870 | 1.752687 |
| 200M | 201,609,249 | 0 | 128 | 141.8289 | 0.709145 | 3,061.7681 | 1.303770 | 1.880943 |
| 200M | 201,609,249 | 0 | 256 | 229.1033 | 1.145516 | 3,233.4751 | 1.194541 | 1.723358 |

Decision:

Choose `random_full`, 201.61M parameters, block 256, batch 8, grad accumulation
2, RoPE, activation checkpointing, Muon, and sampled eval. This is the largest
tested configuration, has the best 200-step sampled validation NLL among the
four candidates, and leaves about `4,954 MiB` of reported allocation headroom
against the `8,188 MiB` card limit.

Resume proof:

The chosen 200M/block-256 configuration was run as an unbroken 400-step
reference, then as an interrupted run stopped after the step-200 checkpoint, and
then resumed from
`experiments\tiny_language_lab\runs\stage55_resume_checkpoints\stage55_resume_interrupted_400_random_full_seed7_step000200.pt`.

| Run | Resume loaded | Resume step | Formation steps | Seconds | Step-400 eval val NLL | Final val NLL | Final bits/char |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbroken | false | 0 | 400 | 477.8673 | 0.986312 | 1.011825 | 1.459755 |
| Resumed | true | 200 | 400 | 316.3387 | 0.986312 | 1.011825 | 1.459755 |

Resume proof result: **PASS**. The resumed run restored at step 200, restored
the optimizer and generator state, completed to step 400, saved a new step-400
checkpoint, and matched the unbroken run's step-400 sampled eval and final
sampled validation NLL.

Long-run plan:

At block 256, each optimizer step sees `4096` training characters. A 50,000-step
seed sees about `204.8M` training characters, clearing the 200M target. At the
measured `1.145516` seconds per step, one 50k seed projects to about `15.91`
hours. Three 50k seeds would exceed the 36 GPU-hour rule before eval and
checkpoint overhead, so the planned package is one full seed and two
reduced-budget replicas:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 50000 -Seed 7
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 20000 -Seed 11
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 20000 -Seed 19
```

Interpretation:

Stage 55's preflight gate proves the selected flagship shape fits the laptop
GPU and that checkpoint resume works on the same optimizer and architecture. It
does not yet prove final NLL, proxy generation quality, or coherence. The long
run must report sampled validation NLL, bits/char, wall clock, checkpoints, and
proxy generation scores, with human review pending.

## Stage 55 - Flagship Long Run Closeout

Date: 2026-07-07 local closeout

Handoff:

This section closes ADR 0013 Phase 4. Stage 53 selected random initialization
for the flagship. Stage 54 selected the sparse order-5 prior for low-budget
tiny-surface work, but that did not change the full-body flagship recipe.

Recipe:

- config `random_full`
- `201,609,249` trainable parameters, `0` frozen parameters
- TinyStories character corpus, `494,094,421` chars total,
  `419,980,257` train chars, `74,114,164` validation chars
- shard-streamed training from
  `experiments/tiny_language_lab/corpus/tinystories_char_shards_500mb`
- block 256, batch 8, grad accumulation 2, RoPE, activation checkpointing
- Muon optimizer, `--muon-lr 0.01`
- sampled eval, `--eval-batches 16`

Primary command shape:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 50000 -Seed 7 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 20000 -Seed 11 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 -Mode stage55-flagship-cell -Budget 20000 -Seed 19 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints
```

The runs were originally launched across resume legs. After a OneDrive
checkpoint rename failure at seed 7 step 35,000, recovery used
`C:\Users\senso\AppData\Local\Temp\cassandra_runs\stage55_flagship_checkpoints`.
Windows Storage Sense can purge `%TEMP%`, so the final seed 7, seed 11, and
seed 19 checkpoints were copied with byte-count verification to
`C:\cassandra_runs\stage55_flagship_checkpoints`. The launcher default now
points there for future continuations.

Run artifacts:

- `experiments/tiny_language_lab/runs/stage55_flagship_200m_b50000_seed7.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage55_flagship_200m_b20000_seed11.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage55_flagship_200m_b20000_seed19.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage55_flagship_200m_b50000_seed7_onedrive_checkpoint_error_20260706.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage55_flagship_generation_quality.md`

Original final checkpoint keep-set before Phase 5 disk demotion:

- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`
- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7.pt`
- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b20000_seed11_random_full_seed11_step020000.pt`
- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b20000_seed11_random_full_seed11.pt`
- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b20000_seed19_random_full_seed19_step020000.pt`
- `C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b20000_seed19_random_full_seed19.pt`

Current-state note, 2026-07-09: during Stage 56, repeated checkpoint-save
failures drove C: close to zero free space. To protect the live Phase 5
checkpoint writes, the Stage 55 no-suffix duplicate checkpoints were pruned
first, and the explicit Stage 55 fp32 training checkpoints above were later
demoted as older cache. Durable Stage 55 evidence remains in the JSONL and
Markdown run artifacts, proxy and human review artifacts, and the checked
Nsight DL Designer ONNX export and manifest. The Stage 55 fp32 training
checkpoint files are no longer expected to be present under
`C:\cassandra_runs`. The seed-7 final fp32 artifact checkpoint remains in the
repo-local Phase 4 artifact package and is the checkpoint used for the Phase 5
letters-only behavior probe:
`experiments/tiny_language_lab/artifacts/phase4/checkpoints/stage55_flagship_200m_b50000_seed7/stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`.

Validation:

| Seed | Budget steps | Resume loaded | Resume step | Formation forwards | Report val NLL | Report bits/char | Last sampled eval val NLL | Last sampled eval bits/char | JSONL seconds | Peak CUDA MiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 50,000 | true | 40,000 | 100,000 | 0.556410 | 0.802730 | 0.551281 | 0.795330 | 16,539.8364 | 4,007.5542 |
| 11 | 20,000 | true | 5,000 | 40,000 | 0.609039 | 0.878657 | 0.579033 | 0.835368 | 21,880.3563 | 4,007.5542 |
| 19 | 20,000 | false | 0 | 40,000 | 0.596864 | 0.861093 | 0.605814 | 0.874005 | 30,702.2723 | 3,233.4751 |

Mean report validation NLL across the mixed-budget package is `0.587438`;
mean bits/char is `0.847493`. This mean mixes one 50k seed with two 20k
replicas, so it is a package summary, not a three-seed 50k estimate. The
reported `seconds` values are per final JSONL invocation. Total wall-clock is
split across resume legs because elapsed time resets per invocation.

Generation proxy:

`experiments/tiny_language_lab/runs/stage55_flagship_generation_quality.md`
scored the saved samples with the deterministic Stage 46 proxy rubric:

| Config | Rows | Mean total | Mean coherence | Mean grammar | Mean relevance | Mean val NLL | Mean bits/char |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_full | 3 | 4.333 | 1.000 | 1.667 | 1.667 | 0.587438 | 0.847493 |

These are deterministic proxy scores only. Human review remains pending, so
this is not a coherence or sample-quality claim.

Nsight DL Designer closeout artifact:

```powershell
$env:PYTHONPATH = 'C:\cassandra_pydeps\onnx' + [IO.Path]::PathSeparator + $env:PYTHONPATH
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 -CheckOnnx
```

The checked export loaded the secured seed-7 step-50,000 checkpoint and wrote:

- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx.data`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.manifest.json`

ONNX checker verification passed with manifest step `50000`, opset `18`,
input `input_ids` shape `[1, 256]`, and output `logits` shape `[1, 256, 33]`.
The wrapper script now defaults to the durable checkpoint path and propagates
Python export failures.

Summary label audit:

The generated run summaries now split the actual training optimizer from the
residual-route selector. Stage 55 rows therefore read `Optimizer = muon` and
`Residual route = adamw`; this is not an AdamW training recipe. Historical
summaries that previously showed only `Residual optim = adamw` were regenerated
from their JSONL rows with the current two-column format.

Interpretation:

Stage 55 completes the ADR 0013 flagship build on the local laptop. The final
full-budget seed reaches sampled report validation NLL `0.556410`
(`0.802730` bits/char) after 50,000 steps and has a resume-capable training
checkpoint outside OneDrive and `%TEMP%`. The two reduced replicas provide
additional run evidence at 20,000 steps. This is a training and proxy-scoring
closeout only; human sample review is still pending.

Human review (2026-07-07):

Mert completed 21 blind A/B votes in the playground
(`runs/human_ab_votes.jsonl`; identities revealed only after each vote; all
votes at temperature 0.8 on the prompt "once upon a time"). Tally of
non-tie head-to-heads:

| Pairing | Wins | Losses | Ties |
| --- | ---: | ---: | ---: |
| flagship vs stage51 25M | 2 | 2 | 0 |
| flagship vs replica seed19 | 3 | 0 | 4 |
| flagship vs replica seed11 | 1 | 1 | 2 |
| replica seed19 vs replica seed11 | 3 | 0 | 1 |
| stage51 25M vs replica seed19 | 1 | 1 | 0 |

Reading, per ADR 0014 D4 and its reversal clause: the vote-count gate is
met (21 of about 20), but the flagship does NOT beat Stage 51 by a clear
majority (2 to 2 direct), so the coherence framing stays withdrawn and the
proxy-score anomaly (flagship `4.333/6` vs Stage 51 `5.667/6` despite `0.38`
better bits/char) now has a SECOND independent signal: blind human judgment
also ranks the 25M model even with the 200M on this prompt. Two protocol
caveats keep this provisional rather than final: only 4 direct
flagship-vs-Stage-51 comparisons exist (2-2 at n=4 is compatible with
chance), and the varied-prompts condition was not met (single prompt).
Within the 200M family the ordering is clean (flagship over seed19 3-0-4;
seed19 over seed11 3-0-1), so more training wins INSIDE a family; the
anomaly is specifically against the smaller model. Named follow-up: a
targeted second round, flagship vs Stage 51 only, 10 to 15 votes across at
least 5 varied prompts, before any final coherence verdict.

## Stage 56 - H022 Broad-Corpus Specialization Gap Closeout

Date: 2026-07-09 local closeout

Verdict:

**H022 CONFIRM, E-data.** The specialization gap is primarily a
data-distribution effect under the registered Stage 56 test. The unchanged
85M character recipe, trained on broad text8 train/valid text, scored well
below the `1.70` CONFIRM line on the held-out text8 TEST split. The 20k
replicas were stable inside the `0.10` bits/char spread guard.

Corpus and contamination:

Stage 56 used standard text8 slicing:

- raw text8 length: `100,000,000` chars
- train shards: chars `[0, 90,000,000)`, nine 10M-char shards
- in-run validation: chars `[90,000,000, 95,000,000)`
- deterministic TEST: chars `[95,000,000, 100,000,000)`
- seed corpus: first `95,000,000` chars
- seed SHA256:
  `a27d3a863dbd8102fb14618aa65e966c40117d9ce852ac02a290b4a5078e0aeb`
- validation SHA256:
  `c9395f11fb185c7cf73f57134be37b6a32c6364ce36e9cec3127d27c7b8bc5dd`
- TEST SHA256:
  `b336f67a7616a0d94ed6cc2049437928aafe01318e76a84abe6918320660f98c`
- contamination assertion:
  `exact 5M-char test segment absent from seed, train, and val text`

The launcher uses `--val-fraction 0.05263157894736842`, because the shorter
decimal `0.05263158` landed one character early in confirm-first checking.
After the 2026-07-09 reboot and cleanup, raw text8 and the generated shards
had been removed; the standard `text8.zip` source was reacquired, extracted to
`experiments\tiny_language_lab\corpus\text8\text8`, and regenerated artifacts
matched the hashes above exactly before seed 19 was relaunched.

Recipe:

- config `random_full`
- `85,097,499` trainable parameters, `0` frozen parameters
- L12, H12, D768, block 256, batch 8, grad accumulation 2
- RoPE, activation checkpointing, Muon optimizer, `--muon-lr 0.01`
- sampled in-run eval with `--eval-batches 16`
- shard-streamed text8 train data from
  `experiments\tiny_language_lab\corpus\text8_char_shards`
- checkpoints written directly to
  `C:\cassandra_runs\stage56_broadchar_checkpoints`

Primary command shape:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 50000 -Seed 7 -CheckpointEvery 5000
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 20000 -Seed 11 -CheckpointEvery 20000
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 20000 -Seed 19 -CheckpointEvery 20000
```

The 20k replicas used `-CheckpointEvery 20000` after the first disk-full
incident so that only the final step checkpoint was saved.

Run artifacts:

- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b50000_seed7.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b20000_seed11.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b20000_seed19.jsonl`
  and `.md`
- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b50000_seed7_text8_test.json`
  and `.md`
- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b20000_seed11_text8_test.json`
  and `.md`
- `experiments/tiny_language_lab/runs/stage56_broadchar_85m_b20000_seed19_text8_test.json`
  and `.md`
- preserved failed-save evidence:
  `stage56_broadchar_85m_b50000_seed7_save_error_20260708.*` and
  `stage56_broadchar_85m_b20000_seed19_save_error_20260708.*`

Final Stage 56 checkpoint keep-set:

- `C:\cassandra_runs\stage56_broadchar_checkpoints\stage56_broadchar_85m_b50000_seed7_random_full_seed7_step050000.pt`
- `C:\cassandra_runs\stage56_broadchar_checkpoints\stage56_broadchar_85m_b20000_seed11_random_full_seed11_step020000.pt`
- `C:\cassandra_runs\stage56_broadchar_checkpoints\stage56_broadchar_85m_b20000_seed19_random_full_seed19_step020000.pt`

The seed 19 no-suffix duplicate checkpoint also existed at closeout and can be
pruned later; the explicit step checkpoint above is the canonical final.

In-run sampled validation:

| Seed | Budget steps | Resume loaded | Resume step | Step-20k sampled val NLL | Final sampled report val NLL | Final sampled report bits/char | JSONL seconds | Peak CUDA MiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 50,000 | true | 45,000 | 1.004619 | 0.997197 | 1.438651 | 4,253.0077 | 1,593.4463 |
| 11 | 20,000 | false | 0 | 1.059212 | 1.034114 | 1.491912 | 17,502.0938 | 1,271.5742 |
| 19 | 20,000 | false | 0 | 1.023510 | 0.998736 | 1.440871 | 17,323.0456 | 1,271.5742 |

Instability guard: PASS. Each final sampled report NLL is better than its own
step-20k sampled validation NLL, so no run is worse than its own 20k value by
more than `0.05`.

Deterministic text8 TEST evaluation:

| Seed | Budget steps | Checkpoint | TEST bits/char | TEST NLL | Chars evaluated | Eval seconds |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 7 | 50,000 | step050000 | 1.485740 | 1.029836 | 4,999,936 | 188.9 |
| 11 | 20,000 | step020000 | 1.532627 | 1.062336 | 4,999,936 | 218.3 |
| 19 | 20,000 | step020000 | 1.529591 | 1.060232 | 4,999,936 | 196.2 |

Decision math:

- Arm A seed 7 TEST bits/char: `1.485740`, which is below the `1.70`
  CONFIRM line.
- Replica spread, seed 11 versus seed 19: `0.003035` bits/char, which is
  below the `0.10` stability limit.
- The KILL guard was not needed because Arm A did not land at or above
  `2.10`.

Operational incidents and cleanup:

- Seed 7 reached step 50,000 on the first attempt, but the checkpoint save
  failed with a PyTorch inline-container write-position error. The failed JSONL
  and Markdown artifacts were preserved with `_save_error_20260708` suffixes.
  Recovery resumed from the seed 7 step-45,000 checkpoint and completed the
  step-50,000 checkpoint cleanly.
- Seed 19 first reached step 20,000 on 2026-07-08 and matched the final sampled
  trajectory, but failed the final checkpoint save when C: hit zero free space.
  Its failed JSONL and Markdown artifacts were preserved with
  `_save_error_20260708` suffixes. A second seed 19 rerun was manually stopped
  at step 6,000 for the user-requested hardware pit stop and wrote no
  checkpoint. The 2026-07-09 post-reboot rerun completed cleanly.
- To recover disk headroom, Stage 56 intermediate checkpoints, Stage 56 smoke
  checkpoints, Stage 55 duplicate no-suffix checkpoints, and then Stage 55
  fp32 final training checkpoints were pruned. This follows the Phase 5 disk
  plan's preference for keeping durable JSONL, Markdown, manifests, and final
  Phase 5 working checkpoints while treating older fp32 training checkpoints as
  cache under pressure.

Interpretation:

The Stage 55 text8 gap was not evidence that the character substrate was
intrinsically unable to model broad text at this scale. With the same
character-level architecture and recipe trained directly on broad text8
train/valid data, TEST bits/char landed around `1.49` to `1.53` under the
registered chunked convention. This confirms the Stage 56 data-distribution
explanation and freezes the Phase 5 developmental experiment on the character
substrate unless Claude writes a later roadmap change. Stage 58 must still wait
for Claude's H024 document; the next Codex-owned workstream is Stage 57 Recipe
v2 gates.

## Stage 57 - Recipe v2 Gates

Date: 2026-07-09 local closeout

Outcome:

Stage 57 locks **Recipe v2 = Recipe v1 plus cosine LR warmdown, checkpoint
retention, fp16 model-only archive mode, and the vocab-union override; fp32
stays as the training precision.** Block 512 is recorded as a sizing input
only, not as a default recipe change.

Code changes:

- `cassandra_tiny_transformer.py` now accepts `--precision {fp32,bf16}`.
  bf16 wraps the training forward and loss in CUDA autocast; eval remains
  outside autocast.
- `--lr-schedule {constant,cosine}` and `--lr-final-frac` implement cosine
  warmdown. With the Stage 57 setting, Muon LR moves from `0.01` to `0.001`.
- `--checkpoint-keep N` prunes older sibling step checkpoints after a durable
  save.
- `--model-only-out` and `--model-only-dtype {fp16,bf16,fp32}` write an
  optimizer-free archive checkpoint.
- `--vocab-chars` and `--vocab-chars-file` let a checkpoint carry a supplied
  union alphabet while training on a subset corpus.

All gates used the Stage 56 text8 train/validation split unless noted,
`random_full`, RoPE, activation checkpointing, Muon `0.01`, sampled eval with
`--eval-batches 16`, and seed 7.

bf16 precision gate:

Acceptance line: adopt bf16 only if the 200-step sampled NLL is within `0.01`
of fp32 **and** throughput is at least `1.4x` fp32.

| Precision | Params | Steps | Final sampled val NLL | Delta vs fp32 | Seconds | Steps/sec | Throughput ratio | Peak CUDA MiB | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 25,247,771 | 200 | 1.663438 | 0.000000 | 56.6861 | 3.5282 | 1.0000 | 494.2148 | baseline |
| bf16 | 25,247,771 | 200 | 1.661443 | -0.001995 | 62.5827 | 3.1958 | 0.9058 | 495.4561 | REJECT |

bf16 satisfied the NLL side but failed the throughput side; it was slower than
fp32. Recipe v2 therefore keeps `--precision fp32`.

Cosine LR gate:

Acceptance line: adopt cosine if the 5000-step cosine arm's final sampled NLL
is at or below the constant arm's.

| LR schedule | Final factor | Params | Steps | Final sampled val NLL | Delta vs constant | Seconds | Steps/sec | Peak CUDA MiB | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| constant | 1.0 | 25,247,771 | 5,000 | 1.139756 | 0.000000 | 1,654.8820 | 3.0214 | 494.2148 | baseline |
| cosine | 0.1 | 25,247,771 | 5,000 | 1.071579 | -0.068177 | 1,675.4442 | 2.9843 | 494.2148 | ADOPT |

Cosine clearly won the final sampled NLL gate. Recipe v2 adopts
`--lr-schedule cosine --lr-final-frac 0.1`.

Checkpoint and model-only archive smoke:

Result: PASS. In a direct tiny CUDA smoke, the step-2 save pruned step 1 under
`--checkpoint-keep 1`; resume from step 2 to step 3 loaded successfully, then
pruned step 2 after saving step 3. The fp16 model-only archive had
`optimizer_state = None`, archive metadata
`{"model_only": true, "dtype": "fp16"}`, and all floating model tensors stored
as `torch.float16`.

Vocab-union smoke:

Result: PASS. A small text8-trained checkpoint was trained from
`corpus\text8_char_shards\train_0000.txt` with the 33-character TinyStories
union vocab:

```text
\n ,.abcdefghijklmnopqrstuvwxyz!?'
```

The checkpoint
`C:\cassandra_runs\stage57_vocab_smoke_checkpoints\checkpoint_step000005.pt`
reported `vocab_size=33` and `vocab_chars_override=true`. It then encoded all
`1,500,001` characters of
`experiments\tiny_language_lab\corpus\tinystories_char_shards\val.txt` and
scored a 49,984-character chunked sample without an out-of-vocab error
(`4.390179` bits/char on that smoke sample). Recipe v2 adopts the vocab union
override for Phase 5 runs that must be scored across text8 and TinyStories.

Block-512 timing row:

Decision role: timing and VRAM input for H024 sizing only.

| Model | Block | Batch | Accum | Precision | Steps | Final sampled val NLL | Seconds | Steps/sec | Peak CUDA MiB |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 85.10M L12 H12 D768 | 512 | 8 | 2 | fp32 | 200 | 1.561000 | 306.5595 | 0.6524 | 1,645.0039 |

Locked Recipe v2 command fragment:

```powershell
--precision fp32 `
--lr-schedule cosine --lr-final-frac 0.1 `
--checkpoint-keep 1 `
--vocab-chars "<phase-specific union alphabet when cross-corpus scoring is required>"
```

`--model-only-out <path> --model-only-dtype fp16` is adopted as an archival
mode, not a default training flag. Block 512 remains a measured sizing option
for Claude's H024 budget arithmetic; default block size stays 256 unless H024
explicitly chooses otherwise.

Artifacts:

- `runs/stage57_bf16_gate_fp32_25m_b200_seed7.jsonl` and `.md`
- `runs/stage57_bf16_gate_bf16_25m_b200_seed7.jsonl` and `.md`
- `runs/stage57_cosine_gate_constant_25m_b5000_seed7.jsonl` and `.md`
- `runs/stage57_cosine_gate_cosine_25m_b5000_seed7.jsonl` and `.md`
- `runs/stage57_block512_timing_85m_b200_seed7.jsonl` and `.md`
- smoke checkpoints under `C:\cassandra_runs\stage57_smoke_checkpoints` and
  `C:\cassandra_runs\stage57_vocab_smoke_checkpoints`

## Phase 5 Behavior Probe - Letters-only Zero-shot Copy

Date: 2026-07-09 local closeout

Outcome:

The behavior axis stays closed. The Phase 4 Stage 55 flagship scored
`0.060547` constrained-choice copy accuracy on the letters-only
memorization-proof probe, versus chance `0.062500` and the registered reopen
line `0.162500`.

Scope:

This was eval-only: no training, no checkpoint writes. Stage 42's original
memorization-proof copy corpus used digits, and the Stage 55 33-character
TinyStories vocabulary has neither digits nor `=`. The Phase 5 probe therefore
uses the same verified key/answer identity construction with letters-only
markers:

- payload alphabet: `abcdefghijklmnop`
- key marker: `key `
- answer marker: `answer `
- case ids: fixed-width base-26 letter sequences such as `aaaa`
- probe corpus: no digits and no equals sign

Commands:

```powershell
python .\experiments\tiny_language_lab\make_letters_copy_probe.py --lines 1024 --seed 20260709
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py --device auto --max-cases 1024
```

Result:

| Metric | Value |
| --- | ---: |
| Checkpoint parameters | 201,609,249 |
| Cases scored | 1,024 |
| Choice candidates | 16 |
| Payload count per candidate | 64 |
| Chance accuracy | 0.062500 |
| Reopen threshold | 0.162500 |
| Constrained-choice copy accuracy | 0.060547 |
| Constrained-choice MRR | 0.217769 |
| Full-vocabulary argmax accuracy | 0.000000 |
| Copy NLL | 3.976691 |
| Eval seconds | 31.0125 |
| Verdict | KEEP_BEHAVIOR_AXIS_CLOSED |

Code and artifact notes:

- `copy_answer_probe` now accepts a backwards-compatible `key_marker`
  parameter. Existing `key=` corpora remain the default.
- `flagship_eval_lib.FINAL_CHECKPOINTS["flagship_200m_50k_seed7"]` now points
  to the repo-local Phase 4 artifact checkpoint, because the old
  `C:\cassandra_runs` Stage 55 copy was pruned during Stage 56 disk recovery.
- Probe corpus and metadata:
  `corpus/letters_copy_probe_seed.txt` and
  `corpus/letters_copy_probe_seed.meta.json`
- Run artifacts:
  `runs/phase5_behavior_letters_probe.json` and
  `runs/phase5_behavior_letters_probe.md`

Interpretation:

The flagship does not show above-chance zero-shot copy behavior under the
letters-only compatibility probe. Per the Phase 5 success criteria, this keeps
the behavior axis closed. The next Codex-owned workstream is D2 open-source
preparation. Stage 58 remains blocked until Claude writes H024.

## Phase 5 D2 Open-source Prep Status

Date: 2026-07-09 local status

Codex completed the release-prep items that do not require user sign-off:

- corpus payloads removed from the Git index with `git rm --cached`, while
  local files remain local and `.meta.json` provenance remains trackable
- `.gitignore` expanded to ignore generated corpus payloads while allowing
  `.meta.json`
- `.githooks/pre-commit` added to block staged additions over 50 MiB, with
  `.gitattributes` pinning hook line endings to LF
- licensing due-diligence notes drafted at
  `docs/phase5-licensing-notes.md`
- flagship model-card draft added at
  `docs/phase5-model-card-draft.md`
- fp16 model-only exports written for the three Stage 56 final step
  checkpoints under `C:\cassandra_runs\phase5_model_only_exports`
- load-back audit confirmed the fp16 archives have no optimizer state and all
  floating model tensors are `torch.float16`

The D2 status and exact export hashes are recorded in
`docs/phase5-d2-prep-status.md`.

Explicitly not done: history surgery, force-push/public push, license choice,
root `LICENSE`/`NOTICE`/`CITATION.cff`, Hub upload, and Stage 58. History
surgery remains gated on explicit user sign-off in a live session. Stage 58
remains gated on Claude writing H024.

## Environment Note - GPU Transition Audit

Date: 2026-06-23

After Stage 35, CUDA became the default for real comparison matrices on the
laptop RTX 4070. The audit is recorded in
`docs/gpu-transition-validity-audit.md`. Exact zero-step full evaluation matched
between CPU and CUDA to displayed precision on the tiny probe, but sampled runs
are not bitwise comparable across devices because the trainer uses
`torch.Generator(device=device)`.

Conclusion: the GPU switch does not resurrect the killed hypotheses. Strong NLL
and behavior kills remain local evidence. CPU wall-clock claims are historical
notes, and past tiny sampled-NLL margins should be rerun under CUDA before they
are used as hard GPU-era decisions.

## Stage 58 Pit Stop - Cold-arm Checkpoint Preservation

Date: 2026-07-10 06:04 local

User requested a hardware pit stop. Codex stopped the active Stage 58 cold-run
Python child and launcher, then verified no `stage58-cold` or
`cassandra_compare.py` process remained. C: free space after stop was
`57012117504` bytes (`53.097` GiB).

Detailed pit-stop manifest:
`experiments/tiny_language_lab/runs/stage58_pitstop_20260710.md`.

Checkpoint directory: `C:\cassandra_runs\stage58_dev_cold_checkpoints`

- `stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step005000.pt`:
  `685649083` bytes, last write `2026-07-10 04:26:41 +03:00`, saved step
  `5000`, formation passes `10000`, saved `args_resume_from = null`
- `stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step010000.pt`:
  `685649147` bytes, last write `2026-07-10 05:51:51 +03:00`, saved step
  `10000`, formation passes `20000`, saved `args_resume_from = null`
- `stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step015000.pt`:
  `685649147` bytes, last write `2026-07-10 02:18:25 +03:00`, saved step
  `15000`, formation passes `30000`, saved `args_resume_from = null`

No `.tmp`, `tmp`, or `partial` checkpoint files were present before or after
stopping.

Lineage guard: the `20260710_030054` launcher logged a `resume_from` path
pointing at `step015000.pt`, but its recorded Python command omitted
`--resume-from`. The newer `step005000.pt` and `step010000.pt` files therefore
belong to a restarted cold-run lineage, not a continuation from `step015000.pt`.
Before resuming Stage 58, audit/fix visible-launcher resume propagation and
choose the intended lineage explicitly.

Follow-up on 2026-07-14 00:16 local:

- Codex fixed `run_phase5_visible.ps1` so the `stage58-cold` branch forwards
  `-ResumeFrom` into the Stage 58 argument builder. Dry-run verification showed
  the generated Python command now includes `--resume-from`.
- A one-step actual resume smoke ran from
  `C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step015000.pt`
  with `-Budget 15001`, `-CheckpointEvery 0`, and `-LrTotalSteps 42000`.
  The report recorded `resume_loaded = true`, `resume_step = 15000`,
  `training_target_steps = 15001`, and `lr_last_factor = 0.74523685`.
- Smoke artifacts:
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b15001_seed7.jsonl`,
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b15001_seed7.md`,
  and `experiments/tiny_language_lab/runs/phase5_stage58-cold_20260714_001541.log`.
- The smoke wrote an isolated final checkpoint,
  `C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b15001_seed7_random_full_seed7.pt`.
  It did not overwrite the earlier `step005000.pt`, `step010000.pt`, or
  `step015000.pt` checkpoint files.
- TinyStories audit: `tinystories_char_seed.meta.json` records the full
  `494094421` character prep and `tinystories_char_shards_500mb`, but the raw
  source file and full shard directory are not currently present on disk. The
  old default `tinystories_char_shards` directory is only `10078510` bytes.
  Codex changed the Stage 58 visible launcher default TinyStories shard path to
  `tinystories_char_shards_500mb` and split readiness checks so COLD can run on
  text8 alone, while CURRICULUM and MIXTURE refuse to run without the full
  TinyStories corpus.

## Stage 58 COLD Arm - Completed Baseline

Date: 2026-07-14 local

Codex resumed the clean Stage 58 COLD lineage from
`C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step015000.pt`
after the launcher resume fix and completed the 42k-step COLD baseline.

Visible launcher command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage58-cold -Budget 42000 -Seed 7 -CheckpointEvery 5000 -LrTotalSteps 42000 -ResumeFrom C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step015000.pt
```

Run artifacts:

- log:
  `experiments/tiny_language_lab/runs/phase5_stage58-cold_20260714_002318.log`
- sampled-run JSONL:
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b42000_seed7.jsonl`
- sampled-run summary:
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b42000_seed7.md`
- final checkpoint:
  `C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt`

Final sampled in-run result:

- `resume_loaded = true`
- `resume_step = 15000`
- `steps = 42000`
- `formation_forward_passes = 84000`
- `val_nll = 0.884350`
- `bits = 1.275847`
- `seconds = 36849.2`
- `peak_cuda_mib = 1593.6689`

Checkpoint progression preserved under
`C:\cassandra_runs\stage58_dev_cold_checkpoints`: `step020000.pt`,
`step025000.pt`, `step030000.pt`, `step035000.pt`, `step040000.pt`, and the
final unsuffixed 42k checkpoint. No partial or temporary checkpoint files were
present after completion.

Deterministic held-out text8 TEST evaluation:

```powershell
python .\experiments\tiny_language_lab\eval_text8.py --device cuda --split test --checkpoint C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt --checkpoint-name stage58_dev_cold_85m_b42000_seed7 --out-stem .\experiments\tiny_language_lab\runs\stage58_dev_cold_85m_b42000_seed7_text8_test
```

- output JSON:
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b42000_seed7_text8_test.json`
- output summary:
  `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b42000_seed7_text8_test.md`
- `chars_evaluated = 4999936`
- `windows = 19531`
- `nll = 0.9408209245512913`
- `bits_per_char = 1.3573176822147175`
- `seconds = 236.3`

Interpretation:

The COLD arm is now evidence-complete as the Stage 58 from-scratch text8
baseline. Its final sampled validation NLL improved from the 25k and 40k
checkpoints, so the instability guard does not trigger. Stage 58 overall is not
complete yet: the next CURRICULUM and MIXTURE arms still require rebuilding the
full `tinystories_char_shards_500mb` corpus from the recorded 500MB TinyStories
source recipe before launch.

## Stage 58 Corpus Recovery and Mixture Prep

Date: 2026-07-14 local

Codex recovered the full TinyStories character corpus required for Stage 58
CURRICULUM and MIXTURE. The downloader wrote:

- source:
  `experiments/tiny_language_lab/corpus/tinystories_raw/TinyStories-train.head500mb.txt`
- download metadata:
  `experiments/tiny_language_lab/corpus/tinystories_raw/TinyStories-train.head500mb.download.json`
- download log:
  `experiments/tiny_language_lab/runs/phase5_tinystories_500mb_download_20260714.log`
- downloaded bytes: `500000000`

The preparation command regenerated
`experiments/tiny_language_lab/corpus/tinystories_char_seed.txt`,
`experiments/tiny_language_lab/corpus/tinystories_char_seed.meta.json`, and
`experiments/tiny_language_lab/corpus/tinystories_char_shards_500mb`.

Preparation evidence:

- clean prep log:
  `experiments/tiny_language_lab/runs/phase5_tinystories_500mb_prepare_clean_20260714.log`
- `input_chars = 499616200`
- `input_records = 1`
- `normalized_chars = 494094421`
- `train_chars_at_recorded_block_size = 419980257`
- `val_chars_at_recorded_block_size = 74114164`
- `observed_vocab_size = 33`
- shard count: `42` train shards plus `val.txt`

Codex also fixed `make_tinystories_corpus.py` so directory discovery ignores
`*.download.json` sidecar metadata files. The first prep pass had included the
download sidecar as a zero-record `source_file`; the clean rerun lists only the
true 500MB text source.

Stage 58 mixture prep then ran through the visible launcher at the actual 42k
arm budget:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage58-prep -Budget 42000
```

Mixture evidence:

- prep log:
  `experiments/tiny_language_lab/runs/phase5_stage58-prep_20260714_112014.log`
- launcher log:
  `experiments/tiny_language_lab/runs/phase5_stage58-prep_20260714_112014_launcher.log`
- metadata:
  `experiments/tiny_language_lab/corpus/mixture_char_shards.meta.json`
- shard directory:
  `experiments/tiny_language_lab/corpus/mixture_char_shards`
- `requested_total_chars = 172032000`
- `written_chars = 172032000`
- `written_tiny_chars = 43008000`
- `written_broad_chars = 129024000`
- `tiny_fraction = 0.25`
- `broad_fraction = 0.75`
- `tiny_reader_wraps = 0`
- `broad_reader_wraps = 1`
- mixture shards: `18`
- `sha256 = c594e57a5dc67aa6e6ee527d561671b8a6997a3816e4b4dc6dac56e0121f09db`

Codex fixed `make_mixture_shards.py` so its default TinyStories source points
at `tinystories_char_shards_500mb` and its source summaries distinguish on-disk
bytes from normalized text characters. Dry-run launcher checks confirmed:

- CURRICULUM phase 1 will run `12500` TinyStories steps with
  `--lr-total-steps 42000`.
- CURRICULUM phase 2 will resume the phase-1 checkpoint with `--steps 29500`,
  `--resume-steps-additional`, and `--lr-total-steps 42000`, producing
  `stage58_dev_curriculum_phase2_85m_b42000_seed7.*` artifacts.
- MIXTURE will run `42000` steps over the generated mixture shard directory.

Remaining Stage 58 work after prep: run CURRICULUM, run deterministic text8
TEST and TinyStories retention evals for the curriculum checkpoint series, run
MIXTURE, run the same deterministic evals, then record the COLD versus
CURRICULUM versus MIXTURE decision.

## Stage 58 CURRICULUM Phase 1 - TinyStories Childhood Complete

Date: 2026-07-14 local

Codex launched CURRICULUM phase 1 through the visible Phase 5 launcher with the
42k global cosine horizon:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage58-curriculum-phase1 -Budget 42000 -Phase1Steps 12500 -LrTotalSteps 42000 -Seed 7 -CheckpointEvery 5000
```

Run artifacts:

- log:
  `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase1_20260714_112332.log`
- launcher log:
  `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase1_20260714_112332_launcher.log`
- sampled-run JSONL:
  `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase1_85m_b12500_seed7.jsonl`
- sampled-run summary:
  `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase1_85m_b12500_seed7.md`
- keep-awake log:
  `experiments/tiny_language_lab/runs/phase5_keep_awake_stage58_curriculum_phase1_python_20260714_112413.log`

Checkpoint directory:
`C:\cassandra_runs\stage58_dev_curriculum_checkpoints`

- `stage58_dev_curriculum_phase1_85m_b12500_seed7_random_full_seed7_step005000.pt`
- `stage58_dev_curriculum_phase1_85m_b12500_seed7_random_full_seed7_step010000.pt`
- `stage58_dev_curriculum_phase1_85m_b12500_seed7_random_full_seed7.pt`

Final phase-1 result:

- `steps = 12500`
- `formation_forward_passes = 25000`
- `lr_total_steps = 42000`
- `lr_last_factor = 0.81723811`
- `val_nll = 0.631930`
- `bits = 0.911683`
- `seconds = 14257.1859`
- `peak_cuda_mib = 1431.7617`

Sampled TinyStories validation curve:

- step 0: `val_nll = 3.682602`, `bits = 5.312871`
- step 5000: `val_nll = 0.694158`, `bits = 1.001458`
- step 10000: `val_nll = 0.630601`, `bits = 0.909764`
- final step 12500: `val_nll = 0.631930`, `bits = 0.911683`

Checkpoint metadata verification confirmed the final phase-1 checkpoint saved
`step = 12500`, `formation_forward_passes = 25000`, `args.steps = 12500`,
`args.lr_total_steps = 42000`, and the
`tinystories_char_shards_500mb` train shard directory.

Codex then ran a one-step phase-2 resume smoke on text8 from the final phase-1
checkpoint:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage58-curriculum-phase2 -Budget 12501 -Phase1Steps 12500 -LrTotalSteps 42000 -Seed 7 -ResumeFrom C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase1_85m_b12500_seed7_random_full_seed7.pt -CheckpointEvery 0
```

Smoke artifacts:

- log:
  `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260714_153939.log`
- sampled-run JSONL:
  `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase2_85m_b12501_seed7.jsonl`
- sampled-run summary:
  `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase2_85m_b12501_seed7.md`
- isolated smoke checkpoint:
  `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b12501_seed7_random_full_seed7.pt`

The smoke recorded `resume_loaded = true`, `resume_step = 12500`,
`resume_steps_additional = true`, `training_target_steps = 12501`,
`args.steps = 1`, and `args.lr_total_steps = 42000`. The resumed text8 eval at
step 12500 was `val_nll = 2.103213`, `bits = 3.034295`; after one broad step
the smoke ended at `val_nll = 1.998551`, `bits = 2.883299`. This verifies the
real phase-2 path before the full continuation run.

## Stage 58 CURRICULUM Phase 2 Pit Stop

Date: 2026-07-14 17:32 local

User requested a pit stop during the full CURRICULUM phase-2 text8
continuation. Codex stopped the active Python training process (`54556`), the
visible launcher (`55056`), and the keep-awake watcher (`54640`). GPU compute
was idle afterward, and C: free space was `166859948032` bytes.

Pit-stop manifest:
`experiments/tiny_language_lab/runs/stage58_curriculum_phase2_pitstop_20260714_1732.md`.

Active run artifacts:

- run log:
  `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260714_154205.log`
- launcher log:
  `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260714_154205_launcher.log`
- keep-awake log:
  `experiments/tiny_language_lab/runs/phase5_keep_awake_stage58_curriculum_phase2_python_20260714_154204.log`

The full phase-2 JSONL and summary
(`stage58_dev_curriculum_phase2_85m_b42000_seed7.*`) were not written because
the run was intentionally stopped before completion.

Last durable resume point:

`C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step015000.pt`

- bytes: `685683117`
- saved step: `15000`
- formation forward passes: `30000`
- `args.steps = 29500`
- `args.lr_total_steps = 42000`
- `args.resume_steps_additional = true`
- `args.resume_from` points to the final phase-1 checkpoint
- checkpoint loss curve includes step 15000:
  `train_nll = 1.093804`, `val_nll = 1.078918`

The active log later reached `step = 17000/42000`, but no checkpoint after step
15000 was written. No `tmp`, `partial`, or `.part` checkpoint files were
present after stopping. The MÜŞAHİT one-shot skip flag was removed after the GPU
run stopped.

Resume safety fix:

Codex patched `run_phase5_visible.ps1` so `stage58-curriculum-phase2` treats a
resume from the phase-1 checkpoint differently from a resume from a phase-2
checkpoint. Dry-runs now verify:

- phase-1 checkpoint resume emits `--steps 29500 --resume-steps-additional`
- preserved phase-2 `step015000.pt` resume emits `--steps 42000` without
  `--resume-steps-additional`

The safe continuation command is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage58-curriculum-phase2 -Budget 42000 -Phase1Steps 12500 -LrTotalSteps 42000 -Seed 7 -ResumeFrom C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step015000.pt -CheckpointEvery 5000
```

## Stage 58 Dose-Matched MIXTURE Correction and Current Status

Date: 2026-07-18 local

The initial Stage 58 MIXTURE shards used the original 1:3 TinyStories:broad
ratio. That was appropriate to the provisional 50,000-step arithmetic, but the
actual Stage 58 budget is 42,000 steps and CURRICULUM phase 1 had already used
12,500 steps. The original untrained mixture would therefore have contained
43,008,000 TinyStories characters, while CURRICULUM contained 51,200,000. No
model trained on the preliminary mixture, so no result was invalidated.

Codex corrected `run_phase5_visible.ps1` to derive mixture weights from
`Phase1Steps : (Budget - Phase1Steps)`. The 42,000-step dry run now emits
`--tiny-weight 25 --broad-weight 59`. The regenerated untrained mixture has:

- total: `172,032,000` characters
- TinyStories: `51,200,000` characters (`0.297619`)
- broad text8: `120,832,000` characters (`0.702381`)
- shards: `18`
- SHA-256: `874f8212a6436a132f57a6986dbdfc8253344e4c9d664109a30149d6104c7f95`

The superseded, untrained 1:3 preparation is preserved in the Phase 5 mid-run
report as ratio `1:3`, `43,008,000` TinyStories characters, and SHA-256
`c594e57a5dc67aa6e6ee527d561671b8a6997a3816e4b4dc6dac56e0121f09db`.

CURRICULUM phase 2 restarted cleanly from the audited durable step-15,000
checkpoint on 2026-07-18. It restored `formation_forward_passes = 30000` and
recorded the expected step-15,000 broad validation NLL `1.078918` before
continuing. A process-bound keep-awake watcher asserts both system and display
activity for the trainer PID. The Stage 58 full comparison remains pending until
CURRICULUM and MIXTURE each receive deterministic text8 TEST evaluation.

See `docs/phase5-developmental-midrun-report.md` for the Phase 4 to Phase 5
comparison, current three-arm ledger, decision rules, and next evidence order.
## Stage 58 Retention Corpus Guard

Date: 2026-07-18 local

Before any Stage 58 retention evaluation, Codex changed
`eval_tinystories_retention.py` to default to the full current curriculum
validation corpus:

`experiments/tiny_language_lab/corpus/tinystories_char_shards_500mb/val.txt`

The previous default targeted the obsolete small-shard validation file. The
current default file is `74,690,293` on-disk characters, matching the full
500 MB TinyStories preparation used by CURRICULUM. Python AST parsing passed;
the active GPU trainer was not used for this source-level verification. Future
retention commands may still pass `--corpus` explicitly, but their default is
now correct and reproducible.
## Stage 58 CURRICULUM Phase 2 Step-20,000 Checkpoint

Date: 2026-07-18 local

The visible CURRICULUM phase-2 continuation reached its first new durable
checkpoint after the audited step-15,000 resume point:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260718_200021.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step020000.pt`
- bytes: `685683245`
- SHA-256: `62F54CC84DF9FAA870C642C8DD98F75B3B24FBE47B0D5B7C7E518A89BDA3D8CD`
- metadata: `step = 20000`, `formation_forward_passes = 40000`, seed `7`,
  fp32, cosine schedule, `lr_total_steps = 42000`, and resume parent
  `..._step015000.pt` with `resume_steps_additional = false`.

The sampled broad-text evaluation at step 20,000 was `val_nll = 1.006270`
(`1.451740` bits/char), improving by `0.072648` NLL from the resumed
step-15,000 value `1.078918`. This is an interim monitoring result only. The
registered instability guard remains the final value against this run's own
step-25,000 sampled value, so no Stage 58 decision is read from this checkpoint.

## Stage 58 CURRICULUM Phase 2 Step-25,000 Guard Checkpoint

Date: 2026-07-18 local

The registered CURRICULUM broad-phase guard point completed under the same
visible continuation:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260718_200021.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step025000.pt`
- bytes: `685683245`
- SHA-256: `9EF44D15539BD5D2F220F36A7F295F6912549E64591822D6F0EA8D3109FFD811`
- metadata: `step = 25000`, `formation_forward_passes = 50000`, seed `7`,
  fp32, cosine schedule, `lr_total_steps = 42000`, and the audited
  `..._step015000.pt` resume parent with `resume_steps_additional = false`.

The sampled broad-text evaluation was `val_nll = 0.985283` (`1.421462`
bits/char), an additional `0.020987` NLL improvement from step 20,000. The
H024 instability guard now has a concrete final line: a final sampled broad
validation NLL greater than `1.035283` would be more than `0.05` worse than
this step-25,000 value and would make the arm inconclusive pending repair.
The continuation had already entered step 26,000 after this checkpoint.

## Stage 58 CURRICULUM Phase 2 Step-30,000 Checkpoint

Date: 2026-07-18 local

The visible continuation advanced through its next durable broad-text point:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260718_200021.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step030000.pt`
- bytes: `685683309`
- SHA-256: `B443CFB7036BF59B3DC7B5A6B6F927AA0401CBE27B42A08929162ABDEB5D23CB`
- metadata: `step = 30000`, `formation_forward_passes = 60000`, seed `7`,
  fp32, cosine schedule, `lr_total_steps = 42000`, and the audited
  `..._step015000.pt` resume parent with `resume_steps_additional = false`.

At step 30,000, sampled broad validation reached `val_nll = 0.914095`
(`1.318760` bits/char), improving by `0.071188` NLL from the registered
step-25,000 guard point. The final stability line remains `1.035283` NLL; this
checkpoint is another interim monitoring result, not the deterministic TEST
decision.

## Stage 58 CURRICULUM Phase 2 Step-35,000 Checkpoint

Date: 2026-07-19 local

The penultimate scheduled phase-2 checkpoint completed under the same visible
continuation:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260718_200021.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step035000.pt`
- bytes: `685683373`
- SHA-256: `6F0B714377F4AB3B5470F23779975C97F085B4F4C6A5CA9CD8D3F52E9DA922DE`
- metadata: `step = 35000`, `formation_forward_passes = 70000`, seed `7`,
  fp32, cosine schedule, `lr_total_steps = 42000`, and the audited
  `..._step015000.pt` resume parent with `resume_steps_additional = false`.

At step 35,000, sampled broad validation reached `val_nll = 0.905480`
(`1.306332` bits/char), a further `0.008615` NLL improvement from step 30,000
and safely below the registered `1.035283` instability threshold. The final
checkpoint and deterministic text8 TEST evaluation remain required.

## Stage 58 CURRICULUM Phase 2 Step-40,000 Checkpoint

Date: 2026-07-19 local

The final scheduled pre-closeout checkpoint completed under the same visible
continuation:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-curriculum-phase2_20260718_200021.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7_step040000.pt`
- bytes: `685683373`
- SHA-256: `F694A885E3746E972ABC50EFFE4408E098270CB459FAD75C75FE2C7CAFA1E772`
- metadata: `step = 40000`, `formation_forward_passes = 80000`, seed `7`,
  fp32, cosine schedule, `lr_total_steps = 42000`, and the audited
  `..._step015000.pt` resume parent with `resume_steps_additional = false`.

At step 40,000, sampled broad validation reached `val_nll = 0.885336`
(`1.277270` bits/char), another `0.020144` NLL improvement from step 35,000
and safely below the registered `1.035283` instability threshold. The final
42,000-step checkpoint, deterministic text8 TEST score, and retention curve
remain required before the CURRICULUM arm is evidence-complete.

## Stage 58 CURRICULUM Seed-7 Arm Complete

Date: 2026-07-19 local

The Stage 58 CURRICULUM seed-7 arm is evidence-complete. It ran phase 1 on
TinyStories for 12,500 steps, then resumed the audited broad-text phase-2
checkpoint at step 15,000 through the common step-42,000 budget. The final
checkpoint is:

`C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7.pt`

- bytes: `685649217`
- SHA-256: `F8A734503143764141C96924ED233F12FBDD9734D984E0DF7886FC5F6DC71CCD`
- final sampled broad validation: `NLL = 0.920510`, `1.328015` bits/char
- H024 guard: passes; the registered limit was `NLL = 1.035283`
- run evidence: `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase2_85m_b42000_seed7.jsonl` and `.md`

Deterministic held-out evaluation used the final checkpoint on text8 TEST:

```powershell
python .\experiments\tiny_language_lab\eval_text8.py --device cuda --split test --checkpoint C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b42000_seed7_random_full_seed7.pt --checkpoint-name stage58_dev_curriculum_phase2_85m_b42000_seed7 --out-stem .\experiments\tiny_language_lab\runs\stage58_dev_curriculum_phase2_85m_b42000_seed7_text8_test
```

It measured `NLL = 0.94435351273717`, `1.3624141296719878` bits/char, across
`4,999,936` held-out characters. The paired COLD score is
`1.3573176822147175`, so the registered primary delta
`CURRICULUM - COLD = +0.005096447457270337` bits/char. This is inside H024's
`[-0.05, +0.05]` practical E-null band. It is a provisional seed-7 reading,
not the Phase 5 developmental verdict: the active MIXTURE arm and required
reduced-budget replicas remain outstanding.

Retention used the uniform full-source TinyStories validation sample at
`1,499,904` characters per point. CURRICULUM went from `0.877782` bits/char at
the end of its TinyStories phase to `3.188760` at broad step 15,000 and
`3.529069` at step 42,000. COLD ended at `3.556502` bits/char. These curves
are descriptive forgetting evidence and do not replace the text8 TEST primary.
The durable reports are
`experiments/tiny_language_lab/runs/stage58_dev_curriculum_retention.{json,md}`
and `experiments/tiny_language_lab/runs/stage58_dev_cold_retention.{json,md}`.

## Stage 58 MIXTURE Step-5,000 Checkpoint

Date: 2026-07-19 local

The visible, dose-matched MIXTURE arm reached its first durable checkpoint
without an interruption:

- log: `experiments/tiny_language_lab/runs/phase5_stage58-mixture_20260719_014620.log`
- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step005000.pt`
- bytes: `685650729`
- SHA-256: `9DC7CF9ABE077B269EF3A297973CBF3DAFBA7B44CD2CF2EBB6D0D03640F49497`
- run recipe: 85,106,721 parameters, seed 7, fp32, Muon, 42,000-step cosine,
  33-character union vocabulary, and corrected 25:59 TinyStories:broad shards.

At step 5,000, sampled broad validation was `NLL = 1.136304`
(`1.639340` bits/char). This is monitoring evidence only. The decision remains
the final deterministic chunked text8 TEST evaluation, with the H024 stability
guard applied to the final sampled broad-validation point.

## Stage 58 MIXTURE Step-10,000 Checkpoint

Date: 2026-07-19 local

The same visible MIXTURE run reached its second durable checkpoint:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step010000.pt`
- bytes: `685650793`
- SHA-256: `F8BB94BEA11EF7475F602415FC6CBAAEE19AB68AB983569C0BBC26F963EB4AEC`
- sampled broad validation: `NLL = 1.066020` (`1.537942` bits/char), improving
  by `0.070284` NLL from step 5,000.

This remains in-run monitoring only. The durable checkpoint is retained on C:
and the process-bound helper continues to assert system and display activity.

## Stage 58 MIXTURE Step-15,000 Checkpoint

Date: 2026-07-19 local

The visible MIXTURE run reached its third durable checkpoint:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step015000.pt`
- bytes: `685650857`
- SHA-256: `839281D62255455BA676C1E2810AD70ACC4D58BE47A4A368644E1E4533A64377`
- sampled broad validation: `NLL = 1.066921` (`1.539241` bits/char).

The final H024 instability check compares the final sampled broad NLL with the
registered step-25,000 value, so this step-15,000 reading is monitoring only.

## Stage 58 MIXTURE Step-20,000 Checkpoint

Date: 2026-07-19 local

The visible MIXTURE run reached its fourth durable checkpoint:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step020000.pt`
- bytes: `685650857`
- SHA-256: `9BC1CCF91A97DEC5D77A3B9D74D08EF802BD52D8E51DC11BFF4F79623285D685`
- sampled broad validation: `NLL = 1.001131` (`1.444326` bits/char), improving
  by `0.065790` NLL from step 15,000.

This is interim monitoring evidence. The registered stability comparison
remains final sampled broad validation against the step-25,000 point.

## Stage 58 MIXTURE Step-25,000 Stability Checkpoint

Date: 2026-07-19 local

The registered MIXTURE mid-budget stability point completed under the same
visible run:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step025000.pt`
- bytes: `685650921`
- SHA-256: `297F366EDB6E53DF8A824E00D6A27F7EC9ECDEE7622D75506D74BA7C55237868`
- sampled broad validation: `NLL = 1.007650` (`1.453731` bits/char).

H024's stability guard is now concrete: final sampled broad validation must not
exceed `NLL = 1.057650`, which is this point plus `0.05`. This checkpoint is
not itself a developmental verdict; final deterministic text8 TEST remains the
primary evidence.

## Stage 58 MIXTURE Step-30,000 Checkpoint

Date: 2026-07-19 local

The visible MIXTURE run reached its next durable checkpoint:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step030000.pt`
- bytes: `685650985`
- SHA-256: `2CAD7AA59C831299133202553C7D2BC502B1B6AB6A741C8F79481495D19CCB84`
- sampled broad validation: `NLL = 0.960756` (`1.386078` bits/char), improving
  by `0.046894` NLL from the registered step-25,000 stability point.

This is interim monitoring evidence only. The final checkpoint must still pass
the `1.057650` sampled-NLL stability line before its deterministic text8 TEST
score can be read.

## Stage 58 MIXTURE Step-35,000 Checkpoint

Date: 2026-07-19 local

The visible MIXTURE run reached its penultimate scheduled checkpoint:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step035000.pt`
- bytes: `685650985`
- SHA-256: `9B915C5F51C8828E49F65DCBC29EE0033BE2FF002A9E5A748E637E7E30B7CE93`
- sampled broad validation: `NLL = 0.950141` (`1.370764` bits/char), improving
  by `0.010615` NLL from step 30,000.

The final sampled broad validation must remain at or below `1.057650` NLL to
pass H024's stability guard before TEST evaluation begins.

## Stage 58 MIXTURE Step-40,000 Checkpoint

Date: 2026-07-19 local

The final scheduled pre-closeout MIXTURE checkpoint completed:

- checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step040000.pt`
- bytes: `685651049`
- SHA-256: `DC5C0DCD79CEBB48AC328D6867C3D50F9DEA052D37B6855B10C6213CD80322E6`
- sampled broad validation: `NLL = 0.917782` (`1.324080` bits/char), improving
  by `0.032359` NLL from step 35,000.

The final 42,000-step checkpoint, final sampled broad validation, deterministic
text8 TEST score, and TinyStories retention curve are still required before
MIXTURE is evidence-complete.

## Stage 58 MIXTURE Seed-7 Arm Training Complete

Date: 2026-07-19 local

The dose-matched MIXTURE arm completed all 42,000 optimizer steps with visible
launch, exit code `0`, and no interruption. Final training evidence:

- final checkpoint: `C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7.pt`
- bytes: `685644413`
- SHA-256: `45D312F976916D86F8021D21C1B45987860FB18BF58B7020E9EE2B6546EDC707`
- final sampled broad validation: `NLL = 0.905836`, `1.306845` bits/char
- registered step-25,000 NLL: `1.007650`; stability ceiling: `1.057650`
- guard result: PASS, with final NLL `0.151814` below the ceiling
- final run reports: `experiments/tiny_language_lab/runs/stage58_dev_mixture_85m_b42000_seed7.{jsonl,md}`

The arm used 85,106,721 parameters, seed 7, fp32, Muon, a continuous 42,000-step
cosine schedule, 33-character union vocabulary, and the corrected 25:59
TinyStories:broad mixture shards. It is now eligible for deterministic text8
TEST and full-source TinyStories retention evaluation; neither result is
inferred from the in-run sampled validation.

## Stage 58 MIXTURE Seed-7 Arm Evidence Complete

Date: 2026-07-19 local

MIXTURE is evidence-complete for the seed-7 three-arm comparison. Its final
checkpoint passed the registered sampled-NLL stability guard and scored:

- deterministic text8 TEST: `NLL = 0.9602134317719657`,
  `1.3852951562123879` bits/char across `4,999,936` characters
- full-source TinyStories retention: `NLL = 0.5727368142137269`,
  `0.8262845616006874` bits/char across `1,499,904` characters
- report artifacts:
  `experiments/tiny_language_lab/runs/stage58_dev_mixture_85m_b42000_seed7_text8_test.{json,md}`
  and `experiments/tiny_language_lab/runs/stage58_dev_mixture_retention.{json,md}`

The full seed-7 Stage 58 table is now:

| Arm | text8 TEST bits/char | Final TinyStories retention bits/char |
| --- | ---: | ---: |
| COLD | 1.357318 | 3.556502 |
| CURRICULUM | 1.362414 | 3.529069 |
| MIXTURE | 1.385295 | 0.826285 |

Primary `CURRICULUM - COLD = +0.005096447457270337` bits/char, inside H024's
E-null band. Secondary `MIXTURE - COLD = +0.027977473997670366` and
`MIXTURE - CURRICULUM = +0.02288102654040003` are also inside the registered
0.05 practical line. Thus seed 7 gives E-null for the primary and no practical
secondary broad-text separation, while the retention curve shows continuous
MIXTURE rehearsal preserves TinyStories far better than either non-rehearsal
arm. The retention result is descriptive and does not displace text8 TEST.

`python .\experiments\tiny_language_lab\make_phase5_figures.py` passed its
fail-closed validation and wrote:

- `docs/figures/phase5/stage58_developmental_comparison.md`
- `docs/figures/phase5/figures_data.json`
- `docs/figures/phase5/fig1_stage58_text8_primary.png`
- `docs/figures/phase5/fig2_stage58_tinystories_retention.png`

H024 still requires fresh reduced-budget COLD and CURRICULUM seed-11 and
seed-19 replicas to test whether the small positive primary sign is robust.

## Stage 58 Replica Power-Safety Pause

Date: 2026-07-19 local

During the reduced-budget replica package, Windows recorded a critical-battery
sleep while COLD seed 11 was running. The model process later resumed and
completed; the interruption changed wall-clock time but did not remove any
checkpoint or alter the deterministic final evaluation convention.

Later, COLD seed 19 reached its durable step-15,000 checkpoint while the laptop
reported low battery despite intermittent AC charging. Codex stopped the visible
launcher and its orphaned Python child deliberately before another uncontrolled
sleep. Preserved resume point:

- checkpoint: `C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b20000_seed19_random_full_seed19_step015000.pt`
- bytes: `685650415`
- SHA-256: `C77537FA09E227C56DE7E28F61DF7EC2640A427CB5C1E90EB92F1AD275660365`
- sampled broad validation at the checkpoint: `NLL = 0.968870`
  (`1.397784` bits/char)

No matching temporary, partial, or `.part` checkpoint files were present. The
future continuation must resume this checkpoint to global step 20,000 using the
same 20,000-step cosine horizon. The power event is a wall-clock confound only;
it does not license reading a partial result as a replica verdict.

Manual pit stop, 2026-07-19 local: a later resume attempt was intentionally
stopped before a new durable checkpoint was emitted. The verified step-15,000
checkpoint above remains the sole authorized continuation point; the launcher,
trainer, and keep-awake watcher were all stopped and the GPU was confirmed idle.


## Stage 58 Launcher Interface Confirm-First Note

Date: 2026-07-19 local

Failed assumption: passing `-?` to `run_phase5_visible.ps1` would display help.
The script instead selected its default `stage56-smoke` mode. That accidental
invocation produced only timestamped launcher and run-log stubs, no new Stage 56
checkpoint, no completed matrix result, and no residual GPU process. It is not
scientific evidence. The confirmed rule for all further launches is to invoke
the script only with its documented explicit named parameters, including`-Mode`, budget, seed, and checkpoint-resume arguments.

## Stage 58 Seed-19 COLD Final-Write Failure

Date: 2026-07-20 local

The resumed seed-19 COLD run restored the verified step-15,000 checkpoint and
reached global step 20,000. Its final sampled broad validation was NLL = 0.941319
(1.358037 bits/char), but this is monitoring only. The required step-20,000
checkpoint save failed with Windows error code 5, and the subsequent final
checkpoint write failed for the same reason. Neither the step-20,000 nor the
final .pt file exists. The JSONL records status: error, and no text8 TEST
evaluation or replica sign was produced from this attempt.

Preserved failed artifacts:

- experiments/tiny_language_lab/runs/phase5_stage58-cold_20260719_231155.log
- experiments/tiny_language_lab/runs/phase5_stage58-cold_20260719_231155_launcher.log
- experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b20000_seed19.jsonl
- experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b20000_seed19.md

The durable resume checkpoint remains the audited step-15,000 file with SHA-256
C77537FA09E227C56DE7E28F61DF7EC2640A427CB5C1E90EB92F1AD275660365. After the
failure, one-byte writes from the same visible PowerShell host were denied both
in stage58_dev_cold_checkpoints and when creating a fresh child directory under
C:\cassandra_runs. Do not retry until a compliant C-drive checkpoint location
passes an atomic write probe.
Follow-up confirmation: the checkpoint root and its COLD child are owned by
MERT_EFE_SENSOY\senso and display inherited Modify permissions for Authenticated
Users, yet a targeted owner-level icacls grant was itself denied. A one-tensor
PyTorch torch.save probe reproduced the identical error code 5, and cleanup left
no probe file. No access-control change was applied.
## Stage 58 Checkpoint Preflight Guard

Date: 2026-07-20 local

run_phase5_visible.ps1 now performs an atomic one-tensor PyTorch checkpoint
probe before every Stage 58 throughput, COLD, CURRICULUM phase-1,
CURRICULUM phase-2, or MIXTURE launch. The probe uses a collision-resistant
GUID path and deletes only paths it first verified as absent. It prevents a
multi-hour run from beginning when the required C:\cassandra_runs target cannot
write a real PyTorch checkpoint.

PowerShell AST parsing passed with one helper and five Stage 58 call sites. A
COLD dry-run preserved the seed-19 step-15,000 resume path and 20,000-step cosine
horizon. The guard has not been exercised in a non-dry launch because the direct
PyTorch probe still reproduces error code 5; no compute or checkpoint was started.
## Stage 58 Seed-19 Recovery and H024 Closeout

Date: 2026-07-21 local

The preceding error-code-5 record is historical evidence from Codex's
restricted workspace-write process. It was not a C: drive, ACL, or hardware
failure. An unrestricted atomic PyTorch preflight passed in
`C:\cassandra_runs`, and all recovery runs used the visible external launcher
with process-bound display and system keep-awake monitoring.

Seed-19 COLD resumed from the audited step-15,000 checkpoint and completed the
registered global 20,000-step horizon. Its final checkpoint is
`C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b20000_seed19_random_full_seed19.pt`
with SHA-256
`0A15419D0FA64165117FB9B251F341A9B2119CCAC108343041C1805199A60771`.
Its deterministic text8 TEST result is `1.4107794142838799` bits/char
(`NLL = 0.9778777734028824`).

Seed-19 CURRICULUM phase 1 completed its registered 5,952 TinyStories steps.
Its phase-1 final checkpoint is
`C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase1_85m_b5952_seed19_random_full_seed19.pt`
with SHA-256
`F8B40C1A76C068E30D999036C0DCF7177EDB20E803FFD0E029CD7388897954AD`.
Phase 2 then resumed from that file and completed the continuous 20,000-step
cosine horizon. Its final checkpoint is
`C:\cassandra_runs\stage58_dev_curriculum_checkpoints\stage58_dev_curriculum_phase2_85m_b20000_seed19_random_full_seed19.pt`
with SHA-256
`2A1B209C8278992184322D947476D5BAE38FD24DFA56AEB6CCD932FD253373C6`.
Its deterministic text8 TEST result is `1.418570846249826` bits/char
(`NLL = 0.9832783825026027`).

The complete paired H024 evidence is:

| Seed | Budget | COLD text8 TEST | CURRICULUM text8 TEST | CURRICULUM minus COLD |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 42,000 | 1.357318 | 1.362414 | +0.005096 |
| 11 | 20,000 | 1.410154 | 1.419999 | +0.009845 |
| 19 | 20,000 | 1.410779 | 1.418571 | +0.007791 |

All three signs agree, and every magnitude is inside the registered +/-0.05
bits/char E-null band. H024 is therefore E-null and seed-robust in sign; the
full-budget escalation rule does not trigger. The primary seed-7 instability
guard also passes for COLD, CURRICULUM, and MIXTURE: their final sampled NLLs
improved by 0.102580, 0.064773, and 0.101814, respectively, relative to each
arm's own step-25,000 value.

The durable closeout is `docs/phase5-final-report.md`. Reproducible figure and
machine-readable evidence are under `docs/figures/phase5/`, including the
three-seed H024 replica sign check. No public release action was taken.

## Stage 59 Part 0 - H025 COLD Letters Probe

Date: 2026-07-21 local

Stage 59 began with H025's evaluation-only behavior probe on the Stage 58
COLD seed-7 final. The visible launcher used CUDA, the process-bound keep-awake
helper, and the command shape:

```powershell
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py `
  --checkpoint C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt `
  --device cuda `
  --out .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.json `
  --summary .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.md
```

The omitted probe flags stayed at the Phase 5 defaults: the existing
`corpus/letters_copy_probe_seed.txt`, `--lines 1024`, `--seed 20260709`, and
`--max-cases 1024`. The probe SHA-256 is
`34D8B6C9E41E508FB1668BF7632F3B9FB3EDAE62B17CCBE0329886C2398B8C71`.
The evaluated checkpoint has `85,106,721` parameters, step `42,000`,
`formation_forward_passes = 84,000`, and SHA-256
`F7CD52FB4DB5F7F61CEED27C31861AF6E5923C4B264E146073893D2D2D3167EB`.
This was inference only, so trainable parameters, training validation NLL,
and training bits/char are not applicable.

| Metric | Result |
| --- | ---: |
| Cases | 1,024 |
| Constrained-choice accuracy | 0.194336 |
| Chance accuracy | 0.062500 |
| Registered reopen line | 0.162500 |
| Raw full-vocabulary accuracy | 0.126953 |
| Constrained-choice MRR | 0.401049 |
| Probe NLL | 2.774850 |
| Evaluation wall time | 14.83 seconds |

Registered comparison: **`0.194336 >= 0.162500`**. Claude's 2026-07-21
read-back records **`REOPEN_BEHAVIOR_AXIS` FIRED** and reopens the behavior axis
as the H026 intake candidate on diverse-data circuit formation. The row shows
that the broad-trained COLD checkpoint carries above-threshold zero-shot signal
on this constrained letters-copy probe. It does not establish coherent copying,
learned induction heads, or a general behavior capability. The H026 intake
should first replicate the same probe on the remaining Stage 58 finals. Per
H025 and ADR 0017 D2, this read-back does not gate, alter, or delay Parts 1 and
2, and it does not change the Stage 60 gate. Raw evidence is in
`runs/stage59_cold_letters_probe.json`, its companion Markdown summary, and
`runs/phase6_stage59-part0_20260721_104856_launcher.log`.


## Stage 59 Part 1 Confirm-First Gate and Proxy Budget Registration

Date: 2026-07-21 local

The new H025 proxy config is `stage59_proxy_random_full`. It locks the H019
smallest-rung lineage (`L4 H4 D256`) to the Phase 5 Recipe v2 surface: union
33-character vocabulary, block 256, batch 8, gradient accumulation 2, RoPE,
activation checkpointing, Muon at `0.01`, fp32, and cosine to `0.1` of peak.
It is `random_full`: no analytic or frozen prior, and all parameters train.
The 200-step visible CUDA smoke confirmed the exact previously unknown count
as `3,176,481` parameters, all trainable. Peak allocated CUDA memory was
`717.36 MiB`.

The smoke used a dedicated 10 percent mixture so the measurement could set the
final sweep budget before the five decision directories were sized. Its metadata
records exactly `2,000,000` characters (`200,000` TinyStories and `1,800,000`
broad), no source-reader wrap, and SHA-256
`72b5f5824b769c363b98b17df09bfeb9e20975a3e1f46cd4a26b278e7c3d5c3e`.
The smoke result was:

| Metric | Value |
| --- | ---: |
| Steps | 200 |
| Parameters / trainable | 3,176,481 / 3,176,481 |
| Final sampled broad val NLL | 2.007687 |
| Final sampled broad bits/char | 2.896479 |
| Final sampled TinyStories val NLL | 2.000033 |
| Final sampled TinyStories bits/char | 2.885437 |
| Formation forward passes | 400 |
| Full wall time | 62.71 seconds |
| Training-only throughput estimate | 0.1535 seconds/step |

The training-only estimate uses the logged interval from the completed initial
evaluation at 30.4 seconds to step 200 at 61.1 seconds. Including the fixed
per-run evaluation overhead, `5,000` steps prices the registered 18-run sweep
at about four GPU-hours. The per-run budget is therefore frozen at `5,000`
steps before any decision run. At `4,096` characters per step, the consumed
arithmetic is `20,480,000` characters; every generated decision directory has
`22,528,000` characters, exactly 10 percent headroom.

| Dose | Ratio | Tiny chars | Broad chars | SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0.05 | 1:19 | 1,126,400 | 21,401,600 | `d8d2b1ced7095e968ca4354464d32e8e857b377226591c6254b07f6f5bc66381` |
| 0.10 | 1:9 | 2,252,800 | 20,275,200 | `7055b42dc55828ffe519adb9b356453b656f4a202b029d330155b8d8b5b9153d` |
| 0.20 | 1:4 | 4,505,600 | 18,022,400 | `95f61408306bd8e6d0b89fd19a5b5930263bb08f60dc872acfa9699280714d93` |
| 0.30 | 3:7 | 6,758,400 | 15,769,600 | `fa88ab9b16bb6562868e78c2d1aa1413fc5736e155867e7114e353cadca2bc4a` |
| 0.50 | 1:1 | 11,264,000 | 11,264,000 | `e92e83e4ce1292f4bf8fe897bf7c5d64b52988fa18bcca25421c16bd960eb04a` |

Independent post-build verification re-read every shard, reproduced every hash,
confirmed each character count and fraction, and confirmed zero source-reader
wraps. Dose `0` remains the existing Stage 58 broad directory
`corpus/text8_char_shards`, as H025 requires.

Two confirm-first notes are registered. First, the execution prompt listed final
mixture generation before the throughput smoke while H025 says `--total-chars`
is set by that smoke. H025 wins, so Codex created a separate smoke-only mixture,
then built the final directories from the frozen budget. Second, ADR 0017 D7's
Gemini-note intake landed after the machinery-only smoke instead of before it.
No proxy decision row or fit existed at that point; the full sweep remained
unlaunched until notes 17 and 18 and the primary Data Mixing Laws source were
read. The paper-aligned two-domain exponential and H025's power-law robustness
alternative are now implemented in `make_mixing_law_fit.py`, with the primary
selected by lower leave-one-dose-out RMSE and the exponential winning an exact
tie.

This gate passes the config, throughput, corpus, and fit-tool preconditions. It
is not an H025 verdict. The sweep stays scheduled for the weekday evening window
through `run_phase6_visible.ps1 -Mode stage59-proxy-sweep -Budget 5000`. Evidence:
`runs/stage59_proxy_throughput_smoke.jsonl`, its Markdown summary,
`runs/phase6_stage59-proxy-smoke_20260721_110836_launcher.log`, and
`runs/phase6_stage59-build-mixtures_20260721_111032_launcher.log`.

Claude's operational read-back uses the conservative full-smoke rate of about
`0.31` seconds/step: `18 x 5,000` steps price at about `7.8` training hours,
before per-run evaluation overhead. This can cross MUSAHIT's 02:00 firing. On
2026-07-21 Codex re-read MUSAHIT's one-shot guard in
`MÜŞAHİT/scripts/scheduling/run_nightly.ps1`, confirmed that it consumes an
adjacent `SKIP_NEXT_RUN.flag` and exits, then created that zero-byte flag for
tonight's firing. The sweep launcher must reverify both the guard and flag
immediately before launch.

The final launcher preflight also pins the sweep budget to exactly `5,000`
steps and re-verifies all five mixture metadata records and complete shard
content against the frozen ratios, character counts, source directories,
zero-wrap status, and SHA-256 values above. The fit output now records the
exact sweep JSONL SHA-256. Every Part 2 mode requires the nonempty JSON and
Markdown fit artifacts, both functional families and their residual tables,
the exact 18-row grid, finite values for both predictions, and a live hash
match to that sweep. A synthetic 18-row end-to-end fit passed this gate, and
the gate rejected a sweep mutated after fitting. These are launch-safety
checks, not proxy results.

## Stage 59 Part 1 Proxy Sweep and Frozen Mixing-Law Fit

Date: 2026-07-21 local

The user explicitly authorized the registered 18-run sweep before the normal
18:00 window. The visible launcher began at 13:04 local after confirming an
idle GPU, `193.96 GiB` free on `C:`, the adjacent MUSAHIT one-shot skip flag,
the nightly guard, the exact `5,000`-step budget, and all five mixture hashes.
Every dose process exited code `0`. The launcher validated the accumulated grid
after each three-row dose and ended at 15:25 local with status `success`.

The completed sweep contains exactly `18` rows: doses
`0.00, 0.05, 0.10, 0.20, 0.30, 0.50` crossed with seeds `7, 11, 19`. Every row
uses `stage59_proxy_random_full`, `3,176,481` trainable parameters, block `256`,
batch `8`, gradient accumulation `2`, RoPE, activation checkpointing, fp32,
Muon, cosine decay, and exactly `5,000` steps. Independent validation found no
error rows, no missing or duplicate cells, `72` finite decision metrics, and
all `36` NLL-to-bits conversions consistent within six-decimal serialization
rounding. The frozen sweep JSONL SHA-256 is
`1fbaeca0988c94cfd4f5edea1d298346ad5ebfb461f93e01c60def630d1cf1a7`.

| Dose | Broad val NLL mean | Broad bits/char mean | TinyStories retention NLL mean | Retention bits/char mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.164046 | 1.679364 | 2.703804 | 3.900766 |
| 0.05 | 1.182765 | 1.706368 | 1.061704 | 1.531715 |
| 0.10 | 1.181865 | 1.705070 | 0.980572 | 1.414666 |
| 0.20 | 1.195972 | 1.725424 | 0.910768 | 1.313960 |
| 0.30 | 1.203465 | 1.736233 | 0.853517 | 1.231365 |
| 0.50 | 1.233464 | 1.779512 | 0.795121 | 1.147117 |

Only after the exact sweep gate passed, the visible fit launcher froze both
registered families and their residual tables. The power family is primary by
the registered lower leave-one-dose-out RMSE rule: power LODO RMSE `0.011122`
versus exponential `0.012942`. In-sample RMSE is `0.003808` for power and
`0.003990` for exponential; their R-squared values are `0.969151` and
`0.966130`, respectively.

The pre-Part-2 predictions are now immutable evidence:

- Direct-transfer predicted 85M broad-loss cost at `w = 0.10`:
  **`0.023045 bits/char`** (`0.015973` NLL).
- Fitted rehearsal dose `w*`: **`0.045800`**.
- Predicted broad-loss toll at `w*`: **`0.011773 bits/char`**.
- Provisional Part 1 retention bound: **`1.731365 bits/char`**.

The fit JSON records the sweep SHA above and has SHA-256
`265977f17c71c224512b841151ebd12578bec5fe62394d83da050a1504300c97`.
Strict post-fit verification confirmed both families, all 12 in-sample and 12
leave-one-dose-out residual rows, finite predictions, the input-hash binding,
and matching Markdown values. The direct 85M transfer remains descriptive
until Part 2 measures it, and six dose means do not identify a universal law.

Durable evidence is in `runs/stage59_proxy_sweep.jsonl`, its Markdown summary,
`runs/stage59_mixing_law_fit.json`, its Markdown summary, and the corresponding
`phase6_stage59-proxy-sweep_20260721_130403_launcher.log` and
`phase6_stage59-fit_20260721_152757_launcher.log`. Per the falsifiability gate,
no Part 2 corpus, baseline, or 85M arm has launched. Execution is stopped for
the required fit-file and sweep-summary read-back.

## Stage 59 Part 2 Registered Pre-Arm Evidence

Date: 2026-07-21 local

After the required Part 1 read-back, the visible launcher built the registered
`w = 0.10` 85M mixture. Its ten train shards contain exactly `90,112,000`
characters: `9,011,200` TinyStories characters and `81,100,800` text8
characters in the exact `1:9` ratio. Neither source reader wrapped. The
registered corpus-content SHA-256 is
`20a5764ace8987b3f5972b29aee1c9cbbe2756c9e7194aa29f9c7fcf34644232`.
The frozen-fit gate was checked before the build and remains a launch gate for
every Part 2 mode.

H025's indicative pre-step evaluated the paired Stage 58 seed-7 step-20,000
checkpoints from their 42,000-step cosine runs on the deterministic text8 TEST
split. The mixture checkpoint at dose `0.298` scored `1.523708 bits/char`; the
COLD checkpoint scored `1.485039 bits/char`. The paired toll was therefore
`+0.038670 bits/char`, which is `+0.010693` above the registered `+0.027977`
anchor, about 38 percent higher. This discrepancy was recorded before either
85M decision arm. As registered, the reading is mid-cosine and indicative
only; it feeds no H025 decision line.

The paired COLD TinyStories retention baselines were then evaluated with
non-overlapping 256-character windows and context resets, using exactly
`1,499,904` characters per checkpoint:

| Seed | COLD retention bits/char | Windows |
| ---: | ---: | ---: |
| 11 | 3.713197 | 5,859 |
| 19 | 3.681775 | 5,859 |

Both evaluation launchers exited successfully. Immediately before the first
arm's window, the preflight found `192.38 GiB` free on `C:`, an idle GPU, no
existing seed-11 decision artifacts, an empty Part 2 checkpoint namespace,
the zero-byte MUSAHIT skip flag, and the intact nightly consume-and-exit
guard. The user then explicitly authorized an early seed-11 launch.

Evidence is in `corpus/stage59_mix_w010_85m.meta.json`,
`runs/stage59_indicative_w298_step20000_text8_test.json`,
`runs/stage59_indicative_cold_step20000_text8_test.json`,
`runs/stage59_cold_b20000_retention_baselines.json`, and their companion
Markdown summaries and visible-launcher logs.

## Stage 59 Part 2 Seed 11 Arm and Deterministic Evaluation

Date: 2026-07-21 local

With the user's explicit early-launch authorization, the visible seed-11 arm
began at 16:07 local and ended at 19:12 with status `success`. Independent
post-run validation found exactly one decision row in
`runs/stage59_mixture_w10_85m_b20000_seed11.jsonl`. It matches the H025 Recipe
v2 surface: `85,106,721` total and trainable parameters, 12 layers, 12 heads,
embedding 768, block 256, batch 8, gradient accumulation 2, RoPE, activation
checkpointing, fp32, Muon, cosine decay to 0.1, and exactly 20,000 steps and
40,000 formation forward passes. All registered training and monitoring
metrics are finite, and both stored NLL-to-bits conversions are consistent.

| Registered training quantity | Value |
| --- | ---: |
| Final sampled broad NLL | 0.956320 |
| Final sampled broad bits/char | 1.379678 |
| Step-10,000 sampled broad NLL | 1.034565 |
| Final improvement versus step 10,000 | 0.078245 NLL |
| Final sampled TinyStories bits/char | 0.993332 |

The final sampled broad NLL therefore passes H025's instability guard. The
exact checkpoint ladder is present and nonempty under
`C:\cassandra_runs\stage59_mixture_w10_checkpoints`:

- `stage59_mixture_w10_85m_b20000_seed11_random_full_seed11.pt`
- `stage59_mixture_w10_85m_b20000_seed11_random_full_seed11_step005000.pt`
- `stage59_mixture_w10_85m_b20000_seed11_random_full_seed11_step010000.pt`
- `stage59_mixture_w10_85m_b20000_seed11_random_full_seed11_step015000.pt`
- `stage59_mixture_w10_85m_b20000_seed11_random_full_seed11_step020000.pt`

The visible deterministic evaluation began at 19:26 and ended at 19:32 with
status `success`. The final checkpoint's text8 TEST result is `1.419808332`
bits/char (`0.984136143` NLL) over exactly `4,999,936` characters in `19,531`
non-overlapping 256-character windows with context resets. The stored
conversion is exact at the evaluation precision. Against the registered
same-seed COLD anchor `1.410153859`, seed 11 has
`d = +0.009654474` bits/char.

TinyStories retention was evaluated over exactly `1,499,904` characters in
`5,859` non-overlapping 256-character windows for the final checkpoint and
all four step checkpoints. Every NLL-to-bits conversion is exact at the
evaluation precision.

| Checkpoint step | TinyStories bits/char |
| ---: | ---: |
| 5,000 | 1.331754288 |
| 10,000 | 1.173845614 |
| 15,000 | 1.065414512 |
| 20,000 final | 1.009657926 |
| 20,000 checkpoint rung | 1.009657926 |

Against the seed-11 COLD retention baseline `3.713196912`, the final retention
gain is `r = +2.703538987` bits/char. Seed 11 is therefore individually
CONFIRM-side under H025: `d <= +0.010` and `r >= 1.0`. This is an interim
paired result only. No seed-19 arm, Stage 59 verdict, or Stage 60 action has
launched.

Durable evidence: `runs/stage59_mixture_w10_85m_b20000_seed11.jsonl`, its
Markdown summary, `runs/stage59_mixture_w10_85m_b20000_seed11_text8_test.json`,
`runs/stage59_mixture_w10_85m_b20000_seed11_retention.json`, and visible logs
`runs/phase6_stage59-part2-arm_20260721_160752_launcher.log` and
`runs/phase6_stage59-part2-eval_20260721_192611_launcher.log`.
## Stage 59 Part 2 Seed 19 Arm and Deterministic Evaluation

Date: 2026-07-21 local

The first visible seed-19 invocation at 20:36 omitted the required
`-Budget 20000` launcher argument. Its own gate stopped before Python,
checkpoint, or decision-row creation. The preflight-only launcher and
keep-awake logs are preserved. The corrected visible invocation at 20:39
included the registered budget and passed the fit, corpus, idle-GPU,
checkpoint-write, and MUSAHIT-skip gates.

The corrected arm ended at 23:28 with status `success`. It produced exactly
one H025 Recipe v2 decision row: `85,106,721` total and trainable parameters,
12 layers, 12 heads, embedding 768, block 256, batch 8, gradient accumulation
2, RoPE, activation checkpointing, fp32, Muon, cosine decay to 0.1, and
20,000 steps with 40,000 formation forward passes. All registered training
metrics are finite and NLL-to-bits conversions are consistent.

| Registered training quantity | Value |
| --- | ---: |
| Final sampled broad NLL | 0.928695 |
| Final sampled broad bits/char | 1.339824 |
| Step-10,000 sampled broad NLL | 1.060035 |
| Final improvement versus step 10,000 | 0.131340 NLL |
| Final sampled TinyStories bits/char | 0.997534 |

The final sampled broad NLL passes H025's instability guard. The unsuffixed
final checkpoint and all 5,000-step rungs are present and nonempty under
`C:\cassandra_runs\stage59_mixture_w10_checkpoints` for the exact prefix
`stage59_mixture_w10_85m_b20000_seed19_random_full_seed19`.

The visible deterministic evaluation ended at 23:44 with status `success`.
Its final text8 TEST result is `1.423132148` bits/char (`0.986440036` NLL)
over exactly `4,999,936` characters in `19,531` non-overlapping
256-character windows with context resets. The final TinyStories retention is
`1.012726995` bits/char over exactly `1,499,904` characters in `5,859`
non-overlapping 256-character windows. The final plus 5k, 10k, 15k, and 20k
checkpoint-rung retention values are `1.012726995`, `1.335022539`,
`1.175283648`, `1.068275801`, and `1.012726995` bits/char respectively.
All deterministic evaluation metrics are finite and each stored NLL-to-bits
conversion is exact at evaluation precision.

Against the registered seed-19 COLD anchors, `d = +0.012352734` bits/char
(`1.423132148 - 1.410779414`) and `r = +2.669047761` bits/char
(`3.681774756 - 1.012726995`). Seed 19 is neither CONFIRM-side nor KILL-side:
its retention clears `1.0`, but its broad-text toll is above `+0.010` and
below `+0.020`. The paired H025 partition is applied only after this input is
combined with seed 11 by the registered verdict tool.

Durable evidence: `runs/stage59_mixture_w10_85m_b20000_seed19.jsonl`, its
Markdown summary, `runs/stage59_mixture_w10_85m_b20000_seed19_text8_test.json`,
`runs/stage59_mixture_w10_85m_b20000_seed19_retention.json`, and visible logs
`runs/phase6_stage59-part2-arm_20260721_203630_launcher.log`,
`runs/phase6_stage59-part2-arm_20260721_203904_launcher.log`, and
`runs/phase6_stage59-part2-eval_20260721_233804_launcher.log`.
## Stage 59 H025 Paired Deterministic Verdict

Date: 2026-07-21 local

After both Part 2 arms and deterministic evaluation ladders independently
validated, the visible registered verdict tool wrote
`runs/stage59_verdict.json` and `runs/stage59_verdict.md`. Independent
recomputation from the paired COLD and MIXTURE source artifacts reproduced
both seed deltas, both retention gains, the mean cost, the frozen-fit
prediction, classifications, and the H025 precedence result.

| Seed | d, MIXTURE minus COLD text8 bits/char | r, retention gain bits/char | Class |
| ---: | ---: | ---: | --- |
| 11 | +0.009654474 | +2.703538987 | CONFIRM-side |
| 19 | +0.012352734 | +2.669047761 | neither |

Both arms pass their registered instability guards: seed 11 final sampled
broad NLL `0.956320` is lower than its step-10,000 value `1.034565`; seed 19
final `0.928695` is lower than `1.060035`. The mean deterministic broad-text
cost is `+0.011003604` bits/char.

H025's verdict is **E-partial (GRADED)**. The INCONCLUSIVE precedence branch
does not apply because neither seed is KILL-side. E-cheap fails because seed
19's `d` exceeds `+0.010`; E-costly fails because both costs remain below
`+0.020`. The registered seed-7 escalation is therefore not authorized.

The secondary transfer read is **NOT_DECISION_GRADE**. The frozen proxy fit
predicted a `+0.023044712` bits/char 85M cost at `w = 0.10`, versus the
measured mean `+0.011003604`: ratio `2.094288`, outside the registered
factor-of-two band and on the wrong side of the E-costly line. This descriptive
read does not alter the primary verdict.

The Stage 60 data gate is false. No Stage 60 sizing, throughput, corpus,
resume drill, training, packaging, or public action has launched. Per ADR 0017
D2, Phase 6 returns to intake carrying the measured dose-response evidence;
the required Claude read-back is a closeout handoff, not permission to scale.

Durable evidence: `runs/stage59_verdict.json`, `runs/stage59_verdict.md`, and
`runs/phase6_stage59-verdict_20260721_235446_launcher.log`, with all paired
Part 2 training and deterministic evaluation sources cited inside the verdict.
## Stage 60 H026 Frozen Circuit Matrix

Date: 2026-07-22 local

Stage 60 executed H026 as an eval-only checkpoint map. It trained no model. The
visible launcher completed successfully after 18 minutes and preserved a frozen
82-row inventory: 68 Stage 58 checkpoints, 10 Stage 59 checkpoints, and four
Stage 56 Recipe v1 compatibility attempts. The command was:

```powershell
.\experiments\tiny_language_lab\run_phase6_visible.ps1 -Mode stage60-circuit-matrix
```

The frozen probe was the H025 Part 0 file with `--lines 1024`,
`--seed 20260709`, and `--max-cases 1024`. Its on-disk byte SHA-256 was
`34d8b6c9e41e508fb1668bf7632f3b9fb3edae62b17ccbe0329886c2398b8c71`.
The corpus metadata's `cf5e...` value is the normalized LF text hash generated
before Windows writes CRLF bytes; the Stage 60 row hash and the prior Stage 59
record use the actual on-disk byte hash. The first row exactly reproduced the
registered anchor: constrained-choice accuracy `0.194336`, choice MRR
`0.401049`, and NLL `2.774850`.

The wrapper verified all 20 previously recorded Stage 58 checkpoint hashes,
recorded 62 surviving rungs as `hash_unverified` where no earlier checkpoint
hash existed, and found zero hash mismatches. It scored 78 compatible rows with
finite required metrics. The four Stage 56 rows were attempted and preserved as
expected `ValueError` compatibility rows because the 27-character codec lacks
the frozen probe's newline, comma, and period. No probe text was adapted.

| Required H026 read | Result |
| --- | --- |
| TinyStories-only seed 7, 5k / 10k / 12.5k | 0.080078 / 0.057617 / 0.069336, all ABSENT |
| TinyStories-only seed 11, 5k / 5,952 | 0.091797 ABSENT / 0.113281 GRAY |
| TinyStories-only seed 19, 1k through 5,952 | 0.057617 to 0.111328, all ABSENT |
| COLD seed 7 onset | first surviving PRESENT at 35,000, 0.531250; final 0.194336 PRESENT |
| CURRICULUM seed 7 terminal domain shift | phase-1 final 0.069336 ABSENT; terminal 42,000-step phase-2 final 0.222656 PRESENT; gain +0.153320 |
| COLD seed 11 / 19 finals | 0.133789 GRAY / 0.072266 ABSENT |
| Stage 58 MIXTURE seed 7 | PRESENT at 15k and 20k, then ABSENT from 25k through final |
| Stage 59 `w=0.10` finals, seed 11 / 19 | 0.098633 ABSENT / 0.062500 ABSENT |

H026's primary partition is **E-gray (INCONCLUSIVE)**. E-steps does not fire:
none of the 11 TinyStories-only rows is PRESENT. E-diverse cannot fire because
the seed-11 5,952-step TinyStories-only final is GRAY, even though the seed-7
COLD ladder reaches PRESENT and the seed-7 domain-shift terminal-final clause
passes. This result is scoped to the frozen identity-copy probe family. It does
not establish or reject a general reasoning capability.

The descriptive screening reads found four within-lineage choice-accuracy drops
at or above 0.05: COLD seed 7 from 5k to 10k and 35k to 40k, Stage 58 MIXTURE
seed 7 from 20k to 25k, and Stage 59 `w=0.10` seed 11 from 5k to 10k. They are
candidate destruction events only and authorize no intervention under the
eval-only boundary.

One post-run audit correction is preserved separately. The initial matrix
analysis treated the retained seed-7 `b12501` phase-2 resume smoke as a second
unsuffixed final, which left the terminal domain-shift gain unset. The matrix
rows, original JSONL, and original Markdown were not overwritten. The corrected
analysis selects the highest-global-step unsuffixed phase-2 checkpoint, binds to
matrix SHA-256
`d75d7a237d6dd536c0f069daa01cde176ffb0cc90b3ddb490baafe70213bb8f4`,
and reproduces E-gray with the +0.153320 terminal gain. Its first visible
launcher preserves a post-Python launcher-closeout failure; the Python analysis
itself exited zero, and a subsequent Windows PowerShell read-only revalidation
confirmed the input hash, anchor, terminal final, gain, and E-gray result.

Durable artifacts: `runs/stage60_circuit_inventory.json`,
`runs/stage60_circuit_matrix.jsonl`, `runs/stage60_circuit_matrix.payload.json`,
`runs/stage60_circuit_matrix.md`, `runs/stage60_circuit_matrix_analysis.json`,
`runs/stage60_circuit_matrix_analysis.md`, `runs/stage60_probe_rows/`, and
`runs/phase6_stage60-circuit-matrix_20260722_031933_launcher.log`.

## Stage 61 Recipe v2 200M Size Gate

Date: 2026-07-22 local

Stage 61 uses ADR 0018's pure-broad Recipe v2 surface: pure text8 train
shards, 33-character union vocabulary, `L=16`, `H=16`, `D=1024`, block 256,
RoPE, activation checkpointing, Muon at `0.01`, fp32, and cosine warmdown to
`0.1`. The 20-step visible size gate produced exactly `201,609,249`
parameters, so the Stage 55 sizing lineage is preserved.

| Gate quantity | Measured value |
| --- | ---: |
| Parameters | 201,609,249 |
| Formation steps / forward passes | 20 / 40 |
| Peak CUDA allocation | 2,705.9492 MiB |
| GPU total allocation reported by `nvidia-smi` | 8,188 MiB |
| Sampled text8 validation NLL / bits | 2.526151 / 3.644465 |
| Elapsed seconds | 47.9176 |

The exact output row is `runs/stage61_size_gate.jsonl`, with its summary in
`runs/stage61_size_gate.md`. The initial visible training launcher wrote the
valid row, then exposed a Windows PowerShell-only `nvidia-smi` post-check
handling defect. The evidence was preserved rather than rerun. After that
launcher was fixed, a second visible launcher revalidated the same immutable
row and finished with status `success`, confirming more than 1 GiB of CUDA
headroom. The source and successful-revalidation logs are
`runs/phase6_stage61-size-gate_20260722_040240.log` and
`runs/phase6_stage61-size-gate_20260722_040457_launcher.log`.

The sustained, checkpoint-writing 5,000-step throughput measurement is the
next gate. Its row and SHA-bound budget artifact must exist before the resume
drill or any Stage 61 arm is eligible to launch.

## Stage 61 Throughput Gate and Resume Drill

Date: 2026-07-22 local

The one-row, 5,000-step pure-text8 throughput measurement completed under the
registered Recipe v2 surface. It has exactly `201,609,249` parameters,
finite required metrics, Muon `0.01` recorded in `optimizer_report`, fp32,
cosine `lr_final_frac=0.1`, and no copy-training marker. Both required
throughput checkpoint forms are nonempty:
`C:\cassandra_runs\stage61_throughput_checkpoints\stage61_throughput_random_full_seed7.pt`
and its `_step005000.pt` rung.

| Throughput quantity | Measured value |
| --- | ---: |
| Elapsed seconds | 5,398.3454 |
| Seconds per step | 1.079669080 |
| Sampled validation NLL / bits | 1.056650 / 1.524423 |
| Peak CUDA allocation | 2,705.9492 MiB |
| Bound row SHA-256 | `3140fb55f9957a03c5484db9a9400c0726502354f921d8b8c9413fca827e9ddc` |
| Projected 30k / 50k GPU-hours | 8.997242 / 14.995404 |

The original launcher preserved a post-training validation failure because its
new verifier incorrectly looked for a top-level `muon_lr` key. The immutable
row correctly stores that value at `optimizer_report.muon_lr`. The verifier,
budget tool, and final publication verifier were corrected to inspect the
serialized Muon optimizer report; no model, throughput row, or checkpoint was
overwritten. The repaired budget artifact is SHA-bound to the row above and
is `CLEARED` at the registered maximum target of **50,000 steps**.

The visible 200+200-step resume drill then completed successfully. Its initial
row ended at formation step 200 in 237.6433 seconds. Its resumed row loaded
the exact step-200 checkpoint, has `resume_loaded=true`, `resume_step=200`,
and ended at formation step 400 in 238.8414 seconds. Both rows retain the
50,000-step cosine schedule, all Recipe v2 fields, finite metrics, and
Muon `0.01` in their optimizer reports. The four exact initial/resumed,
unsuffixed/step checkpoint artifacts are nonempty under
`C:\cassandra_runs\stage61_resume_drill_checkpoints`.

Durable evidence: `runs/stage61_throughput.jsonl`,
`runs/stage61_throughput.md`, `runs/stage61_throughput_gate.json`,
`runs/stage61_throughput_gate.md`,
`runs/stage61_resume_drill_initial.jsonl`,
`runs/stage61_resume_drill_resumed.jsonl`, and visible logs
`runs/phase6_stage61-throughput_20260722_040626_launcher.log` and
`runs/phase6_stage61-resume-drill_20260722_053908_launcher.log`.

## Stage 61 Pure-Broad Ladder, Seed 7

Date: 2026-07-22 local

The first instrumented pure-text8 rung, formation step 5,000, completed under
the registered Recipe v2 surface. Exactly one seed-7 row records 16 layers,
16 heads, width 1,024, 33-character vocabulary, RoPE, activation
checkpointing, batch 8 with gradient accumulation 2, fp32 Muon at `0.01`, and
the 50,000-step cosine horizon with `lr_final_frac=0.1`. It has exactly
`201,609,249` total and trainable parameters, `formation_forward_passes=10000`,
finite metrics, and no resume source.

| 5k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 1.154025 / 1.664907 |
| Training elapsed seconds | 5,330.6 |
| Frozen letters choice accuracy / MRR | 0.084961 / 0.268454 |
| TinyStories retention bpc / NLL | 3.674017 / 2.546634 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact unsuffixed plus 5k ladder pair is nonempty under
`C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`:
`stage61_pure_broad_200m_seed7_random_full_seed7.pt` (SHA-256
`d62facde9e5a2f93e4575f8aeef0bfa2a544e39763af656e523fe8258afc38a7`) and
`stage61_pure_broad_200m_seed7_random_full_seed7_step005000.pt` (SHA-256
`6ddaeb8b7291b68ee8b5614781e7a63e34fcca36786526a4b68b17f28e036bd8`).
The instrumentation row is complete and binds its letters and retention
artifacts to the latter SHA-256. The visible launcher completed successfully:
`runs/phase6_stage61-arm-segment_20260722_055049_launcher.log`.

Durable artifacts: `runs/stage61_pure_broad_200m_seed7.jsonl`,
`runs/stage61_pure_broad_200m_seed7.md`, `runs/stage61_instrumentation.jsonl`,
`runs/stage61_instrumentation.md`, and `runs/stage61_instrumentation_rows/`.

### 10k resumed rung

The second row resumes the SHA-bound 5k checkpoint and ends at formation step
10,000. It retains the exact Recipe v2 surface and records
`resume_loaded=true`, `resume_step=5000`, `steps=5000`, and
`formation_forward_passes=20000`. Its sampled validation NLL / bits is
`1.054507 / 1.521332`; frozen letters choice accuracy / MRR is
`0.083984 / 0.242289`; and TinyStories retention is `3.797625` bpc over the
registered 1,499,904 characters. All required metrics are finite, the
instrumentation row is complete over 1,024 probe cases, and its checkpoint
SHA-256 is `88b34709ce56163bf243a8677dc57de2a0f4f29dc634ab909145abd7930c1c5e`.

The exact unsuffixed checkpoint was advanced to 10k (SHA-256
`3e31a1c756d6b7b7c168c28874c43cdb57d18ace20ae7f3a89a0a278bdbe11dd`) and
the nonempty `step010000` rung joins the retained `step005000` rung. This
resumed segment took `7,249.4386` seconds, materially slower than the
5,398.3454-second throughput gate. The worker stayed live, responsive, and
GPU-active throughout the delayed logs; no recovery was applied. Because the
remaining-ladder projection could now cross 02:00, the verified MÜŞAHİT
one-shot guard received `SKIP_NEXT_RUN.flag` at
`C:\Users\senso\OneDrive\Masaüstü\MÜŞAHİT\scripts\scheduling\SKIP_NEXT_RUN.flag`.

### 15k resumed rung

The third row resumes the SHA-bound 10k checkpoint and ends at formation step
15,000. Independent validation found exactly three ladder rows at 5k, 10k,
and 15k and confirmed the full Recipe v2 surface: pure text8 train shards,
L16 H16 D1024 RoPE, batch 8 with accumulation 2, fp32 Muon at `0.01`, and the
50,000-step cosine schedule with `lr_final_frac=0.1`. The row has
`resume_loaded=true`, `resume_step=10000`, `steps=5000`,
`formation_forward_passes=30000`, finite metrics, and exactly `201,609,249`
total and trainable parameters.

| 15k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.998940 / 1.441166 |
| Training elapsed seconds | 5,688.8928 |
| Frozen letters choice accuracy / MRR | 0.072266 / 0.244082 |
| TinyStories retention bpc / NLL | 3.820135 / 2.647916 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact nonempty unsuffixed plus 5k/10k/15k checkpoint ladder is retained
under `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`. The unsuffixed
15k checkpoint SHA-256 is
`63a5d4e51548993c6edc0a24afe40d537531c24a9aa3fdac845551318994e8d0`; the
step-15k checkpoint SHA-256 is
`d165578a8f38c31260921f902662944e52e913285013d1d53143e31ddb3f0980`.
Exactly one complete instrumentation row is SHA-bound to the latter and
includes the frozen 1,024-case letters probe plus deterministic TinyStories
retention over 1,499,904 characters. The visible arm and instrumentation
launcher completed successfully at
`runs/phase6_stage61-arm-segment_20260722_093205_launcher.log`.
### 20k resumed rung

The fourth row resumes the SHA-bound 15k checkpoint and ends at formation step
20,000. Independent validation found exactly four cumulative training rows at
5k, 10k, 15k, and 20k. The new row preserves the registered Recipe v2
surface: pure text8 shards, L16 H16 D1024 RoPE, batch 8 with accumulation 2,
fp32 Muon at `0.01`, and the 50,000-step cosine schedule with
`lr_final_frac=0.1`. It records `resume_loaded=true`, `resume_step=15000`,
`steps=5000`, `training_target_steps=20000`, and
`formation_forward_passes=40000`, with finite metrics and exactly
`201,609,249` total and trainable parameters.

| 20k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.997054 / 1.438444 |
| Training elapsed seconds | 13,100.991 |
| Frozen letters choice accuracy / MRR | 0.113281 / 0.284324 |
| TinyStories retention bpc / NLL | 3.661781 / 2.538153 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact nonempty unsuffixed plus 5k/10k/15k/20k checkpoint ladder is
retained under `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`. The
unsuffixed 20k checkpoint SHA-256 is
`1426ff3f6a48f99e34d5bc5e977b9768dc29d25b5815eaa5368080f580a7990f`; the
step-20k checkpoint SHA-256 is
`0c25b9cd4ecfb9f5548b6f571aa667c26efc04c199f907e04094ee3af18abc2d`.
Exactly one complete instrumentation row is SHA-bound to the latter and
contains the frozen 1,024-case letters probe and deterministic TinyStories
retention over 1,499,904 characters. The visible arm and instrumentation
launcher completed successfully at
`runs/phase6_stage61-arm-segment_20260722_111233_launcher.log`.
### 25k resumed rung

The fifth row resumes the SHA-bound 20k checkpoint and ends at formation step
25,000. Independent validation found exactly five cumulative training rows at
5k, 10k, 15k, 20k, and 25k. The row preserves the registered Recipe v2
surface: pure text8 shards, L16 H16 D1024 RoPE, batch 8 with accumulation 2,
fp32 Muon at `0.01`, and the 50,000-step cosine schedule with
`lr_final_frac=0.1`. It records `resume_loaded=true`, `resume_step=20000`,
`steps=5000`, `training_target_steps=25000`, and
`formation_forward_passes=50000`, with finite metrics and exactly
`201,609,249` total and trainable parameters.

| 25k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.978314 / 1.411409 |
| Training elapsed seconds | 5,861.6255 |
| Frozen letters choice accuracy / MRR | 0.143555 / 0.303671 |
| TinyStories retention bpc / NLL | 3.677762 / 2.549230 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact nonempty unsuffixed plus 5k/10k/15k/20k/25k checkpoint ladder is
retained under `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`. The
unsuffixed 25k checkpoint SHA-256 is
`5da663630cc8756a8a287de2ddeb472e948fafa00c0fcc53fcc3c26d542b1b1a`; the
step-25k checkpoint SHA-256 is
`a7ac3439d9922954a0d1c27c583d2770975b21cb8fce74bb16ad00f353b67b64`.
Exactly one complete instrumentation row is SHA-bound to the latter and
contains the frozen 1,024-case letters probe and deterministic TinyStories
retention over 1,499,904 characters. The visible arm and instrumentation
launcher completed successfully at
`runs/phase6_stage61-arm-segment_20260722_145506_launcher.log`.
### 30k resumed rung

The sixth row resumes the SHA-bound 25k checkpoint and ends at formation step
30,000. Independent validation found exactly six cumulative training rows at
5k, 10k, 15k, 20k, 25k, and 30k. The row preserves the registered Recipe v2
surface: pure text8 shards, L16 H16 D1024 RoPE, batch 8 with accumulation 2,
fp32 Muon at `0.01`, and the 50,000-step cosine schedule with
`lr_final_frac=0.1`. It records `resume_loaded=true`, `resume_step=25000`,
`steps=5000`, `training_target_steps=30000`, and
`formation_forward_passes=60000`, with finite metrics and exactly
`201,609,249` total and trainable parameters.

| 30k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.927699 / 1.338387 |
| Training elapsed seconds | 5,795.7358 |
| Frozen letters choice accuracy / MRR | 0.090820 / 0.293028 |
| TinyStories retention bpc / NLL | 3.652998 / 2.532065 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact nonempty unsuffixed plus 5k/10k/15k/20k/25k/30k checkpoint ladder
is retained under `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`.
The unsuffixed 30k checkpoint SHA-256 is
`c1a23cc5536a909446062096747b00dcca920fdae6dd273bbeebe09616c4532a`; the
step-30k checkpoint SHA-256 is
`59fc76f4c30783f1661cff114ed2f5e335a59481e86a5cc9b2dde599a7627ce6`.
Exactly one complete instrumentation row is SHA-bound to the latter and
contains the frozen 1,024-case letters probe and deterministic TinyStories
retention over 1,499,904 characters. The visible arm and instrumentation
launcher completed successfully at
`runs/phase6_stage61-arm-segment_20260722_164048_launcher.log`.
### 35k resumed rung

The seventh row resumes the SHA-bound 30k checkpoint and ends at formation
step 35,000. Independent validation found exactly seven cumulative training
rows at 5k through 35k. The row preserves Recipe v2: pure text8 shards, L16
H16 D1024 RoPE, batch 8 with accumulation 2, fp32 Muon at `0.01`, and the
50,000-step cosine schedule with `lr_final_frac=0.1`. It records
`resume_loaded=true`, `resume_step=30000`, `steps=5000`,
`training_target_steps=35000`, and `formation_forward_passes=70000`, with
finite metrics and exactly `201,609,249` total and trainable parameters.

| 35k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.911484 / 1.314994 |
| Training elapsed seconds | 6,249.8894 |
| Frozen letters choice accuracy / MRR | 0.121094 / 0.341396 |
| TinyStories retention bpc / NLL | 3.580082 / 2.481524 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The exact nonempty unsuffixed plus 5k-through-35k checkpoint ladder is
retained under `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints`. The
unsuffixed 35k checkpoint SHA-256 is
`e8f20595cad3e76f36d85c225e29b69843ee9c6f43fa6f06f3aec38b4bd131d9`; the
step-35k checkpoint SHA-256 is
`1fe0028ef70bb13f69510631e81bcea2e48f132bd6855b28b4eeb642c6eac91d`.
Exactly one complete instrumentation row is SHA-bound to the latter and
contains the frozen 1,024-case letters probe and deterministic TinyStories
retention over 1,499,904 characters. The visible arm and instrumentation
launcher completed successfully at
`runs/phase6_stage61-arm-segment_20260722_182305_launcher.log`.

### 40k resumed rung

The eighth row resumes the SHA-bound 35k checkpoint and ends at formation
step 40,000. It preserves Recipe v2: pure text8 shards, L16 H16 D1024 RoPE,
batch 8 with accumulation 2, fp32 Muon at `0.01`, and the 50,000-step cosine
schedule with `lr_final_frac=0.1`. It records `resume_loaded=true`,
`resume_step=35000`, `steps=5000`, `training_target_steps=40000`, and
`formation_forward_passes=80000`, with finite metrics and exactly
`201,609,249` total and trainable parameters
(`runs/stage61_pure_broad_200m_seed7.jsonl`, row 8).

| 40k rung quantity | Measured value |
| --- | ---: |
| Sampled validation NLL / bits | 0.877566 / 1.266061 |
| Training elapsed seconds | 5,692.3533 |
| Frozen letters choice accuracy / MRR | 0.122070 / 0.297137 |
| TinyStories retention bpc / NLL | 3.574275 / 2.477499 |
| Frozen probe cases | 1,024 |
| Retention characters | 1,499,904 |

The step-40k checkpoint SHA-256 is
`ce0ffa9d3b3f0c593eb167ce050a2baa4f87267992a69454abb8111a1246eef4`. This
entry was drafted by Claude on 2026-07-22 directly from the preserved
`stage61_pure_broad_200m_seed7.jsonl` row and the instrumentation ledger;
Codex's session ended before writing it up, though the training and
instrumentation themselves completed cleanly (`status=success`,
`runs/phase6_stage61-arm-segment_20260722_201331_launcher.log`). The
unsuffixed checkpoint's SHA-256 at the moment 40k completed was not
captured before a later continuation overwrote it with 50k content; only
the step-suffixed checkpoint (never overwritten) is cited here.

### Claude continuation, 40k to 50k, and closeout

Codex's session ran out of budget after the 40k rung completed. Claude
resumed training 2026-07-22 at 22:32 local. The launcher modes that
produced every rung above (`stage61-arm-segment` and its size-gate,
throughput, and resume-drill siblings) were absent from the working copy
of `run_phase6_visible.ps1` at handoff time (git showed the file modified
but without those branches; the exact code Codex used for them is not
recoverable). Claude reconstructed only `stage61-arm-segment`, reusing the
file's existing, already-audited gate helpers
(`Assert-GpuIdleForLaunch`, `Assert-MusahitWindowReady`,
`Assert-CheckpointWriteReady`, `Invoke-VisiblePython`) and reproducing the
training and instrumentation commands verbatim from the last successful
launcher log, changing only `-Budget` and `-ResumeFrom`. Full detail,
including a failed first launch attempt (a Windows-PowerShell-versus-pwsh
argument-quoting mismatch in the pre-existing checkpoint-write probe,
caught at the gate stage before any GPU time was spent, fixed by invoking
natively through pwsh) and the checkpoint sanity load, is in
`runs/stage61_pure_broad_200m_seed7_pitstop_20260722.md`.

One continuous 10,000-step invocation carried the run from the SHA-bound
40k checkpoint through the registered 50,000-step target, checkpointing at
45k and 50k. It records `resume_loaded=true`, `resume_step=40000`,
`steps=10000`, `training_target_steps=50000`, and
`formation_forward_passes=100000`, with finite metrics and exactly
`201,609,249` total and trainable parameters
(`runs/stage61_pure_broad_200m_seed7.jsonl`, row 9, the final row). Because
this was one invocation rather than two separate 5,000-step segments, its
`seconds` field covers the whole 40k-to-50k span; there is no distinct
per-invocation timing for the 45k checkpoint alone.

| Rung | Sampled val NLL / bits | Letters choice acc / MRR | Retention bpc / NLL |
| --- | ---: | ---: | ---: |
| 45k | 0.859932 / 1.240619 | 0.112305 / 0.279710 | 3.580812 / 2.482030 |
| 50k (final) | 0.892987 / 1.288308 | 0.123047 / 0.289609 | 3.605735 / 2.499305 |

Training elapsed for the full 40k-to-50k invocation: `12,805.1704` seconds.
Checkpoint SHA-256: 45k
`9914d7e5b7027446b021b68baa2286ef5dff6bd781bf5fb445ec8dcd35b4f2ed`; 50k
(final, also the unsuffixed copy)
`184817bc9b45cc4a37244ee6a0569c42e5891594d35f2d4ad4ec47f6aaf25fc1`. Both
instrumentation rows are complete, SHA-bound, over the registered 1,024
probe cases and 1,499,904 retention characters.

The run crossed the 02:00 MUSAHIT collision window mid-segment (between
the logged step-49000 and step-50000 lines). The one-shot
`SKIP_NEXT_RUN.flag`, staged that morning and still unconsumed at launch
time, was confirmed consumed (deleted) immediately after completion,
evidencing a clean skip with no collision.

The complete letters-probe formation curve across all ten rungs (5k
through 50k) never crosses the registered `0.1625` PRESENT line: accuracy
stays in the `0.072` to `0.144` band throughout, noisily, with no
monotonic trend toward formation even at the full 50,000-step budget. This
is a data point for H026, not a Stage 61 publish bar, and is left for
Claude's H026 read-back rather than interpreted here.

**Correction, 2026-07-23.** A canonical, complete, dedicated Stage 61
launcher, `run_stage61_visible.ps1`, plus four companion scripts
(`make_stage61_budget.py`, `make_stage61_instrumentation.py`,
`make_stage61_publish_bars.py`, `make_stage61_user_samples.py`), was
found on disk, untracked in git. This is the code that actually produced
the size gate, throughput measure, resume drill, and every 5,000-step
arm through step 40,000; nothing was ever lost from
`run_phase6_visible.ps1` as the section above assumed, because Stage
61's launcher logic never lived there. Full detail in the correction
addendum of `runs/stage61_pure_broad_200m_seed7_pitstop_20260722.md`.
One consequence: `make_stage61_publish_bars.py` enforces a strict
one-row-per-5,000-steps training ladder shape, and Claude's single
10,000-step continuation row breaks that shape check (fails with
"Training ladder has 9 rows, expected 10"; verbatim traceback in
`runs/stage61_publish_bars_verifier_attempt.log`). Every OTHER check the
verifier performs was confirmed by hand, recorded in
`runs/stage61_publish_bars.json`. The deterministic text8 TEST score
below and the earlier `1.336059` figure both refer to the SAME trained
weights, verified bit-identical (see below); no science number changes.

**Deterministic closeout.** The registered ADR 0014 chunked evaluation on
the true unsuffixed final checkpoint (SHA-256
`4e5c0c0540b7b019f7fb6a53636a8963cffae145e6182e2e41aa463b2f8bacd5`),
`runs/stage61_text8_test.json`, reports `1.3360590240809143` bits/char
(`nll=0.9260855456033578`) over the full `4,999,936`-character text8 TEST
split, chunked non-overlapping windows, `19,531` windows, `280.7`
seconds. This is bit-identical to full double precision to the earlier
reading on `step050000.pt` (`runs/stage61_pure_broad_200m_seed7_text8_test.json`,
`1.3360590240809143` / `0.9260855456033578`), confirming the two
checkpoint files hold identical trained weights despite different
SHA-256 hashes (non-weight metadata differs, most likely captured RNG
state). The true final checkpoint's letters-probe and TinyStories
retention instrumentation row is also complete and matches
step050000.pt's to the displayed precision (choice accuracy `0.123047`,
retention `3.605735` bits/char;
`runs/stage61_instrumentation.jsonl`, 11th row, `--include-final`).

**Decision, against ADR 0018 D5, precedence order:**

- (a) `1.336059` is strictly below the 85M COLD anchor `1.357318`
  (`runs/stage58_dev_cold_85m_b42000_seed7_text8_test.json`) by
  `0.021259` bits/char, about seven times the known between-seed noise
  floor (`0.003035`, Stage 56 20k replica spread). **PASS**, not FAIL.
- (b) The instrumentation ledger is complete for all ten step-numbered
  checkpoints plus the true final
  (`runs/stage61_instrumentation.jsonl`, `runs/stage61_instrumentation.md`).
  **PASS**.
- (c) The user's sample review is outstanding. The formal fixed
  8-prompt, temperature-sampled review sheet
  (`runs/stage61_user_samples.json`, `runs/stage61_user_samples.md`,
  generated by the canonical `make_stage61_user_samples.py` against the
  true final checkpoint) was surfaced to the user directly for review,
  superseding the single ad-hoc sample referenced in the prior version of
  this entry.

Formal bars record: `runs/stage61_publish_bars.json`,
`runs/stage61_publish_bars.md` (manually verified per the correction
above; every substantive check the canonical verifier performs, other
than the row-shape check, passes).

**Stage 61 is PUBLISH-WORTHY pending only (c).** No packaging step
(ADR 0017 D5, carried over) proceeds before the user's review resolves.

Interpretation: this is a step-budget-matched, not compute-matched,
capacity comparison against the 85M COLD anchor (the 200M model used
50,000 steps at a larger per-step cost than the 85M model's 42,000-step
run). It is a real, noise-clearing improvement on the primary metric, not
proof that scale is efficient at this budget; the model card states the
comparison's exact terms. It says nothing about the letters-probe circuit
question, which H026 owns separately.

## Stage 62 - H027 Context-Utilization Probe

Date: 2026-07-23

Handoff:

Stage 62 implements H027. The Stage 61 flagship writes locally fluent but
topically drifting text (the user's "off-topic, like ADHD" review). That drift
is invisible to NLL, so this stage builds a probe that measures whether the
model actually USES its preceding context, and uses the answer to pick the
Phase 6 coherence intervention: a longer context window (if the model uses
context and the window is the wall) versus a subword substrate (if it ignores
the window it already has).

Code change:

New `experiments/tiny_language_lab/eval_context_utilization.py`, reusing
`flagship_eval_lib.load_model`, `encode_fast`, and `LOG2E`, and the text8 split
loader from `eval_text8.ensure_text8`. For each held-out text8 passage it scores
per-target-character NLL under the passage's TRUE first `L_c` chars versus a
RANDOM other passage's first `L_c` chars (a Sattolo derangement, so no passage
sees its own context), with the whole `L_c + L_t` sequence inside one window.
`U = NLL_random - NLL_true` per target offset, bucketed by distance into the
target, with 2,000-resample bootstrap CIs. It also runs a synthetic deep-copy
sensitivity anchor, an `L_c` dose curve, and a double-random control.

Verification:

- Double-random control deep-bucket `U = -0.004342` bits/char, CI
  `[-0.008509, -0.000098]`: with two wrong contexts the signal collapses to
  zero, so the probe does not manufacture a difference.
- Synthetic deep-copy anchor deep-bucket `U = +0.857891` bits/char, CI
  `[+0.843835, +0.872035]`: the deep bucket registers a large signal when deep
  context is genuinely informative, so a null would indict the model, not the
  probe.
- Near-boundary buckets are strongly positive and decay monotonically
  (`+5.72`, `+1.36`, `+0.64`, `+0.37`, `+0.21`), the expected sharp-nearby,
  fuzzy-far-away shape.

Primary command shape:

    python .\experiments\tiny_language_lab\eval_context_utilization.py `
      --checkpoint C:\cassandra_runs\stage61_pure_broad_200m_checkpoints\stage61_pure_broad_200m_seed7_random_full_seed7.pt `
      --split test --n 4096 --context-len 192 --target-len 64 `
      --bucket-edges 4 8 16 32 --seed 20260723 --device cuda `
      --synthetic-anchor --control --context-len-sweep 64 128 192 `
      --out runs\stage62_context_util_flagship.json `
      --summary runs\stage62_context_util_flagship.md

Artifacts:

- `runs/stage62_context_util_flagship.json` and `.md` (N=4096, seed 20260723,
  checkpoint SHA-256 `4e5c0c05...`).
- Dev smoke at N=128 preserved as `runs/stage62_smoke.json` / `.md`.

Results (flagship, 201,609,249 params, block 256, text8 TEST split, N=4096):

| Target chars into segment | U (bits/char) | 95% CI |
| --- | ---: | --- |
| 1-4 | +5.723206 | [+5.657148, +5.800318] |
| 5-8 | +1.357182 | [+1.320663, +1.395129] |
| 9-16 | +0.639015 | [+0.619614, +0.659362] |
| 17-32 | +0.368219 | [+0.356334, +0.380070] |
| 33-64 (deep) | +0.205289 | [+0.198204, +0.212034] |

`L_c` dose curve (deep bucket): `L_c=64` gives `+0.136498`, `L_c=128` gives
`+0.179659`, `L_c=192` gives `+0.205289`; deep utilization RISES with more
available context, so the model's appetite is not saturated at the window edge.

Decision:

Stage 62 reads **E-uses-context** under H027. The deep-bucket `U` CI lower bound
`+0.198204` is far above the `0.05` bits/char line; the control is null and the
synthetic anchor confirms deep sensitivity, so the reading is trustworthy. The
flagship uses its context strongly out to 33-64 characters deep, which spans
nearly the entire 256-character window. The topical drift is therefore a
context-WINDOW-SIZE limit, not a failure to use context: the model uses
everything it can see (about 45 words) and drifts when a longer generation runs
past the window and the opening falls out of view. The rising dose curve is
direct evidence that a larger window would be used, not wasted. Per H027 this
selects the LONGER-CONTEXT char arm (block 512), the cheapest intervention,
which reuses the entire char eval and probe stack unchanged, over the expensive
subword-substrate arm. Recorded in ADR 0019.

Interpretation:

This measures context USE on text8, a proxy for topical coherence, not a human
coherence judgment; the Stage 61 sample sheet and the user's review remain the
ground truth for "reads better." It does not prove a block-512 model will
actually be more coherent to a reader: a bigger window raises the ceiling, but
the model must still learn to use the extra span, and the 8GB card bounds how
far block size can grow before O(n^2) attention becomes prohibitive (Stage 57's
block-512 timing row sizes it). It says nothing about the substrate question for
the specialization gap (Stage 56 owns that) or the copy-circuit question (H026
owns that), though all three concern long-range attention.

### Workstream 2: sampling-grid diagnostic

Date: 2026-07-23

Code change: `flagship_eval_lib.sample_text` gained nucleus (`top_p`) and
`repetition_penalty` controls (backward compatible; the defaults reproduce the
prior temperature-plus-top-k behavior). New
`experiments/tiny_language_lab/make_stage62_sampling_grid.py` regenerates three
fixed prompts across a decoding grid on the flagship, so the same prompt reads
across settings. Artifacts: `runs/stage62_sampling_grid.json` and `.md`.

Finding: decoding trades one failure mode for another around a single
underlying deficit, and cannot manufacture topical coherence.

- The temperature-0.8 no-truncation baseline (the Stage 61 review-sheet
  default) admits a wild low-probability tail: rare near-garbage tokens
  (aircraft designations, broken strings such as "brira s mi two mi six") that
  read as the worst of the drift. Nucleus truncation (`top_p 0.9`) removes
  them, so that part of the perceived incoherence WAS a sampling artifact.
- The residual TOPIC drift survives tighter decoding. At `top_p 0.9` the model
  still wanders subject to subject clause by clause (constitution to Caesar to
  the British crown to Kafka). Lowering temperature does not fix it; it
  collapses the model into REPETITION LOOPS instead ("the united states of
  america and the united states of america ..."; greedy degenerates to "the
  continent and the continent ..."). A mild repetition penalty (`1.3`)
  suppresses the loops but not the drift.
- So the decoder can only choose between wander (high temperature) and loop
  (low temperature), two symptoms of the same missing long-range plan; neither
  is coherence. This is independent BEHAVIORAL corroboration of the H027 and
  ADR 0019 window-limit finding: a model that cannot see its whole generation
  has no global plan for any decoder to exploit.

Practical recommendation (not a science claim): the most readable setting on
this grid is about temperature `0.7` to `0.8` with `top_p 0.9` and a mild
repetition penalty, which avoids both the garbage tail and the loops, and is a
better default for any future sample sheet than temperature `0.8` with no
truncation. The canonical Stage 61 review sheet is left unchanged so the
pending publish review judges a fixed artifact; the user may prefer the grid's
tighter setting.