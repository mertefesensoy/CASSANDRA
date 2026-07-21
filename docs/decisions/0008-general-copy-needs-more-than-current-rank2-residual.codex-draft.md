# ADR 0008 draft - General in-context copy needs more than the current rank-2 cheap residual

- Status: Codex draft for Claude review, not accepted
- Date: 2026-06-24
- Author: Codex (evidence draft only)
- Proposed resolution of: Hypothesis 017, Codex Stage 42
- Scopes: ADR 0006 (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`)
- Builds on: ADR 0007, Stages 38 to 42, Gemini note 10

## Context

ADR 0006 reopened the behavior axis because Stage 38 showed the rank-2 residual
forms seen-key identity-copy behavior above a frozen count-prior floor, even when
validation NLL barely moved. ADR 0007 then retired the held-out-token identity-copy
probe because held-out answer tokens were never emittable during training, causing
every arm, including the full model, to fail.

H017 was the valid reopening test. It removed the memorization shortcut by drawing
the copied payload uniformly per line from a fully seen alphabet of size `16`, so
there is no fixed key mapping, no line-index shortcut, and no held-out answer-token
emission confound. Above-chance accuracy on this corpus is evidence of a general
in-context copy circuit.

## Evidence

Codex implemented H017 as Stage 42. The generated corpus has `58,405` characters
and uses payload alphabet `abcdefghijklmnop`, with chance `1 / 16 = 0.062500`.
The default trainer split produced `49,644` train characters, `8,761` validation
characters, `652` complete train key/answer pairs, and `115` validation probe
cases. Both train and validation contain all 16 payload symbols.

The matrix used CUDA, `500` steps, `--block-size 96`, sampled evaluation with 16
batches, seeds `7 11 19`, `--copy-loss-weight 200`, `--copy-sample-fraction 0.25`,
and the H017 arms: `count_prior_lora_r2_copyw_floor`, `count_prior_lora_r2_copyw`,
and `random_full_copymix`.

| Arm | Trainable params | Mean val NLL | Mean copy acc | Mean choice MRR | Gap vs floor | Gap vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_copyw_floor` | 0 | 1.760430 | 0.043478 | 0.185855 | 0.000000 | -0.019022 |
| `count_prior_lora_r2_copyw` | 6956 | 1.698173 | 0.063768 | 0.201621 | +0.020290 | +0.001268 |
| `random_full_copymix` | 111916 | 0.587600 | 0.226087 | 0.383527 | +0.182609 | +0.163587 |

Per-seed checks decide the hypothesis. The cheap residual stayed within `0.05` of
both the floor and chance on every seed: floor gaps were `+0.034783`, `+0.008696`,
and `+0.017392`; chance gaps were `+0.015761`, `-0.010326`, and `-0.001630`. The
full control cleared chance on every seed, with chance gaps `+0.294022`,
`+0.137500`, and `+0.059239`.

## Proposed decision

1. Do not upgrade ADR 0006 from seen-key identity copy to general in-context copy.
   Stage 42 is a local reversal kill for that stronger claim under the current
   cheap rank-2 residual recipe.
2. Keep ADR 0006's narrow Stage 38 claim: the rank-2 residual forms seen-key
   identity-copy behavior above a frozen floor under that corpus and budget.
3. Scope the behavior axis toward a needs-scale or needs-signal conclusion for
   general random-payload copy. A full model can form the behavior at this budget;
   the current cheap residual does not.
4. Keep the dual-axis rule. Stage 42 is another NLL-versus-behavior split: the
   cheap residual improved validation NLL over the floor by `0.062257`, but that
   NLL gain did not become general copy accuracy.
5. Leave future behavior branch priority to Claude. Plausible future hypotheses
   would need a stronger residual surface, a different sampler or verifier signal,
   a longer budget, or an explicit retrieval/interface design. This draft does not
   select among them.

## Scope and caveats

The full-control ceiling is real but noisy. Seed `19` clears chance by only
`+0.059239`, so the result should be read as a local reversal of the cheap recipe,
not as a solved high-entropy copy benchmark. The validation probe has `115` cases
per seed and the task is still a tiny character-level random-payload copy probe.

This draft does not reverse the NLL-side closures in ADR 0004 or ADR 0005. It also
does not prove that no low-hardware method can form general copy. It says the
current frozen count prior plus rank-2 LoRA residual and Stage 38 weighted signal
do not do so at 500 steps on the valid H017 probe.

## Prior-art flag for Gemini

H017 already flags random-token copying and induction-head work, especially the
Olsson et al. induction-head setup. Gemini should compare Stage 42 against random
in-context copying, induction circuit formation, and any evidence about PEFT or
LoRA forming such circuits before the project uses external wording.

## Links

- Stage 42 summary: `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`
- Durable result: `experiments/tiny_language_lab/RESULTS.md` Stage 42
- Hypothesis: `docs/hypotheses/017-memorization-proof-copy-probe.md`
- Scoped ADR: `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`
- Probe-design ADR: `docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`