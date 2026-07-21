# Phase 5 Developmental Mid-Run Report

Date: 2026-07-21 local

Status: COMPLETE for the empirical Phase 5 program. All three Stage 58 seed-7
arms, the full-source retention curves, and both reduced-budget H024 sign
replicas are evidence-complete. The release-preparation boundary remains
user-gated.

## Executive Read

Phase 4 established that a 201.6M model can learn TinyStories well, but it also
measured a large specialization gap. Phase 5 narrows that gap to a
data-distribution effect, locks a measured recipe, and answers the fixed-compute
developmental question.

The completed Stage 58 primary comparison is COLD 1.357318 versus CURRICULUM
1.362414 text8 TEST bits/char at 85.11M parameters and 42,000 steps. The
registered primary delta is +0.005096 bits/char (CURRICULUM - COLD), inside
H024's practical E-null band. Reduced-budget seed 11 gives +0.009845 and seed
19 gives +0.007791. All three signs favor COLD while every magnitude remains
inside E-null: the registered H024 result is **E-null, seed-robust in sign**.

## Replica Status Update

The reduced-budget seed-11 and seed-19 packages are evidence-complete:

| Seed | COLD TEST bits/char | CURRICULUM TEST bits/char | CURRICULUM minus COLD |
| ---: | ---: | ---: | ---: |
| 11 | 1.410154 | 1.419999 | +0.009845 |
| 19 | 1.410779 | 1.418571 | +0.007791 |

The earlier seed-19 error-code-5 event was isolated to Codex's restricted
workspace-write process. An unrestricted atomic PyTorch preflight passed in
`C:\cassandra_runs`; the compliant visible external launcher then completed the
COLD recovery and a fresh CURRICULUM phase-1 plus continuous phase-2 package.
The two reduced-budget signs agree with seed 7, so H024 does not require the
full-budget escalation condition.

## Phase 4 To Phase 5

| Dimension | Phase 4 closeout | Phase 5 progress |
| --- | --- | --- |
| Core question | Can a larger model learn the narrow TinyStories register reliably? | Does acquisition order improve broad-text learning at fixed compute? |
| Main model | 201.61M, 50,000 steps, TinyStories | 85.11M, 42,000 steps, text8 and TinyStories |
| Main metric | TinyStories validation bits/char | Held-out text8 TEST bits/char |
| Key measured result | 0.812571 chunked TinyStories validation bits/char for the seed-7 flagship | 1.357318 chunked text8 TEST bits/char for the matched COLD baseline |
| Generalization signal | Flagship zero-shot text8: 2.8817 bits/char | H022 broad-training anchor: 1.485740 at 50k; Recipe v2 COLD: 1.357318 at 42k |
| What changed the plan | Stage 53 showed a frozen analytic prior becomes a late-training handicap | Stage 56 confirmed the specialization gap is primarily data distribution, so the char substrate remains valid |
| Current scientific test | Completed | COLD versus CURRICULUM is primary; MIXTURE is the dose-matched secondary order contrast |

The Phase 4 and Phase 5 headline scores are not a leaderboard comparison. They
use different models, budgets, corpora, and evaluation roles. Their useful
connection is causal: Phase 4 exposed the narrow-to-broad gap, Stage 56 showed
that direct broad training closes it, and Stage 58 asks whether a narrow learned
starting point earns back the broad-text exposure it displaces.

## Completed Phase 5 Work

| Workstream | Result | Evidence state |
| --- | --- | --- |
| Stage 56, H022 broad-corpus test | CONFIRM. Seed 7 scored 1.485740 text8 TEST bits/char at 50,000 steps, below the 1.70 line; 20k replicas differed by 0.003035 bits/char | Complete and documented |
| Stage 57 Recipe v2 | fp32 retained; bf16 rejected on throughput; cosine adopted after a -0.068177 sampled-NLL gain; checkpoint retention and 33-character vocab smoke tests passed | Complete and locked |
| Behavior probe | 0.060547 copy accuracy versus 0.062500 chance; behavior axis stays closed | Complete and documented |
| D2 release preparation | Corpus untracked, guardrails and due-diligence docs drafted, model-only exports audited; history surgery and public release remain user-gated | Release-ready work in progress |
| Stage 58 corpus work | Full 500 MB TinyStories preparation, union vocabulary, deterministic mixture generator, and retention evaluator implemented and verified | Ready |

## Stage 58 Controlled Comparison

Each arm uses the same 85,106,721-parameter model, 42,000 optimizer steps,
4,096 training characters per step, fp32, Muon, RoPE, block size 256, a
continuous 42,000-step cosine schedule, and the same 33-character union
vocabulary. That fixes the total training budget at 172,032,000 characters per
arm. The 42,000-step budget is authorized by ADR 0015, which delegates exact
steps to the sustained Recipe v2 throughput measurement: 5,000 steps required
4,992.0457 seconds, projecting 42,000 steps to about 11.65 GPU-hours.

| Arm | TinyStories characters | Broad characters | Status | Deterministic text8 TEST |
| --- | ---: | ---: | --- | ---: |
| COLD | 0 | 172,032,000 | Complete | 1.357318 |
| CURRICULUM | 51,200,000 | 120,832,000 | Complete | 1.362414 |
| MIXTURE | 51,200,000 | 120,832,000 | Complete | 1.385295 |

COLD completed with a final sampled broad-validation NLL of 0.884350 and 1.275847
sampled bits/char. The deterministic full TEST score is the decision score.

CURRICULUM phase 1 learned TinyStories from random initialization for 12,500
steps, reaching 0.911683 sampled TinyStories validation bits/char. Its phase-2
resume smoke confirmed the checkpoint lineage and the one continuous 42,000-step
cosine horizon. The active continuation restored the audited step-15,000
checkpoint with `formation_forward_passes = 30000` and broad validation NLL
1.078918 before continuing. It reached a new durable step-20,000 checkpoint
with `formation_forward_passes = 40000`, broad validation NLL `1.006270`, and
recorded SHA-256 `62F54CC84DF9FAA870C642C8DD98F75B3B24FBE47B0D5B7C7E518A89BDA3D8CD`.
The registered step-25,000 broad-phase guard point then reached validation NLL
`0.985283` and SHA-256 `9EF44D15539BD5D2F220F36A7F295F6912549E64591822D6F0EA8D3109FFD811`.
The final sampled broad validation must not exceed `1.035283` NLL or the arm
becomes inconclusive under H024's instability rule. This remains monitoring,
not a developmental verdict. The step-30,000 continuation checkpoint then
reached broad validation NLL `0.914095` with SHA-256
`B443CFB7036BF59B3DC7B5A6B6F927AA0401CBE27B42A08929162ABDEB5D23CB`.
Step 35,000 then reached broad validation NLL `0.905480` with SHA-256
`6F0B714377F4AB3B5470F23779975C97F085B4F4C6A5CA9CD8D3F52E9DA922DE`.
Step 40,000 then reached broad validation NLL `0.885336` with SHA-256
`F694A885E3746E972ABC50EFFE4408E098270CB459FAD75C75FE2C7CAFA1E772`.

## CURRICULUM Completion Update

The final step-42,000 checkpoint has SHA-256
`F8A734503143764141C96924ED233F12FBDD9734D984E0DF7886FC5F6DC71CCD` and
sampled broad validation NLL `0.920510`, safely below the `1.035283` guard.
Its deterministic full text8 TEST score is `1.3624141296719878` bits/char
over 4,999,936 characters. Against COLD, this is `+0.005096447457270337`
bits/char: the registered seed-7 primary outcome is provisional E-null.

The matching full-source TinyStories retention series is complete for COLD and
CURRICULUM. CURRICULUM retained `0.877782` bits/char at the end of phase 1,
then measured `3.188760` at broad step 15,000 and `3.529069` at step 42,000.
COLD measured `3.556502` at step 42,000. These curves describe catastrophic
forgetting under the fixed-budget protocol; they do not override text8 TEST.

## Dose Control Correction

The initial untrained mixture used the original 1:3 ratio, appropriate to the
superseded 50,000-step arithmetic but not to the measured 42,000-step budget.
It would have supplied 43,008,000 TinyStories characters, while CURRICULUM had
already committed to 51,200,000. No model was trained on that preliminary
mixture. Its provenance was recorded as ratio 1:3 and SHA-256
`c594e57a5dc67aa6e6ee527d561671b8a6997a3816e4b4dc6dac56e0121f09db`.

The launcher now derives mixture weights from `Phase1Steps : (Budget -
Phase1Steps)`. At 12,500 and 42,000 it emits 25:59, and the regenerated
mixture has exactly 51,200,000 TinyStories and 120,832,000 broad characters.
The final mixture SHA-256 is
`874f8212a6436a132f57a6986dbdfc8253344e4c9d664109a30149d6104c7f95`.
This restores the equal-dose control required to interpret MIXTURE versus
CURRICULUM as an order comparison, subject to H024's already-registered
learning-rate exposure caveat.

## MIXTURE Completion Update

MIXTURE completed its common 42,000-step budget with final sampled broad NLL
`0.905836`, passing its registered `1.057650` stability ceiling. Its deterministic
text8 TEST result is `1.3852951562123879` bits/char over 4,999,936 characters.
The secondary deltas are `+0.027977` versus COLD and `+0.022881` versus
CURRICULUM, both within the registered 0.05 practical band.

Its full-source TinyStories retention curve is the planned rehearsal contrast:
`1.170210` at step 5,000, descending to `0.826285` at step 42,000. By contrast,
COLD and CURRICULUM finish at `3.556502` and `3.529069`. This is a large
retention separation, but it remains descriptive rather than the primary
broad-text decision metric. The fail-closed generator has written the three-arm
comparison Markdown, plotted-data JSON, and two figures under `docs/figures/phase5/`.

## Decision Rules

The primary delta is `CURRICULUM - COLD` on deterministic text8 TEST
bits/char.

- At or below -0.05: E-curriculum. The learned childhood earns back its cost.
- At or above +0.05: E-interfere. The learned childhood repeats the Stage 53
  interference pattern.
- Between -0.05 and +0.05: E-null at this budget and dose.

MIXTURE comparisons use the same 0.05 practical line but remain secondary. The
TinyStories retention curve is descriptive evidence about forgetting and
rehearsal, not a substitute for the primary broad-text score.

## Run Safety And Next Evidence

MIXTURE completed cleanly with visible launch, its process-bound display/system
keep-awake helper, and C: remaining far above the launch gate. Its corrected
25:59 shards preserve exact TinyStories exposure parity with CURRICULUM; the
one-night schedule skip was consumed before the run began.

The deterministic text8 TEST reports, full-source retention curves, three-arm
comparison Markdown, figures, and plotted-data JSON now exist for seed 7.
Seed 11 has completed its reduced COLD versus CURRICULUM sign check. Remaining
execution is to restore a compliant checkpoint root, rerun seed-19 COLD from
the preserved step-15,000 checkpoint, evaluate it deterministically, then run
the paired seed-19 CURRICULUM package and apply H024's sign rule.

Once all three text8 and retention reports exist, regenerate the Phase 5
comparison figures, Markdown table, and plotted-data JSON with:

```powershell
python .\experiments\tiny_language_lab\make_phase5_figures.py
```

## Primary Artifacts

- `experiments/tiny_language_lab/RESULTS.md`
- `experiments/tiny_language_lab/runs/stage58_dev_cold_85m_b42000_seed7_text8_test.md`
- `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase1_85m_b12500_seed7.md`
- `experiments/tiny_language_lab/runs/stage58_dev_curriculum_phase2_85m_b42000_seed7_text8_test.md`
- `experiments/tiny_language_lab/runs/stage58_dev_cold_retention.md`
- `experiments/tiny_language_lab/runs/stage58_dev_curriculum_retention.md`
- `experiments/tiny_language_lab/runs/stage58_dev_mixture_85m_b42000_seed7_text8_test.md`
- `experiments/tiny_language_lab/runs/stage58_dev_mixture_retention.md`
- `docs/figures/phase5/stage58_developmental_comparison.md`
- `docs/figures/phase5/fig1_stage58_text8_primary.png`
- `docs/figures/phase5/fig2_stage58_tinystories_retention.png`
- `experiments/tiny_language_lab/corpus/mixture_char_shards.meta.json`
- `experiments/tiny_language_lab/make_phase5_figures.py`
- `docs/hypotheses/024-developmental-acquisition-order.md`
## 2026-07-21 Closeout Addendum

The empirical Phase 5 program is complete. The full closeout, stability table,
replica figure, and interpretation limits are in `docs/phase5-final-report.md`.
The three figures and their machine-readable evidence are in
`docs/figures/phase5/`. This addendum supersedes the dated "Run Safety And Next
Evidence" execution paragraph above. No public release action is implied by
this result.