# Phase 5 Completion Audit

Date: 2026-07-21 local

Status: COMPLETE for the empirical Phase 5 program. This audit records the
evidence supporting that closeout. It is a status ledger, not a new decision
record, and does not supersede ADR 0015 or H024.

## Completion Bar

Phase 5 is complete only when all of these are true:

1. The specialization gap has a clean measured cause.
2. Recipe v2 is locked from completed gates.
3. The developmental question has a three-arm, fixed-budget answer with
   deterministic text8 TEST scores and TinyStories retention curves.
4. Required replicas have tested the deciding sign, with any inconclusive case
   escalated according to H024.
5. Durable evidence states the verdict, caveats, and next decision.
6. The repository is release-ready, subject only to the user's public-release
   choices and explicit history-surgery approval.

## Evidence Ledger

| Requirement | Current evidence | Status |
| --- | --- | --- |
| H022 cause of specialization gap | Stage 56 seed 7 scored 1.485740 text8 TEST bits/char, below the 1.70 CONFIRM line; seed-11 and seed-19 20k replicas differed by 0.003035 | Complete |
| Recipe v2 | bf16 rejected on throughput, cosine adopted after a -0.068177 sampled-NLL gain, checkpoint retention and union-vocab smoke tests passed | Complete |
| Behavior probe | 0.060547 constrained-choice copy accuracy versus 0.062500 chance; behavior axis stays closed | Complete |
| COLD primary arm | 85.11M, 42,000 steps, deterministic text8 TEST 1.357318 bits/char | Complete |
| CURRICULUM primary arm | Final checkpoint SHA-256 `F8A734...6DC71CCD`; sampled NLL 0.920510 passes the 1.035283 guard; deterministic text8 TEST 1.362414 | Complete |
| MIXTURE primary arm | Final checkpoint SHA-256 `45D312...6EDC707`; sampled NLL 0.905836 passes the 1.057650 guard; deterministic text8 TEST 1.385295 | Complete |
| Retention protocol | COLD, CURRICULUM, and MIXTURE full-source curves completed at 1,499,904 characters per point; MIXTURE final is 0.826285 bits/char | Complete |
| Three-arm decision | Seed-7 primary COLD versus CURRICULUM delta is +0.005096 bits/char, E-null; secondary MIXTURE deltas are +0.027977 versus COLD and +0.022881 versus CURRICULUM | Complete |
| Replica sign check | Seed 11 gives +0.009845 and seed 19 gives +0.007791 for CURRICULUM minus COLD. Both signs agree with the seed-7 +0.005096 primary delta; all three magnitudes are inside E-null | Complete, seed-robust in sign; no escalation required |
| Durable Stage 58 closeout | Seed-7 three-arm figures plus a fail-closed three-seed replica sign panel, Markdown, and JSON evidence are written under docs/figures/phase5; RESULTS and closeout reports record the verdict and caveats | Complete |
| Release-ready D2 preparation | Corpus is untracked, cleanup tiers executed, license notes and model-card draft written, three fp16 model-only exports audited | Complete except user-gated actions |

## Current Safety Evidence

- COLD, CURRICULUM, and MIXTURE checkpoints wrote to `C:\cassandra_runs`, not
  OneDrive or `%TEMP%`.
- The phase-2 continuation restored the durable step-15,000 checkpoint with
  `formation_forward_passes = 30000`.
- It reached an audited step-20,000 checkpoint with `formation_forward_passes
  = 40000`, SHA-256
  `62F54CC84DF9FAA870C642C8DD98F75B3B24FBE47B0D5B7C7E518A89BDA3D8CD`, and
  sampled broad validation NLL `1.006270`.
- Its registered step-25,000 guard checkpoint recorded sampled broad validation
  NLL `0.985283` and SHA-256
  `9EF44D15539BD5D2F220F36A7F295F6912549E64591822D6F0EA8D3109FFD811`.
  The final arm is unstable only if its sampled NLL exceeds `1.035283`.
- Step 30,000 then recorded sampled broad validation NLL `0.914095` and
  SHA-256 `B443CFB7036BF59B3DC7B5A6B6F927AA0401CBE27B42A08929162ABDEB5D23CB`.
- Step 35,000 then recorded sampled broad validation NLL `0.905480` and
  SHA-256 `6F0B714377F4AB3B5470F23779975C97F085B4F4C6A5CA9CD8D3F52E9DA922DE`.
- Step 40,000 then recorded sampled broad validation NLL `0.885336` and
  SHA-256 `F694A885E3746E972ABC50EFFE4408E098270CB459FAD75C75FE2C7CAFA1E772`.
- CURRICULUM final checkpoint SHA-256 is
  `F8A734503143764141C96924ED233F12FBDD9734D984E0DF7886FC5F6DC71CCD`; its
  sampled broad validation NLL `0.920510` passes the registered `1.035283`
  stability line, and its deterministic text8 TEST score is `1.362414`.
- COLD and CURRICULUM retention reports use the same uniform full-source
  1,499,904-character TinyStories validation sample at every point. Their final
  values are `3.556502` and `3.529069` bits/char respectively.
- Process-bound helpers asserted system and display activity for MIXTURE training
  and both MIXTURE GPU evaluations, then stopped naturally with their watched
  processes.
- The launch gate has substantial free space above the 15 GiB threshold.
- The corrected 25:59 MIXTURE corpus began from random initialization only after
  dose control was verified, so the correction invalidated no model result.
- Seed-19 COLD's resumed step-20,000 compute attempt is preserved as failed
  evidence only: its final checkpoint and deterministic TEST do not exist after
  Windows error code 5 denied checkpoint writes. The audited step-15,000 resume
  checkpoint is unchanged; one-byte writes to its current checkpoint directory
  and to a fresh child of C:\cassandra_runs are currently denied.

## 2026-07-21 Recovery Clarification

The preceding error-code-5 notes are historical process-containment evidence,
not the current state of `C:\cassandra_runs`. An unrestricted atomic PyTorch
preflight passed without changing ACLs, demonstrating that Codex's restricted
workspace-write execution context caused the failed writes. The visible external
launcher then completed the seed-19 COLD recovery and a fresh paired
CURRICULUM run. Their final checkpoint SHA-256 values are, respectively,
`0A15419D0FA64165117FB9B251F341A9B2119CCAC108343041C1805199A60771` and
`2A1B209C8278992184322D947476D5BAE38FD24DFA56AEB6CCD932FD253373C6`; their
deterministic text8 TEST scores are `1.410779` and `1.418571` bits/char.
## Execution Closeout

1. An unrestricted atomic PyTorch preflight passed in `C:\cassandra_runs`.
   The former error-code-5 condition was isolated to Codex's restricted
   workspace-write process, not a disk, ACL, or hardware failure.
2. Seed-19 COLD resumed from the audited step-15,000 checkpoint, wrote its
   step-20,000 and final checkpoints, and scored `1.410779` deterministic text8
   TEST bits/char.
3. Seed-19 CURRICULUM restarted fresh at the registered 5,952-step childhood
   dose, used one-thousand-step recovery checkpoints, completed its continuous
   20,000-step horizon, and scored `1.418571` deterministic text8 TEST
   bits/char.
4. The seed-19 paired delta is `+0.007791`; it agrees with seeds 7 and 11.
   H024 therefore does not trigger full-budget seed-11/19 escalation.
5. The final evidence package and caveats are recorded in
   `docs/phase5-final-report.md` for Claude's closeout ADR.

## User-Gated Release Actions

These are deliberately not autonomous and do not authorize any public push:

- Explicit sign-off before bundle backup, history surgery, or force-push.
- Code-license choice after review of `docs/phase5-licensing-notes.md`.
- Round-2 flagship-versus-Stage-51 A/B votes for the release narrative.
- Public-release go or no-go after the scientific closeout.

Historical text in `docs/phase5-d2-prep-status.md` that says Stage 58 had not
started is a dated July 9 snapshot. The current Stage 58 state is authoritative
in `experiments/tiny_language_lab/RESULTS.md`, this audit, and the live run
artifacts.
