# Open-Source Release Plan and Post-Phase-5 Disk End-State

Date: 2026-07-08 · expands ADR 0015 D2 into an executable sequence.
Owners are marked per item: (M) Mert, (X) Codex, (C) Claude. Nothing goes
public without Mert present (ADR 0015 hard rule). The release plan and the
disk plan are one plan: the Hub becomes the archive, the laptop keeps only
working copies, and no hardware move is needed.

## Part A · Release preparation

### A1 · Repo and history surgery (the release blocker; also 1.2 GB back)

> **Decision update, 2026-07-21 (Mert, in session):** the public repo will
> be a FRESH INIT with a single initial commit of the cleaned current
> tree. `git filter-repo` surgery is therefore no longer on the critical
> path: the existing private origin (verified PRIVATE on GitHub this day)
> simply remains the full-history archive, and the corpus blobs in its
> history never become public. Item 3 below is superseded except for the
> bundle backup, which stays recommended before any destructive local
> operation. Item 4 becomes: create the fresh public repo, stage the
> curated tree, one initial commit, push, Mert present.
> Also decided 2026-07-21: the code license is **Apache-2.0** (A2 item 2
> executed: root `LICENSE`, `NOTICE` with the Muon credit, and
> `CITATION.cff` are written).

1. (X) `git rm --cached` every tracked corpus file, commit together with
   the updated `.gitignore`. Do this before ANY other commit.
2. (X) Add a pre-commit guard blocking staged files over 50 MB.
3. (M+X) History surgery, Mert present: `git bundle create
   ..\cassandra-backup.bundle --all` (keep the bundle on the external
   drive once it arrives; the OneDrive cloud quota is only 5 GB, so
   OneDrive is NOT a backup target for anything in this project), then
   `git filter-repo` stripping `experiments/tiny_language_lab/corpus/`
   blobs, then verify with `git count-objects -vH` (expect `.git` to drop
   from `1.26 GB` to tens of MB).
4. (M) Push to a FRESH GitHub repository for the public release and keep
   the old remote as a private archive. A fresh remote avoids force-push
   hazards entirely.

### A2 · Licensing and provenance (facts by Codex, choice by Mert)

1. (X) `docs/phase5-licensing-notes.md`: code license candidates (Apache
   2.0 recommended for the patent grant, MIT as the simpler alternative);
   the TinyStories dataset license AS LISTED on its Hugging Face page and
   what it implies for releasing weights trained on it; text8's Wikipedia
   (CC BY-SA) lineage; provenance of any adapted code (the Muon optimizer
   lineage must be credited).
2. (M) Choose the code license; (C) writes LICENSE, NOTICE, and
   CITATION.cff accordingly.
3. Fallback already decided (ADR 0015 reopen item 4): if dataset terms
   are incompatible with weights release, the release narrows to
   code-plus-recipe and says so.

### A3 · Artifact packaging on Hugging Face Hub (this IS the disk relief)

> **Update 2026-07-23:** the first model to publish is the **Stage 61
> pure-broad 200M text8 flagship** (text8 TEST `1.336059` bits/char), not the
> Stage 55 TinyStories flagship this plan was first written around. Its package
> is prepared and upload-ready at `C:\cassandra_runs\stage61_release\`
> (weights license Apache-2.0, user-chosen; recommended inference top-p 0.9,
> user-chosen). The operative, current instructions are
> `docs/release/stage61-flagship/UPLOAD.md`; the model card is
> `docs/release/stage61-flagship/MODEL_CARD.md`; the reproducible packager is
> `experiments/tiny_language_lab/make_stage61_release.py`. The generic A3/A4
> steps below still describe the multi-model end state.

1. (M) Create a Hugging Face account and namespace (about 10 minutes;
   the only step no one can do for you besides sign-offs).
2. (X) Export every keep-set model to fp16 safetensors, model-only:
   flagship 200M about `400 MB`; the 20k replicas about `400 MB` each;
   Stage 56 85M about `170 MB`; Stage 51 25M about `50 MB`. Ship each
   with: config json (layer, head, dim, block, alphabet), the 33-char
   codec as a plain chars list, a minimal `load_model` snippet built on
   `flagship_eval_lib.py`, and the model card (A4).
3. (X) Also upload the full fp32 resume-capable checkpoints (one per
   released model) to a PRIVATE Hub repo as the training archive. This
   is what lets the laptop delete them locally without losing the
   ability to continue training.
4. (C) Model repos to create: the flagship (with its ONNX and Nsight
   manifest attached), the Stage 56 broad-corpus 85M, and the Stage 51
   25M reference. Corpora are NOT uploaded: TinyStories is already on
   the Hub and text8 is public; we ship the prep scripts and meta files
   only (cleaner license posture, zero storage).

### A4 · Public documentation set

1. (C) Public README rewrite: the project story (55+ stages, the
   falsifiability ethos), the headline results table, figures 1 to 5,
   a quickstart (load a model, generate, run the playground), and the
   three-AI-roles workflow as a feature.
2. (C) Model cards grown from `docs/phase4-flagship-evaluation-report.md`
   with the ADR 0015 D6 rule enforced: every number carries the repo
   command that regenerates it. Limitations section carries the
   specialization gap and the A/B coherence status verbatim.
3. (X) Scrub pass: no `C:\Users\senso` paths in released docs, no
   personal artifacts; (M) decides whether `AGENT.md` and the `.claude/`
   automation ship (recommendation: ship them; an AI-operated lab with
   its skills and hooks visible is a distinguishing feature).
4. (C) `REPRODUCING.md`: the command sequence from corpus download to
   flagship eval, smoke-tested on a clean clone by (X).

### A5 · Gates before the button is pressed

- The A/B round-2 verdict is folded into every model card (it shapes the
  coherence wording either way).
- Phase 5 closeout ADR accepted.
- Licensing notes reviewed and license chosen.
- Clean-clone reproduction smoke passed.
- (M) Final go or no-go. "Prepared but unreleased" is a valid end state.

### A6 · Optional demo (post-release)

Playground as a Hugging Face Space on CPU with the fp16 85M model
(about 30 chars/sec generation, adequate for a demo). Stretch goal, not
a gate.

## Part B · Disk end-state after Phase 5 (no new hardware)

Peak CASSANDRA footprint hit about `65 GB` and nearly killed a run. The
end-state budget after the steps below is about `5 GB` local:

| Location | Now | End state | How |
| --- | ---: | ---: | --- |
| `C:\cassandra_runs` | ~19 GB | ~2 GB | After the H022 verdict records: prune Stage 56 seed-7 intermediates (9 x 654 MB) and smoke checkpoints (~1.9 GB), keep finals. After Hub upload (A3): delete local fp32 finals for released models, keep only the CURRENT phase's working checkpoints. Recipe v2's `--checkpoint-keep` stops future accumulation at the source. |
| Repo corpora | ~1.3 GB | ~0 GB | After Phase 5 closeout: delete local TinyStories seed plus shards and the text8 set; both are rebuildable by script (documented in each meta json; text8 re-downloads in a minute, TinyStories in about 30). Keep scripts and meta files. |
| `.git` | 1.26 GB | tens of MB | A1 history surgery. |
| Runs evidence (jsonl, md, logs) | <20 MB | <20 MB | Keep forever; this is the record. |
| Pagefile | 30.1 GB | ~16 GB | (M) It ballooned under memory pressure; a reboot lets Windows shrink it, or cap it at 16 GB in System settings. This alone returns about 14 GB and is independent of CASSANDRA. |

Prune points during the remainder of Phase 5, so the peak never rebuilds:

1. After the H022 verdict is recorded: the Stage 56 intermediate and
   smoke checkpoints (about 7.6 GB back). The 192 MB save-error `.tmp`
   stays: preserved evidence.
2. After each H024 arm finishes: keep that arm's final checkpoint only
   (Recipe v2 `--checkpoint-keep 2` during the run).
3. At closeout, in order: Hub uploads verified (checksums), THEN local
   fp32 deletions, THEN corpora deletion. Never delete before the upload
   is verified, mirroring the ONNX-before-delete rule that already
   worked once.

The escape hatch you asked about (moving the folder to other hardware)
becomes unnecessary under this plan, but for completeness: if it were
ever needed, the move-set is only `C:\cassandra_runs` plus the repo;
everything else is rebuildable. An external drive is strictly worse than
the Hub for this purpose: unversioned, unshareable, and one more thing
to lose.

## Sequencing summary

- **Now, during Phase 5 (no user needed):** A1 items 1 and 2, A2 item 1,
  A4 items 2 and 3 drafting, B prune point 1 once H022 records.
- **One sitting with Mert (about an hour):** A1 items 3 and 4, A2 item 2,
  A3 item 1, pagefile cap.
- **At Phase 5 closeout:** A3 uploads, A4 finalization, A5 gates,
  B closeout deletions.

## Links

- `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md` (D2, D6)
- `docs/phase5-success-criteria.md` (stop-and-ask triggers incl. the new disk gate)
- `docs/phase4-flagship-evaluation-report.md` (model-card source)
- `docs/phase4-flagship-midrun-report.md` (the executed cleanup tiers)
