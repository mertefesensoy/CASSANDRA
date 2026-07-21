# Phase 5 D2 Prep Status

Date: 2026-07-09

This note records the open-source preparation state after Stage 56, Stage 57,
and the eval-only behavior probe. It is release-ready prep, not a release.
Nothing has been pushed publicly.

## Completed By Codex

- Corpus payloads were removed from the Git index with `git rm --cached`.
  Local files were not deleted by that operation.
- `.gitignore` now ignores generated corpus `.txt` and `.json` payloads while
  allowing `.meta.json` provenance files.
- `.githooks/pre-commit` blocks staged additions over 50 MiB, and local
  `core.hooksPath` is set to `.githooks`.
- `.gitattributes` pins `.githooks/*` to LF line endings.
- `docs/phase5-licensing-notes.md` records code-license candidates, dataset
  license facts, text8/Wikipedia lineage, and Muon provenance.
- `docs/phase5-model-card-draft.md` drafts the flagship model card from the
  Phase 4 evaluation report and Phase 5 behavior probe.
- The Stage 56 checkpoint-cleanup tiers already executed during the disk
  emergency: Stage 56 intermediates and smoke checkpoints were pruned; Stage 55
  C-drive fp32 checkpoint copies were demoted under pressure after durable
  evidence and the repo-local seed-7 artifact checkpoint were preserved.
- The Phase 5 behavior probe completed and kept the behavior axis closed:
  `0.060547` constrained-choice accuracy versus chance `0.062500` and reopen
  threshold `0.162500`.

## fp16 Model-only Exports

These are optimizer-free archival exports of the three Stage 56 final step
checkpoints. They were written to `C:\cassandra_runs`, not OneDrive or `%TEMP%`.
Load-back audit confirmed `optimizer_state is None`, archive metadata is
`{"model_only": true, "dtype": "fp16"}`, and all floating model tensors are
`torch.float16`.

| Source final | Model-only archive | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `stage56_broadchar_85m_b50000_seed7_random_full_seed7_step050000.pt` | `C:\cassandra_runs\phase5_model_only_exports\stage56_broadchar_85m_b50000_seed7_fp16_model_only.pt` | 172,230,943 | `c9c7b0fa281f4671d80176d8edffe93ce4ad95128195f3c1f04555bcaab95b44` |
| `stage56_broadchar_85m_b20000_seed11_random_full_seed11_step020000.pt` | `C:\cassandra_runs\phase5_model_only_exports\stage56_broadchar_85m_b20000_seed11_fp16_model_only.pt` | 172,230,750 | `fdc8a943db403f0bccc41016a221b682447bcdea2fe4c103bc421eff40001e6e` |
| `stage56_broadchar_85m_b20000_seed19_random_full_seed19_step020000.pt` | `C:\cassandra_runs\phase5_model_only_exports\stage56_broadchar_85m_b20000_seed19_fp16_model_only.pt` | 172,230,750 | `96795e237162ff6a2ca8f40d854102aca7f8209ca22bd33d55406cafc1a97c68` |

## Explicitly Not Done

- History surgery was not run. It requires explicit user sign-off in the live
  session before `git bundle create ..\cassandra-backup.bundle --all` and
  `git filter-repo`.
- No force-push or public push was attempted.
- No license choice was made. The user chooses the license after reviewing the
  licensing notes.
- No root `LICENSE`, `NOTICE`, or `CITATION.cff` was created.
- No Hugging Face Hub upload was attempted.
- Stage 58 was not started. It remains gated on Claude writing H024.

## Remaining User/Claude Gates

- User: explicit history-surgery sign-off.
- User: code license choice and release go/no-go.
- User: Round-2 A/B votes for the flagship coherence read.
- Claude: H024 hypothesis document before Stage 58.
- Claude: final public README, `LICENSE`, `NOTICE`, `CITATION.cff`, and final
  model-card wording after user decisions.
