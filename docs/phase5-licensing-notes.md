# Phase 5 Licensing Notes

Date: 2026-07-09

Purpose: factual due diligence for a future public release. This is not legal
advice and does not choose the project license. The user chooses the license
after reviewing these notes.

## Current Repo State

- No root `LICENSE` file is present as of this note.
- Generated corpus payloads are being removed from the Git index before the
  next commit. The intended release posture is scripts plus metadata, not local
  corpus copies.
- No public push or force-push is allowed without the user present.

## Code License Candidates

| Candidate | Facts | Release paperwork impact |
| --- | --- | --- |
| Apache-2.0 | OSI-approved; includes copyright license, express patent grant, redistribution conditions, NOTICE handling when applicable, and patent-litigation termination. | Strong default if the user wants an explicit patent grant. Needs `LICENSE`; add `NOTICE` if attribution notices are required. |
| MIT | OSI-approved; short permissive license allowing use, copy, modification, publication, distribution, sublicensing, and sale if the copyright and permission notice are included. | Simplest permissive option. No express patent grant in the license text. |
| BSD-3-Clause | OSI-approved; permissive source/binary redistribution terms plus a no-endorsement clause. | Slightly more restrictive than MIT on endorsement language. No express patent grant in the license text. |

The release plan currently notes Apache-2.0 as the recommended option because
of the patent grant, with MIT as the simpler alternative. That remains a
recommendation to review, not a choice already made.

## Dataset Provenance

### TinyStories

- The Hugging Face dataset card for `roneneldan/TinyStories` lists
  `license: cdla-sharing-1.0`.
- The same card describes the dataset as short stories synthetically generated
  by GPT-3.5 and GPT-4, with `TinyStories-train.txt` as the training file used
  for the paper's models.
- CDLA-Sharing-1.0 defines "Data", "Computational Use", and "Results". It says
  Results do not include more than a de minimis portion of the underlying Data.
- CDLA-Sharing-1.0 grants rights to use and publish Data subject to its
  conditions. If publishing received Data or Enhanced Data, Section 3 requires
  publishing under the same agreement, preserving attribution or legal notices
  where present, and not adding further restrictions on the Data.
- CDLA-Sharing-1.0 Section 3.5 says the agreement imposes no obligations or
  restrictions on use or publication of Results.

Implication for Cassandra:

- Do not publish local TinyStories corpus files as part of the repo or model
  package.
- If released model weights are treated as computational Results and do not
  contain more than a de minimis portion of TinyStories text, CDLA-Sharing-1.0
  appears to distinguish them from published Data. This is a factual reading of
  the license text, not a legal conclusion.
- Model cards should still disclose TinyStories as the training source and link
  the dataset card and license.

### text8

- The lab uses text8 as the 100,000,000-character cleaned Wikipedia benchmark,
  with the standard first 90M characters for train, next 5M for validation, and
  final 5M for test.
- The direct benchmark source is Matt Mahoney's text data page. The local
  metadata records the downloaded artifact hashes and split boundaries.
- text8's content lineage is Wikipedia. Wikimedia's current Terms of Use state
  that contributed text is licensed under CC BY-SA 4.0 and GFDL, and that reuse
  must comply with the underlying licenses.

Implication for Cassandra:

- Do not publish local text8 files or reconstructed text excerpts in the repo
  or model package.
- Model cards for Stage 56 or future text8-trained weights should disclose the
  Wikipedia/text8 lineage and cite the benchmark source.
- Before public release of text8-trained weights, keep a conservative note that
  the underlying text is share-alike licensed and that the legal treatment of
  trained weights should be reviewed. A code-plus-recipe release remains the
  fallback if dataset terms are judged incompatible with weight release.

## Third-party Code and Algorithm Provenance

- Muon optimizer: Cassandra implements a single-device Muon-with-aux-Adam path
  adapted to this trainer. The upstream `KellerJordan/Muon` repository presents
  Muon as an optimizer for hidden layers, recommends AdamW for embeddings and
  heads, and is MIT licensed. Any public release should credit the Muon
  lineage in `NOTICE`, `README`, or both.
- PyTorch, Gradio, NumPy, and other installed dependencies are used as
  dependencies, not vendored source. Their licenses should be captured in a
  dependency notice only if binaries or vendored code are distributed.
- RoPE, transformer blocks, activation checkpointing, and n-gram priors are
  implemented locally as standard algorithms or project-specific code. No
  external checkpoint or pretrained model is loaded.

## Release Posture Checklist

- Add chosen root `LICENSE`.
- Add `NOTICE` if Apache-2.0 is chosen or if the release wants a consolidated
  attribution file regardless of license.
- Add `CITATION.cff` after the final release title/authors are chosen.
- Keep generated corpora ignored and untracked; ship scripts and `.meta.json`
  provenance only.
- Do not include sample generations that reproduce long source passages from
  TinyStories or text8.
- Make every model card list training data, evaluation data, and limitations.
- Do not push publicly until the user gives the release go/no-go.

## Sources

- TinyStories dataset card:
  <https://huggingface.co/datasets/roneneldan/TinyStories>
- TinyStories raw README with license metadata:
  <https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/README.md>
- CDLA-Sharing-1.0:
  <https://cdla.dev/sharing-1-0/>
- Matt Mahoney text data page:
  <https://mattmahoney.net/dc/textdata.html>
- Wikimedia Foundation Terms of Use:
  <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>
- OSI MIT license:
  <https://opensource.org/license/mit>
- OSI Apache-2.0 license:
  <https://opensource.org/license/apache-2.0>
- OSI BSD-3-Clause license:
  <https://opensource.org/license/bsd-3-clause>
- KellerJordan/Muon:
  <https://github.com/KellerJordan/Muon>
