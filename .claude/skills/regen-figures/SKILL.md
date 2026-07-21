---
name: regen-figures
description: Regenerate the phase evaluation figures from raw run artifacts. Use after new results land (a stage finishes, a benchmark reruns) so report figures never go stale relative to the data.
---

# regen-figures · figures are built from artifacts, never drawn by hand

When to use: any figure-bearing report is about to be read, shared, or
released, and results have changed since the figures were last built.

## Exact steps, in order

1. Run the generator from the repo root:
   `python .\experiments\tiny_language_lab\make_phase4_figures.py`
   (or the current phase's successor script). It parses the training
   logs, the Stage 52 jsonls, and the validation and text8 result jsons;
   missing optional inputs are skipped gracefully.
2. Check the console listing: it must name every expected figure and
   print no matplotlib warnings ("No artists with labels" means a parser
   found no data; treat it as a failure, not cosmetics).
3. Open `docs/figures/phase4/figures_data.json` and verify the plotted
   numbers against their sources (it exists precisely so every plotted
   value is auditable without reading pixels).
4. Visually inspect each PNG for overlapping labels or truncated titles
   before it goes into a document.
5. Figures and `figures_data.json` are committable (small PNGs); the raw
   inputs under `runs/` stay gitignored.

## Full example of a good final output (real console output and data file, 2026-07-07)

```text
[figures] wrote 5 figures to C:\...\CASSANDRA\docs\figures\phase4
  - fig1_flagship_learning_curves.png
  - fig2_capacity_ladder.png
  - fig3_h019_crossover.png
  - fig4_gpt1_comparison.png
  - fig5_efficiency_frontier.png
```

And the audit file (`docs/figures/phase4/figures_data.json`, excerpt):

```json
{
  "stage52_random_full_bits": {
    "random_full_b2000": {
      "3176481": 1.3979801507916507,
      "10672929": 1.3473463638399177,
      "25253921": 1.3303769038706348,
      "85106721": 1.3249408289565652
    }
  },
  "text8_flagship_bits_per_char": 2.881723,
  "figures": [
    "fig1_flagship_learning_curves.png",
    "fig2_capacity_ladder.png",
    "fig3_h019_crossover.png",
    "fig4_gpt1_comparison.png",
    "fig5_efficiency_frontier.png"
  ]
}
```

## Mistakes to avoid (each one actually happened here)

- **The training logs are UTF-16.** Windows PowerShell's `Tee-Object`
  writes UTF-16 with a BOM; reading them as UTF-8 made every regex miss
  and produced an EMPTY learning-curve figure that still saved without
  error. The parser must sniff the BOM. Watch for silently empty plots.
- **`runs/` is gitignored, so file-search tools skip it.** Globbing for
  run artifacts through gitignore-respecting tools returns nothing; list
  the directory directly.
- **Regenerate after late-arriving results.** The GPT comparison figure
  was first built before the text8 benchmark finished and had to be
  rebuilt to include it; a figure snapshot taken mid-pipeline is a stale
  claim.
- **Prefer the `_sharded` rerun files where both variants exist** in the
  Stage 52 jsonls (the plain prior files predate the OOM fix).
- **Never hand-edit a figure or its numbers.** If a figure is wrong, fix
  the parser or the source artifact and regenerate; the data json exists
  so discrepancies are catchable.
