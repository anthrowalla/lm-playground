# Way forward — corpus mix and model size

Considerations recorded 2026-09-13, pending the `medium_ethnographic_v4` run
and the OCM-tagging eval. Nothing here is decided. Companion documents:
`q_larger.md` (the 1.3B/epoch math) and `progress.md` (history and results).

## The question

What text mix should future corpora carry, and is the medium (303M) model
sufficient — or do we grow the corpus *and* move to a larger model?

Three candidate directions:

1. **High-quality multidomain (general English) mix.** Helps fluency and
   robustness, not OCM tagging directly. If added, keep the val split
   ethnographic-only so the metric keeps tracking the actual task.
2. **Rely on the journals.** The American Anthropologist run (1880s–2005,
   ~+5–10% tokens) is stylistically closer to what analysts actually feed the
   model — AAA-style theoretical/review prose vs eHRAF's descriptive
   ethnography. This is a *coverage* argument, not a scale argument; by
   itself it cannot sustain much more training (see `q_larger.md`).
3. **Broadly related academic sources + a larger model.** Statistics, logic,
   linguistics, philosophy, economics, and coded classifications of the
   cultures involved (e.g. SCCS/Ethnographic-Atlas-style coded data, which
   could be formatted like the OCM codebook as reference docs). This is the
   only path that could grow effective tokens toward the ~10B+ a 1.3B model
   wants — and the intended use shifts: scientific/educational purposes
   beyond HRAF analyst-assist.

## Working hypothesis

The medium transformer is likely sufficient for many HRAF purposes. The
larger model is interesting mainly for scientific/educational uses — with
the caveat from `q_larger.md` that a larger model on a thin corpus *loses*
to a well-trained smaller one, so option 3 stands or falls with actually
building the corpus to match.

## Decision gates

- **Gate 0 — OCM-tagging eval (precision/recall vs gold paragraph tags on
  held-out ethnographies).** This is the project's real capacity test. All
  corpus-mix and size decisions should be justified against it, not
  perplexity.
- **Gate 1 — v3 vs v4 on the tagging eval.** Did named tags + journal text
  help? If tagging errors drop, corpus-text levers are confirmed; if not,
  the mix question changes shape (the lever may be task fine-tuning, not
  pretraining text).
- **Gate 2 — where do the remaining errors come from?**
  - Tagging/structure errors → option 3's related-domain sources are the
    better mix (linguistics, coded classifications, methods/statistics
    texts teach exactly the surrounding conceptual vocabulary).
  - Fluency/robustness errors → option 1's general multidomain mix.
- **Gate 3 — size.** Move beyond medium only if (a) the tagging eval shows a
  capacity limit at 303M, AND (b) the corpus has grown toward the ~10B+
  effective-token regime a 1.3B model needs — or the larger model is
  accepted as a deliberately data-thinner scientific/educational experiment
  (~1.5 days compute at the medium recipe for ~4x fewer tokens/param).

## Sequencing proposal

1. v4 finishes → convert → quantize → serve.
2. Build the OCM-tagging eval; run v3 vs v4.
3. Choose corpus v5 mix per Gate 2; whichever direction, val stays
   ethnographic-only.
4. Revisit model size only after the eval says the medium is the bottleneck.
