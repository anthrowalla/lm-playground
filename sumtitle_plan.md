# Summarization-adjacent fine-tunes (branch `sum-title`, off `small-v5-tag`)

The corpus has no prose summaries (the doc `synopsis` field is subject
keywords), but two summarizing supervisions exist and are exploited here,
one per run, both on the small (118M) base while medium stays gated on
analyst feedback:

## Run 1 — section → title path

Every section's paragraphs paired with its real full title path (~95% of
sections are titled; `[No Title]` skipped; val societies excluded).
Trigger reuses the corpus's own continuation distribution: the model saw
`...text\n\n<|sec|>TITLE PATH\n...` throughout pretraining, so the task is
prompt = section text + `\n\n` + `<|sec|>` (marker forced), target = title
path + `\n\n` (loss masked to it; stop string `\n\n`, exactly the tagging
FT mechanics). No new tokens, no embedding surgery. The task is abstractive
compression to a topical descriptor; ambiguity is irreducible (many valid
paths), so the metric is token-F1 against the gold path (not exact match).

- `prepare_titles.py` — build examples from the tree; head-pack a section's
  paragraphs into the prompt up to the token budget; target ~200k examples.
- `train_ft.py --eval-task titles` — in-training greedy title eval on a
  fixed val-society sample; token-F1 checkpointing (same best-state logic).
- `configs/small_ethnographic_v5_title.toml` — lr 2e-5, 12000 steps
  (~2.3 h; 64-90M tokens of examples, a handful of epochs).
- `scripts/title_eval.py` — full-val post-hoc eval against a served GGUF
  (port 8087 planned): token-F1, exact match, sample generations.

## Run 2 — culture-summary pairing (the `-000` docs)

356 societies have an `hdocid -000` culture-summary doc: concise tagged
prose meant as a backdrop to all other docs of that society. The pairing is
by OCM code overlap, not by reference: for each culture-summary paragraph
(non-`000` codes, real prose) take the best-matching section from the other
documents of the same society by code Jaccard. Prompt = plain-text task
frame (no trigger collisions in the corpus):

    Source ({society}):
    {source text}

    Summary:
    {culture-summary paragraph}

Target = the summary paragraph (loss masked to it). Expected ~10-20k pairs
(thin, but task FTs work at that scale with replay); ~10k steps. No
automatic gold — quality is judged by inspection and later by analysts;
the checkable failure modes are society-prior echo (predicting generic
summary text regardless of source) and verbatim copying.

## Results (2026-09-17, first round)

**Run 1 full-path variant: learned, but wrong granularity.** 194,498
examples (113M tokens), 12000 steps (~2.3 h at half speed — shared GPU
with run 2). Training loss fell 2.0 → ~0.9 (replay batches ~2.9), but
in-training title F1 sat flat at ~0.02 across all six evals. Sample
generations show *why*: the model produces document-appropriate,
topically-adjacent full paths — Kuanyama-style paths for a Kuanyama doc,
"THE CHIEF'S HOUSE" where gold said "THE CHIEF AND HIS CHIEFDOM",
"HOUSING" where gold said "SHELTERS, HUTS, AND HOUSES" — near-synonyms
that word-bag F1 cannot credit, plus long book-prefix ancestors that are
book-identifiable but section-irrelevant. Full paths are the wrong target:
too much irreducible ambiguity, too little credit per near-miss. **Pivot:
retrain with leaf-title targets** (`--granularity leaf`, same recipe).
Base zero-shot anchor (leaf scoring, 2,000 val sections): **F1 0.054**
(P 0.033 / R 0.149).

**Run 2 culture-summary pairing: society-prior echo confirmed.** 10,568
pairs from 344 societies, 3000 steps. Probes on 8088 (q8_0, port 8088):

- fo32-002 (Nkundó parent-child affection) → generic Mongo culture-summary
  boilerplate ("The Mongo are a Bantu-speaking people who live in the
  Congo Basin…"), not a summary of the source.
- fo32-011 (pregnancy/physique) → unrelated bride-price text with heavy
  118M repetition loops.
- st13 (Island Carib, its `-000` doc never seen by FT or corpus): format
  transfers (society-named opening sentence) then degrades into repetition.

The model learned *which society*, not *what the source says* — with all
targets of a society sharing its summary voice, predicting "some Mongo
summary text" already minimizes loss. The pairing signal (top-2 Jaccard)
was too weak to overcome the society prior at 118M. Recorded as a negative
result; a stronger signal (source section's own title path + codes in the
prompt, or tighter Jaccard thresholds) would be the next lever, if any.

## Decision context

If the title task lands (clear token-F1 lift over the base model's
zero-shot continuation), summarizing-style conditioning works at 118M and
the same recipe ports to medium with the tagging FT after analyst
feedback. The summary-pair run additionally tests whether topic-aligned
pairing is enough supervision for prose summarization without gold.
