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

## Decision context

If the title task lands (clear token-F1 lift over the base model's
zero-shot continuation), summarizing-style conditioning works at 118M and
the same recipe ports to medium with the tagging FT after analyst
feedback. The summary-pair run additionally tests whether topic-aligned
pairing is enough supervision for prose summarization without gold.
