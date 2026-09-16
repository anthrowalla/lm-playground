# Tagging fine-tune, small-model variant (v5-small → v5-small-tag)

Small-model (118M) dress rehearsal of `finetune_plan.md` — same dataset,
same strategy — run on branch `small-v5-tag` while the HRAF analyst review
pends. Purpose: validate the fine-tune tooling and acceptance gates cheaply,
and measure how much of the tagging task a 118M base carries. The medium
(303M) fine-tune then launches on `main` after analyst feedback.

Branch-specific files (duplicates; originals on `main` stay untouched):

- `configs/small_ethnographic_v5.toml` — small dims (768/2688/12L, GQA 12/4)
  on the v5 corpus; same 27k-step token budget as medium v5 (~884.7M tokens,
  ~2.1 epochs of train.bin).
- `finetune_plan_small.md` — this plan.
- Planned next: `train_ft.py` (duplicate of `train.py` + `--task-ft` masked
  loss / example-aligned batches / replay mix) and `prepare_finetune.py`
  (tokenized prompt/target pairs from the tree, LOO section groups) — shared
  logic, built here first, then ported to `main` for the medium run.

## Stage 0 — small v5 base (no v5-convention small model exists yet)

- `train.py --config configs/small_ethnographic_v5.toml`, ~44.9k tok/s →
  **~5.5 h**; val curve persisted in
  `checkpoints/small_ethnographic_v5/eval_history.jsonl`.
- Convert → quantize → serve on **8085** (v5-medium stays 8083, v4 8082,
  v3 8080).
- Baseline the full eval matrix (forced/free × none/sec-loo + sec-oracle)
  on the v5 val with the v5 tokenizer — this anchors every FT delta.
- Note: `tag_eval.py` currently has no small-specific knobs; the model dir
  and tokenizer paths are the only changes.

## Stages 1+ — identical to `finetune_plan.md`

Same task format (70% sec-loo / 30% cold, masked loss on target tokens),
same data rules (v5 train split only, ~800k examples, 15–20% replay),
same acceptance gates (echo-the-union, error-propagation probe, fluency
check, free ≈ forced). Differences:

- LR sweep matters more at 118M: start 2e-5, fall back 1e-5 if F1 is
  unstable; 1 epoch unchanged.
- FT serve port **8086**; run dir `checkpoints/small_ethnographic_v5_tag`.
- Expectation to test: small base sec-loo F1 will land below medium's
  0.468; the question is whether the FT closes a similar *fraction* of the
  LOO→oracle gap. If yes, the strategy generalizes across scale and the
  medium run is de-risked.

## Decision gate

If small-FT hits its gates, port the FT tooling to `main` unchanged and
launch the medium run after analyst feedback. If small-FT fails a gate,
the failure mode (prior-gaming vs format-overfit vs fluency loss)
diagnoses the spec before burning the medium run.
