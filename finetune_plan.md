# Tagging fine-tune spec (v5 → v5-tag)

Drafted 2026-09-15. **Launch is gated on the HRAF analyst review**
(`eHRAF_analysis_v5_a.md`) — their deviation classes may redirect the
objective. Everything below is ready to execute once that gate clears.

## Goal

Improve OCM code selection on the analyst-assist flow without losing what
v5 already does well. Baselines to beat (v5, 21,957-para cross-culture
val): forced+none F1 0.364, free+none 0.362, forced+sec-loo **0.468**
(P 0.491 / R 0.446), sec-oracle 0.539. Realistic target: sec-loo
F1 ≈ 0.50–0.55 — the LOO→oracle gap is the closable part; structural
misses (implicit content the model never sees) are not.

## Base

- Checkpoint `checkpoints/medium_ethnographic_v5` (HF safetensors, 303M).
- Tokenizer `data/ethnographic_v5/` unchanged — no new special tokens.
- Full fine-tune (see `lora_plan.md` for why not LoRA).

## Task format (train on the deployment condition)

Example = optional context header + paragraph + target tag block:

- **sec-loo class (~70%)**: `<|sec|>{title path}\n<|ocm|>{LOO union, named}\n\n`
  + text + `\n` + target `<|ocm|>{named codes}` — LOO union over the
  *other* paragraphs of the same (hdoc, section) group, gold at train
  time. Single-paragraph sections get the bare header (no union line),
  matching eval sec-loo behavior.
- **cold class (~30%)**: no header; text + `\n` + target.

Conventions match the corpus exactly: codes sorted by `(len, c)`, named
form, `\n\n` after the target line so greedy decoding stops naturally.

**Loss masking**: loss only on the target tokens (`<|ocm|>` + codes +
trailing `\n\n`), never on prompt/context — this is the fundamental change
from base LM training.

## Data

- **v5 train split only** — never the val societies; the existing eval
  stays clean. Rebuild examples from the tree via
  `prepare_ethno_tree.parse_file` grouped by (hdoc, section path).
- ~800k examples (stratified across the 8 areas), 1 epoch — ≈ 3–4 h,
  not another 12 h run.
- **Replay ~15–20%**: raw LM text (ethnography + AA journals + codebook)
  with full-LM loss interleaved — preserves fluency and the unconditional
  tagging behavior (free ≈ forced).

## Trainer changes

- New `prepare_finetune.py`: emit tokenized (prompt, target) id pairs.
- `train.py --task-ft`: example-aligned batches (pad-to-longest), masked
  loss, mixed replay stream. `torch.compile` still fine.
- Optimizer: AdamW, **lr 2e-5** (range 1e-5–3e-5), cosine to 10%,
  warmup 100 steps, grad clip 1.0, 1 epoch.
- **Checkpoint selection by tagging F1, not val loss**: every N steps,
  torch-side greedy tag decode (forced marker) on a fixed ~500-paragraph
  val sample; micro-F1 + mean codes/para appended to `eval_history.jsonl`.

## Eval plan (acceptance)

1. Full 4-variant matrix + sec-oracle on the GGUF (new port), compared
   against the v5 baseline table in `progress.md`.
2. **Echo-the-union baseline**: predict the header codes verbatim — the
   FT model must clearly beat this at sec-loo, proving it filters the
   prior rather than parrots it (prior-gaming guard).
3. **Error-propagation probe**: sec-loo variant fed v5's own predicted
   sibling codes instead of gold — quantifies the no-gold deployment gap
   (currently unmeasured).
4. Fluency check: fixed prose prompts before/after; repetition comparison.
5. Accept iff: sec-loo F1 ≥ 0.50 **and** free ≈ forced (within 0.01)
   **and** fluency not visibly degraded.

## Risks / guardrails

- Prior-gaming → echo-union baseline + oracle-gap tracking.
- Format overfit / precision collapse → low LR, 1 epoch, replay mix,
  F1-based checkpoint (not last checkpoint).
- Serving continuity: v5 stays on 8083; FT serves on **8084** as a
  separate run dir (`checkpoints/medium_ethnographic_v5_tag`).

## Open items

- Analyst review outcomes → possible hard-negative emphasis on confused
  code pairs (e.g. 412↔372) or objective tweaks before launch.
- Subset size/stratification details; whether Culture Summary docs'
  sections belong in the FT mix.
