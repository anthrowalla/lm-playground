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

### Stage 0 result (2026-09-15, full 21,957-para cross-culture val)

Base trained 27000/27000 steps; final val_loss 3.1295 (min 3.0930 @ 22.5k;
medium v5 was 3.003). Eval matrix (forced marker unless noted):

| variant            | P     | R     | F1    | exact |
|--------------------|-------|-------|-------|-------|
| forced + none      | 0.391 | 0.312 | 0.347 | 0.107 |
| free + none        | 0.395 | 0.309 | 0.346 | 0.104 |
| forced + sec-loo   | 0.432 | 0.430 | 0.431 | 0.154 |
| forced + sec-oracle| 0.492 | 0.509 | 0.500 | 0.190 |

vs medium v5 (0.364 / 0.362 / 0.468 / 0.539): the 118M base lands within
~5%, free ≈ forced holds here too, LOO gives the same +24% relative lift
and captures ~55% of oracle headroom (medium ~59%). The LOO-conditioning
effect is stable across scale — the fine-tune question is well-posed at
both sizes. FT anchor: sec-loo F1 0.431 (gate: ≥ ~0.46).

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

## Stage 1 result (small FT, 2026-09-17)

Dataset: 799,340 examples (548,564 sec-loo / 251,436 cold ≈ 69/31),
177.2M tokens. Run: 31000 steps ~5.5 h (lr 2e-5, warmup 100, 20% replay);
in-training tagging F1 reached ~0.48 by step 2000, plateaued 0.50–0.526,
**best 0.5259 @ step 28000** (P 0.566 / R 0.491), no late collapse.
Best-checkpoint GGUF (q8_0, 126 MB) served on 8086.

Full-val matrix (21,957 paras, forced marker unless noted):

| variant             | small base | **small FT** | Δ      | medium pre-FT |
|---------------------|------------|--------------|--------|---------------|
| forced + none       | 0.347      | 0.381        | +0.034 | 0.364         |
| free + none         | 0.346      | 0.379        | +0.033 | 0.362         |
| forced + sec-loo    | 0.431      | **0.507**    | +0.076 | 0.468         |
| forced + sec-oracle | 0.500      | 0.546        | +0.046 | 0.539         |
| sec-loo, sibling-PRED unions | — | **0.492** | −0.015 vs gold-union | — |

### Gate verdicts

- **sec-loo ≥ ~0.46** — PASS (0.507). The 118M FT beats medium pre-FT on
  every variant.
- **free ≈ forced** — PASS (0.379 vs 0.381). The FT tags unconditionally;
  the marker-forcing crutch is not load-bearing.
- **echo-the-union guard** — PASS, and informative: on the 1,575 trap
  paragraphs (gold == union, where echoing is the right answer) the FT
  echoes 80.9% of the time (base 88.9%). On the 19,830 paragraphs that
  require real discrimination (gold ⊊ union) FT F1 is 0.490 vs base 0.407,
  and the FT steps *outside* the union more often than the base (22.2% vs
  13.7%) while raising precision (0.549 vs 0.432 overall). The gain is
  discrimination skill, not prior-gaming.
- **error-propagation probe** (`scripts/error_prop.py`, sec-loo with unions
  over siblings' *predicted* codes — the deployment condition) — F1 0.492
  vs 0.507 with gold unions: only −0.015 (−3%). The section prior almost
  fully survives feeding the model its own predictions; the flow degrades
  gracefully in the incremental loop.
- **fluency** — PASS. Free prose continuation shows the same repetition
  loop as the base model (side-by-side on 8085/8086), i.e. a 118M-scale
  property, not FT damage; tag decode is clean (median output = one code
  line).

## Decision gate

If small-FT hits its gates, port the FT tooling to `main` unchanged and
launch the medium run after analyst feedback. If small-FT fails a gate,
the failure mode (prior-gaming vs format-overfit vs fluency loss)
diagnoses the spec before burning the medium run.

**Verdict (2026-09-17): all gates pass.** The FT's sec-loo (0.507) lands
above the *base model's own oracle bound* (0.500) and inside the spec's
medium target band (0.50–0.55); it captures 66% of the remaining LOO→oracle
headroom ((0.507−0.431)/(0.546−0.431)). The strategy — LOO-conditioned
masked-loss FT + LM replay — generalizes across scale; tooling is ready to
port to `main`. Medium FT remains gated on HRAF analyst feedback
(`finetune_plan.md`).
