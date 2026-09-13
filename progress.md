# Progress log

Project history and measured results. The companion documents are `README.md`
(pipeline reference) and `q_larger.md` (the "would 1.3B pay off" analysis).

## Origin

This project began as an evaluation of [`../aarambh-studio`](../aarambh-studio),
a Rust framework for developing small LLMs. The plan was to follow its specs for
tiny, small, and medium models. Training speed in Rust measured **14-15x slower**
than conventional Python-based tools, so the work moved here into a
PyTorch + llama.cpp pipeline. The two repositories are formally independent,
each with its own `origin`; lm-playground inherits aarambh-studio's model
sizing and general architecture intent, not its code.

## Timeline

### 2026-09-11 — pipeline established

- `3d9d7a4` PyTorch training + llama.cpp serving pipeline. Tiny Shakespeare
  smoke test end to end: `prepare.py` (ByteLevel BPE tokenizer, uint16 bins)
  → `train.py` → `convert.py` (GGUF) → `llama-quantize` → `llama-server`.
- `3e8f9c5` ethnographic corpus builder (`prepare_ethno.py`) with
  per-document metadata, on an eHRAF-style CSV export.
- `a94a788` medium config: batch 32 / accum 1 (same effective batch,
  ~6x throughput vs the previous setting).

### 2026-09-12 — corpus v2/v3, small + medium trained

- `c6ed02c` corpus v2: markup filter, ByteLevel decoder, small ethnographic config.
- `be5c843` corpus v3: per-paragraph OCM tags (`<|ocm|>131 133` — bare codes,
  the human analysts' assignments), `<|div|>` division markers, OWC/field dates.
  Purpose: teach the model the analysts' coding scheme so it can assign codes
  to new text.
- `767624d` / `6d9fba1` corpus v3.1: OCM codebook document (codes, names,
  definitions from `data/ethnographic/ocmdefs.txt`) embedded as reference
  docs; 4-digit code fix.
- `5e60d77` README, MIT license, corpus scaling notes.

Checkpoints completed: `tiny_shakespeare`, `small_ethnographic` (118M),
`medium_ethnographic` (303M), each converted to f16 + Q8_0 GGUF.

### 2026-09-13 — analysis, corpus v4, v4 training run

- `d2b0a92` measured answer on epoch extension (see `q_larger.md`):
  medium-v3.1 val loss went 5.04 → 3.27 over 20k steps (1.8 epochs) but the
  last ~10 evals sit at 3.27-3.29 — **flattened**. Extending epochs would buy
  only a few hundredths; the corpus, not the schedule, is the binding
  constraint. Consequence: **1.3B "large" is not worth it** at current corpus
  size (~367M unique tokens); it needs ~10B+ effective tokens (~25x current).
  Real levers: OCM-tagging eval, task fine-tuning, fresh domain text.
- `dcf0a8f` `--tag-names`: optional OCM code-name expansion in paragraph tags
  (`<|ocm|>131 LOCATION 133 TOPOGRAPHY...`) — hoping the model learns the
  labels along with the codes.
- `7c450d7` American Anthropologist JSTOR TEI extractor (`prepare_aa.py`).
- `9078ea1` corpus v4 = eHRAF ethnography + **114 years of American
  Anthropologist** (1880s-2005) + named OCM tags + codebook reference docs.
- `b4ed1fc` `make serve` parameterized (`MODEL`/`PORT`/`THREADS`).

**In progress:** `medium_ethnographic_v4` (303M, 27000 steps, batch 32,
seq 1024, ~1.64 s/step ≈ 12.3 h). Started 09:53; ETA ~22:15 same day. The run
survives terminal interruption (detached process); only `training_state.pt`
appears in the checkpoint dir until it finishes.

## Measured results (DGX Spark / GB10)

Throughput ladder (bf16, `torch.compile`, seq 1024, batch 32):

| model  | params | tok/s |
|--------|--------|-------|
| small  | 118M   | 44.9k |
| medium | 303M   | 20.3k |
| medium | 303M   | 20.7k @ batch 64 (+2% → compute-bound) |

Scaling exponent vs params ≈ 0.85. Serving (llama.cpp, CPU): medium q8_0
decodes 70-73 tok/s with `-t 10`; `-t -1` collapses to 7.7 tok/s — always pin
`THREADS` to the physical core count (10 on the GB10).

Val loss: medium-v3.1 5.04 → 3.27 over 20k steps, then flat (see above).

## Next steps (from `q_larger.md`)

1. **OCM-tagging eval** — precision/recall vs gold paragraph tags on held-out
   ethnographies. This, not perplexity, is the project's real capacity test;
   scale decisions should be justified by it.
2. Compare medium v3 vs v4 checkpoints on that eval (did the named tags and
   journal text help?).
3. Task fine-tuning on the tagging objective if base-model tagging is weak.
4. Fresh domain text (more journals) is worth adding for coverage, not scale;
   general English (~20-30%) would help fluency only — keep val
   ethnographic-only if added.

## Open items

- `train.py` does not persist eval/loss history — stdout only. The v4 run's
  log was lost to a session interruption; save eval history at `eval_every`
  before the next long run.
- scheduled check-in for the v4 run (~22:15 on 2026-09-13) to convert →
  quantize → serve on completion.
