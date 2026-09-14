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

**Completed:** `medium_ethnographic_v4` (303M, 27000 steps, batch 32,
seq 1024) ran 09:53 → 22:04 (~12.2 h, ~1.62 s/step), all 27000 steps,
cosine schedule fully decayed (final LR 3e-5 = min ratio 0.1). Converted to
`medium-v4-f16.gguf` (607M) and `medium-v4-q8_0.gguf` (307.5 MiB, 8.5 BPW);
smoke test with `llama-cli -t 10`: coherent generation at ~165 tok/s.
Final val loss not recoverable — stdout log was lost to a session
interruption (see Open items).

### First tagging demonstration (2026-09-13, evening)

The v4 named-tag hypothesis — that training with labels after the codes
(`--tag-names`) would make the model reproduce labels when analyzing new
text — **confirmed qualitatively**. Fed an unseen subsistence paragraph,
the model greedily emitted:

```
222 COLLECTING 224 HUNTING AND TRAPPING 226 FISHING 241 TILLAGE 262 DIET
462 DIVISION OF LABOR BY GENDER
```

Plausible codes for the passage, correct code→name pairings, then it
resumed prose. A second paragraph (annual ceremony, genealogies, land
inheritance disputes before a council of kinsmen) produced `423 REAL
PROPERTY 428 INHERITANCE 613 LINEAGES 627 INFORMAL IN-GROUP JUSTICE` —
again on target. `scripts/tag_test.py` (first resident of `scripts/`) runs
this against llama-server `/completion` (temperature 0, `special: True`),
in both buffered and SSE-streaming (`--stream --live`) modes at ~170 tok/s.
Two nuances: the model skipped the `<|ocm|>` marker itself and emitted
code+name pairs directly — parsers should match the CODE NAME pattern,
not the marker; and this remains qualitative — the precision/recall eval
vs gold tags (`wayforward.md` Gate 0) is still the real test.

### First quantitative tag eval — forced marker (2026-09-14)

`scripts/tag_eval.py` scores all 6,732 tagged val paragraphs, cold-start,
greedy, predictions filtered to the 743 authoritative OCM codes. Forced
marker (prompt = text + `\n<|ocm|>`, isolating code selection):

| model | micro P | micro R | F1 | exact match |
|---|---|---|---|---|
| v4 (unseen val) | 0.533 | 0.264 | 0.353 | 0.079 |
| v3 (contaminated: likely trained on these) | 0.517 | 0.297 | 0.377 | 0.075 |

Reads as rough parity with contamination favoring v3 — v4 leads precision
and exact-match on truly unseen data. Decode-design findings from the
smoke runs: free decoding doubles as citation-continuation (AA-journal
mode competes with tagging on cold-start prompts; forcing the marker
removes it, F1 0.157 → 0.328); greedy beats temperature 0.6 (0.328 vs
0.280); the model under-generates (~1.7-2 codes vs analyst ~3.4) but its
picks are salient (P ≈ 0.53). The single-culture caveat applies (6
Eastern Toraja documents).

Full-run results, all 6,732 tagged val paragraphs (2026-09-14):

| config | micro P | micro R | F1 | exact |
|---|---|---|---|---|
| v4 forced | 0.533 | 0.264 | 0.353 | 0.079 |
| v3 forced (contaminated) | 0.517 | 0.297 | 0.377 | 0.075 |
| v4 free | 0.508 | 0.088 | 0.150 | 0.025 |
| v3 free (contaminated) | 0.516 | 0.297 | 0.377 | 0.075 |

**The free-decode contrast is the headline**: v3 free ≈ v3 forced
(0.377 = 0.377) — trained purely on text→bare-codes, v3 tags
unconditionally. v4 free collapses (0.150): the AA-journal text taught
citation continuation, which competes with tagging on cold-start
prompts. The fluency v4 gained (see qualitative comparison above) came
at the cost of unconditional tagging. Practical consequence: the
analyst-assist flow should force the `<|ocm|>` marker (cheap, and
equivalent to how the corpus is structured) — with it, v4 is at parity
with v3 on unseen data.

### Qualitative v4 vs v3 comparison (2026-09-14)

- Fluency markedly better in v4, attributed to the American Anthropologist
  journal text. v3's noticeable repetition (whole sentences / long phrases)
  is nearly gone — residuals are word- and syntactic-level.
- Residual flaws are sentential-logic ("we got up at 5, and went to bed")
  while the overall entry frame holds; occasional pronoun gender-agreement
  slips, possibly ambiguous-antecedent artefacts.
- Tagging pertinent in all ~2 dozen informal trials so far.

### The val split is held-out ethnography — mapped and verified

The assumption that no ethnography was held back is wrong, usefully:
`prepare_ethno.py` holds out a document-aligned val split from the corpus
tail, and `--extra-corpus` (journals) is deliberately inserted *before*
the ethnography so val stays on-task (per the flag's help text). v4:
train = 462.7M tokens, val = 1.23M tokens of **unseen** ethnographic
documents.

`scripts/val_docs.py` reconstructs exactly which documents those are,
self-verified two ways: val.bin decoded with the lossless ByteLevel
tokenizer must be the literal suffix of `corpus.txt`, and the decoded
documents must match fresh `format_document()` regenerations from the
source CSV (`et43texts/ethnotext_v430.csv`). Output:
`data/ethnographic_v4/val_docs.jsonl` — **6 documents, 6,742 paragraphs
(6,732 tagged), 466 distinct gold codes**.

**Caveat — the slice is single-culture**: all 6 val documents are Eastern
Toraja (OWC og11, field dates 1892-1932, the Adriani & Kruyt Celebes
monographs; the CSV tail clusters hard). A Gate 0 eval on this slice is
quantitative but measures one culture. For a general eval, the next
corpus build should hold out a stratified sample (spread across OWC
regions/cultures), or fresh held-out entries can be sourced.

## Measured results (DGX Spark / GB10)

Throughput ladder (bf16, `torch.compile`, seq 1024, batch 32):

| model  | params | tok/s |
|--------|--------|-------|
| small  | 118M   | 44.9k |
| medium | 303M   | 20.3k |
| medium | 303M   | 20.7k @ batch 64 (+2% → compute-bound) |

Scaling exponent vs params ≈ 0.85. Serving (llama.cpp, CPU, medium q8_0):
`-t 10` decodes ~150 tok/s and `-t -1` ~73 tok/s on an **idle** machine
(re-measured 2026-09-13 evening with llama-cli); the earlier figures of
70-73 vs 7.7 tok/s were taken **during the v4 training run** — concurrent
load compounds the oversubscription penalty to ~9×. Always pin `THREADS`
to the physical core count (10 on the GB10), and don't benchmark while
training.

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
