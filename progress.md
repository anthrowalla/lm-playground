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

### Corpus v5 — tree-driven rebuild with section structure (2026-09-14)

The pre-CSV corpus at `~/et43texts/ethnotext_v430` (AREA/SOCIETY/hdoc.txt,
7,024 files) turned out to document everything the CSV flattening lost:
`## section::eid; section.parent::P; TITLE; <`/`>` hierarchy (~266k sections,
3+ levels), per-SRE metadata lines, SRE types (`p` 2.16M, `bibl.item` 334k,
`enote` 51k), and `{POST}` post-text (figures, tables, notes). The CSV was a
1:1 row dump of every `## ocms::` line — so the old pipeline had silently
trained on bibliography entries and endnotes (notes ~70% duplicated inline +
backmatter), and left `{...}` markup remnants in kept paragraphs.

`prepare_ethno_tree.py` rebuilds from the tree with the intended design:

- **p-type SREs only** — bibliography items and endnotes dropped;
- **nested-aware `{...}` stripping** — the POST scaffolding is gone;
- **`<|sec|>` headers** on section entry carrying the full title path
  (ancestors joined " / ") and the **union of OCM codes** over all paragraphs
  under that path — encoding the analyst observation that codes persist
  across a section;
- **hdocid-keyed documents** (basenames verified unique, 7,024/7,024);
- **stratified whole-society val holdout** — 12 societies across all eight
  OWC areas (fc07 Mende, fy08 Tanala, ru41 Nenets, ef05 Montenegrins,
  st13 Island Carib, ma10 Basseri + ml01, mj04 Bedouin, nf12 Stoney,
  nt18 Tewa Pueblos, oj13 Kwoma, sk15 Enxet/Enlhet), all English,
  replacing the single-culture Toraja tail split;
- tokenizer trained on the train stream only; `<|sec|>` is a new special
  token (`sec_id` in meta.json).

Measured: 6,865 content documents (159 skipped: 9 no-hdocid oddballs, 110
bibliography/endnote-only files, 40 figure/table-only files), 2,052,162
tagged paragraphs, 222,331 section headers (all carrying unions) →
**train.bin 419.4M tokens, val.bin 3.578M tokens**; val round-trip
(decode == corpus_val.txt) PASS. Note ~4.4 bytes/token — named tags and
union lines add markup tokens relative to raw prose. Sanity anchors:
og11-000 kept 32 paragraphs (old 41 incl. non-p types), og11-002 kept
2,132 (old 2,134).

v5 is the baseline for the next training run and for the
section-conditioned work in `wayforward.md`: the eval set is now
cross-culture, and `val_docs.jsonl` carries each paragraph's section path
plus gold tags for section-conditioned evaluation.

### Zero-training section-context ceiling check (2026-09-14)

The `wayforward.md` ceiling check, run before any v5 training: does section
context help a model that has never seen `<|sec|>`? v4 (forced marker,
greedy) on the og11 slice reparsed from the tree in v5 form
(`scripts/og11_slice.py`, 6,658 p-only stripped paragraphs) — the one slice
v4 never trained on. Four paired prompt variants:

| context | micro P | micro R | F1 | exact |
|---|---|---|---|---|
| none (baseline) | 0.526 | 0.298 | 0.381 | 0.071 |
| section title path | 0.580 | 0.300 | 0.395 | 0.083 |
| title + LOO section union | 0.500 | 0.487 | **0.493** | 0.083 |
| title + full union (oracle) | 0.527 | 0.536 | **0.531** | 0.092 |

Context variants prepend `SECTION: {title path}` and, for the union rows,
`SECTION CODES: {named codes}`. **LOO** = union over the *sibling*
paragraphs' gold codes (leave-one-out — deployable in an incremental assist
flow: earlier paragraphs' tags become the prior for the next); **oracle**
adds the target's own tags (ceiling only). Results land in
`results/ceiling_og11_*.jsonl`.

Readings:

- **+29% relative F1 with zero training** (0.381 → 0.493) from sibling-code
  context alone. Recall jumps 0.298 → 0.487 at a modest precision cost —
  the model's chronic under-generation (~2 codes vs analyst ~3.4) is
  largely a *prior* problem, and the section union is the prior. Mean
  predicted codes/para rises to ~3.3 vs gold 3.4.
- **LOO captures ~75% of the oracle headroom** ((0.493−0.381)/(0.531−0.381))
  — the target's own tags add little beyond its siblings'. Sibling codes
  predict paragraph codes almost as well as the paragraph itself.
- Title path alone is mild (+0.014 F1, mostly precision).
- Baseline shifted 0.353 → 0.381 vs the old CSV-text eval — the cleaned
  (stripped, p-only) text is slightly easier for v4; all variants are
  paired on identical text, so deltas are clean.

Caveats: single-culture slice (og11); LOO uses *gold* sibling codes —
deployment feeds model/analyst-produced codes instead (error propagation
unmeasured, but the analyst is in the loop in that flow); oracle is
leakage by construction. Direct implication: the section-conditioned
direction is validated — v5's `<|sec|>` + union markup should internalize
this prior, and even the *current* v4 + forced marker + LOO prompt is a
usable analyst-assist flow at F1 ≈ 0.49.

### v5 training run (2026-09-14 → 15)

`medium_ethnographic_v5` (303M, same hypers as v4: 27000 steps, batch 32,
seq 1024) ran ~12 h, all steps, ~20.4k tok/s. **Eval history persisted for
the first time** (`checkpoints/medium_ethnographic_v5/eval_history.jsonl`,
54 evals): 4.96 @ 500 → 3.42 @ 5k → min **3.0027 @ 24k**, flattened at
3.02-3.06 through step 27k; final val_loss 3.0367. Not comparable to v4's
3.27 — the val sets differ (cross-culture stratified holdout vs Toraja
tail). Converted to `medium-v5-f16.gguf` (606.8M) and
`medium-v5-q8_0.gguf` (307.5 MiB, 8.50 BPW); serving on port 8083
(`-t 10 -c 8192 --parallel 4`), v4 still on 8082, v3 on 8080.

### The decisive eval — v5 on the cross-culture val (2026-09-15)

Full OCM-tagging eval of v5 on its own held-out val (`val_docs.jsonl`,
21,957 tagged paragraphs, 12 societies across all eight OWC areas,
cold-start, greedy, 743-code whitelist). New `--context sec-*` variants in
`scripts/tag_eval.py` prepend the *training-native* header
(`<|sec|>title path\n<|ocm|>union\n\n`); LOO = union over sibling
paragraphs' gold codes (deployable incrementally), oracle = full section
union (leakage by construction).

| run | context | micro P | micro R | F1 | exact |
|---|---|---|---|---|---|
| v5 forced | none | 0.397 | 0.336 | 0.364 | 0.107 |
| v5 **free** | none | 0.398 | 0.331 | 0.362 | 0.104 |
| v5 forced | sec-loo | 0.491 | 0.446 | **0.468** | 0.186 |
| v5 forced | sec-oracle | 0.553 | 0.525 | **0.539** | 0.225 |
| v4 forced (og11 ref) | none | 0.526 | 0.298 | 0.381 | 0.071 |
| v4 **free** (og11 ref) | none | 0.526 | 0.297 | 0.380 | 0.070 |
| v4 forced (og11 ref) | LOO prose | 0.500 | 0.487 | 0.493 | 0.083 |
| v4 forced (og11 ref) | oracle prose | 0.527 | 0.536 | 0.531 | 0.092 |

The runs, briefly:

- **v5 forced / none** — bare paragraph on a cold start with the `<|ocm|>`
  marker appended, so the model only selects codes; v5's baseline on its
  clean cross-culture holdout.
- **v5 free / none** — no marker forced: the model must decide on its own
  to emit a tag block (or continue prose). Measures unconditional tagging,
  the zero-scaffolding analyst flow.
- **v5 forced / sec-loo** — prepends the training-native
  `<|sec|>title path` header plus `<|ocm|>` union of the *sibling*
  paragraphs' gold codes (leave-one-out). The deployable incremental flow:
  earlier paragraphs' tags become the prior for the next.
- **v5 forced / sec-oracle** — same header, but the union covers the whole
  section including the target paragraph's own gold codes (what the
  training format literally carries). Ceiling only — leakage by
  construction.
- **v4 forced / none (og11 ref)** — v4's baseline on the og11 slice
  (reparsed clean; the one text v4 never trained on), for cross-model
  reference. Not paired with the v5 rows: different eval sets.
- **v4 free / none (og11 ref)** — same without the forced marker, on
  cleaned text; the run that revealed the old 0.150 free-collapse was a
  text-format artifact (see Readings).
- **v4 forced / LOO prose (og11 ref)** — the zero-training ceiling-check
  variant: section context injected as prose lines (`SECTION: …`,
  `SECTION CODES: …`) with the leave-one-out sibling union; v4 never saw
  `<|sec|>` in training.
- **v4 forced / oracle prose (og11 ref)** — same prose form with the full
  gold section union; v4's leakage ceiling.

Readings:

- **v5 tags unconditionally**: free ≈ forced (0.362 ≈ 0.364, identical to
  three decimals in P) — the v3 behavior, now on genuinely clean data. No
  marker forcing needed in the analyst flow.
- **The earlier v4 free-collapse was a text-format effect, not (only)
  journal competition.** v4 free on the *cleaned* og11 text scores 0.380 ≈
  its forced 0.381 — the 0.150 free collapse measured on the old
  CSV-format val text doesn't appear once paragraphs are stripped and
  p-only. Both models tag freely on tree-cleaned text; the 0.150 headline
  is revised.
- **The section prior is model- and format-robust**: v5's native
  `<|sec|>` header gains +29% relative F1 (0.364 → 0.468), mirroring v4's
  +29% from prose-injected context (0.381 → 0.493). Unlike v4, v5 pays no
  precision for the recall — P and R rise together (0.397/0.336 →
  0.491/0.446), and exact match jumps 74% (0.107 → 0.186).
- **LOO captures ~59% of the oracle headroom on v5**
  ((0.468−0.364)/(0.539−0.364); v4 captured ~75%). Oracle F1 0.539 /
  exact 0.225 on cross-culture data.
- Best deployable flow: v5, free decode, section header with LOO union
  built from earlier paragraphs' tags — F1 ≈ 0.47 with no marker forcing.

Caveats: v4 rows are on og11 (single culture, clean for v4; in v5's
train), v5 rows on its cross-culture val (clean for v5) — cross-model
deltas are indicative, not paired. LOO uses gold sibling codes;
deployment feeds model/analyst codes (analyst-in-the-loop). Oracle is
leakage by construction.



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

### Language-floor pivot — mixed-corpus v6 + pretrained-base plan (2026-09-18)

The v5 models do the analysis skill but are poor generators: hallucinated
ethnographic prose, conjunction-flip repetition loops, pronoun-gender errors,
contradictory phrases. Diagnosis is data scale — v5-small saw 419M tokens,
~0.18x the token budget implied by Chinchilla-style ratios for 118M params.
Two approved responses:

- **Track A (from scratch, launched)** — corpus v6, a mixed general/domain
  corpus built by `prepare_mix.py` from three streams: FineWeb-Edu
  sample/10BT rows (fetched by `scripts/fetch_general.py`, 4 shards,
  6.98 GB chars), Project Gutenberg classics in the requested domains
  (philosophy/logic/sociology/political economy: Plato, Aristotle-adjacent
  logicians, Hobbes, Spinoza, Kant, Hume, Locke, Russell, Mill x2, Marx/Engels,
  Adam Smith, Nietzsche x2, Montaigne, Marcus Aurelius, Machiavelli, Descartes
  — 17 of 18 candidates fetched, one 404 gracefully skipped), and the v5
  domain stream (`corpus_train.txt` verbatim: codebook x3 + AA journals +
  tagged ethnography; the 12 val societies remain excluded, so all v5 task
  evals stay valid). Fresh 32k BPE (same specials incl. `<|ocm|>`, `<|sec|>`).
  Measured mix: **general 76.1% / classics 1.0% (repeated x6) / domain 22.9%**,
  train 1.996B tokens (3.85 GB bin), val 27.2M tokens (domain val first +
  classics head + 22.7k general docs); val round-trip decode PASS;
  `mix_report.json` carries the counts. Build took ~6 min (~5M tok/s encode).
- **Track B (pretrained base, planned)** — `pretrained_adapt_plan.md`:
  SmolLM2-360M (Apache-2.0, Llama arch — our state dicts already match)
  + `<|ocm|>`/`<|sec|>` special tokens + mean-init embedding resize →
  re-vectorized v5 streams → DAPT (~0.5-0.6B tokens, domain + general
  replay, lr 1e-4, ~9 h) → task FTs re-run with the full gate matrix plus
  fluency/perplexity gates. Needs `train.py --init` + an HF-safetensors
  start-point script. Decision gate after both tracks: B wins fluency by
  construction; if tagging gates match, B becomes the serving line and A the
  ablation/understanding track.

Training: `configs/small_mixed_v6.toml` (small 118M dims, batch 64, 30,400
steps = 1 epoch ≈ 2.0B tokens, lr 3e-4 cosine, warmup 500, eval_every 5000
with eval history persisted). Healthy start: 45.9k tok/s (matches the
throughput ladder), GPU 96%, ETA ~12 h. Note this supersedes the earlier
"general English ~20-30% would help fluency only" note — the goal changed
from tagging optimization to building a language floor; tagging evals still
run against the untouched v5 val societies.

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

- ~~`train.py` does not persist eval/loss history~~ — fixed; appends
  `eval_history.jsonl` in the run dir. First exercised by the v5 run.
- ~~v5 eval: the decisive comparison~~ — done 2026-09-15 (see the table
  above). Deployable flow: v5 + LOO section header, F1 ≈ 0.47.
- Untested levers next: task fine-tuning on the tagging objective;
  error-propagation test (LOO fed *model* codes instead of gold).
