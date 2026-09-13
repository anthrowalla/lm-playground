# Would a "Large" (1.3B) run pay off?

Notes from the 2026-09-12 discussion, pending evaluation of the small and
medium models. Nothing here is decided — the point is to record the math so
the decision later is empirical, not vibes.

## The Large scale

aarambh-studio's `wikitext103_large` config: hidden 2048, ffn 6656, 24L,
32H/8KV GQA, tied embeddings, vocab 32000 → **~1.30B params**.

## Throughput ladder measured on GB10 (batch 32, seq 1024, compiled)

| model   | params | tok/s    |
|---------|--------|----------|
| small   | 118M   | 44.9k    |
| medium  | 303M   | 20.3k    |
| medium  | 303M   | 20.7k @ batch 64 (+2% → compute-bound, ladder closed) |

Scaling runs slightly better than 1/params (2.21× slowdown for 2.57×
params, exponent ≈ 0.85). Extrapolating medium → large (4.29× params):
**~4.7–5.9k tok/s**.

Same recipe as medium (20k steps × 32 × 1024 = 655M tokens):
**≈ 31–39 h, call it ~1.5 days** (medium ≈ 9 h, so ~4×). Memory is a
non-issue — AdamW + bf16 activations for 1.3B fit easily in 128 GB unified;
purely compute-bound.

## Is the corpus thin? (yes, for both sizes)

Corpus v3.1 = **367.3M unique tokens**.

- Chinchilla-style ideal ≈ 20 tokens/param:
  - medium (303M) wants ~6B tokens — corpus is ~1.2 tokens/param
  - large (1.3B) wants ~26B tokens — corpus is ~0.3 tokens/param

So the medium is *also* data-thin; the difference is that it sits in the
recoverable zone. Current run sees each token only ~1.8× (655M / 367M).
Data-constrained scaling (Muennighoff et al. 2023): repetition up to
~4 epochs is nearly as good as fresh data; returns decay after that.

## Would 1.3B pay off at this corpus size? — No

Even with +20% general English (~440M unique tokens), 4 epochs =
~1.8B tokens seen ≈ **1.4 tokens/param** for 1.3B. In the data-constrained
regime an undertrained large model roughly matches or *loses to* a
well-trained smaller model — 4× the compute to likely break even with
medium. 1.3B starts making sense around **~10B+ effective tokens**, i.e.
~25× the current corpus. Journal additions cannot reach that.

## What does move the needle

- **Epochs, not size**: extending medium to ~48k steps (~1.5B tokens,
  ~4 epochs, ~24 h) is a legitimate ~2.2× effective-data increase for free.
  Empirical test: if val loss is still descending meaningfully at step 20k
  of the medium-v3 run, epochs are buying something. (Note: no loss log is
  persisted to disk — worth saving eval history next run.)
  **Measured answer (2026-09-13)**: mostly flattened. Medium-v3.1 val went
  5.04 → 3.27 over 20k steps (1.8 epochs), but the last ~10 evals sit at
  3.27–3.29, gains < 0.02 per 500 steps. Extending epochs would buy only a
  few hundredths — the corpus, not the schedule, is the binding constraint.
  Real levers now: the tagging eval, task fine-tuning, and/or new text.
- **Fresh domain tokens > repeats at the margin**: the American
  Anthropologist run (1880s–2005) plus other journals ≈ +5–10%
  (~20–40M tokens). Worth adding for *coverage*, not scale — AAA-style
  theoretical/review prose is stylistically different from eHRAF's
  descriptive ethnography and likely closer to what analysts will actually
  feed the model.
- **General English (20–30%)** helps fluency/robustness, not OCM tagging
  directly; if added (v4), keep val ethnographic-only so the metric tracks
  the actual task.

## Practical ladder

1. Evaluate small vs medium (GGUF sampling + OCM-tagging eval).
2. Extend medium with epochs; fold in journals + general mix as corpus v4
   if the val curve says tokens still help.
3. Revisit 1.3B only if the tagging task itself turns out to need more
   capacity.

**The real capacity test is the OCM-tagging eval** (precision/recall vs
gold paragraph tags on held-out ethnographies), not perplexity. LM scale
should be justified by that metric, since analyst-assist tagging is the
project goal.
