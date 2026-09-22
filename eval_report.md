# OCM tagging evals — report and legend

Consolidated results of the paragraph-tagging evaluations run on the eHRAF-derived
corpus, from the first v4 probes through the Track B (SmolLM2-360M DAPT) stage-5
fine-tune acceptance (2026-09-14 → 2026-09-22). Raw rows live in `results/`
(gitignored); this file is the summary of record.

---

## 1. Legend — what the terms mean

### Task and scoring

| Term | Meaning |
|---|---|
| **Paragraph tagging eval** | Predict OCM subject codes for each tagged paragraph of the **v5 validation set**: 21,957 paragraphs from 12 held-out societies covering all 8 geographic areas (`data/ethnographic_v5/val_docs.jsonl`). |
| **gold** | The HRAF analyst codes for a paragraph (~1–5 codes; gold sizes {1: 6860, 2: 5520, 3: 4177, 4: 2654, 5+: 2746}). |
| **micro P / R / F1** | Precision, recall and F1 pooled over all code decisions of all paragraphs (one big tp/fp/fn count), not averaged per paragraph. |
| **exact** | Fraction of paragraphs where the predicted code *set* equals the gold set exactly. |
| **greedy** | Decoding at temperature 0. All evals in this report are greedy. |
| **743-code whitelist** | Predictions are dropped unless they appear in `data/ethnographic/ocmdefs.txt` (the published OCM codebook). |
| **prompt** | Optional section header + paragraph text (head-truncated to 900 tokens; headers are never truncated) + `<|ocm|>` marker. |

### Prompt variants (what the model is shown)

| Variant | Prompt construction | What it measures |
|---|---|---|
| **forced, none** | Bare paragraph + forced `<|ocm|>` marker. | Code selection from text alone. Forcing separates selection from the decision to tag at all. |
| **free, none** | Bare paragraph, model decides whether to emit `<|ocm|>`. | Spontaneous tagging. **free ≈ forced** means the model tags unconditionally (desirable for the analyst-assist flow). |
| **sec-loo (gold unions)** | Native v5 header: `<|sec|>` section title path + `<|ocm|>` union of the *other* paragraphs' gold codes in the same section (leave-one-out). | Value of realistic section context — the codes a colleague or an earlier pass would have produced for sibling paragraphs. The primary deployment-shaped number. |
| **sec-oracle** | Same header, but the union *includes* the paragraph's own gold codes. | Leakage ceiling — upper bound if section context were perfect. Not achievable; defines the headroom. |
| **sibpred** (error-propagation probe) | sec-loo, but the unions are built from the model's **own predictions** on the sibling paragraphs instead of gold. | Survival when the model is fed its own errors — the realistic no-human-in-the-loop flow. The **sibpred delta** (gold-union F1 − sibpred F1) is the error-propagation penalty. |
| **LOO headroom capture** | (sec-loo − forced) / (sec-oracle − forced). | How much of the oracle headroom the realistic LOO context recovers. |

### Echo-union trap guard

| Term | Meaning |
|---|---|
| **trap** | A paragraph whose gold set *equals* the sibling union — echoing the header verbatim would score perfectly. ~7.6% of evaluable rows. |
| **non-trap** | All other rows; F1 here is the honest test of skill (echoing cannot help). |
| **echo rate** | Fraction of traps where the model's prediction is exactly the union (pred == union); "subset" counts predictions strictly inside the union. |
| **echo ceiling** | Mean F1 if the model echoed the union on *every* row. Both a lure ceiling and a sanity bound — every model in this report scores far above it. |

### Protocol variants and context-length experiments

| Term | Meaning |
|---|---|
| **excluded protocol** | Original serving setup: this llama.cpp build clamps each slot's context to the model's declared training context (`n_ctx_slot = min(n_ctx_seq, n_ctx_train)`), so paragraphs whose sec-union header overflowed 1,024 tokens got HTTP 400 and were written as error rows, zero-scored. |
| **giant rows** | Those 1,134 paragraphs (5.2%). Not long *text* — paragraph bodies are capped at 900 tokens; what overflows is the union header (up to ~2,000 tokens of code-name pairs) under SmolLM2's BPE, which is ~42% less compact than the v5 tokenizer. |
| **8k protocol** | After fixing an export bug (`save_hf` wrote `max_position_embeddings: 1024`; native SmolLM2-360M is 8192) and re-converting: all 21,957 rows are servable, zero exclusions. The definitive protocol. |
| **non-giant subset** | The 20,823 rows that never overflowed — the only row set directly comparable across both protocols. |
| **position-shift probe** | Normal rows re-run with ~1,250 tokens of document-filler prepended so the paragraph itself lands beyond position 1,250 (prompts up to 1,870 tokens). Separates *long-position* degradation from *header-content* dilution. |
| **parity check** | Old vs re-exported GGUF on identical prompts. Run-to-run same-server reruns differ on ~10% of predictions (continuous-batching nondeterminism, summary F1 stable ±0.002); llama-server greedy is not bit-stable. |

### Models

| Name | Arch | Origin |
|---|---|---|
| **medium v5** | 303M from scratch | v5 corpus (sections + `<|sec|>` markup), 27k steps |
| **small v5** | 118M from scratch | same corpus |
| **small v5-tag** | 118M | small v5 + tagging FT (LOO-conditioned, masked loss, 20% replay) |
| **small v6** | 118M from scratch | mixed corpus (76% FineWeb / 1% classics / 23% domain) — Track A language-floor attempt |
| **DAPT base** | 362M (SmolLM2-360M + mean-init vocab resize to 49,154) | pretrained base + domain-adaptive pretraining (78% domain / 22% FineWeb replay) — Track B |
| **DAPT-tag** | 362M | DAPT base + the same stage-5 tagging FT recipe |

---

## 2. Cross-model gate matrix

Full v5 val, greedy, forced marker, native `<|sec|>` header format for sec-* rows.
Medium/small/v6 rows cover all 21,957 paragraphs (their tokenizers fit the 1,024
slot everywhere). DAPT rows: original protocol = 1,134 giant rows zero-scored;
8k = full 21,957.

| Model | forced | free | sec-loo | sec-oracle | LOO headroom |
|---|---|---|---|---|---|
| medium v5 (303M) | 0.364 | 0.362 | 0.468 | 0.539 | ~55% |
| small v5 (118M) | 0.347 | 0.346 | 0.431 | 0.500 | ~48% |
| small v5-tag (FT) | 0.381 | 0.379 | 0.507 | 0.546 | ~71% |
| small v6 mixed (118M) | 0.310 | 0.306 | 0.400 | 0.476 | ~54% |
| **DAPT-tag (FT, orig. protocol)** | **0.406** | **0.404** | **0.541**† | **0.587**‡ | ~75% |
| DAPT base, sec-loo only | — | — | 0.392† / 0.378 (8k) | — | — |

† 20,823 rows (1,134 giant excluded, zero-scored). ‡ 20,820 rows (1,137 excluded).

Best on every variant: **DAPT-tag**. free ≈ forced holds for every model — all of
them tag unconditionally once the tagging skill is installed. Detail rows for the
DAPT-tag forced variant: P 0.555-class precision with exact match 0.232 (vs medium
v5's 0.186 — nearly double).

## 3. Track B acceptance — DAPT-tag vs DAPT base (8k protocol, all 21,957 rows, zero exclusions)

| Eval | DAPT-tag | DAPT base | Δ |
|---|---|---|---|
| sec-loo, gold unions | **0.516** (P .538 / R .497, exact .232) | 0.378 (P .307 / R .492, exact .142) | +0.138 |
| sec-loo, own-pred unions (deployment) | **0.398** (P .402 / R .394, exact .138) | 0.252 (P .194 / R .358, exact .076) | **+0.146** |
| non-giant subset, gold unions | 0.541 (exact .244) | 0.392 | +0.149 |
| non-giant subset, sibpred | 0.417 (exact .144) | 0.263 (exact .080) | +0.154 |
| giant rows, gold unions @8k | 0.032 (P .047 / R .024) | 0.015 (P .021 / R .011) | (both ≈ 0) |
| giant rows, sibpred @8k | 0.103 (exact .022) | not run | — |

Readings:

- The deployment-shaped number (sibpred) is decisive: **+0.146 over the base**.
  The base collapses under its own noisy unions (precision 0.194, 85k false
  positives — it echoes junk from the header); the FT holds 0.398.
- **FT fed its own errors (0.417 non-giant) still beats the base fed *gold*
  unions (0.392)** — the fine-tuned model is a net win in the fully autonomous
  flow, not just the oracle-ish one.
- **Flagged robustness regression**: the FT's own error-propagation penalty is
  −0.124 (0.541 → 0.417), far worse than the small-v5-tag precedent (−0.015).
  Leading hypothesis: the fat tokenizer overflowed large-union examples out of
  the 1,024-token FT budget, so the model never learned to read wide/noisy code
  walls. Both this and the giant-row collapse (0.032) point to the same round-2
  remedy: train on capped or prediction-built unions.

## 4. Echo-union trap guard (original protocol, 20,823 evaluable rows)

| Metric | DAPT-tag | DAPT base |
|---|---|---|
| traps (gold == sibling union) | 1,575 | 1,575 |
| non-trap rows | 19,248 | 19,248 |
| F1, all evaluable rows | 0.595 | 0.463 |
| F1, traps only | 0.946 | 0.979 |
| F1, non-trap only | **0.567** | 0.420 |
| echo rate in traps (pred == union) | 83.2% (1,310) | 94.2% (1,484) |
| … + pred strictly inside union | +237 | +84 |
| echo ceiling (F1 if always echoing) | 0.321 | 0.321 |

**PASS.** Echoing the header everywhere would score only 0.321 — below both
models. On the rows where echoing fails, the FT scores 0.567 vs the base's 0.420,
and it echoes traps less than the base does (83% vs 94%): the section prior is
used as a *prior*, not parroted. (Small-v5-tag precedent: non-trap 0.490 vs
base 0.407.)

## 5. Context-length experiments (Track B)

| Probe | Result | Interpretation |
|---|---|---|
| Export metadata | native SmolLM2-360M `max_position_embeddings` = **8192**; our export said 1024 | the 1,024-slot clamp was an artifact of our `save_hf`, not a model limit |
| Parity after re-export | **0 / 20,823** prediction flips on shared rows; same-server reruns flip ~10% of preds with summary F1 stable | re-export clean; serving noise, not weight drift |
| Giant rows @8k | FT F1 0.032, base 0.015 (were unservable before) | servable but no skill there — see next two rows |
| Position-shift probe (n=300, paragraphs pushed past position 1,250) | F1 **0.619 vs 0.662** unshifted (−7% relative) | long *positions* survive DAPT at 1,024; ~94% of skill retained |
| Giant rows under sibpred @8k | FT 0.103 | smaller self-generated headers work where 2,000-token gold walls fail |
| Conclusion | — | the collapse is **header dilution** (code wall never seen in FT), not position death; DAPT at 1,024 did not destroy long-context competence |

## 6. Historical baselines (for context)

| Eval | Result |
|---|---|
| First forced-marker eval, v4 (6,732 paras, og11 slice) | v4 forced 0.353 / free 0.150 (journal-citation interference, later traced mostly to text-format); v3 0.377 both (contaminated — likely trained on these paragraphs) |
| Ceiling check, v4 + section context (og11, 6,658 paras) | baseline 0.381 → +title 0.395 → +LOO 0.493 → oracle 0.531 (validated the whole section-conditioning direction) |
| Small-v5-tag acceptance gates | trap non-trap F1 0.490 vs base 0.407; sibpred 0.492 vs 0.507 (−0.015); all pass |
| Leaf-title FT on small v5 (summarization-adjacent) | word-bag F1 0.224 vs base zero-shot 0.054; exact 0.159 vs 0.001 (2,498 sections) |
| Sumpairs (culture-summary) FT | negative result — society-prior echo; retired |

## 7. Verdict and open items

- **Track B stage-5 tagging FT: ACCEPTED** (trap guard PASS, error propagation
  PASS with the flagged −0.124 penalty but a decisive deployment-matched win,
  fluency PASS, free≈forced PASS).
- **Best current flow**: DAPT-tag + native sec header + LOO/own-pred unions →
  F1 ≈ 0.40–0.52 depending on where the sibling codes come from.
- **Round-2 FT candidates**: train on capped or prediction-built unions (fixes
  both the error-propagation penalty and the giant-row collapse); leaf-title FT
  on the DAPT base (follow-on, not started).
- **Serving notes**: SmolLM2-adapted GGUFs need request-level `ignore_eos`
  (their emitted `<|ocm|>` halts llama-server); this llama.cpp build clamps slot
  context to declared training context — the 8k GGUFs (`smollm2-*-8k-q8_0.gguf`)
  supersede the originals; CPU build stays the deployment standard, CUDA build
  is eval-only (6.2× throughput).
