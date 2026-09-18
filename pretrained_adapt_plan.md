# Adaptation plan: pretrained base instead of from-scratch (option B)

Status: planned, not started. Approved in principle 2026-09-18; execution waits on
the v6 from-scratch run (option A) clearing the GPU.

## Motivation

The v5 models do the analysis skill (OCM tagging) but are poor generators:
hallucinated ethnographic prose, conjunction-flip repetition loops, pronoun-gender
errors, contradictory phrases. Root cause is data scale, not architecture: v5-small
saw 419M tokens, ~0.18x the Chinchilla-adjacent budget for 118M params (~2.4B
tokens). Narrow-domain training at sub-optimal token counts buys task behavior but
not a stable language model underneath. Two responses were approved:

- **A** (running): from-scratch small on a ~2.1B-token mixed corpus (general +
  classics + domain). Tests whether our own pipeline can reach a serviceable
  language floor given ~5x more data.
- **B** (this doc): start from an established pretrained generator at small/medium
  scale and adapt it to the ethnographic domain. Tests the same goal against a
  model whose language floor cost ~11T tokens of someone else's compute.

B is expected to win fluency by a wide margin; the open question is whether task
FTs on top of it match or beat our from-scratch tagging gates.

## Base choice

**Primary: `HuggingFaceTB/SmolLM2-360M`** (sibling 135M / 1.7B exist if we want
smaller/larger).

- Apache-2.0 — clean for redistribution; GGUF sharing unproblematic.
- Llama architecture (`LlamaForCausalLM`) — our `Transformer` state-dict keys
  already match this; RoPE + GQA + SwiGLU + RMSNorm + tied embeddings, all of
  which `train.py` implements. Head dim is the standard 64.
- Trained on ~11T tokens of curated web/educational text (FineWeb-Edu-heavy mix —
  the same distribution we are buying in option A).
- 49,152-vocab BPE tokenizer, byte-level, 8192 context (we will keep seq_len 1024
  to match our pipeline; no RoPE changes needed at shorter context).

**Alternative: `meta-llama/Llama-3.2-1B`** — stronger but gated + community
license (use restrictions, 700M-MAU clause); 128k vocab makes embedding + KV
memory much heavier at our serving setup. Only if 360M proves too small.

No eHRAF contamination risk: eHRAF is license-restricted and not in SmolLM2's
training mix, so our held-out societies remain genuinely held out.

## Stages

### 0. Fetch + inspect

`hf_hub_download` the SmolLM2-360M snapshot (safetensors + tokenizer). Verify:
exact `config.json` dims, state-dict key names against our `Transformer`,
`tie_word_embeddings`, rope_theta. Check our llama.cpp clone converts smollm2
(support landed long before our clone; `git -C llama.cpp pull` if not).

### 1. Tokenizer: theirs + our specials

Start from the SmolLM2 tokenizer. Add exactly two special tokens: `<|ocm|>`,
`<|sec|>` (the only corpus-native specials the v5 tree stream uses besides
`<|bos|>/<|eos|>`, which map onto the model's existing special/`<|endoftext|>`
tokens). Pad/unk map onto existing tokens (byte-level BPE has total byte
coverage). Meta emitted as usual.

### 2. Embedding resize (mean-init)

`vocab 49152 -> 49154`: new rows = mean of existing embedding rows (tied matrix
feeds both embedding and lm_head). This is the only weight surgery required.

### 3. Re-vectorize the v5 corpus

Re-encode `data/ethnographic_v5/corpus_train.txt` / `corpus_val.txt` with the
extended tokenizer (same `encode_file` path). No text changes — the streams are
already correct (val societies excluded from train, codebook x3 + AA journal
included). Re-run `prepare_finetune.py`-family scripts with `--tokenizer`
pointing at the extended tokenizer for the FT stages.

### 4. DAPT (domain-adaptive pretraining)

Continue pretraining from the initialized weights on a mix of:

- v5 domain stream (~419M tokens)
- general replay slice from the v6 general corpus (~100-200M tokens) to prevent
  catastrophic forgetting of the language floor

~0.5-0.6B tokens total, 1 epoch, batch 64, seq 1024, lr 1e-4 cosine to 1e-5,
warmup 200. At SmolLM2-360M scale (~17k tok/s on GB10 by interpolation from
medium 303M = 20.3k) that is ~8-9.5 h. Checkpoint + val loss history to disk
(the eval-history gap that `train.py` still has).

`train.py` needs two small additions to run this: `--init` (load a start state
dict instead of fresh init) and the mean-init resize helper
(`scripts/adapt_init.py`: HF safetensors -> our `.pt` start point, with key
verification). Both are mechanical.

### 5. Task FTs re-run

Port the branch tooling to main (`prepare_finetune.py`, `train_ft.py`,
`scripts/error_prop.py`, title/sumpair preparers). Then:

- Tagging FT: same recipe (masked target loss + replay, lr 2e-5, best-F1
  checkpointing).
- Leaf-title FT: same recipe.
- Eval: full gate matrix from `finetune_plan_small.md` (sec-loo, free~forced,
  echo-the-union guard, error-propagation probe) **plus** two new gates that
  matter for B specifically:
  - **fluency probe**: free generation vs v5-small and v6-A small on fixed
    prompts; compare repetition loops, pronoun/agreement errors by inspection
    and n-gram repetition rate.
  - **perplexity delta**: held-out general text + held-out ethnographic text,
    B-adapted vs from-scratch smalls. B should dominate general-text PPL by
    construction; domain PPL is the real signal of DAPT quality.

### 6. Serve

`convert.py` -> f16 GGUF -> Q8_0 -> `llama-server` with explicit THREADS=10.
Spot-check `<|ocm|>`/`<|sec|>`-forced continuations against ports 8085/8086
baselines.

## Decision gate

After stage 5, compare B-adapted vs v5-small vs v6-A small on: tagging gates,
leaf-title F1, fluency probe, val losses. Expected outcome: B wins fluency
decisively and matches or beats tagging; if so, B becomes the primary line for
analyst-facing serving and the from-scratch line (A) remains the
understanding/ablation track.

## Risks

- **Vocab 49152** raises embedding memory (~47M tied params) — fits where
  medium 303M already trains; KV cache grows proportionally at serve time.
- **Special-token interaction**: added tokens match before the byte-level
  pre-tokenizer, so raw `<|ocm|>131` strings tokenize correctly; verify with a
  round-trip check on a corpus slice (same check as `prepare_ethno_tree.py`).
- **Catastrophic forgetting during DAPT** is the main scientific risk — the
  general replay slice and low LR are the mitigations; fluency probe before/after
  DAPT quantifies it.
- **llama.cpp conversion** of a resized-vocab smollm2 is untested upstream —
  verify early (stage 0) rather than after the 9 h DAPT run.
