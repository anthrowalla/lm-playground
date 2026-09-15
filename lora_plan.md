# LoRA — considerations and deferral

Assessed 2026-09-15 in the context of the planned tagging fine-tune (see
`finetune_plan.md`). Conclusion: **defer** — first tagging fine-tune is a
full fine-tune with replay; revisit LoRA when a second task or a much
larger base model shows up.

## Pipeline friction

- **Hand-rolling**: `train.py` is a from-scratch Llama, not HF — LoRA has
  to be implemented ourselves (wrap target `nn.Linear`s with low-rank A/B,
  freeze base). The forward math is ~40 lines, but save/resume/`save_hf`
  all assume full state dicts, so checkpoint plumbing grows too.
- **The GGUF path**: serving needs merged weights. `W += (α/r)·BA` before
  `convert_hf_to_gguf` is easy math, but it means one merged export per
  adapter — so the classic LoRA perk (hot-swap task adapters in one
  server) mostly evaporates. Keeping separate adapters in llama.cpp
  (`--lora`) works but adds a serving mode we've never exercised.

## Benefit mismatch at 303M

- LoRA's memory argument is void — the GB10 already full-fine-tunes this
  size in bf16.
- The forgetting argument (preserve the AA-journal fluency) is real, but a
  **replay mix + low LR achieves the same with zero new code**, and LoRA
  doesn't guarantee fluency retention either — we'd still have to measure
  it.
- Quality: for a task this far from the LM objective, full FT at 303M
  typically edges out LoRA slightly, and it's the simpler thing to get
  right first.

## When LoRA would earn its place

- Several task adapters over one shared base (tagging, date normalization,
  section-header prediction…).
- A base model too big for full fine-tuning.

Neither is true today.
