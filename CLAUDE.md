# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal PyTorch training + llama.cpp serving pipeline for small decoder-only LMs, developed on an NVIDIA DGX Spark (GB10, 20-core). Current focus: an ethnographic (eHRAF-derived) corpus with Outline of Cultural Materials (OCM) subject-code tags, aimed at analyst-assist paragraph classification. Model sizes (tiny/small/medium) follow the specs of the sibling Rust project `../aarambh-studio`, where this work started — that repo's training was 14-15x slower, hence this Python port. The two repos are formally independent with separate origins.

## Commands

Uses `.venv/bin/python` (managed by `uv`). There are no tests or linters.

```sh
make deps          # uv venv + torch (cu130) + deps; clones llama.cpp (depth 1) if absent
make prepare       # tiny Shakespeare smoke test: tokenizer + train.bin/val.bin
make train         # train with configs/tiny_shakespeare.toml
make gguf          # HF checkpoint -> f16 GGUF (tiny default)
make quantize      # build llama.cpp (CPU-only) + Q8_0 quantize
make serve MODEL=<gguf> [PORT=8080] [THREADS=10]   # llama-server, OpenAI-compatible
```

Ethnographic runs bypass the Makefile defaults (which are tiny-Shakespeare-hardcoded):

```sh
.venv/bin/python prepare_ethno.py --csv <export.csv> --out data/ethnographic_vN \
    --ocm-labels data/ethnographic/ocmdefs.txt [--tag-names]
.venv/bin/python prepare_aa.py ...                  # American Anthropologist JSTOR TEI
.venv/bin/python train.py --config configs/<name>.toml
.venv/bin/python convert.py checkpoints/<run> --outfile <run>/<name>-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize <f16.gguf> <q8_0.gguf> Q8_0
```

**Serving: always pass an explicit `THREADS` (10 on the GB10), never `-t -1`.** Thread
oversubscription halves decode even idle (~73 vs ~150 tok/s on medium q8_0) and collapses
it ~9x under concurrent load (e.g. an active training run). Pass physical core count on other hosts.

## Architecture

Pipeline: `prepare*.py` → `train.py` → `convert.py` → `llama-quantize` → `llama-server`.

- **Data convention** (nanoGPT-style): tokenizer + text packed into uint16 `train.bin`/`val.bin` via `np.memmap`; `meta.json` carries vocab size and special-token ids (e.g. `ocm_id` for `<|ocm|>`). Batches are random offsets into the memmap.
- **`prepare_ethno.py`** builds the eHRAF corpus: per-document metadata headers, per-paragraph OCM tags (`<|ocm|>131 133`, or with `--tag-names`: `<|ocm|>131 LOCATION 133 TOPOGRAPHY...`), `<|div|>` division markers, and the OCM codebook (codes + names + definitions from `ocmdefs.txt`) embedded as reference docs, repeated `--ocm-labels-repeat` (3) times. `prepare_aa.py` adds American Anthropologist journal text from JSTOR TEI.
- **`train.py`** is a from-scratch Llama-style GQA transformer (RoPE, SwiGLU, RMSNorm, pre-norm residual blocks, tied embeddings). The module's state-dict keys deliberately match `LlamaForCausalLM` (see the `Transformer` docstring) — that is what makes HF tooling accept it. bf16 autocast + `torch.compile`, AdamW + warmup/cosine schedule, TOML configs with `[model]` and `[train]` sections. `save_hf` exports `config.json` + `model.safetensors` + tokenizer files so llama.cpp's `convert_hf_to_gguf` accepts it unchanged (wrapped by `convert.py`).
- **`convert.py`** shells out to `llama.cpp/convert_hf_to_gguf.py` — the llama.cpp clone must exist (`make deps`).

## Constraints & operational notes

- **The eHRAF corpus is not redistributed** — users export their own CSV. `data/ethnographic/ocmdefs.txt` (OCM codes/names/definitions, published by HRAF at Yale) is included only as a reference-document source. The local `llama.cpp/` clone is also not committed.
- llama.cpp is built with `GGML_CUDA=OFF` — serving is CPU-based by design on this host.
- Training runs are long (medium ≈ 9-12 h) and are typically launched in the background; the process survives terminal/session interruption — check `ps aux | grep train.py` and GPU utilization before assuming a run died. Progress estimate: ~1.64 s/step at medium scale.
- **Known gap: `train.py` logs only to stdout** — no eval/loss history is persisted to disk, so an interrupted session loses the log. Saving eval history at `eval_every` is a wanted fix.
- Measured throughput (GB10, bf16, compiled, seq 1024): small 118M ≈ 44.9k tok/s; medium 303M ≈ 20.3k tok/s (compute-bound, batch 64 adds only +2%).
