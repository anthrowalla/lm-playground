# lm-playground

A minimal PyTorch training + [llama.cpp](https://github.com/ggml-org/llama.cpp)
serving pipeline for small decoder-only language models, developed on an
NVIDIA DGX Spark (GB10). Built for experimenting with domain-specific models —
currently an ethnographic corpus with Outline of Cultural Materials (OCM)
subject-code tagging, aimed at assisting HRAF-style paragraph classification.

## Pipeline

| step | file | what it does |
|------|------|--------------|
| prepare | `prepare.py` | trains a ByteLevel BPE tokenizer and packs text into uint16 `train.bin`/`val.bin` |
| prepare (ethno) | `prepare_ethno.py` | builds the ethnographic corpus from an eHRAF-style CSV: per-document metadata headers, per-paragraph OCM tags (`...text...\n<\|ocm\|>131 133`), division markers, and an OCM codebook (codes, names, definitions) |
| train | `train.py` | HF-Llama-compatible GQA transformer — bf16 autocast, `torch.compile`, AdamW + cosine schedule, HF safetensors export |
| convert | `convert.py` | HF checkpoint → GGUF (wraps llama.cpp's `convert_hf_to_gguf`) |
| serve | llama.cpp | `llama-quantize` + `llama-server` (OpenAI-compatible API) |

Configs live in `configs/` (tiny smoke → small → medium). A typical
end-to-end smoke run:

```sh
make deps prepare train gguf quantize serve
```

### Serving notes

`make serve` takes overrides — `MODEL`, `PORT`, and `THREADS`:

```sh
make serve MODEL=checkpoints/medium_ethnographic_v3/medium-v3-q8_0.gguf
```

**Always pass an explicit thread count (`THREADS`, default 10); never `-t -1`.**
On the 20-core GB10, letting llama-server use every core collapses decode
throughput ~9× from thread oversubscription — `medium q8_0` measures
7.7 tok/s with `-t -1` but 70–73 tok/s with `-t 10` (8 and 4 threads give
68 and 55). The same oversubscription will bite on any many-core host, so
when serving elsewhere, pass the physical core count.

## The ethnographic corpus

The eHRAF-derived corpus is **not redistributed** — export your own CSV and
build the bins with `prepare_ethno.py`:

```sh
.venv/bin/python prepare_ethno.py --csv <export.csv> --out data/ethnographic_v3 \
    --ocm-labels data/ethnographic/ocmdefs.txt
```

`data/ethnographic/ocmdefs.txt` (OCM codes, names, and definitions) is
included as a reference document source; the corpus text itself is not.

## Measured throughput (DGX Spark / GB10, bf16, torch.compile, seq 1024)

| model | params | tok/s |
|-------|--------|-------|
| small | 118M   | 44.9k |
| medium | 303M  | 20.3k (compute-bound; batch 64 adds only +2%) |

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — GGUF conversion,
  quantization, and inference serving; cloned locally by `make deps`, not
  redistributed in this repo.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the uint16
  `train.bin`/`val.bin` packing convention follows its example; the tiny
  Shakespeare smoke-test data comes from
  [char-rnn](https://github.com/karpathy/char-rnn).
- The Outline of Cultural Materials is published by the Human Relations Area
  Files at Yale University; this repo ships only code — bring your own
  eHRAF export.

## License

MIT — see [LICENSE](LICENSE).
