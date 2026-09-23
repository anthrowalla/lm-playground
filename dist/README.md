# Analyst-assist server distribution (GPU-free, x86_64 Linux)

Self-contained llama.cpp server + minimal web UI for local, CPU-only use on a
RedHat-compatible host (RHEL / Rocky / Alma / Fedora).

## Requirements

    sudo dnf install -y gcc gcc-c++ make cmake git

No GPU, driver, or CUDA toolkit required.

## Build & install

    make

fetches the llama.cpp source at a pinned upstream commit (the one the models
were validated against), builds the CPU-only `llama-server`, and installs it
into `./runtime/`. If fetch-by-SHA is blocked on your network, clone
llama.cpp into `dist/llama.cpp` yourself (checkout the commit in the
Makefile's `LLAMA_REF`) and re-run `make`.

## Add a model

    cp /path/to/model-q8_0.gguf models/     # e.g. smollm2-dapt-tag-8k-q8_0.gguf

Models are not included in this package. Whichever single `.gguf` sits in
`models/` is the one served (override with `MODEL=`).

## Run

    ./serve.sh                          # open http://127.0.0.1:8080 in a browser
    PORT=8090 THREADS=16 ./serve.sh     # overrides

- Binds to `127.0.0.1` only by default — reachable from the host itself, not the
  network. Use `HOST=0.0.0.0 ./serve.sh` to expose it to the local network.
- `THREADS` defaults to all cores; ~10 is plenty for a 362M model. Never
  lower it below 1 or pass negative values.
- `CTX` (default 8192) is the prompt + generation context window; the models
  are validated up to 8192 tokens.
- The web UI (`webapp/`) is served at `/`. Tick **section mode** to analyze a
  whole section: first line is the section title, then blank-line-separated
  paragraphs. Each paragraph is tagged with an OCM union carried forward from
  the predictions so far (the leave-one-out flow fed with the model's own
  earlier answers), and results are listed `a) b) c)…` in the output box.
  For API use, POST JSON directly to `/completion`:

      curl -s http://127.0.0.1:8080/completion \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Preserving food is critical for most populations.","n_predict":64}'

## Layout

    Makefile          build + local install (see targets: fetch, build, install, serve, clean)
    serve.sh          start script (PORT / THREADS / CTX / MODEL env overrides)
    webapp/index.html the two-textarea UI (POSTs to /completion)
    runtime/          installed binaries (created by make)
    models/           drop .gguf models here (created by make)
