# Analyst-assist server distribution (GPU-free, x86_64 Linux)

Self-contained llama.cpp server + minimal web UI for local, CPU-only use on a
RedHat-compatible host (RHEL / Rocky / Alma / Fedora).

## Requirements

    ./dependencies.sh

installs git, make and cmake, plus a gcc-toolset where the system compiler
is too old. No GPU, driver, or CUDA toolkit required.

**RHEL 8: the system GCC 8.5 is too old** for the pinned llama.cpp
(undefined `std::filesystem` links, CTAD and `<iomanip>` build errors).
`dependencies.sh` installs gcc-toolset-12, and the Makefile activates it
itself — it also wipes `llama.cpp/build` when its cached compiler no
longer matches, so plain `make` just works and nothing needs sourcing by
hand. Manual equivalent: install `gcc-toolset-12-gcc{,-c++}`, `rm -rf
llama.cpp/build`, `source /opt/rh/gcc-toolset-12/enable` in the same
shell as `make`.

(CentOS 7's GCC 4.8 is unsupported.)

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
- Behind a reverse proxy, mount it at a subpath with trailing slashes on both
  sides, e.g. `ProxyPass /demo/ http://127.0.0.1:8080/`. The web app resolves
  its API calls relative to the page URL, so no proxy rewriting is needed.
- `THREADS` defaults to all cores; ~10 is plenty for a 362M model. Never
  lower it below 1 or pass negative values.
- `THREADS_BATCH` (default = `THREADS`) sizes the prompt-processing pool
  separately: prompt processing is compute-bound and likes more threads than
  decode (memory-bandwidth-bound) — e.g. `THREADS=8 THREADS_BATCH=16` on a
  dual-socket host.
- `CTX` (default 8192) is the prompt + generation context window; the models
  are validated up to 8192 tokens.
- The web UI (`webapp/`) is served at `/`. Every OCM code in the results is
  clickable and opens an in-page dialog with that code's definition from the
  codebook (`webapp/ocmdefs.txt`). The **Sample** menu holds 10 gold sections (one
  per work) picked for HRAF analyst feedback: with **section mode** ticked,
  selecting a sample inserts the whole section; otherwise it inserts the
  first paragraph and a **Paragraph** spinner appears to pick any of them.
  Tick **section mode** to analyze a
  whole section: first line is the section title, then blank-line-separated
  paragraphs. Each paragraph is tagged with an OCM union carried forward from
  the predictions so far (the leave-one-out flow fed with the model's own
  earlier answers), and results are listed `a) b) c)…` in the output box,
  followed by the aggregate OCM set for the whole section (`+` attached to
  a code's label when seen twice, `++` for three or more).
  For API use, POST JSON directly to `/completion`:

      curl -s http://127.0.0.1:8080/completion \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Preserving food is critical for most populations.","n_predict":64}'

## Layout

    Makefile          build + local install (see targets: fetch, build, install, serve, clean)
    dependencies.sh   install build deps (git/make/cmake; gcc-toolset where the
                      system GCC is too old, i.e. RHEL 8)
    serve.sh          start script (PORT / THREADS / CTX / MODEL env overrides)
    demodist.tgz      tarball of just these package files, for building on a
                      remote host (regenerate from the repo root after edits)
    webapp/index.html the two-textarea UI (POSTs to /completion)
    webapp/ocmdefs.txt OCM codebook for the clickable definitions
                       (verbatim copy of the dev repo's data/ethnographic/ocmdefs.txt)
    webapp/samples.js  the 10 gold feedback sections (brief eHRAF excerpts,
                       cleared as fair use)
    runtime/          installed binaries (created by make)
    models/           drop .gguf models here (created by make)
