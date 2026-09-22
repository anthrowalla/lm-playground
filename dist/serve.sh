#!/usr/bin/env bash
# Start the analyst-assist llama.cpp server (CPU only) with the webapp UI.
# Env overrides: PORT (8080), THREADS (nproc), CTX (8192), MODEL (gguf path).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
THREADS="${THREADS:-$(nproc)}"
CTX="${CTX:-8192}"
MODEL="${MODEL:-$(ls models/*.gguf 2>/dev/null | head -n1 || true)}"

if [ -z "$MODEL" ]; then
  echo "No model found. Copy a .gguf into models/ (or set MODEL=/path/to/model.gguf)." >&2
  exit 1
fi
if [ ! -x runtime/bin/llama-server ]; then
  echo "Server not built. Run: make" >&2
  exit 1
fi

echo "serving $MODEL on http://127.0.0.1:$PORT (threads=$THREADS ctx=$CTX)"
export LD_LIBRARY_PATH="$PWD/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec runtime/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -t "$THREADS" \
  -c "$CTX" \
  --parallel 1 \
  --path webapp
