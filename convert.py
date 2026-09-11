"""Convert a checkpoint to GGUF, forcing the gpt2 (ByteLevel) pre-tokenizer.

prepare.py always trains a ByteLevel BPE, but llama.cpp's converter identifies
pre-tokenizers by matching against a hardcoded table of known production
tokenizers, which a small custom BPE never matches. This wrapper pins the
detection to llama.cpp's "gpt-2" pre-type, which implements the ByteLevel regex
(older llama.cpp releases call this pre-type "gpt2").

Usage: same CLI as llama.cpp/convert_hf_to_gguf.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "llama.cpp"))

import conversion.base  # noqa: E402
import convert_hf_to_gguf  # noqa: E402

conversion.base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"

if __name__ == "__main__":
    convert_hf_to_gguf.main()
