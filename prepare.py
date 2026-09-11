"""Train a BPE tokenizer and pack a text corpus into uint16 bin files.

Outputs <out>/train.bin, <out>/val.bin (uint16 token streams), tokenizer files
loadable by transformers/llama.cpp, and meta.json with vocab metadata.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

PAD, BOS, EOS, UNK = "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"
SPECIALS = [PAD, BOS, EOS, UNK]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="plain-text corpus")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--val-tokens", type=int, default=1_048_576, help="tokens held out for val")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    text = Path(args.input).read_text(encoding="utf-8")

    base = Tokenizer(models.BPE(unk_token=None))
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    base.train([args.input], trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
        model_max_length=1_000_000,
    )
    ids = tokenizer.encode(text)
    print(f"corpus: {len(text):,} chars -> {len(ids):,} tokens")

    n_val = min(args.val_tokens, max(4096, len(ids) // 10))
    val_ids = np.array(ids[:n_val], dtype=np.uint16)
    train_ids = np.array(ids[n_val:], dtype=np.uint16)
    val_ids.tofile(out / "val.bin")
    train_ids.tofile(out / "train.bin")

    tokenizer.save_pretrained(out)
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.convert_tokens_to_ids(PAD),
        "bos_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_id": tokenizer.convert_tokens_to_ids(EOS),
        "unk_id": tokenizer.convert_tokens_to_ids(UNK),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/train.bin ({len(train_ids):,}), {out}/val.bin ({len(val_ids):,})")
    print(f"vocab_size={meta['vocab_size']} (requested {args.vocab_size})")


if __name__ == "__main__":
    main()
