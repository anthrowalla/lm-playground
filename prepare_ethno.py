"""Build the ethnographic corpus from eHRAF-style CSV into trainable bins.

Streams the CSV, splits it into documents wherever the 'title' column changes
(chronological/document order), prepends a one-line metadata header per
document, and writes <|bos|> ... <|eos|> wrapped text. Then trains a BPE
tokenizer and packs uint16 bins exactly like prepare.py, with the val split
aligned to a document boundary near the end of the corpus.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from prepare import BOS, EOS, PAD, UNK, SPECIALS

HEADER_FIELDS = [
    ("title", "Title"),
    ("byline", "Author"),
    ("culture", "Culture"),
    ("place", "Place"),
    ("coverage", "Coverage"),
    ("pub.date", "Published"),
    ("pub.lang", "Language"),
    ("pub.type", "Type"),
]

SKIP_TEXT = {"", "none"}
MARKUP = re.compile(r"\{[^{}]*\}")


def is_markup_line(text: str) -> bool:
    """True for lines that are nothing but eHRAF markup blocks ({POST},
    {fig: {graphic: ...}}, ...), allowing nesting."""
    remainder = MARKUP.sub("", text)
    while remainder != text:
        text, remainder = remainder, MARKUP.sub("", remainder)
    return not remainder.strip()


def clean(value: str | None) -> str:
    return (value or "").replace("|", "/").replace("\n", " ").strip()


def iter_documents(csv_path: str):
    """Yield (header, paragraphs) per title change, preserving row order."""
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cur_title, fields, paragraphs = None, {}, []
        for row in reader:
            text = (row.get("text") or "").strip()
            title = clean(row.get("title"))
            if title != cur_title:
                if cur_title is not None:
                    yield fields, paragraphs
                cur_title, fields, paragraphs = title, {}, []
            if not text or text.lower() in SKIP_TEXT or is_markup_line(text):
                continue
            for col, label in HEADER_FIELDS:
                value = clean(row.get(col))
                if value:
                    fields[label] = value
            paragraphs.append(text)
        if cur_title is not None:
            yield fields, paragraphs


def format_document(fields: dict, paragraphs: list[str]) -> str:
    header = " | ".join(f"{label}: {fields[label]}" for _, label in HEADER_FIELDS if label in fields)
    return f"<|bos|>{header}\n\n" + "\n\n".join(paragraphs) + "\n<|eos|>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="ethnographic corpus CSV")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--val-tokens", type=int, default=1_048_576, help="tokens held out for val")
    parser.add_argument("--corpus", default=None, help="keep the intermediate plain-text corpus here")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus) if args.corpus else out / "corpus.txt"

    n_docs, n_chars = 0, 0
    with corpus_path.open("w", encoding="utf-8") as f:
        for fields, paragraphs in iter_documents(args.csv):
            doc = format_document(fields, paragraphs)
            f.write(doc)
            n_docs += 1
            n_chars += len(doc)
    print(f"corpus: {n_docs:,} documents, {n_chars:,} chars -> {corpus_path}")

    base = Tokenizer(models.BPE(unk_token=None))
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    base.train([str(corpus_path)], trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
        model_max_length=1_000_000,
    )

    # Stream-encode the corpus in batches to keep memory flat.
    eos_id = tokenizer.convert_tokens_to_ids(EOS)
    total = 0
    with corpus_path.open(encoding="utf-8") as f, (out / "train.bin").open("wb") as dst:
        batch, size = [], 0
        for line in f:
            batch.append(line)
            size += len(line)
            if size >= 8_000_000:
                for ids in tokenizer(batch).input_ids:
                    dst.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    total += len(ids)
                batch, size = [], 0
        if batch:
            for ids in tokenizer(batch).input_ids:
                dst.write(np.asarray(ids, dtype=np.uint16).tobytes())
                total += len(ids)
    print(f"tokens: {total:,}")

    # Move the final val block to val.bin, aligned to the last document
    # boundary at or before the cut.
    ids = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    cut = len(ids) - args.val_tokens
    if cut <= 0:
        raise SystemExit(f"corpus ({len(ids):,} tokens) smaller than val split")
    boundaries = np.nonzero(ids[: cut + 1] == eos_id)[0]
    boundary = int(boundaries[-1])
    np.asarray(ids[boundary + 1 :], dtype=np.uint16).tofile(out / "val.bin")
    train_len = boundary + 1
    del ids
    with (out / "train.bin").open("r+b") as f:
        f.truncate(train_len * 2)

    tokenizer.save_pretrained(out)
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.convert_tokens_to_ids(PAD),
        "bos_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_id": eos_id,
        "unk_id": tokenizer.convert_tokens_to_ids(UNK),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/train.bin ({train_len:,}), {out}/val.bin ({total - train_len:,})")
    print(f"vocab_size={meta['vocab_size']} (requested {args.vocab_size})")


if __name__ == "__main__":
    main()
