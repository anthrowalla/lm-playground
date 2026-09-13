"""Build the ethnographic corpus from eHRAF-style CSV into trainable bins.

Streams the CSV, splits it into documents wherever the 'title' column changes
(chronological/document order), and writes <|bos|> ... <|eos|> wrapped text:
a one-line metadata header, paragraphs each tagged with their OCM subject
codes (<|ocm|>... after the paragraph text), and <|div|> markers at division
(chapter-scale) boundaries. Then trains a BPE tokenizer and packs uint16 bins
exactly like prepare.py, with the val split aligned to a document boundary
near the end of the corpus.
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

OCM, DIV = "<|ocm|>", "<|div|>"

HEADER_FIELDS = [
    ("title", "Title"),
    ("byline", "Author"),
    ("culture", "Culture"),
    ("place", "Place"),
    ("coverage", "Coverage"),
    ("pub.date", "Published"),
    ("pub.lang", "Language"),
    ("pub.type", "Type"),
    ("owcs", "OWC"),
    ("field.date", "Field dates"),
]

SKIP_TEXT = {"", "none"}
NULL_DATES = {"no date", "not applicable", "not specified", "n/a", "unknown"}
MARKUP = re.compile(r"\{[^{}]*\}")
OCM_CODES = re.compile(r"#?(\d{3,4})(?!\d)")


def is_markup_line(text: str) -> bool:
    """True for lines that are nothing but eHRAF markup blocks ({POST},
    {fig: {graphic: ...}}, ...), allowing nesting."""
    remainder = MARKUP.sub("", text)
    while remainder != text:
        text, remainder = remainder, MARKUP.sub("", remainder)
    return not remainder.strip()


def clean(value: str | None) -> str:
    return (value or "").replace("|", "/").replace("\n", " ").strip()


def load_code_names(defs_path: Path) -> dict[str, str]:
    """Map OCM code string -> official name, parsed from the definitions file."""
    names: dict[str, str] = {}
    for block in defs_path.read_text(encoding="utf-8").split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            m = re.match(r"(\d{3,4})\s+(.+?)\s*$", lines[0])
            if m:
                names[m.group(1)] = m.group(2)
    return names


def iter_documents(csv_path: str, names: dict[str, str] | None = None):
    """Yield (fields, paragraphs) per title change, preserving row order.

    Paragraphs carry their OCM codes appended as <|ocm|>131 133 (or
    <|ocm|>131 LOCATION 133 TOPOGRAPHY AND GEOLOGY when a name map is
    given), and a standalone <|div|> entry precedes each new division
    within a document.
    """
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cur_title, fields, paragraphs = None, {}, []
        prev_div = None
        for row in reader:
            text = (row.get("text") or "").strip()
            title = clean(row.get("title"))
            if title != cur_title:
                if cur_title is not None:
                    yield fields, paragraphs
                cur_title, fields, paragraphs = title, {}, []
                prev_div = None
            if not text or text.lower() in SKIP_TEXT or is_markup_line(text):
                continue
            for col, label in HEADER_FIELDS:
                value = clean(row.get(col))
                if label == "Field dates":
                    value = value.lower()
                    if value in NULL_DATES:
                        continue
                if value:
                    fields[label] = value
            division = clean(row.get("division"))
            if division and division != prev_div:
                if prev_div is not None:
                    paragraphs.append(DIV)
                prev_div = division
            codes = list(dict.fromkeys(OCM_CODES.findall(row.get("ocms") or "")))
            if codes:
                if names:
                    expanded = [f"{c} {names[c]}" if c in names else c for c in codes]
                else:
                    expanded = codes
                paragraphs.append(f"{text}\n{OCM}{' '.join(expanded)}")
            else:
                paragraphs.append(text)
        if cur_title is not None:
            yield fields, paragraphs


def format_document(fields: dict, paragraphs: list[str]) -> str:
    header = " | ".join(f"{label}: {fields[label]}" for _, label in HEADER_FIELDS if label in fields)
    return f"<|bos|>{header}\n\n" + "\n\n".join(paragraphs) + "\n<|eos|>\n"


def format_codebook(defs_path: Path) -> str:
    """Codebook document: one paragraph per OCM code — <|ocm|>CODE NAME,
    then its definition and (when present) related-term cross references,
    teaching code -> name -> meaning."""
    entries: list[str] = []
    for block in defs_path.read_text(encoding="utf-8").split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        m = re.match(r"(\d{3,4})\s+(.+?)\s*$", lines[0])
        if not m:
            if entries:
                entries[-1] += "\n" + "\n".join(lines)
            continue
        parts = [f"{OCM}{m.group(1)} {m.group(2)}"]
        for line in lines[1:]:
            if line.startswith("Summary - "):
                line = line[len("Summary - "):]
            parts.append(line)
        entries.append("\n".join(parts))
    fields = {
        "Title": "Outline of Cultural Materials (OCM) subject codes and definitions",
        "Language": "English",
        "Type": "Reference",
    }
    return format_document(fields, entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="ethnographic corpus CSV")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--val-tokens", type=int, default=1_048_576, help="tokens held out for val")
    parser.add_argument("--corpus", default=None, help="keep the intermediate plain-text corpus here")
    parser.add_argument("--ocm-labels", default=None, help="OCM definitions file (blocks: CODE NAME / Summary - ... / Related Terms - ...) to embed as reference docs")
    parser.add_argument("--ocm-labels-repeat", type=int, default=3, help="how many times to repeat the codebook")
    parser.add_argument("--tag-names", action="store_true", help="expand OCM codes in paragraph tags with their names (requires --ocm-labels)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus) if args.corpus else out / "corpus.txt"
    names = load_code_names(Path(args.ocm_labels)) if args.tag_names and args.ocm_labels else None
    if args.tag_names and not args.ocm_labels:
        raise SystemExit("--tag-names requires --ocm-labels")

    n_docs, n_chars = 0, 0
    with corpus_path.open("w", encoding="utf-8") as f:
        if args.ocm_labels:
            for _ in range(max(1, args.ocm_labels_repeat)):
                doc = format_codebook(Path(args.ocm_labels))
                f.write(doc)
                n_docs += 1
                n_chars += len(doc)
        for fields, paragraphs in iter_documents(args.csv, names):
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
        special_tokens=[*SPECIALS, OCM, DIV],
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
        additional_special_tokens=[OCM, DIV],
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
        "ocm_id": tokenizer.convert_tokens_to_ids(OCM),
        "div_id": tokenizer.convert_tokens_to_ids(DIV),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/train.bin ({train_len:,}), {out}/val.bin ({total - train_len:,})")
    print(f"vocab_size={meta['vocab_size']} (requested {args.vocab_size})")


if __name__ == "__main__":
    main()
