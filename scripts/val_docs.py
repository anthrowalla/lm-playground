"""Reconstruct the held-out val documents (with gold OCM tags) for the tagging eval.

prepare_ethno.py holds out a document-aligned tail of the packed corpus as
val.bin, and inserts --extra-corpus (journal) text before the ethnography so
the val split stays on-task. This script identifies exactly which eHRAF
documents that tail contains and emits them as JSONL with per-paragraph gold
OCM tags, ready for the precision/recall eval (wayforward.md Gate 0).

Method (self-verifying, both checks must pass):
  1. val.bin is decoded back to text with the saved ByteLevel tokenizer
     (lossless) and must be the literal suffix of corpus.txt;
  2. the decoded documents must match fresh format_document() regenerations
     from the source CSV, in order.

Paragraphs are emitted from the CSV side, so they are the true logical
paragraphs even when blank lines inside one would be ambiguous in the
serialized corpus.

Usage:
    .venv/bin/python scripts/val_docs.py [--csv /home/mf1/et43texts/ethnotext_v430.csv]
        [--data-dir data/ethnographic_v4] [--ocm-labels data/ethnographic/ocmdefs.txt]
        [--out data/ethnographic_v4/val_docs.jsonl] [--scan-mb 64]
"""

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_ethno as pe

DEFAULT_CSV = "/home/mf1/et43texts/ethnotext_v430.csv"


def normalize_newlines(text: str) -> str:
    # prepare_ethno read the corpus in universal-newlines mode; replicate.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_tags(paragraph: str) -> tuple[str, list[list[str]]]:
    """Split 'text\\n<|ocm|>131 LOCATION 133 TOPOGRAPHY' into (text, [[code,
    name], ...])."""
    if pe.OCM not in paragraph:
        return paragraph, []
    text, _, tag = paragraph.partition(pe.OCM)
    pairs, code, name = [], None, []
    for t in tag.split():
        if t.isdigit() and 3 <= len(t) <= 4:
            if code is not None:
                pairs.append([code, " ".join(name)])
            code, name = t, []
        elif code is not None:
            name.append(t)
    if code is not None:
        pairs.append([code, " ".join(name)])
    return text, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=DEFAULT_CSV, help="eHRAF export CSV used at build time")
    ap.add_argument("--data-dir", default="data/ethnographic_v4")
    ap.add_argument("--ocm-labels", default="data/ethnographic/ocmdefs.txt")
    ap.add_argument("--out", default=None,
                    help="output JSONL (default <data-dir>/val_docs.jsonl)")
    ap.add_argument("--scan-mb", type=int, default=64,
                    help="how much of corpus.txt to read for the suffix check")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "val_docs.jsonl"

    val_ids = np.fromfile(data_dir / "val.bin", dtype=np.uint16).tolist()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    val_text = tok.decode(val_ids, skip_special_tokens=False)
    n_docs = val_text.count(pe.BOS)
    print(f"decoded val.bin: {len(val_ids):,} tokens, {n_docs} documents, "
          f"{len(val_text):,} chars")

    # --- check 1: decoded val text is the suffix of corpus.txt ------------
    tail = normalize_newlines(
        (data_dir / "corpus.txt").read_bytes()[-args.scan_mb * (1 << 20):]
        .decode("utf-8"))
    if not tail.endswith(val_text):
        raise SystemExit("verification failed: decoded val text is not the "
                         "suffix of corpus.txt")
    print("verified: decoded val text is the exact suffix of corpus.txt")

    # --- check 2: documents match fresh CSV regeneration ------------------
    # The split lands right after the boundary <|eos|> token, so the newline
    # ending that line leads val; the regenerated documents don't have it.
    val_docs_text = val_text.lstrip("\n")
    names = pe.load_code_names(Path(args.ocm_labels))
    docs = deque(maxlen=n_docs)  # (fields, paragraphs, serialized)
    n_csv_docs = 0
    for fields, paragraphs in pe.iter_documents(args.csv, names):
        docs.append((fields, paragraphs, pe.format_document(fields, paragraphs)))
        n_csv_docs += 1
    regenerated = "".join(d[2] for d in docs)
    if normalize_newlines(regenerated) != val_docs_text:
        raise SystemExit("verification failed: val documents do not match the "
                         f"last {n_docs} CSV documents")
    print(f"verified: matches the last {n_docs:,} of {n_csv_docs:,} CSV "
          "documents")

    # --- emit JSONL with gold tags ----------------------------------------
    cultures, dates, owcs = Counter(), Counter(), Counter()
    n_paras = n_tagged = 0
    with out.open("w", encoding="utf-8") as f:
        for fields, paragraphs, _ in docs:
            paras = []
            for p in paragraphs:
                text, pairs = parse_tags(p)
                n_paras += 1
                n_tagged += bool(pairs)
                paras.append({"text": text, "tags": pairs})
            f.write(json.dumps({"fields": fields, "paragraphs": paras},
                               ensure_ascii=False) + "\n")
            cultures[fields.get("Culture", "?")] += 1
            dates[fields.get("Field dates", "?")] += 1
            for o in fields.get("OWC", "").replace("|", " ").split():
                owcs[o] += 1

    print(f"wrote {out}: {n_docs} docs, {n_paras:,} paragraphs "
          f"({n_tagged:,} tagged)")
    print("\ntop cultures:", cultures.most_common(10))
    print("\nfield dates:", dates.most_common(12))
    print("\nOWC regions:", owcs.most_common(12))


if __name__ == "__main__":
    main()
