"""Build the ethnographic corpus v5 directly from the eHRAF text directory tree.

Unlike prepare_ethno.py (CSV-driven, structure-free), this walks
AREA/SOCIETY/hdoc.txt and recovers the document hierarchy the CSV flattened:

- paragraphs are p-type SREs only (bibliography items and endnotes are
  dropped — the old pipeline silently included both, notes even duplicated);
- {POST}-style {...} markup is stripped nested-aware from kept text;
- entering a section emits a <|sec|> header carrying the full title path
  (ancestors joined " / ", [No Title] components skipped) and the union of
  OCM codes over all paragraphs under that path — the analyst observation
  that codes persist across a section;
- documents are keyed by hdocid (verified unique across the tree);
- the val split is a stratified whole-society holdout spread across all
  eight OWC areas, written as a separate stream — no more single-culture
  tail split;
- files without ## hdocid:: (*_collection_materials.txt, *_hraf.docbook.txt)
  are skipped.

Usage:
    .venv/bin/python prepare_ethno_tree.py --tree ~/et43texts/ethnotext_v430 \
        --out data/ethnographic_v5 --ocm-labels data/ethnographic/ocmdefs.txt \
        --tag-names --extra-corpus data/aa/aa_corpus.txt
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from prepare import BOS, EOS, PAD, UNK, SPECIALS
from prepare_ethno import (HEADER_FIELDS, MARKUP, NULL_DATES, OCM, OCM_CODES,
                           SKIP_TEXT, clean, format_codebook, load_code_names)

SEC = "<|sec|>"
CONTAINER_NAMES = {"front": "Front matter", "back": "Back matter", "bibliography": "Bibliography"}
NO_TITLE = {"", "[no title]"}
SECTION_OPEN = re.compile(r"^## ([a-z.]+)::([^;]+);(.*?)<\s*$")
SECTION_CLOSE = re.compile(r"^## ([a-z.]+)::([^;]+);.*>\s*$")
PARENT = re.compile(r"section\.parent::([^;]*);")
SRE_LINE = re.compile(r"^\{[^}]*\}\{([a-z.]+)\}\[([^\]]*)\]\s?(.*)$")
BYTES_PER_TOKEN_ESTIMATE = 8.4  # measured on v4 ethnographic text


def strip_markup(text: str) -> str:
    """Remove {...} blocks nested-aware, then collapse the leftover whitespace."""
    prev = None
    while prev != text:
        prev = text
        text = MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def title_component(title: str) -> str | None:
    title = title.strip()
    return title if title and title.lower() not in NO_TITLE else None


def parse_file(fp: Path) -> tuple[str, dict, list] | None:
    """Parse one document file -> (hdoc, fields, paragraphs).

    paragraphs = [(path_tuple, text, codes)] for kept p-type SREs, in order.
    Returns None for files without a hdocid (collection materials, docbook).
    """
    lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    hdoc_m = re.match(r"^## hdocid::([^;]+);", lines[0]) if lines else None
    if not hdoc_m:
        return None
    hdoc = hdoc_m.group(1).strip()

    stack: list[tuple[str, str | None]] = []  # (eid, path title or None)
    paragraphs: list[tuple[tuple[str, ...], str, list[str]]] = []
    fields: dict = {}
    ocms_codes: list[str] = []

    for line in lines:
        if line.startswith("## ocms::"):
            ocms_codes = list(dict.fromkeys(
                OCM_CODES.findall(line.split(";", 1)[0])))
            if not fields:
                for seg in line.split(";")[1:]:
                    k, _, v = seg.partition("::")
                    fields[k.strip()] = clean(v)
            continue
        m = SRE_LINE.match(line)
        if m:
            kind, _, text = m.groups()
            codes = list(dict.fromkeys(ocms_codes))
            ocms_codes = []
            if kind != "p":
                continue
            text = strip_markup(text)
            if not text or text.lower() in SKIP_TEXT:
                continue
            path = tuple(t for _, t in stack if t)
            paragraphs.append((path, text, codes))
            continue
        if line.startswith("## section::") or line.startswith("## body::") \
                or line.startswith("## front::") or line.startswith("## back::") \
                or line.startswith("## bibliography::"):
            m = SECTION_OPEN.match(line)
            if m:
                kind, eid, rest = m.groups()
                title = title_component(PARENT.sub("", rest).strip(" ;"))
                if kind == "body":
                    stack.append((eid, None))
                else:
                    stack.append((eid, title or CONTAINER_NAMES.get(kind, kind)))
                continue
            m = SECTION_CLOSE.match(line)
            if m and stack:
                eid = m.group(2)
                while stack and stack[-1][0] != eid:
                    stack.pop()
                if stack:
                    stack.pop()

    if not paragraphs:
        return None
    # document fields mirror the old CSV columns, from the first SRE meta line
    out_fields = {}
    for key, label in HEADER_FIELDS:
        value = fields.get(key, "")
        if label == "Field dates":
            value = value.lower()
            if value in NULL_DATES:
                continue
        if value:
            out_fields[label] = value
    return hdoc, out_fields, paragraphs


def format_items(paragraphs: list, names: dict[str, str] | None) -> tuple[list[str], Counter]:
    """Interleave <|sec|> headers (lazy, on path change) with tagged paragraphs.

    Returns (items, stats) where stats counts headers/paragraphs/unions.
    """
    unions: dict[tuple[str, ...], set] = {}
    for path, _, codes in paragraphs:
        if not codes:
            continue
        for i in range(len(path) + 1):
            unions.setdefault(path[:i], set()).update(codes)

    def tag_str(codes):
        if names:
            return " ".join(f"{c} {names[c]}" if c in names else c for c in codes)
        return " ".join(codes)

    items, last_path = [], None
    stats = Counter()
    for path, text, codes in paragraphs:
        if path != last_path:
            last_path = path
            union = sorted(unions.get(path, ()), key=lambda c: (len(c), c))
            header = f"{SEC}{' / '.join(path)}".rstrip()
            if union:
                items.append(f"{header}\n{OCM}{tag_str(union)}")
                stats["headers_with_union"] += 1
            else:
                items.append(header)
                stats["headers_plain"] += 1
        if codes:
            items.append(f"{text}\n{OCM}{tag_str(codes)}")
            stats["tagged_paragraphs"] += 1
        else:
            items.append(text)
            stats["untagged_paragraphs"] += 1
    return items, stats


def format_document(fields: dict, items: list[str]) -> str:
    header = " | ".join(f"{label}: {fields[label]}" for _, label in HEADER_FIELDS if label in fields)
    return f"<|bos|>{header}\n\n" + "\n\n".join(items) + "\n<|eos|>\n"


def iter_tree(tree: Path):
    for fp in sorted(tree.glob("*/*/*.txt")):
        doc = parse_file(fp)
        if doc is None:
            yield fp, None
        else:
            yield fp, doc


def select_val_societies(sizes: dict[str, dict[str, int]], target_chars: int,
                         seed: int) -> tuple[set[str], list]:
    """Pick whole societies per area, greedy toward target_chars per area."""
    rng = random.Random(seed)
    chosen, report = set(), []
    for area in sorted(sizes):
        socs = [(s, n) for s, n in sizes[area].items() if n > 0]
        per_area = target_chars / len(sizes)
        small = [x for x in socs if x[1] <= per_area * 1.5]
        rng.shuffle(small)
        acc, picks = 0, []
        for soc, n in small:
            if acc >= per_area:
                break
            picks.append((soc, n))
            acc += n
        if not picks and socs:
            picks.append(min(socs, key=lambda x: x[1]))
        for soc, n in picks:
            chosen.add(f"{area}/{soc}")
        report.append((area, picks, acc))
    return chosen, report


def encode_file(src: Path, dst: Path, tokenizer) -> int:
    total = 0
    with src.open(encoding="utf-8") as f, dst.open("wb") as out:
        batch, size = [], 0
        for line in f:
            batch.append(line)
            size += len(line)
            if size >= 8_000_000:
                for ids in tokenizer(batch).input_ids:
                    out.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    total += len(ids)
                batch, size = [], 0
        if batch:
            for ids in tokenizer(batch).input_ids:
                out.write(np.asarray(ids, dtype=np.uint16).tobytes())
                total += len(ids)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", required=True, help="ethnotext directory tree (AREA/SOCIETY/hdoc.txt)")
    parser.add_argument("--out", required=True, help="output directory, e.g. data/ethnographic_v5")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--val-mb", type=float, default=10.0,
                        help="target val holdout size in MB of kept paragraph text (~1.2M tokens)")
    parser.add_argument("--ocm-labels", default=None, help="OCM definitions file for the codebook + named tags")
    parser.add_argument("--tag-names", action="store_true", help="expand OCM codes with their names")
    parser.add_argument("--ocm-labels-repeat", type=int, default=3)
    parser.add_argument("--extra-corpus", action="append", default=[],
                        help="plain-text corpus file(s) (e.g. journal text) inserted after the codebook (train only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0, help="only first N files (smoke)")
    args = parser.parse_args()
    if args.tag_names and not args.ocm_labels:
        raise SystemExit("--tag-names requires --ocm-labels")
    names = load_code_names(Path(args.ocm_labels)) if args.tag_names and args.ocm_labels else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = Path(args.tree).expanduser()

    # pass 1: sizes for the stratified split
    sizes: dict[str, dict[str, int]] = {}
    skipped_files, n_docs = [], 0
    for i, (fp, doc) in enumerate(iter_tree(tree)):
        if args.limit_files and i >= args.limit_files:
            break
        area, soc, _ = fp.relative_to(tree).parts
        if doc is None:
            skipped_files.append(str(fp.relative_to(tree)))
            continue
        hdoc, fields, paragraphs = doc
        sizes.setdefault(area, {}).setdefault(soc, 0)
        sizes[area][soc] += sum(len(t) for _, t, _ in paragraphs)
        n_docs += 1
    total_chars = sum(n for area in sizes.values() for n in area.values())
    val_societies, report = select_val_societies(sizes, args.val_mb * 1e6, args.seed)
    print(f"pass 1: {n_docs:,} documents, {total_chars:,} chars of kept paragraph text")
    print(f"val holdout: {len(val_societies)} societies across {len(report)} areas "
          f"(target {args.val_mb:.1f} MB ~ {args.val_mb * 1e6 / BYTES_PER_TOKEN_ESTIMATE / 1e6:.2f}M tokens)")
    for area, picks, acc in report:
        names_str = ", ".join(f"{s} ({n/1e6:.2f}MB)" for s, n in picks) or "-"
        print(f"  {area:9s} {acc/1e6:5.2f}MB: {names_str}")
    if skipped_files:
        print(f"skipped {len(skipped_files)} files (no hdocid, no p-type SREs, "
              f"or no kept text after markup strip): {', '.join(skipped_files[:6])}"
              + (" ..." if len(skipped_files) > 6 else ""))

    # pass 2: write train and val text streams
    stats = Counter()
    train_path, val_path = out / "corpus_train.txt", out / "corpus_val.txt"
    val_docs_path = out / "val_docs.jsonl"
    n_train_docs = n_val_docs = 0
    with train_path.open("w", encoding="utf-8") as ftr, \
            val_path.open("w", encoding="utf-8") as fva, \
            val_docs_path.open("w", encoding="utf-8") as fvd:
        if args.ocm_labels:
            for _ in range(max(1, args.ocm_labels_repeat)):
                ftr.write(format_codebook(Path(args.ocm_labels)))
        for extra in args.extra_corpus:
            ftr.write(Path(extra).read_text(encoding="utf-8"))
        for i, (fp, doc) in enumerate(iter_tree(tree)):
            if args.limit_files and i >= args.limit_files:
                break
            if doc is None:
                continue
            hdoc, fields, paragraphs = doc
            area, soc, _ = fp.relative_to(tree).parts
            items, st = format_items(paragraphs, names)
            stats.update(st)
            is_val = f"{area}/{soc}" in val_societies
            doc_text = format_document(fields, items)
            if is_val:
                fva.write(doc_text)
                n_val_docs += 1
                fvd.write(json.dumps({
                    "hdoc": hdoc,
                    "society": soc,
                    "area": area,
                    "fields": fields,
                    "paragraphs": [
                        {"text": t,
                         "tags": [[c, names.get(c, "") if names else ""] for c in codes],
                         "section": " / ".join(path)}
                        for path, t, codes in paragraphs
                    ],
                }, ensure_ascii=False) + "\n")
            else:
                ftr.write(doc_text)
                n_train_docs += 1
    print(f"pass 2: train {n_train_docs:,} docs, val {n_val_docs:,} docs "
          f"({stats['tagged_paragraphs']:,} tagged + {stats['untagged_paragraphs']:,} untagged paragraphs, "
          f"{stats['headers_with_union']:,} section headers with union, {stats['headers_plain']:,} plain)")

    # tokenizer: train on the train stream only, so val text never shapes merges
    base = Tokenizer(models.BPE(unk_token=None))
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[*SPECIALS, OCM, SEC],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    base.train([str(train_path)], trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
        additional_special_tokens=[OCM, SEC],
        model_max_length=1_000_000,
    )
    tokenizer.save_pretrained(out)

    train_tokens = encode_file(train_path, out / "train.bin", tokenizer)
    val_tokens = encode_file(val_path, out / "val.bin", tokenizer)

    # self-check: the ByteLevel tokenizer is lossless, so val.bin must decode
    # back to corpus_val.txt exactly.
    decoded = tokenizer.decode(tokenizer(val_path.read_text(encoding="utf-8")).input_ids,
                               skip_special_tokens=False)
    ok = decoded == val_path.read_text(encoding="utf-8")
    print(f"val round-trip check: {'PASS' if ok else 'FAIL'}")

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.convert_tokens_to_ids(PAD),
        "bos_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_id": tokenizer.convert_tokens_to_ids(EOS),
        "unk_id": tokenizer.convert_tokens_to_ids(UNK),
        "ocm_id": tokenizer.convert_tokens_to_ids(OCM),
        "sec_id": tokenizer.convert_tokens_to_ids(SEC),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/train.bin ({train_tokens:,} tokens), {out}/val.bin ({val_tokens:,} tokens)")
    print(f"vocab_size={meta['vocab_size']} (requested {args.vocab_size})")


if __name__ == "__main__":
    main()
