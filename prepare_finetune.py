"""Build tokenized prompt/target pairs for the OCM-tagging fine-tune.

Sources the v5 corpus tree directly (p-SREs, section paths) — never the
val societies. Each example: [optional <|sec|> header] + text + "\n" +
<|ocm|> marker, target = the paragraph's named code line + "\n\n" (loss is
masked everywhere except the target). sec-loo examples (default 70%) carry
the training-native header with the leave-one-out union of the section's
OTHER paragraphs' gold codes — the deployment condition; cold examples
(30%) carry no header. Single-paragraph sections can only be cold.

Output: uint16 token bin + int64 offsets + int32 prompt lengths, consumed
by train_ft.py.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_ethno import OCM, load_code_names
from prepare_ethno_tree import SEC, parse_file


def build_examples(paragraphs, names, sec_frac, rng):
    """Yield (prompt_text_or_None, text, codes) with class assignment."""
    groups = defaultdict(list)
    for path, text, codes in paragraphs:
        if codes:
            groups[path].append((text, codes))
    for path, items in groups.items():
        sec = " / ".join(path) if path else ""
        for k, (text, codes) in enumerate(items):
            if sec and rng.random() < sec_frac and len(items) > 1:
                sib = set()
                for j, (_, cs) in enumerate(items):
                    if j != k:
                        sib.update(cs)
                named = " ".join(f"{c} {names.get(c, c)}"
                                 for c in sorted(sib, key=lambda c: (len(c), c)))
                prefix = (f"{SEC}{sec}\n{OCM}{named}\n\n" if named
                          else f"{SEC}{sec}\n\n")
            else:
                prefix = ""
            tag = OCM + " ".join(f"{c} {names.get(c, c)}" for c in codes)
            yield prefix, text, tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=str(Path.home() / "et43texts/ethnotext_v430"))
    ap.add_argument("--out", default="data/ethnographic_v5/ft")
    ap.add_argument("--tokenizer", default="data/ethnographic_v5/tokenizer.json")
    ap.add_argument("--ocm-labels", default="data/ethnographic/ocmdefs.txt")
    ap.add_argument("--val-docs", default="data/ethnographic_v5/val_docs.jsonl",
                    help="excluded (val) societies derived from its hdoc prefixes")
    ap.add_argument("--target", type=int, default=800_000, help="tagged examples")
    ap.add_argument("--sec-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=0, help="smoke: only N files")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    names = load_code_names(Path(args.ocm_labels))
    val_socs = {json.loads(l)["hdoc"].split("-")[0]
                for l in open(args.val_docs, encoding="utf-8") if l.strip()}
    print(f"excluding {len(val_socs)} val societies: {sorted(val_socs)}")

    files = sorted(Path(args.tree).glob("*/*/*.txt"))
    rng = random.Random(args.seed)
    per_area = defaultdict(list)
    for fp in files:
        if fp.parent.name in val_socs:
            continue
        per_area[fp.parent.parent.name].append(fp)
    if args.limit_files:
        per_area = {a: fps[:args.limit_files] for a, fps in per_area.items()}

    # stratify the target across areas by their tagged-paragraph share
    ex_by_area = {}
    for area, fps in sorted(per_area.items()):
        examples = []
        for fp in fps:
            doc = parse_file(fp)
            if doc is None:
                continue
            examples.extend(build_examples(doc[2], names, args.sec_frac, rng))
        ex_by_area[area] = examples
        print(f"{area}: {len(examples):,} examples from {len(fps)} files", flush=True)

    total = sum(len(v) for v in ex_by_area.values())
    quota = {a: max(1, round(args.target * len(v) / total))
             for a, v in ex_by_area.items()}
    picked = []
    for area, examples in ex_by_area.items():
        rng.shuffle(examples)
        picked.extend(examples[:quota[area]])
    rng.shuffle(picked)
    picked = picked[:args.target]

    ocm_id = tok.token_to_id(OCM)
    toks, lens, plens = [], [], []
    cls = Counter()
    n_trunc = n_skip = 0
    for prefix, text, tag in picked:
        cls["sec-loo" if prefix else "cold"] += 1
        # assemble exactly like tag_eval: prefix ids, text ids, marker,
        # target — so train and eval prompts tokenize identically
        pre_ids = tok.encode(prefix, add_special_tokens=False).ids
        txt_ids = tok.encode(text + "\n", add_special_tokens=False).ids
        tgt_ids = tok.encode(tag + "\n\n", add_special_tokens=False).ids
        budget = 1024 - len(pre_ids) - 1 - len(tgt_ids)
        if budget < 32:
            n_skip += 1
            continue
        if len(txt_ids) > budget:
            txt_ids = txt_ids[:budget]  # head-truncate, as tag_eval does
            n_trunc += 1
        ids = pre_ids + txt_ids + [ocm_id] + tgt_ids
        toks.extend(ids)
        lens.append(len(ids))
        plens.append(len(ids) - len(tgt_ids))  # loss starts at the codes

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    np.array(toks, dtype=np.uint16).tofile(out / "ft.bin")
    offsets.tofile(out / "ft_offsets.npy")
    np.array(plens, dtype=np.int32).tofile(out / "ft_plens.npy")
    print(f"wrote {out}: {len(lens):,} examples {dict(cls)}, "
          f"{len(toks):,} tokens ({n_trunc:,} head-truncated, {n_skip} skipped)")


if __name__ == "__main__":
    main()
