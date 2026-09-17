"""Build culture-summary pairing examples (run 2 of sumtitle_plan.md).

For each society with an `hdocid -000` culture-summary doc, pair every
usable summary paragraph (real prose, no `000` code) with the best-matching
section from that society's OTHER documents, scored by OCM-code Jaccard.
Prompt = plain-text task frame (no trigger collisions in the corpus):

    Source ({society}):
    {section text}

    Summary:
    {culture-summary paragraph}

Loss is masked to the target paragraph. Val societies are excluded.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_ethno_tree import parse_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=str(Path.home() / "et43texts/ethnotext_v430"))
    ap.add_argument("--out", default="data/ethnographic_v5/ft_sumpairs")
    ap.add_argument("--tokenizer", default="data/ethnographic_v5/tokenizer.json")
    ap.add_argument("--val-docs", default="data/ethnographic_v5/val_docs.jsonl")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-target-toks", type=int, default=280)
    ap.add_argument("--min-jaccard", type=float, default=0.05)
    ap.add_argument("--sources-per-para", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-societies", type=int, default=0, help="smoke")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    val_socs = {json.loads(l)["hdoc"].split("-")[0]
                for l in open(args.val_docs, encoding="utf-8") if l.strip()}
    print(f"excluding {len(val_socs)} val societies: {sorted(val_socs)}")

    tree = Path(args.tree)
    sum_docs = sorted(tree.glob("*/*/*-000.txt"))
    socs = []
    for fp in sum_docs:
        soc = fp.parent.name
        if soc not in val_socs:
            socs.append(fp)
    if args.limit_societies:
        socs = socs[:args.limit_societies]
    print(f"{len(socs)} culture-summary docs (societies)")
    budget_max = 1024

    n_pairs = n_nosrc = n_short = n_long = 0
    toks, lens, plens = [], [], []
    for fp in socs:
        soc = fp.parent.name
        doc = parse_file(fp)
        if doc is None:
            continue
        # usable summary paragraphs: real prose, at least one non-000 code
        sums = []
        for path, text, codes in doc[2]:
            cs = {c for c in codes if c != "000"}
            if cs and len(text.split()) >= args.min_words:
                sums.append((cs, text))
        if not sums:
            continue
        # source sections: other docs of the same society
        secs = []
        for sfp in sorted(fp.parent.glob("*.txt")):
            if sfp.name == fp.name:
                continue
            sdoc = parse_file(sfp)
            if sdoc is None:
                continue
            groups = defaultdict(list)
            for path, text, codes in sdoc[2]:
                if path:
                    groups[tuple(path)].append((text, codes))
            for path, items in groups.items():
                cs = {c for _, codes in items for c in codes if c != "000"}
                text = "\n\n".join(t for t, _ in items)
                if cs and len(text.split()) >= 32:
                    secs.append((cs, text))
        if not secs:
            n_nosrc += len(sums)
            continue
        for scodes, stext in sums:
            ranked = sorted(
                ((len(scodes & cs) / len(scodes | cs), text) for cs, text in secs),
                key=lambda t: -t[0])
            ranked = [(j, text) for j, text in ranked[:args.sources_per_para]
                      if j >= args.min_jaccard]
            if not ranked:
                n_nosrc += 1
                continue
            t_ids = tok.encode(stext, add_special_tokens=False).ids
            if len(t_ids) > args.max_target_toks:
                n_long += 1
                continue
            culture = doc[1].get("culture", soc)
            pre_ids = tok.encode(f"Source ({culture}):\n",
                                 add_special_tokens=False).ids
            post_ids = tok.encode("\n\nSummary:\n", add_special_tokens=False).ids
            nl = len(tok.encode("\n\n", add_special_tokens=False).ids)
            for j, src in ranked:
                s_ids = tok.encode(src, add_special_tokens=False).ids
                if len(s_ids) > budget_max - len(pre_ids) - len(post_ids) - len(t_ids):
                    n_short += 1
                    continue
                ids = pre_ids + s_ids + post_ids + t_ids + \
                    tok.encode("\n\n", add_special_tokens=False).ids
                toks.extend(ids)
                lens.append(len(ids))
                plens.append(len(pre_ids) + len(s_ids) + len(post_ids))
                n_pairs += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    np.array(toks, dtype=np.uint16).tofile(out / "ft.bin")
    offsets.tofile(out / "ft_offsets.npy")
    np.array(plens, dtype=np.int32).tofile(out / "ft_plens.npy")
    print(f"wrote {out}: {n_pairs:,} pairs ({n_nosrc} no-source, "
          f"{n_long} target too long, {n_short} too short), {len(toks):,} tokens")


if __name__ == "__main__":
    main()
