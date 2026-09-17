"""Build tokenized section-title examples for the summarization-adjacent FT.

Task (run 1 of sumtitle_plan.md): prompt = a section's paragraphs (head-
packed to the token budget) + "\n\n" + <|sec|>, target = the section's full
title path + "\n\n" (loss masked to the target). This mirrors the corpus's
own continuation distribution ("...text\n\n<|sec|>TITLE PATH\n...") and the
tagging FT's forced-marker mechanics. Sources the v5 tree directly; val
societies are excluded.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_ethno_tree import SEC, parse_file

MARKER = "\n\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=str(Path.home() / "et43texts/ethnotext_v430"))
    ap.add_argument("--out", default="data/ethnographic_v5/ft_titles")
    ap.add_argument("--tokenizer", default="data/ethnographic_v5/tokenizer.json")
    ap.add_argument("--val-docs", default="data/ethnographic_v5/val_docs.jsonl")
    ap.add_argument("--target", type=int, default=200_000, help="examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=0, help="smoke: only N files")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    val_socs = {json.loads(l)["hdoc"].split("-")[0]
                for l in open(args.val_docs, encoding="utf-8") if l.strip()}
    print(f"excluding {len(val_socs)} val societies: {sorted(val_socs)}")

    sec_id = tok.token_to_id(SEC)
    tail_ids = tok.encode(MARKER, add_special_tokens=False).ids

    files = sorted(Path(args.tree).glob("*/*/*.txt"))
    rng = random.Random(args.seed)
    per_area = defaultdict(list)
    for fp in files:
        if fp.parent.name in val_socs:
            continue
        per_area[fp.parent.parent.name].append(fp)
    if args.limit_files:
        per_area = {a: fps[:args.limit_files] for a, fps in per_area.items()}

    # (text, path) per area: one example per section (same-path sections
    # within a doc merge)
    ex_by_area = {}
    for area, fps in sorted(per_area.items()):
        examples = []
        for fp in fps:
            doc = parse_file(fp)
            if doc is None:
                continue
            groups = defaultdict(list)
            for path, text, _ in doc[2]:
                if path:
                    groups[tuple(path)].append(text)
            for path, texts in groups.items():
                examples.append(("\n\n".join(texts), " / ".join(path)))
        ex_by_area[area] = examples
        print(f"{area}: {len(examples):,} titled sections from {len(fps)} files",
              flush=True)

    total = sum(len(v) for v in ex_by_area.values())
    quota = {a: max(1, round(args.target * len(v) / total))
             for a, v in ex_by_area.items()}
    picked = []
    for area, examples in ex_by_area.items():
        rng.shuffle(examples)
        picked.extend(examples[:quota[area]])
    rng.shuffle(picked)
    picked = picked[:args.target]

    toks, lens, plens = [], [], []
    n_skip = 0
    n_over = 0
    for text, path in picked:
        txt_ids = tok.encode(text, add_special_tokens=False).ids
        tgt_ids = tok.encode(path + MARKER, add_special_tokens=False).ids
        # total must fit seq_len: text + "\n\n" + <|sec|> + target
        budget = 1024 - len(tail_ids) - 1 - len(tgt_ids)
        if len(txt_ids) < 32 or budget < 32:
            n_skip += 1
            continue
        if len(txt_ids) > budget:
            txt_ids = txt_ids[:budget]
            n_over += 1
        prompt_ids = txt_ids + tail_ids + [sec_id]
        ids = prompt_ids + tgt_ids
        toks.extend(ids)
        lens.append(len(ids))
        plens.append(len(prompt_ids))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    np.array(toks, dtype=np.uint16).tofile(out / "ft.bin")
    offsets.tofile(out / "ft_offsets.npy")
    np.array(plens, dtype=np.int32).tofile(out / "ft_plens.npy")
    assert max(lens) <= 1024, f"example length {max(lens)} exceeds seq_len"
    print(f"wrote {out}: {len(lens):,} examples, {len(toks):,} tokens "
          f"({n_over:,} head-truncated, {n_skip} skipped)")


if __name__ == "__main__":
    main()
