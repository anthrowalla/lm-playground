"""Emit the og11 (Eastern Toraja) slice — v4's held-out val — as eval JSONL
with section paths, parsed from the corpus tree with the v5 (p-only,
markup-stripped) conventions.

Output paragraphs carry text, gold tags [[code, name], ...] and the section
title path, so tag_eval.py can run section-context variants against v4 on
the one slice it never trained on.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_ethno import load_code_names
from prepare_ethno_tree import parse_file


def main() -> None:
    tree = Path.home() / "et43texts/ethnotext_v430/Asia/og11"
    names = load_code_names(Path("data/ethnographic/ocmdefs.txt"))
    out = Path("data/ethnographic_v5/og11_slice.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_para = 0
    with out.open("w", encoding="utf-8") as f:
        for fp in sorted(tree.glob("*.txt")):
            doc = parse_file(fp)
            if doc is None:
                continue
            hdoc, fields, paragraphs = doc
            f.write(json.dumps({
                "hdoc": hdoc,
                "fields": fields,
                "paragraphs": [
                    {"text": t,
                     "tags": [[c, names.get(c, "") if names else ""] for c in codes],
                     "section": " / ".join(path)}
                    for path, t, codes in paragraphs
                ],
            }, ensure_ascii=False) + "\n")
            n_para += len(paragraphs)
    print(f"wrote {out}: {n_para:,} paragraphs")


if __name__ == "__main__":
    main()
