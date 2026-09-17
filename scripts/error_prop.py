"""Error-propagation probe for sec-loo tagging evals.

The sec-loo eval feeds GOLD sibling unions; in deployment the sibling codes
come from the model itself. This script rebuilds the sec-loo prefixes with
unions over siblings' PREDICTED codes taken from a prior tag_eval run
(--pred-rows, same enumeration: tagged paragraphs in val_docs order) and
scores the result. The F1 delta vs the gold-union sec-loo number measures
how much of the section prior survives feeding the model its own errors.

Usage:
    .venv/bin/python scripts/error_prop.py --url http://localhost:8086/completion \
        --tokenizer checkpoints/small_ethnographic_v5_tag/tokenizer.json \
        --data data/ethnographic_v5/val_docs.jsonl \
        --pred-rows results/tag_eval_smallv5tag_sec_loo.jsonl \
        --out results/tag_eval_smallv5tag_sec_loo_sibpred.jsonl
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_ethno import OCM, load_code_names
from prepare_ethno_tree import SEC
from tag_eval import evaluate_one, print_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/ethnographic_v5/val_docs.jsonl")
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pred-rows", required=True,
                    help="tag_eval output whose pred codes become the unions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ocm-labels", default="data/ethnographic/ocmdefs.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-prompt", type=int, default=900)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    names = load_code_names(Path(args.ocm_labels))

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    items = []
    for di, d in enumerate(recs):
        for p in d["paragraphs"]:
            if p["tags"]:
                items.append({"doc": di, **p})
    preds = [json.loads(l) for l in open(args.pred_rows) if l.strip()]
    assert len(preds) == len(items), \
        f"pred rows {len(preds)} != tagged paragraphs {len(items)}"

    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[(it["doc"], it.get("section", ""))].append(i)
    prefixes = [""] * len(items)
    n_ctx = 0
    for i, it in enumerate(items):
        sec = it.get("section", "")
        if not sec:
            continue
        sib = {c for j in groups[(it["doc"], sec)] if j != i
               for c in preds[j]["pred"]}
        named = " ".join(f"{c} {names[c]}" if c in names else c
                         for c in sorted(sib, key=lambda c: (len(c), c)))
        prefixes[i] = (f"{SEC}{sec}\n{OCM}{named}\n\n" if named
                       else f"{SEC}{sec}\n\n")
        n_ctx += 1
    print(f"sibling-PRED unions: {n_ctx:,}/{len(items):,} paragraphs get a prefix")

    tagged = list(enumerate(items))
    if args.limit:
        tagged = tagged[:args.limit]
    done = set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for l in out.open():
            if l.strip():
                done.add(json.loads(l).get("idx"))
    todo = [(i, r) for i, r in tagged if i not in done]
    print(f"0 already done, {len(todo):,} to go ({len(tagged):,} tagged total)",
          flush=True)
    if not todo:
        print_summary("sibpred", [json.loads(l) for l in out.open() if l.strip()])
        return

    marker_id = tok.token_to_id(OCM)
    valid = set(re.findall(r"(?m)^(\d{3,4})\s",
                           Path(args.ocm_labels).read_text(encoding="utf-8")))
    print(f"{len(valid)} valid OCM codes loaded from {args.ocm_labels}")
    t0 = time.time()
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(evaluate_one, tok, args.url, i, rec,
                          args.n_predict, args.max_prompt, valid, True, 0.0,
                          prefixes[i], marker_id)
                for i, rec in todo]
        for fut in futs:
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if n % 2000 == 0:
                rate = n / (time.time() - t0)
                print(f"{n:,}/{len(todo):,}  {rate:.1f}/s  "
                      f"ETA {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    print_summary("sibpred", [json.loads(l) for l in out.open() if l.strip()])


if __name__ == "__main__":
    main()
