"""Section-title eval: score generated title paths against gold on val docs.

For each unique (doc, section) in the val JSONL, prompts the model with the
section's paragraphs (head-truncated) + "\n\n" + forced <|sec|> marker,
greedily decodes up to the stop string, and scores word-bag P/R/F1 against
the gold full title path (exact match on normalized words too). Deeply
ambiguous task — F1, not exact match, is the headline.

Usage:
    .venv/bin/python scripts/title_eval.py --url http://localhost:8087/completion \
        --tokenizer checkpoints/small_ethnographic_v5_title/tokenizer.json \
        --out results/title_eval_smallv5title.jsonl
    .venv/bin/python scripts/title_eval.py --summary --out results/...
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_ethno_tree import SEC


def norm_words(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def leaf(s):
    return s.split(" / ")[-1]


def score_rows(rows):
    tp = fp = fn = exact = 0
    for r in rows:
        if "error" in r:
            continue
        g, p = set(norm_words(leaf(r["gold"]))), set(norm_words(leaf(r["pred"])))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += norm_words(leaf(r["pred"])) == norm_words(leaf(r["gold"]))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, exact / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/ethnographic_v5/val_docs.jsonl")
    ap.add_argument("--url", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-text", type=int, default=800,
                    help="head-truncate the section text to this many tokens")
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary:
        rows = [json.loads(l) for l in out.open() if l.strip()]
        prec, rec, f1, exact = score_rows(rows)
        print(f"== titles: {len(rows):,} sections ==")
        print(f"word-bag P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}  exact {exact:.3f}")
        return

    tok = Tokenizer.from_file(args.tokenizer)
    sec_id = tok.token_to_id(SEC)
    tail_ids = tok.encode("\n\n", add_special_tokens=False).ids

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    items = []
    for di, d in enumerate(recs):
        groups = defaultdict(list)
        for p in d["paragraphs"]:
            if p.get("section"):
                groups[p["section"]].append(p["text"])
        for sec, texts in groups.items():
            items.append({"doc": di, "hdoc": d["hdoc"], "section": sec,
                          "text": "\n\n".join(texts)})
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} val sections to score")

    done = set()
    if out.exists():
        for l in out.open():
            if l.strip():
                done.add(json.loads(l).get("idx"))
    todo = [(i, it) for i, it in enumerate(items) if i not in done]
    print(f"0 already done, {len(todo):,} to go", flush=True)

    def work(k):
        i, it = todo[k]
        txt_ids = tok.encode(it["text"], add_special_tokens=False).ids
        prompt_ids = txt_ids[:args.max_text] + tail_ids + [sec_id]
        t0 = time.time()
        r = requests.post(args.url, json={
            "prompt": prompt_ids,
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "special": True,
            "stop": ["\n\n"],
        }, timeout=600)
        r.raise_for_status()
        content = r.json().get("content", "")
        return {"idx": i, "hdoc": it["hdoc"], "gold": it["section"],
                "pred": content.split("\n\n")[0], "trunc": len(txt_ids) > args.max_text,
                "ms": round((time.time() - t0) * 1000)}

    t0 = time.time()
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(work, range(len(todo))):
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if n % 2000 == 0:
                rate = n / (time.time() - t0)
                print(f"{n:,}/{len(todo):,}  {rate:.1f}/s  "
                      f"ETA {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    prec, rec, f1, exact = score_rows(
        [json.loads(l) for l in out.open() if l.strip()])
    print(f"word-bag P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}  exact {exact:.3f}")


if __name__ == "__main__":
    main()
