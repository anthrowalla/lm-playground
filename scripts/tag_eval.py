"""OCM-tagging eval: score model-emitted tags against gold on held-out val paragraphs.

Reads the gold JSONL produced by scripts/val_docs.py, prompts the model with
each paragraph text (cold-start, no document context — the analyst-assist
deployment scenario), greedily decodes the tag block, and scores emitted codes
against gold with micro precision/recall/F1 plus exact-set-match. Output rows
append to a JSONL file as they complete, so a run can be resumed after an
interruption (already-present indexes are skipped).

The same script serves v4 and v3 models:
  v4: --url http://localhost:8082/completion --tokenizer tokenizers/ethnographic_v4/tokenizer.json
  v3: --url http://localhost:8080/completion --tokenizer data/ethnographic_v3/tokenizer.json
(v3 was trained on bare codes; the parser handles both bare and named form.
Note v3's own val window differed, so it likely saw these paragraphs during
training — contamination in v3's favor.)

Usage:
    .venv/bin/python scripts/tag_eval.py --limit 25          # smoke
    .venv/bin/python scripts/tag_eval.py                     # full run
    .venv/bin/python scripts/tag_eval.py --summary           # score existing output
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tokenizers import Tokenizer


def parse_pairs(tag: str) -> list[list[str]]:
    """'222 COLLECTING 224 HUNTING' or bare '222 224' -> [[222, COLLECTING], ...]"""
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
    return pairs


def evaluate_one(tok: Tokenizer, url: str, idx: int, rec: dict,
                 n_predict: int, max_prompt: int,
                 valid_codes: set[str] | None, force_marker: bool,
                 temperature: float) -> dict:
    prompt = rec["text"] + "\n"
    if force_marker:
        # append the tag marker so the model only has to pick codes —
        # separates code selection from the free-decode decision to tag
        prompt += "<|ocm|>"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    trunc = len(ids) > max_prompt
    if trunc:
        ids = ids[:max_prompt]
    t0 = time.time()
    r = requests.post(url, json={
        "prompt": ids,
        "n_predict": n_predict,
        "temperature": temperature,
        "special": True,
        "stop": ["\n\n"],
    }, timeout=600)
    elapsed = time.time() - t0
    r.raise_for_status()
    content = r.json().get("content", "")
    pairs = parse_pairs(content)
    codes = [c for c, _ in pairs]
    if valid_codes:
        codes = [c for c in codes if c in valid_codes]
    return {
        "idx": idx,
        "title": rec.get("fields", {}).get("Title", "?"),
        "gold": [c for c, _ in rec["tags"]],
        "pred": list(dict.fromkeys(codes)),
        "trunc": trunc,
        "raw": content,
        "n_prompt": len(ids),
        "ms": round(elapsed * 1000),
    }


def score(rows: list[dict]) -> dict:
    tp = fp = fn = exact = 0
    by_n = Counter()
    for r in rows:
        if "error" in r:
            continue
        g, p = set(r["gold"]), set(r["pred"])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += g == p
        by_n[min(len(g), 5)] += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"n": len(rows), "micro_P": p, "micro_R": r, "micro_F1": f,
            "exact_match": exact / len(rows) if rows else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "gold_size_hist": dict(sorted(by_n.items()))}


def print_summary(label: str, rows: list[dict]) -> None:
    s = score(rows)
    n_err = sum("error" in r for r in rows)
    print(f"== {label}: {s['n']:,} paragraphs ({n_err} errors) ==")
    print(f"micro P {s['micro_P']:.3f}  R {s['micro_R']:.3f}  "
          f"F1 {s['micro_F1']:.3f}  exact {s['exact_match']:.3f}")
    print(f"tp {s['tp']:,}  fp {s['fp']:,}  fn {s['fn']:,}  "
          f"gold sizes {s['gold_size_hist']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/ethnographic_v4/val_docs.jsonl")
    ap.add_argument("--url", default="http://localhost:8082/completion")
    ap.add_argument("--tokenizer",
                    default="tokenizers/ethnographic_v4/tokenizer.json")
    ap.add_argument("--out", default="results/tag_eval_v4.jsonl")
    ap.add_argument("--label", default="v4")
    ap.add_argument("--limit", type=int, default=0, help="only first N (0=all)")
    ap.add_argument("--skip", type=int, default=0, help="skip first N tagged")
    ap.add_argument("--max-prompt", type=int, default=900,
                    help="head-truncate prompts to fit the 1024 context")
    ap.add_argument("--ocm-labels", default="data/ethnographic/ocmdefs.txt",
                    help="authoritative code list; predictions outside it are dropped")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel requests (llama-server slots)")
    ap.add_argument("--summary", action="store_true",
                    help="score an existing output file and exit")
    ap.add_argument("--force-marker", action="store_true",
                    help="append <|ocm|> to the prompt (score code selection)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; small sampling can lengthen code lists")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.summary:
        rows = [json.loads(l) for l in out.open() if l.strip()]
        print_summary(args.label, rows)
        return

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    # flatten documents to tagged paragraphs, carrying doc fields along
    items = [{"fields": d.get("fields", {}), **p}
             for d in recs for p in d["paragraphs"] if p["tags"]]
    tagged = list(enumerate(items))
    done = set()
    if out.exists():
        for l in out.open():
            if l.strip():
                done.add(json.loads(l).get("idx"))
    todo = [(i, r) for i, r in tagged if i not in done]
    if args.skip:
        todo = todo[args.skip:]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done):,} already done, {len(todo):,} to go "
          f"({len(tagged):,} tagged total)")

    tok = Tokenizer.from_file(args.tokenizer)
    valid = None
    defs = Path(args.ocm_labels)
    if defs.exists():
        import re
        valid = set(re.findall(r"(?m)^(\d{3,4})\s", defs.read_text(encoding="utf-8")))
        print(f"{len(valid)} valid OCM codes loaded from {defs}")
    started = time.time()
    n_done = 0
    with out.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        def work(item):
            i, rec = item
            for attempt in range(3):
                try:
                    return evaluate_one(tok, args.url, i, rec,
                                        args.n_predict, args.max_prompt,
                                        valid, args.force_marker,
                                        args.temperature)
                except Exception as e:  # noqa: BLE001 - keep the run alive
                    if attempt == 2:
                        return {"idx": i, "error": str(e)}
                    time.sleep(2 * (attempt + 1))
        for res in ex.map(work, todo):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 100 == 0:
                rate = n_done / (time.time() - started)
                eta = (len(todo) - n_done) / rate / 60
                print(f"{n_done:,}/{len(todo):,}  {rate:.1f}/s  "
                      f"ETA {eta:.0f} min", flush=True)

    rows = [json.loads(l) for l in out.open() if l.strip()]
    print_summary(args.label, rows)


if __name__ == "__main__":
    main()
