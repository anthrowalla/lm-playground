"""Qualitative OCM-tagging test against llama-server /completion.

Feeds a paragraph (with trailing newline, as in corpus v4) to the model and
greedily decodes what follows — with `special: True` so the model may emit
<|ocm|>CODE NAME ... tags. Token-level pieces are printed to see exactly how
the tag is assembled. First v4 result: the model reproduces codes *and*
labels after unseen paragraphs, confirming the named-tag training worked.

Usage:
    .venv/bin/python scripts/tag_test.py "paragraph text" [--n-predict 24]
        [--url http://localhost:8082/completion]
        [--tokenizer data/ethnographic_v4/tokenizer.json]
        [--stream] [--live]
"""

import argparse
import json

import requests
from tokenizers import Tokenizer


def tag(paragraph, tok, url, n_predict=24):
    resp = requests.post(url, json={
        "prompt": tok.encode(paragraph + "\n").ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "special": True,
    })
    r = resp.json()
    if "content" not in r:
        print(resp.status_code, r)
        raise SystemExit("server returned an error — see above")
    content = r["content"]
    pieces = [tok.decode([t], skip_special_tokens=False)
              for t in tok.encode(content).ids]
    return content, pieces


def tag_stream(paragraph, tok, url, n_predict=24, live=False):
    resp = requests.post(url, json={
        "prompt": tok.encode(paragraph + "\n").ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "special": True,
        "stream": True,
    }, stream=True)
    pieces, timings = [], {}
    for line in resp.iter_lines():
        if not line:
            continue
        data = line.decode("utf-8")
        if data == "data: [DONE]":
            break
        chunk = json.loads(data.removeprefix("data: "))
        if "content" in chunk:
            pieces.append(chunk["content"])
            if live:
                print(chunk["content"], end="", flush=True)
        if chunk.get("stop"):
            timings = chunk.get("timings", {})
    if live:
        print()
    return "".join(pieces), timings


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paragraph")
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--url", default="http://localhost:8082/completion")
    ap.add_argument("--tokenizer",
                    default="data/ethnographic_v4/tokenizer.json")
    ap.add_argument("--stream", action="store_true",
                    help="use SSE streaming endpoint handling")
    ap.add_argument("--live", action="store_true",
                    help="print tokens as they arrive, report tok/s")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    if args.stream:
        content, timings = tag_stream(args.paragraph, tok, args.url,
                                      args.n_predict, live=args.live)
        if not args.live:
            print("TAGGED :", repr(content))
        if timings:
            print(f"{timings['predicted_per_second']:.1f} tok/s")
    else:
        content, pieces = tag(args.paragraph, tok, args.url, args.n_predict)
        print("PROMPT :", args.paragraph[:90], "...")
        print("TAGGED :", repr(content))
        print("PIECES :", pieces)


if __name__ == "__main__":
    main()
