"""Build the Track-B DAPT corpus: v5 domain stream + FineWeb-Edu replay slice.

Re-encodes data/ethnographic_v5/corpus_train.txt / corpus_val.txt with the
extended SmolLM2 tokenizer (tokenizers/smollm2_360m_ethno). The v5 streams'
literal <|bos|> doc markers are remapped to <|endoftext|> (SmolLM2 id 0);
everything else in the streams is used verbatim.

    .venv/bin/python scripts/prepare_dapt.py \
        --domain data/ethnographic_v5 --general data/general --out data/adapt_b
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_mix import BinWriter, GEN_BYTES_PER_TOKEN, collect_general

from transformers import PreTrainedTokenizerFast

BOS5 = "<|bos|>"
EOT = "<|endoftext|>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", default="data/ethnographic_v5")
    ap.add_argument("--general", default="data/general")
    ap.add_argument("--tokenizer", default="tokenizers/smollm2_360m_ethno")
    ap.add_argument("--out", default="data/adapt_b")
    ap.add_argument("--replay-tokens", type=float, default=1.5e8)
    ap.add_argument("--val-replay-tokens", type=float, default=5e6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 10 ** 9

    replay_chars = int(args.replay_tokens * GEN_BYTES_PER_TOKEN)
    val_replay_chars = int(args.val_replay_tokens * GEN_BYTES_PER_TOKEN)
    print(f"pass 1: collecting FineWeb-Edu replay (~{args.replay_tokens/1e6:.0f}M tok)")
    gen_train, gen_val = collect_general(Path(args.general), replay_chars,
                                         val_replay_chars, rng)
    print(f"collected: replay train {len(gen_train):,} docs, val {len(gen_val):,} docs")

    stats = Counter()
    tw = BinWriter(out / "train.bin")
    domain_docs = [EOT + p for p in
                   (Path(args.domain) / "corpus_train.txt").read_text(encoding="utf-8").split(BOS5)
                   if p.strip()]
    B = 256
    for i in range(0, len(domain_docs), B):
        stats["domain"] += tw.add_texts(tokenizer, domain_docs[i:i + B])
        if (i // B) % 400 == 0:
            print(f"  train.bin: {tw.total/1e6:.0f}M tokens (domain doc {i:,}/{len(domain_docs):,})")
    stats["replay"] = tw.add_texts(tokenizer, gen_train)
    n_train = tw.close()

    vw = BinWriter(out / "val.bin")
    val_domain = (Path(args.domain) / "corpus_val.txt").read_text(encoding="utf-8")
    val_domain = EOT + val_domain.replace(BOS5, EOT)
    val_texts = [val_domain]
    stats["val_domain"] = vw.add_texts(tokenizer, [val_domain])
    val_texts.extend(gen_val)
    stats["val_replay"] = vw.add_texts(tokenizer, gen_val)
    n_val = vw.close()

    ids = []
    for seq in tokenizer(val_texts[:400]).input_ids:
        ids.extend(seq)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    ok = decoded == "".join(val_texts[:400])
    print(f"val round-trip check: {'PASS' if ok else 'FAIL'}")
    assert ok

    meta = {
        "vocab_size": 49154,
        "pad_id": 0, "bos_id": 0, "eos_id": 0, "unk_id": 0,
        "ocm_id": 49152, "sec_id": 49153,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    report = {
        "train_tokens": n_train,
        "val_tokens": n_val,
        "streams": dict(stats),
        "stream_share": {k: round(stats[k] / (stats["domain"] + stats["replay"]), 4)
                         for k in ("domain", "replay")},
        "replay_docs_train": len(gen_train),
        "replay_docs_val": len(gen_val),
        "tokenizer": args.tokenizer,
    }
    (out / "dapt_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
