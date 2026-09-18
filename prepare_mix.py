"""Build mixed corpus v6: general web (FineWeb-Edu) + Gutenberg classics + ethnographic domain.

Everything is re-encoded with a fresh 32k BPE trained on a sample of the mix.
Streams:

- general: FineWeb-Edu sample/10BT rows (fetched by scripts/fetch_general.py),
  subsampled to a char budget, shuffled per batch, ~2.5% diverted to val;
- classics: Gutenberg texts, repeated --classics-repeat times in train for
  exposure (they are a small fraction of the mix by design);
- domain: data/ethnographic_v5/corpus_train.txt verbatim (codebook x3 + AA
  journal + tagged ethnographic paragraphs; the 12 val societies are already
  excluded) and corpus_val.txt on the val side.

Usage:
    .venv/bin/python prepare_mix.py --domain data/ethnographic_v5 \
        --general data/general --out data/mixed_v6 --train-tokens 2.10e9
Smoke:
    .venv/bin/python prepare_mix.py --domain data/ethnographic_v5 \
        --general data/general --out data/mixed_v6_smoke --smoke
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from prepare import BOS, EOS, PAD, UNK, SPECIALS
from prepare_ethno import OCM
from prepare_ethno_tree import SEC

DOC_SPLIT = "<|bos|>"
VAL_RATE = 0.025
# rough chars-per-token for budgeting; exact counts are measured at encode time
GEN_BYTES_PER_TOKEN = 4.2
CLASSIC_BYTES_PER_TOKEN = 4.4


def encode_texts(tokenizer, texts: list[str]) -> tuple[int, list]:
    """Batch-encode docs -> (token count, flat uint16 bytes per doc list)."""
    chunks = tokenizer(texts).input_ids
    total = 0
    blobs = []
    for ids in chunks:
        blobs.append(np.asarray(ids, dtype=np.uint16).tobytes())
        total += len(ids)
    return total, blobs


class BinWriter:
    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.total = 0

    def add_blobs(self, blobs) -> None:
        for b in blobs:
            self.f.write(b)
        # total updated by caller (needs count)

    def add_texts(self, tokenizer, texts: list[str]) -> int:
        total = 0
        CH = 2048
        for i in range(0, len(texts), CH):
            n, blobs = encode_texts(tokenizer, texts[i:i + CH])
            self.add_blobs(blobs)
            total += n
        self.total += total
        return total

    def close(self) -> int:
        self.f.close()
        return self.total


def collect_general(general_dir: Path, train_char_budget: int, val_char_budget: int,
                    rng: random.Random) -> tuple[list[str], list[str]]:
    """Subsample FineWeb-Edu rows into (train, val) text lists."""
    shards = sorted(general_dir.glob("fineweb/**/*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {general_dir}/fineweb")
    train, val = [], []
    n_train_chars = n_val_chars = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            texts = batch.column("text").to_pylist()
            rng.shuffle(texts)
            for t in texts:
                n = len(t)
                if n < 200 or n > 100_000:
                    continue
                if rng.random() < VAL_RATE and n_val_chars < val_char_budget:
                    val.append(t)
                    n_val_chars += n
                elif n_train_chars < train_char_budget:
                    train.append(t)
                    n_train_chars += n
            if n_train_chars >= train_char_budget and n_val_chars >= val_char_budget:
                print(f"  general: budget hit at {shard.name}")
                return train, val
        print(f"  general: {shard.name} done "
              f"(train {n_train_chars/1e9:.2f}GB, val {n_val_chars/1e6:.0f}MB)")
    return train, val


def load_classics(general_dir: Path) -> list[str]:
    fps = sorted((general_dir / "classics").glob("*.txt"))
    texts = [fp.read_text(encoding="utf-8") for fp in fps]
    return texts


def split_domain_docs(path: Path) -> list[str]:
    """Split a v5 corpus file into documents on <|bos|> (keeping the marker)."""
    text = path.read_text(encoding="utf-8")
    parts = [p for p in text.split(DOC_SPLIT) if p.strip()]
    return [DOC_SPLIT + p for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="data/ethnographic_v5")
    parser.add_argument("--general", default="data/general")
    parser.add_argument("--out", default="data/mixed_v6")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--train-tokens", type=float, default=2.10e9)
    parser.add_argument("--val-general-tokens", type=float, default=25e6)
    parser.add_argument("--classics-repeat", type=int, default=6)
    parser.add_argument("--tokenizer-out", default="tokenizers/mixed_v6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    domain = Path(args.domain)
    general = Path(args.general)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_dir = Path(args.tokenizer_out)
    tok_dir.mkdir(parents=True, exist_ok=True)

    train_char_budget = 50e6 if args.smoke else int(
        (args.train_tokens - 0) * GEN_BYTES_PER_TOKEN * 0.79)
    val_char_budget = 2e6 if args.smoke else int(args.val_general_tokens * GEN_BYTES_PER_TOKEN)
    if not args.smoke:
        # reserve room for classics x repeat + domain by shrinking general to
        # the remainder of the token target (domain tokens are measured below)
        domain_est = (domain / "train.bin").stat().st_size // 2
        classics_chars = sum(p.stat().st_size for p in (general / "classics").glob("*.txt"))
        classics_est = int(classics_chars / CLASSIC_BYTES_PER_TOKEN) * args.classics_repeat
        train_char_budget = int((args.train_tokens - domain_est - classics_est)
                                * GEN_BYTES_PER_TOKEN)
        print(f"budgets: general train {train_char_budget/1e9:.2f}GB chars "
              f"(domain ~{domain_est/1e6:.0f}M tok, classics ~{classics_est/1e6:.0f}M tok x{args.classics_repeat})")

    print("pass 1: collecting FineWeb-Edu rows")
    gen_train, gen_val = collect_general(general, train_char_budget, val_char_budget, rng)
    classics = load_classics(general)
    print(f"collected: general train {len(gen_train):,} docs "
          f"({sum(map(len, gen_train))/1e9:.2f}GB), val {len(gen_val):,} docs, "
          f"classics {len(classics)} files ({sum(map(len, classics))/1e6:.0f}MB)")

    # tokenizer on a proportional sample of the mix
    sample_path = out / "_tok_sample.txt"
    domain_head = 2e6 if args.smoke else 60e6
    general_head = 4e6 if args.smoke else 130e6
    with sample_path.open("w", encoding="utf-8") as f:
        size = 0
        with (domain / "corpus_train.txt").open(encoding="utf-8") as ftr:
            while size < domain_head:
                chunk = ftr.read(1_000_000)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
        for t in classics:
            f.write(t + "\n")
        limit = domain_head + general_head + sum(map(len, classics))
        for t in gen_train:
            if size > limit:
                break
            f.write(t + "\n")
            size += len(t)

    base = Tokenizer(models.BPE(unk_token=None))
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[*SPECIALS, OCM, SEC],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    base.train([str(sample_path)], trainer)
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
    tokenizer.save_pretrained(tok_dir)
    sample_path.unlink()
    print(f"tokenizer trained (vocab {tokenizer.vocab_size}), saved to {out} and {tok_dir}")

    # encode train: general + classics x repeat + domain
    stats = Counter()
    tw = BinWriter(out / "train.bin")
    stats["general"] = tw.add_texts(tokenizer, gen_train)
    classic_ids = None
    if classics:
        classic_ids = encode_texts(tokenizer, classics)[1]
        classic_tok = sum(len(b) // 2 for b in classic_ids)
        for i in range(args.classics_repeat):
            tw.add_blobs(classic_ids)
            stats["classics"] += classic_tok
    domain_docs = split_domain_docs(domain / "corpus_train.txt")
    B = 256
    for i in range(0, len(domain_docs), B):
        stats["domain"] += tw.add_texts(tokenizer, domain_docs[i:i + B])
        if (i // B) % 400 == 0:
            print(f"  train.bin: {tw.total/1e9:.2f}B tokens (domain doc {i:,}/{len(domain_docs):,})")
    n_train = tw.close()

    # encode val: domain val + classics head + general val
    vw = BinWriter(out / "val.bin")
    val_texts = []
    val_domain_text = (domain / "corpus_val.txt").read_text(encoding="utf-8")
    val_texts.append(val_domain_text)
    stats["val_domain"] = vw.add_texts(tokenizer, [val_domain_text])
    classic_head = "\n".join(classics)[:200_000]
    if classic_head:
        val_texts.append(classic_head)
        stats["val_classics"] = vw.add_texts(tokenizer, [classic_head])
    val_texts.extend(gen_val)
    stats["val_general"] = vw.add_texts(tokenizer, gen_val)
    n_val = vw.close()

    # round-trip check: ByteLevel tokenizer is lossless
    ids = []
    enc = tokenizer(val_texts).input_ids
    for seq in enc:
        ids.extend(seq)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    ok = decoded == "".join(val_texts)
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

    total = sum(stats[s] for s in ("general", "classics", "domain"))
    report = {
        "smoke": args.smoke,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "streams": dict(stats),
        "stream_share": {k: round(v / total, 4) for k, v in stats.items()
                         if k in ("general", "classics", "domain")},
        "general_docs_train": len(gen_train),
        "general_docs_val": len(gen_val),
        "classics_files": len(classics),
        "classics_repeat": args.classics_repeat,
        "roundtrip_pass": ok,
        "meta": meta,
    }
    (out / "mix_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/train.bin ({n_train:,} tok), val.bin ({n_val:,} tok)")
    print(f"mix: general {stats['general']/total:.1%}, classics {stats['classics']/total:.1%}, "
          f"domain {stats['domain']/total:.1%}")


if __name__ == "__main__":
    main()
