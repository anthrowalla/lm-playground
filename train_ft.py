"""Task fine-tune trainer: train.py + the tagging task.

Loads tokenized prompt/target pairs (prepare_finetune.py output), pads
example-aligned batches (right-pad; causal attention keeps pads inert),
masks loss to the target codes, and interleaves raw-LM replay batches from
the base train.bin (full loss) to preserve fluency and unconditional
tagging. Adds an in-training greedy tagging-F1 eval on a fixed val sample
(sec-loo prompts, forced <|ocm|> marker) and keeps the best-F1 state_dict.

Architecture/optimizer/schedule are unchanged from train.py. Branch
duplicate of train.py per finetune_plan_small.md.
"""

import argparse
import json
import random
import re
import shutil
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from prepare_ethno import load_code_names
from prepare_ethno_tree import OCM, SEC
from tag_eval import parse_pairs
from train import Transformer, lr_at

PAD_IGNORE = -100


class TagDataset:
    """Padded example batches from ft.bin + offsets + prompt lengths."""

    def __init__(self, path: Path):
        self.toks = np.fromfile(path / "ft.bin", dtype=np.uint16)
        self.offsets = np.fromfile(path / "ft_offsets.npy", dtype=np.int64)
        self.plens = np.fromfile(path / "ft_plens.npy", dtype=np.int32)

    def __len__(self):
        return len(self.plens)

    def batch(self, idx, device):
        rows = [self.toks[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)
                for i in idx]
        w = max(len(ids) for ids in rows) - 1
        x = torch.zeros((len(rows), w), dtype=torch.int64)   # pad id 0
        y = torch.full((len(rows), w), PAD_IGNORE, dtype=torch.int64)
        for b, ids in enumerate(rows):
            pl = int(self.plens[idx[b]])
            x[b, :len(ids) - 1] = torch.from_numpy(ids[:-1])
            tgt = torch.from_numpy(ids[1:].copy())
            tgt[:pl - 1] = PAD_IGNORE  # mask prompt; train on codes + "\n\n"
            y[b, :len(ids) - 1] = tgt
        return x.to(device), y.to(device)


class ReplayDataset:
    """Full-LM batches sampled from the base train.bin (train.py style)."""

    def __init__(self, bin_path: str, seq: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq = seq

    def batch(self, bs, device):
        ix = torch.randint(len(self.data) - self.seq - 1, (bs,))
        x = torch.stack([torch.from_numpy(self.data[i:i + self.seq].astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.seq].astype(np.int64))
                         for i in ix])
        return x.to(device), y.to(device)


def build_eval_sample(tok, val_docs, names, ocm_id, n=250, seed=0, max_prompt=900):
    """Fixed sec-loo sample: (prompt_ids + [ocm_id], gold codes)."""
    rng = random.Random(seed)
    recs = [json.loads(l) for l in open(val_docs, encoding="utf-8")]
    groups = defaultdict(list)
    items = []
    for di, d in enumerate(recs):
        for p in d["paragraphs"]:
            if p["tags"]:
                groups[(di, p.get("section", ""))].append(p)
                items.append((di, p))
    rng.shuffle(items)
    sample = []
    for di, p in items[:n]:
        sibs = {c for o in groups[(di, p.get("section", ""))] if o is not p
                for c, _ in o["tags"]}
        prefix = ""
        if p.get("section") and sibs:
            named = " ".join(f"{c} {names.get(c, c)}"
                             for c in sorted(sibs, key=lambda c: (len(c), c)))
            prefix = f"{SEC}{p['section']}\n{OCM}{named}\n\n"
        ids = tok.encode(prefix, add_special_tokens=False).ids + \
            tok.encode(p["text"] + "\n", add_special_tokens=False).ids
        ids = ids[:max_prompt]
        sample.append((ids + [ocm_id], [c for c, _ in p["tags"]]))
    return sample


@torch.no_grad()
def greedy_new(model, tok, ids, device, max_new=40, stop="\n\n"):
    """Greedy-decode new tokens after ids, stopping at `stop`."""
    x = torch.tensor([ids], device=device)
    n = len(ids)
    for _ in range(max_new):
        logits = model(x[:, -model.cfg["seq_len"]:])[:, -1]
        nxt = int(logits.argmax(-1))
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if stop and stop in tok.decode(x[0, n:].tolist()):
            break
    return tok.decode(x[0, n:].tolist()).split("\n\n")[0]


@torch.no_grad()
def tagging_f1(model, tok, sample, valid, device, max_new=40):
    """Greedy-decode the tag block per sample; micro P/R/F1 vs gold."""
    model.eval()
    tp = fp = fn = 0
    for ids, gold in sample:
        out = greedy_new(model, tok, ids, device, max_new)
        codes = list(dict.fromkeys(c for c, _ in parse_pairs(out)
                                   if c in valid))
        g, p = set(gold), set(codes)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    model.train()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def norm_words(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


@torch.no_grad()
def title_f1(model, tok, sample, valid, device, max_new=48):
    """Greedy-decode the title path per sample; word-bag P/R/F1 vs gold."""
    model.eval()
    tp = fp = fn = exact = 0
    for ids, gold in sample:
        out = greedy_new(model, tok, ids, device, max_new)
        g, p = set(norm_words(" ".join(gold))), set(norm_words(out))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += norm_words(out) == norm_words(" ".join(gold))
    model.train()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def build_title_sample(tok, val_docs, sec_id, n=200, seed=0, budget=800):
    """Fixed title sample from val societies: (prompt_ids ending <|sec|>, gold path)."""
    rng = random.Random(seed)
    recs = [json.loads(l) for l in open(val_docs, encoding="utf-8")]
    groups = defaultdict(list)
    for di, d in enumerate(recs):
        for p in d["paragraphs"]:
            if p.get("section"):
                groups[(di, p["section"])].append(p["text"])
    keys = list(groups)
    rng.shuffle(keys)
    tail_ids = tok.encode("\n\n", add_special_tokens=False).ids
    sample = []
    for k in keys[:n]:
        txt_ids = tok.encode("\n\n".join(groups[k]), add_special_tokens=False).ids
        sample.append((txt_ids[:budget] + tail_ids + [sec_id], k[1]))
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-ckpt", required=True, help="HF dir with model.safetensors")
    ap.add_argument("--ft-data", default="data/ethnographic_v5/ft")
    ap.add_argument("--replay-bin", default="data/ethnographic_v5/train.bin")
    ap.add_argument("--replay-frac", type=float, default=0.2)
    ap.add_argument("--tokenizer", default="data/ethnographic_v5/tokenizer.json")
    ap.add_argument("--val-docs", default="data/ethnographic_v5/val_docs.jsonl")
    ap.add_argument("--ocm-labels", default="data/ethnographic/ocmdefs.txt")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-task", choices=["tags", "titles", "none"],
                    default="tags")
    ap.add_argument("--max-steps", type=int, help="override train.steps")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    raw = tomllib.loads(Path(args.config).read_text())
    t, c = raw["train"], raw["model"]
    if args.max_steps:
        t["steps"] = args.max_steps
    device = raw.get("device", "cuda")
    torch.manual_seed(t["seed"])

    meta = json.loads(Path(raw["meta_path"]).read_text())
    c["vocab_size"] = meta["vocab_size"]
    model = Transformer(c).to(device)
    model.load_state_dict(load_file(str(Path(args.init_ckpt) / "model.safetensors")))
    print(f"init from {args.init_ckpt}")

    train_data = TagDataset(Path(args.ft_data))
    replay = ReplayDataset(args.replay_bin, c["seq_len"])
    print(f"ft examples: {len(train_data):,}; replay stream {len(replay.data):,} tokens")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t["lr"], betas=(t["beta1"], t["beta2"]), fused=device.startswith("cuda"))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    names = load_code_names(Path(args.ocm_labels))
    ocm_id = tok.token_to_id(OCM)
    valid = set(re.findall(r"(?m)^(\d{3,4})\s",
                           Path(args.ocm_labels).read_text(encoding="utf-8")))
    if args.eval_task == "titles":
        eval_sample = build_title_sample(tok, args.val_docs,
                                         tok.token_to_id(SEC), args.eval_n)
        eval_fn, metric = title_f1, "title"
        print(f"title-F1 eval on {len(eval_sample)} fixed val-society sections")
    elif args.eval_task == "none":
        eval_sample, eval_fn, metric = None, None, "none"
    else:
        eval_sample = build_eval_sample(tok, args.val_docs, names, ocm_id, args.eval_n)
        eval_fn, metric = tagging_f1, "tagging"
        print(f"tagging-F1 eval on {len(eval_sample)} fixed sec-loo samples")

    out = Path(raw["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if raw.get("compile", True) and not args.no_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    order = np.random.permutation(len(train_data))
    ptr = 0
    model.train()
    tokens_per_step = t["batch"] * c["seq_len"]
    t0 = time.time()
    for step in range(t["steps"]):
        lr = lr_at(step, t)
        for group in opt.param_groups:
            group["lr"] = lr
        if random.random() < args.replay_frac:
            x, y = replay.batch(t["batch"], device)
        else:
            if ptr + t["batch"] > len(order):
                order = np.random.permutation(len(train_data))
                ptr = 0
            x, y = train_data.batch(order[ptr:ptr + t["batch"]], device)
            ptr += t["batch"]
        with torch.autocast(device, torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, c["vocab_size"]).float(),
                                   y.view(-1), ignore_index=PAD_IGNORE)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % t["log_every"] == 0:
            rate = tokens_per_step * t["log_every"] / (time.time() - t0)
            t0 = time.time()
            print(f"step={step + 1} loss={loss.item():.4f} lr={lr:.6f} "
                  f"grad_norm={gnorm:.3f} tok/s={rate:,.0f}", flush=True)
        if t["eval_every"] and eval_fn and (step + 1) % t["eval_every"] == 0:
            p, r, f1 = eval_fn(model, tok, eval_sample, valid, device)
            print(f"{metric} P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)
            with (out / "eval_history.jsonl").open("a") as ef:
                ef.write(json.dumps({"step": step + 1, f"{metric}_P": p,
                                     f"{metric}_R": r, f"{metric}_F1": f1}) + "\n")
            best = 0.0
            if (out / "best_f1.txt").exists():
                best = float((out / "best_f1.txt").read_text())
            if f1 > best:
                (out / "best_f1.txt").write_text(f"{f1:.4f}")
                unwrapped = getattr(model, "_orig_mod", model)
                torch.save({"model": unwrapped.state_dict(), "step": step},
                           out / "best_f1_state.pt")
                print(f"new best F1 {f1:.4f} — saved best_f1_state.pt", flush=True)

    unwrapped = getattr(model, "_orig_mod", model)
    save_file({k: v.to(torch.bfloat16).contiguous()
               for k, v in unwrapped.state_dict().items()},
              out / "model.safetensors")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        src = Path(args.init_ckpt) / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"saved HF checkpoint to {out}")
    if eval_fn:
        p, r, f1 = eval_fn(model, tok, eval_sample, valid, device)
        print(f"final {metric} P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
