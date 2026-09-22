"""Build the Track-B adaptation start point from a pretrained SmolLM2 snapshot.

Loads the HF safetensors, verifies every key against our Llama-compatible
Transformer, mean-init resizes the (tied) embedding matrix by two rows for
<|ocm|>/<|sec|>, and emits:

  <out>/start_state.pt   fp32 state dict for train.py --init
  <out>/hf/              config.json + model.safetensors + tokenizer
                         (ready for convert_hf_to_gguf / llama.cpp)
  <tokenizer-out>/       extended tokenizer (tokenizers/ on the repo path)

Also runs a logits parity check (ours vs HF LlamaForCausalLM) on the resized
model to catch key-mapping / RoPE mistakes before the multi-hour DAPT run.

    .venv/bin/python scripts/adapt_init.py \
        --snapshot <hf snapshot dir> --out checkpoints/adapt_b_init \
        --tokenizer-out tokenizers/smollm2_360m_ethno
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import AddedToken, Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Transformer, save_hf

OCM, SEC = "<|ocm|>", "<|sec|>"


def our_cfg_from_hf(hf_cfg: dict) -> dict:
    heads = hf_cfg["num_attention_heads"]
    return {
        "vocab_size": hf_cfg["vocab_size"] + 2,
        "hidden": hf_cfg["hidden_size"],
        "ffn": hf_cfg["intermediate_size"],
        "layers": hf_cfg["num_hidden_layers"],
        "heads": heads,
        "kv_heads": hf_cfg["num_key_value_heads"],
        "head_dim": hf_cfg.get("head_dim", hf_cfg["hidden_size"] // heads),
        "norm_eps": hf_cfg["rms_norm_eps"],
        "rope_theta": hf_cfg["rope_theta"],
        "seq_len": 1024,
        "tie_embeddings": hf_cfg["tie_word_embeddings"],
    }


def extend_tokenizer(snapshot: Path, out_dir: Path) -> dict:
    tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tok.add_special_tokens(
        [AddedToken(OCM, special=True, normalized=False),
         AddedToken(SEC, special=True, normalized=False)]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(snapshot / name, out_dir / name)
    ids = {s: tok.token_to_id(s) for s in (OCM, SEC)}
    enc = tok.encode(f"{OCM}131 133{SEC}Title")
    assert enc.ids[0] == ids[OCM], enc.ids
    dec = tok.decode(enc.ids, skip_special_tokens=False)
    assert dec == f"{OCM}131 133{SEC}Title", dec
    print(f"tokenizer: +{OCM}={ids[OCM]} +{SEC}={ids[SEC]}, round-trip PASS")
    return {
        "vocab_size": tok.get_vocab_size(),
        "pad_id": 0, "bos_id": 0, "eos_id": 0, "unk_id": 0,
        "ocm_id": ids[OCM], "sec_id": ids[SEC],
    }


def load_hf_state(snapshot: Path) -> dict:
    hf = load_file(snapshot / "model.safetensors")
    state = {}
    for k, v in hf.items():
        if k == "lm_head.weight":
            continue  # tied to embed_tokens
        assert k.startswith("model."), f"unexpected key: {k}"
        state[k[len("model."):]] = v.float()
    return state


def parity_check(model: Transformer, snapshot: Path) -> None:
    from transformers import LlamaForCausalLM
    hf_model = LlamaForCausalLM.from_pretrained(snapshot, torch_dtype=torch.float32)
    with torch.no_grad():
        w = hf_model.model.embed_tokens.weight
        hf_model.resize_token_embeddings(model.cfg["vocab_size"])
        mean = w.mean(dim=0, keepdim=True).expand(2, -1)
        hf_model.model.embed_tokens.weight[-2:] = mean
        if hf_model.config.tie_word_embeddings:
            hf_model.tie_weights()
        hf_model.eval()
        torch.manual_seed(0)
        x = torch.randint(0, model.cfg["vocab_size"] - 2, (1, 64))
        ours = model(x)
        theirs = hf_model(x).logits
    err = (ours - theirs).abs().max().item()
    print(f"parity check vs HF LlamaForCausalLM: max |logit err| = {err:.2e}")
    assert err < 5e-3, "logits diverge from HF — key mapping or RoPE mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default="checkpoints/adapt_b_init")
    ap.add_argument("--tokenizer-out", default="tokenizers/smollm2_360m_ethno")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    snapshot = Path(args.snapshot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hf_cfg = json.loads((snapshot / "config.json").read_text())
    cfg = our_cfg_from_hf(hf_cfg)

    meta = extend_tokenizer(snapshot, Path(args.tokenizer_out))
    assert meta["vocab_size"] == cfg["vocab_size"], (meta["vocab_size"], cfg["vocab_size"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))

    state = load_hf_state(snapshot)
    model = Transformer({**cfg, "vocab_size": hf_cfg["vocab_size"]})
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        emb = model.embed_tokens.weight
        mean = emb.mean(dim=0, keepdim=True)
        model.embed_tokens.weight = torch.nn.Parameter(
            torch.cat([emb, mean.expand(2, -1)], dim=0).contiguous())
    model.cfg = cfg
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {n / 1e6:.1f}M params, vocab {cfg['vocab_size']} "
          f"(rows {cfg['vocab_size'] - 2} mean-init x2)")

    if not args.skip_parity:
        parity_check(model, snapshot)

    torch.save({k: v for k, v in model.state_dict().items()}, out / "start_state.pt")
    save_hf(model, out / "hf", meta, Path(args.tokenizer_out))
    print(f"start point: {out / 'start_state.pt'}")
    print(f"HF dir for GGUF: {out / 'hf'}")


if __name__ == "__main__":
    main()
