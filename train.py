"""Minimal Llama-style trainer for tiny/small/medium decoders.

Trains a GQA decoder (RMSNorm, RoPE, SwiGLU, tied embeddings) from TOML
configs and saves HF-Llama-compatible checkpoints (config.json +
model.safetensors) that convert_hf_to_gguf.py accepts directly.
"""

import argparse
import json
import math
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast


class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float, max_seq_len: int):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos", torch.cat((cos, cos), dim=-1), persistent=False)
        self.register_buffer("sin", torch.cat((sin, sin), dim=-1), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        cos = self.cos[None, None, :seq, :].to(x.dtype)
        sin = self.sin[None, None, :seq, :].to(x.dtype)
        return x * cos + self._rotate(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, heads, kv_heads = cfg["hidden"], cfg["heads"], cfg["kv_heads"]
        self.n_heads, self.n_kv = heads, kv_heads
        self.head_dim = cfg.get("head_dim", hidden // heads)
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.rope = Rotary(self.head_dim, cfg["rope_theta"], cfg["seq_len"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(y)


class Mlp(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden"], cfg["ffn"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden"], cfg["ffn"], bias=False)
        self.down_proj = nn.Linear(cfg["ffn"], cfg["hidden"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = Mlp(cfg)
        self.input_layernorm = nn.RMSNorm(cfg["hidden"], cfg["norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(cfg["hidden"], cfg["norm_eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Transformer(nn.Module):
    """HF-Llama-compatible module: state-dict keys match LlamaForCausalLM."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden"])
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg["layers"]))
        self.norm = nn.RMSNorm(cfg["hidden"], cfg["norm_eps"])
        self.cfg = cfg
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg["layers"]))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)


def lr_at(step: int, t: dict) -> float:
    warmup, total = t["warmup"], t["steps"]
    if step < warmup:
        return t["lr"] * (step + 1) / warmup
    decay = max(0, step - warmup) / max(1, total - warmup)
    ratio = t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * decay))
    return t["lr"] * ratio


def get_batch(data: np.memmap, batch: int, seq: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i : i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, device: str, t: dict, batches: int = 10) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, t["batch"], model.cfg["seq_len"], device)
        with torch.autocast(device, torch.bfloat16, enabled=device.startswith("cuda")):
            losses.append(F.cross_entropy(model(x).view(-1, model.cfg["vocab_size"]), y.view(-1)))
    model.train()
    return torch.stack(losses).mean().item()


@torch.no_grad()
def sample(model, data, device: str, tokenizer, tokens: int = 200) -> str:
    model.eval()
    seq = model.cfg["seq_len"]
    x, _ = get_batch(data, 1, seq, device)
    for _ in range(tokens):
        with torch.autocast(device, torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(x)[:, -1]
        x = torch.cat([x[:, 1:], logits.argmax(-1, keepdim=True)], dim=1)
    model.train()
    return tokenizer.decode(x[0].tolist()[-tokens:])


def save_hf(model: Transformer, out: Path, meta: dict, tokenizer_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    c = model.cfg
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": c["vocab_size"],
        "hidden_size": c["hidden"],
        "intermediate_size": c["ffn"],
        "num_hidden_layers": c["layers"],
        "num_attention_heads": c["heads"],
        "num_key_value_heads": c["kv_heads"],
        "head_dim": c.get("head_dim", c["hidden"] // c["heads"]),
        "max_position_embeddings": c["seq_len"],
        "rms_norm_eps": c["norm_eps"],
        "rope_theta": c["rope_theta"],
        "tie_word_embeddings": c["tie_embeddings"],
        "attention_bias": False,
        "bos_token_id": meta["bos_id"],
        "eos_token_id": meta["eos_id"],
        "pad_token_id": meta["pad_id"],
        "torch_dtype": "bfloat16",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
        out / "model.safetensors",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tokenizer_dir / name).exists():
            shutil.copy2(tokenizer_dir / name, out / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, help="override train.steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    args = parser.parse_args()

    raw = tomllib.loads(Path(args.config).read_text())
    t, c = raw["train"], raw["model"]
    if args.max_steps:
        t["steps"] = args.max_steps
    if args.no_compile:
        raw["compile"] = False
    device = raw.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(t["seed"])

    meta = json.loads(Path(raw["meta_path"]).read_text())
    c["vocab_size"] = meta["vocab_size"]
    model = Transformer(c).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params on {device} (vocab {c['vocab_size']})")

    train_data = np.memmap(raw["train_bin"], dtype=np.uint16, mode="r")
    val_data = np.memmap(raw["val_bin"], dtype=np.uint16, mode="r")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": t["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=t["lr"],
        betas=(t["beta1"], t["beta2"]),
        fused=device.startswith("cuda"),
    )

    start = 0
    out = Path(raw["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "training_state.pt").exists():
        state = torch.load(out / "training_state.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"] + 1
        print(f"resumed from step {start}")

    if raw.get("compile", True) and hasattr(torch, "compile"):
        model = torch.compile(model)

    model.train()
    tokens_per_step = t["batch"] * c["seq_len"]
    t0, last_tokens = time.time(), 0
    for step in range(start, t["steps"]):
        lr = lr_at(step, t)
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(t.get("accum", 1)):
            x, y = get_batch(train_data, t["batch"], c["seq_len"], device)
            with torch.autocast(device, torch.bfloat16, enabled=device.startswith("cuda")):
                loss = F.cross_entropy(model(x).view(-1, c["vocab_size"]), y.view(-1))
            (loss / t.get("accum", 1)).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % t["log_every"] == 0:
            now = time.time()
            rate = tokens_per_step * t["log_every"] / (now - t0 - last_tokens) if step > start else 0.0
            last_tokens = now - t0
            print(
                f"step={step + 1} loss={loss.item():.4f} lr={lr:.6f} "
                f"grad_norm={grad_norm:.4f} tok/s={rate:,.0f}",
                flush=True,
            )
        if t["eval_every"] and (step + 1) % t["eval_every"] == 0:
            print(f"val_loss={evaluate(model, val_data, device, t):.4f}", flush=True)
        if t["save_every"] and (step + 1) % t["save_every"] == 0:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                out / "training_state.pt",
            )

    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": t["steps"] - 1},
        out / "training_state.pt",
    )
    unwrapped = getattr(model, "_orig_mod", model)
    save_hf(unwrapped, out, meta, Path(raw["meta_path"]).parent)
    print(f"saved HF checkpoint to {out}")
    print(f"final val_loss={evaluate(model, val_data, device, t):.4f}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path(raw["meta_path"]).parent)
    print("sample:", sample(unwrapped, val_data, device, tokenizer)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
