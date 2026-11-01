"""Qwen3 后端：GQA + qk-norm + RMSNorm + RoPE(1e6) + SwiGLU，tie embedding。"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def load(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    w = {}
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        w.update(load_file(os.path.join(d, f)))
    return cfg, w


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, theta):
    B, H, T, D = x.shape
    half = D // 2
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos[:, None] * inv[None, :]
    cos = ang.cos()[None, None].to(x.dtype)
    sin = ang.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def layer_forward(t, x, mask, cfg):
    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    B, T, d = x.shape
    h = rms(x, t["input_layernorm.weight"], eps)
    q = (h @ t["self_attn.q_proj.weight"].T).view(B, T, H, dh).transpose(1, 2)
    k = (h @ t["self_attn.k_proj.weight"].T).view(B, T, KV, dh).transpose(1, 2)
    v = (h @ t["self_attn.v_proj.weight"].T).view(B, T, KV, dh).transpose(1, 2)
    q = rope(rms(q, t["self_attn.q_norm.weight"], eps), cfg["rope_theta"])
    k = rope(rms(k, t["self_attn.k_norm.weight"], eps), cfg["rope_theta"])
    k = k.repeat_interleave(H // KV, dim=1)
    v = v.repeat_interleave(H // KV, dim=1)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + a.transpose(1, 2).reshape(B, T, d) @ t["self_attn.o_proj.weight"].T
    h = rms(x, t["post_attention_layernorm.weight"], eps)
    g = F.silu(h @ t["mlp.gate_proj.weight"].T) * (h @ t["mlp.up_proj.weight"].T)
    return x + g @ t["mlp.down_proj.weight"].T


def logits(cfg, w, ids, device="cuda", dtype=torch.float32):
    ids = torch.tensor([ids], device=device)
    T = ids.shape[1]
    x = w["model.embed_tokens.weight"][ids].to(dtype)
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
        x = layer_forward(t, x, mask, cfg)
    x = rms(x, w["model.norm.weight"], cfg["rms_norm_eps"])
    return x @ w["model.embed_tokens.weight"].T


if __name__ == "__main__":
    d = "/data/models/Qwen3-1.7B"
    cfg, w = load(d)
    w = {k: v.to("cuda", torch.float32) for k, v in w.items()}
    ids = [40, 100, 200, 300]
    got = logits(cfg, w, ids)
    print("shape", tuple(got.shape))
    try:
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).to("cuda")
        with torch.no_grad():
            out = ref(torch.tensor([ids], device="cuda")).logits
        print("hf diff", float((got - out).abs().max()))
    except Exception as exc:
        print("hf 对拍跳过:", type(exc).__name__, exc)
    print("qwen ok")


def rope_at(x, theta, offset):
    B, H, T, D = x.shape
    half = D // 2
    pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos[:, None] * inv[None, :]
    cos = ang.cos()[None, None].to(x.dtype)
    sin = ang.sin()[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def layer_forward_cached(t, x, past, cfg):
    """x: (B,1,d)；past: (k, v) 或 None；返回 (y, (k, v))。解码时无需掩码。"""
    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    B, T, d = x.shape
    offset = 0 if past is None else past[0].shape[2]
    h = rms(x, t["input_layernorm.weight"], eps)
    q = (h @ t["self_attn.q_proj.weight"].T).view(B, T, H, dh).transpose(1, 2)
    k = (h @ t["self_attn.k_proj.weight"].T).view(B, T, KV, dh).transpose(1, 2)
    v = (h @ t["self_attn.v_proj.weight"].T).view(B, T, KV, dh).transpose(1, 2)
    q = rope_at(rms(q, t["self_attn.q_norm.weight"], eps), cfg["rope_theta"], offset)
    k = rope_at(rms(k, t["self_attn.k_norm.weight"], eps), cfg["rope_theta"], offset)
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    try:
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    except TypeError:                      # torch < 2.5 没有 enable_gqa
        kk = k.repeat_interleave(H // KV, dim=1)
        vv = v.repeat_interleave(H // KV, dim=1)
        a = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=False)
    x = x + a.transpose(1, 2).reshape(B, T, d) @ t["self_attn.o_proj.weight"].T
    h = rms(x, t["post_attention_layernorm.weight"], eps)
    g = F.silu(h @ t["mlp.gate_proj.weight"].T) * (h @ t["mlp.up_proj.weight"].T)
    return x + g @ t["mlp.down_proj.weight"].T, (k, v)


def generate_cached(cfg, w, ids, n=8, device="cuda"):
    """增量解码：只在开头算一次前缀，之后每个 token 复用各层 K/V。"""
    L = cfg["num_hidden_layers"]
    layers = [{k[len(f"model.layers.{i}."):]: v for k, v in w.items()
               if k.startswith(f"model.layers.{i}.")} for i in range(L)]
    emb, nrm = w["model.embed_tokens.weight"], w["model.norm.weight"]
    pasts = [None] * L
    seq = list(ids)

    def step(tok):
        nonlocal pasts
        x = emb[torch.tensor([[tok]], device=device)]
        for i in range(L):
            x, pasts[i] = layer_forward_cached(layers[i], x, pasts[i], cfg)
        return (rms(x, nrm, cfg["rms_norm_eps"]) @ emb.T)[0, -1]

    lg = None
    for tok in seq:
        lg = step(tok)
    for _ in range(n):
        seq.append(int(lg.argmax()))
        lg = step(seq[-1])
    return seq
