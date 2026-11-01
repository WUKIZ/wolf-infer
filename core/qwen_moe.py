"""Qwen1.5-MoE 后端：60 专家 top-4 + 共享专家，无 GQA，无 qk-norm。"""
import os
import sys

import torch
import torch.nn.functional as F

try:
    from .qwen_backend import rms, rope
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_backend import rms, rope


def norm32(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w


def mlp(x, gate, up, down):
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def attention(t, x, cfg):
    H = cfg["num_attention_heads"]
    dh = cfg.get("head_dim") or cfg["hidden_size"] // H
    eps = cfg["rms_norm_eps"]
    B, T, d = x.shape
    h = norm32(x, t["input_layernorm.weight"], eps)
    def lin(x, name):
        b = t.get("self_attn." + name + ".bias")
        y = x @ t["self_attn." + name + ".weight"].T
        return y if b is None else y + b

    q = lin(h, "q_proj").view(B, T, H, dh).transpose(1, 2)
    k = lin(h, "k_proj").view(B, T, H, dh).transpose(1, 2)
    v = lin(h, "v_proj").view(B, T, H, dh).transpose(1, 2)
    q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
    Tq = q.shape[2]
    mask = torch.full((Tq, Tq), float('-inf'), device=q.device, dtype=q.dtype).triu(1)
    w = torch.softmax(q @ k.transpose(-1, -2) / (dh ** 0.5) + mask, dim=-1)
    a = w @ v
    return x + a.transpose(1, 2).reshape(B, T, d) @ t["self_attn.o_proj.weight"].T


def moe_block(t, x, cfg):
    """MoE 前向：命中的专家权重堆成 (K,...) 一次算完，避免逐专家 12 次小 GEMM。"""
    eps, E = cfg["rms_norm_eps"], cfg["num_experts"]
    top = cfg["num_experts_per_tok"]
    B, T, d = x.shape
    h = norm32(x, t["post_attention_layernorm.weight"], eps)
    flat = h.reshape(-1, d)
    probs = torch.softmax(flat @ t["mlp.gate.weight"].T, dim=-1)
    vals, idx = torch.topk(probs, top, dim=-1)
    # 概率下限：低概率专家的权重本就接近 0，剔掉可让并集变小、bmm 的 K 变小；
    # floor=0 时不剔除，行为与原来完全一致
    floor = cfg.get('moe_prob_floor', 0.0)
    keep = None
    if floor > 0.0:
        keep = vals >= floor
        vals = torch.where(keep, vals, torch.zeros_like(vals))

    out = torch.zeros(flat.shape, device=flat.device, dtype=torch.float32)
    active = sorted(set((idx if keep is None else idx[keep]).reshape(-1).tolist()))
    if active:
        pre = ["mlp.experts.%d." % e for e in active]
        gw = torch.stack([t[q + "gate_proj.weight"].clone() for q in pre])
        uw = torch.stack([t[q + "up_proj.weight"].clone() for q in pre])
        dw = torch.stack([t[q + "down_proj.weight"].clone() for q in pre])
        act = F.silu(torch.matmul(flat, gw.transpose(1, 2))) * torch.matmul(flat, uw.transpose(1, 2))
        eo = torch.matmul(act, dw.transpose(1, 2)).float()
        for k, e in enumerate(active):
            wk = (vals * (idx == e)).sum(-1).float()
            out += eo[k] * wk[:, None]

    sh = mlp(flat, t["mlp.shared_expert.gate_proj.weight"],
             t["mlp.shared_expert.up_proj.weight"], t["mlp.shared_expert.down_proj.weight"])
    gate_s = torch.sigmoid(flat @ t["mlp.shared_expert_gate.weight"].T)
    out = out + sh.float() * gate_s
    return x + out.to(x.dtype).reshape(B, T, d)


def load_moe(d, device="cuda", dtype=torch.bfloat16):
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(d, "config.json")))
    w = {}
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        w.update(load_file(os.path.join(d, f)))
    return cfg, {k: v.to(device, dtype) for k, v in w.items()}


def logits(cfg, w, ids, device="cuda"):
    ids_t = torch.tensor([ids], device=device)
    x = w["model.embed_tokens.weight"][ids_t]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
        x = attention(t, x, cfg)
        x = moe_block(t, x, cfg)
    x = rms(x, w["model.norm.weight"], cfg["rms_norm_eps"])
    return x @ w["lm_head.weight"].T


if __name__ == "__main__":
    d = "/data/models/Qwen1.5-MoE-A2.7B"
    cfg, w = load_moe(d)
    g = logits(cfg, w, [40, 100, 200, 300])
    p = torch.softmax(g[0, -1].float(), -1)
    print("shape", tuple(g.shape), "entropy", float(-(p * p.clamp_min(1e-9).log()).sum()))
    print("top5", [int(i) for i in g[0, -1].topk(5).indices])
    print("moe model ok")
