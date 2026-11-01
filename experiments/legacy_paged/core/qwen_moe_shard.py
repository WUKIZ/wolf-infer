"""Qwen1.5-MoE 按专家落盘。

每层：注意力、两个 norm、router、共享专家 → 常驻（dense）；
每个专家（gate/up/down 三张量）→ 独立文件，一片 = 一个专家。
这样 top-4 只需换入 4 片，而不是 60 片。
"""
import os
import sys

import torch
from core.tmpdir import TMP

try:
    from .shard import Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shard import Shard

EXPERT_KEYS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def write_experts(w, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dense, shards = {}, []
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        t = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
        dense[L] = {k: v for k, v in t.items() if not k.startswith("mlp.experts.")}
        for e in range(cfg["num_experts"]):
            pre = f"mlp.experts.{e}."
            path = os.path.join(out_dir, f"L{L:02d}e{e:02d}.bin")
            raw = [t[pre + s].contiguous().view(torch.uint8).numpy().tobytes() for s in EXPERT_KEYS]
            with open(path, "wb") as f:
                for b in raw:
                    f.write(b)
            shards.append(Shard(len(shards), path, 0, sum(len(b) for b in raw),
                                L, L + 1, expert=e))
    return dense, shards


if __name__ == "__main__":
    from qwen_moe import load_moe
    d = "/data/models/Qwen1.5-MoE-A2.7B"
    cfg, w = load_moe(d, device="cpu", dtype=torch.bfloat16)
    dense, shards = write_experts(w, cfg, TMP + "/moe_shards")
    per = sum(s.nbytes for s in shards if s.layer_from == 0)
    print("layers", len(dense), "shards", len(shards),
          "one-layer experts", per // 1048576, "MB",
          "one expert", shards[0].nbytes // 1024, "KB")


def read_expert(mv, cfg, dtype):
    import numpy as np
    d, h = cfg["hidden_size"], cfg["moe_intermediate_size"]
    n = d * h
    b = torch.from_numpy(np.frombuffer(mv, dtype=np.uint16).copy())
    b = b.view(torch.bfloat16) if hasattr(torch, "bfloat16") else b
    return b[:n].view(h, d), b[n:2 * n].view(h, d), b[2 * n:].view(d, h)


def paged_moe(layer, t, x, cfg, pager, by_layer, dtype=torch.bfloat16):
    """只换入路由命中的专家，算完立即释放租用。"""
    from qwen_moe import mlp, norm32
    B, T, d = x.shape
    eps = cfg["rms_norm_eps"]
    h = norm32(x, t["post_attention_layernorm.weight"], eps)
    flat = h.reshape(-1, d)
    probs = torch.softmax(flat @ t["mlp.gate.weight"].T, dim=-1)
    vals, idx = torch.topk(probs, cfg["num_experts_per_tok"], dim=-1)
    out = torch.zeros(flat.shape, device=flat.device, dtype=torch.float32)
    for e in sorted({int(i) for i in idx.flatten()}):
        rows, slots = (idx == e).nonzero(as_tuple=True)
        shard = by_layer[(layer, e)]
        with pager.lease(shard) as mv:
            wg, wu, wd = read_expert(mv, cfg, dtype)
            g = wg.to(flat.device).clone()
            u = wu.to(flat.device).clone()
            dn = wd.to(flat.device).clone()
            del mv, wg, wu, wd
            import gc
            gc.collect()
        eo = mlp(flat[rows], g, u, dn).float()
        del g, u, dn
        out.index_add_(0, rows, eo * vals[rows, slots][:, None].float())
    sh = mlp(flat, t["mlp.shared_expert.gate_proj.weight"],
             t["mlp.shared_expert.up_proj.weight"], t["mlp.shared_expert.down_proj.weight"])
    gate_s = torch.sigmoid((flat @ t["mlp.shared_expert_gate.weight"].T).float())
    return x + (out + sh.float() * gate_s).to(x.dtype).reshape(B, T, d)


def write_dense(dense, out_dir):
    """每层稠密部分单独存盘，便于进程只加载自己负责的层。"""
    paths = {}
    for L, t in dense.items():
        p = os.path.join(out_dir, f"L{L:02d}.dense.pt")
        torch.save(t, p)
        paths[L] = p
    return paths


def load_dense(path, device="cuda", dtype=torch.bfloat16):
    return {k: v.to(device, dtype) for k, v in torch.load(path, map_location="cpu").items()}
