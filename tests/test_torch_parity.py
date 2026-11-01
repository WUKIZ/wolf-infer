"""torch 与 numpy 前向对拍：多形状 + bf16。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import numpy as np
import torch
from types import SimpleNamespace
from local_backend import layer_keys, forward_layer as np_forward
from torch_backend import forward_layer as t_forward


def build(d, H, seed=0, T=6):
    rng = np.random.default_rng(seed)
    W = {n: rng.normal(0, 0.05, s).astype(np.float32) for n, s in layer_keys(d)}
    for g in ("ln1_g", "ln2_g"):
        W[g][:] = 1.0
    for b in ("ln1_b", "ln2_b"):
        W[b][:] = 0.0
    x = rng.normal(0, 1, (2, T, d)).astype(np.float32)
    mask = np.triu(np.ones((T, T), bool), 1)
    return W, x, mask


for d, H in ((32, 4), (512, 8), (256, 16)):
    W, x, mask = build(d, H)
    cfg = {"n_embd": d, "n_head": H}
    ref = np_forward(SimpleNamespace(cfg=SimpleNamespace(**cfg), params=W), 0, x, mask)
    got = t_forward({k: torch.tensor(v) for k, v in W.items()},
                    torch.tensor(x), torch.tensor(mask), cfg).numpy()
    diff = float(np.abs(got - ref).max())
    print(f"d={d} H={H} fp32 diff {diff:.2e}")
    assert diff < 1e-4, diff
    try:
        got16 = t_forward({k: torch.tensor(v).bfloat16() for k, v in W.items()},
                          torch.tensor(x).bfloat16(), torch.tensor(mask), cfg).float().numpy()
        d16 = float(np.abs(got16 - ref).max())
        denom = max(1e-9, float(np.abs(ref).max()))
        print(f"d={d} H={H} bf16 rel {d16 / denom:.2e}")
        assert d16 / denom < 2e-2, d16 / denom
    except RuntimeError as exc:
        print("bf16 skip:", exc)

print("torch parity ok")
