"""torch 执行后端：与 local_backend 同一契约，算子跑在 torch 上。

契约：
    run(device, blocks, tokens) -> tokens
    tokens = {"x": (B,T,d) 张量, "mask": bool(T,T), "layer": int}
    blocks[0] = 该层权重字节，顺序同 local_backend.layer_keys(d)，float32
    计算在 float32 上做，权重的存放 dtype 由 dtype 参数决定（bf16 省显存）

数值已在 CPU 上与 numpy 对拍一致（fp32 1e-6 / bf16 相对 6e-3，见 tests/test_torch_parity.py）；DCU 上未验证的只剩设备放置与吞吐，与数值无关。
"""
from __future__ import annotations
import math
import os
import sys

try:
    from .local_backend import layer_keys
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from local_backend import layer_keys

try:
    import torch
except ImportError:                      # 登录节点没有 torch，导入失败不影响别处
    torch = None


def layer_tensors(block, d: int, device: str, dtype=None) -> dict:
    """一片字节 → 该层张量，放到 device 上。"""
    if torch is None:
        raise RuntimeError("torch 不可用")
    flat = torch.frombuffer(bytearray(block), dtype=torch.float32)
    out, off = {}, 0
    for name, shape in layer_keys(d):
        n = 1
        for s in shape:
            n *= s
        out[name] = flat[off:off + n].reshape(shape).to(device=device, dtype=dtype or torch.float32)
        off += n
    if off != flat.numel():
        raise ValueError(f"层字节数不符：用了 {off}，收到 {flat.numel()}")
    return out


def forward_layer(W: dict, x, mask, cfg, compute_dtype=None) -> object:
    """compute_dtype=None 用输入自身 dtype；给 torch.float32 则升精度计算再降回。"""
    out_dtype = getattr(x, "dtype", None)
    if compute_dtype is not None and torch is not None:
        x = x.to(compute_dtype)
        W = {k: v.to(compute_dtype) for k, v in W.items()}
    d, H = cfg["n_embd"], cfg["n_head"]
    dh = d // H
    B, T = x.shape[0], x.shape[1]

    def ln(t, g, b):
        mu = t.mean(-1, keepdim=True)
        var = t.var(-1, unbiased=False, keepdim=True)
        return (t - mu) / torch.sqrt(var + 1e-5) * g + b

    h1 = ln(x, W["ln1_g"], W["ln1_b"])
    qkv = h1 @ W["qkv"]
    q, k, v = qkv.split(d, dim=-1)
    q = q.reshape(B, T, H, dh).permute(0, 2, 1, 3)
    k = k.reshape(B, T, H, dh).permute(0, 2, 1, 3)
    v = v.reshape(B, T, H, dh).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    bias = torch.zeros((T, T), dtype=scores.dtype, device=scores.device).masked_fill(
        mask.to(scores.device), float("-1e9"))
    att = torch.softmax(scores + bias, dim=-1)
    y = (att @ v).permute(0, 2, 1, 3).reshape(B, T, d)
    x = x + y @ W["proj"]
    h2 = ln(x, W["ln2_g"], W["ln2_b"])
    out = x + torch.relu(h2 @ W["fc"]) @ W["fc2"]
    return out.to(out_dtype) if out_dtype is not None else out


def make_run(cfg: dict, dtype=None):
    """返回可直接交给 Supervisor 的 run；blocks 为空表示权重已常驻 tokens["W"]。"""
    def run(dev: str, blocks, tokens):
        layer = tokens["layer"]
        W = layer_tensors(blocks[0], cfg["n_embd"], dev, dtype) if blocks else tokens["W"][layer]
        tokens["x"] = forward_layer(W, tokens["x"], tokens["mask"], cfg)
        return tokens
    return run


if __name__ == "__main__":
    if torch is None:
        print("skip: 本机没有 torch（DCU 容器里才有）")
        raise SystemExit(0)
    import numpy as np
    from types import SimpleNamespace
    from local_backend import forward_layer as np_forward

    d, H, cfg = 32, 4, {"n_embd": 32, "n_head": 4}
    rng = np.random.default_rng(0)
    W_np = {n: rng.normal(0, 0.05, s).astype(np.float32) for n, s in layer_keys(d)}
    W_np["ln1_g"][:] = 1.0
    W_np["ln1_b"][:] = 0.0
    W_np["ln2_g"][:] = 1.0
    W_np["ln2_b"][:] = 0.0
    x = rng.normal(0, 1, (1, 4, d)).astype(np.float32)
    mask = np.triu(np.ones((4, 4), bool), 1)

    ref = np_forward(SimpleNamespace(cfg=SimpleNamespace(n_embd=d, n_head=H), params=W_np), 0, x, mask)
    tl = {n: torch.tensor(v) for n, v in W_np.items()}
    got = forward_layer(tl, torch.tensor(x), torch.tensor(mask), cfg).numpy()
    diff = float(np.abs(got - ref).max())
    print("numpy vs torch diff", diff)
    assert diff < 1e-4, diff
    print("torch backend ok")
