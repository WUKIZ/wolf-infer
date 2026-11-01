"""把 Wolf400 按层落盘并逐层执行：20 片 × 78.6 MB，预算可小于模型总量。"""
import math
import os
import sys

import torch
from core.tmpdir import TMP

try:
    from .shard import ModelSpec, Shard
    from .supervisor import Supervisor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shard import ModelSpec, Shard
    from supervisor import Supervisor

ORDER = ("n1.w", "a.qkv.weight", "a.o.weight", "n2.w", "f.w1.weight", "f.w2.weight", "f.w3.weight")


def write_shards(w: dict, n_layer: int, out_dir: str) -> ModelSpec:
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for i in range(n_layer):
        buf = b"".join(w[f"b.{i}.{n}"].detach().float().cpu().numpy().tobytes() for n in ORDER)
        path = os.path.abspath(os.path.join(out_dir, f"layer_{i:04d}.bin"))
        with open(path, "wb") as fh:
            fh.write(buf)
        shards.append(Shard(i, path, 0, len(buf), i, i + 1))
    return ModelSpec("wolf400", 0, n_layer, 6000, tuple(shards))


def _rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope(x, theta):
    B, H, T, D = x.shape
    half = D // 2
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def layer_forward(t, x, mask, meta):
    n_head, theta = meta["n_head"], meta["theta"]
    B, T, d = x.shape[0], x.shape[1], x.shape[-1]
    dh = d // n_head
    h = _rms(x, t["n1.w"])
    q, k, v = (h @ t["a.qkv.weight"].T).split(d, dim=-1)
    q = q.view(B, T, n_head, dh).transpose(1, 2)
    k = k.view(B, T, n_head, dh).transpose(1, 2)
    v = v.view(B, T, n_head, dh).transpose(1, 2)
    q, k = _rope(q, theta), _rope(k, theta)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    a = torch.softmax(s.masked_fill(mask, float("-1e9")), dim=-1) @ v
    x = x + a.transpose(1, 2).reshape(B, T, d) @ t["a.o.weight"].T
    h = _rms(x, t["n2.w"])
    g = torch.nn.functional.silu(h @ t["f.w1.weight"].T) * (h @ t["f.w3.weight"].T)
    return x + g @ t["f.w2.weight"].T


def make_run(cfg: dict):
    def run(dev, blocks, tokens):
        layer = tokens["layer"]
        flat = torch.frombuffer(bytearray(blocks[0]), dtype=torch.float32)
        off, t = 0, {}
        for name in ORDER:
            shape = tuple(cfg["shapes"][name])
            num = 1
            for s in shape:
                num *= s
            t[name] = flat[off:off + num].reshape(shape).to(dev, cfg["dtype"])
            off += num
        tokens["x"] = layer_forward(t, tokens["x"], tokens["mask"], cfg)
        return tokens
    return run


def logits(model, ids, budget_bytes, devices=("cuda",), theta=10000.0):
    w = model.w
    n_layer, n_head = model.n_layer, model.n_head
    shapes = {n: tuple(w[f"b.0.{n}"].shape) for n in ORDER}
    cfg = {"n_head": n_head, "theta": theta, "dtype": w["b.0.n1.w"].dtype, "shapes": shapes}
    spec = _spec(model)
    T = len(ids)
    x = _emb(model)[torch.tensor(ids, device=w["b.0.n1.w"].device)][None]
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
    sup = Supervisor(spec, list(devices), budget_bytes, make_run(cfg))
    out = sup.forward({"x": x, "mask": mask, "layer": 0})
    sup.close()
    return _rms(out["x"], _lnf(model)) @ _head(model).T

_SPEC = {}


def _spec(model, path=TMP + "/wolf400_shards"):
    if path not in _SPEC:
        _SPEC[path] = write_shards(model.w, model.n_layer, path)
    return _SPEC[path]


def generate(model, ids, n=16, budget_bytes=160_000_000, theta=10000.0, temp=0.8, window=32):
    seq = list(ids)
    for _ in range(n):
        lg = logits(model, seq[-window:], budget_bytes, theta=theta)[0, -1] / temp
        seq.append(int(torch.multinomial(torch.softmax(lg, -1), 1)))
    return seq


def _tops(m):
    return [v for k, v in m.w.items() if not k.startswith("b.")]


def _lnf(m):
    return next(v for v in _tops(m) if v.dim() == 1)


def _emb(m):
    return [v for v in _tops(m) if v.dim() == 2][0]


def _head(m):
    return [v for v in _tops(m) if v.dim() == 2][1]


if __name__ == "__main__":
    from llama_backend import Wolf400

    m = Wolf400("/data/wolf400M.pt")
    ids = [1, 2, 3, 4]
    ref = m.logits(ids)
    got = logits(m, ids, budget_bytes=3 * 78_644_480)
    print("dense vs sharded diff", float((got - ref).abs().max()))
    print("sharded ok")
