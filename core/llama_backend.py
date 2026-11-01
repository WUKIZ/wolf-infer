"""LLaMA 式后端：吃 wolf400M.pt 这类权重（RMSNorm + RoPE + SwiGLU，head 不共享）。

结构来自权重键名与 train400.py 的两行 FFN：
  每层 n1.w / a.qkv.weight(3d,d) / a.o.weight / n2.w / f.w1,w3(h,d) / f.w2(d,h)
  FFN = w2(silu(w1(x)) * w3(x))；无位置嵌入键 → RoPE；cfg = [n_layer, n_head, n_embd]
"""
import math
import torch
import torch.nn.functional as F


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _get(w, name):
    """容忍键名里的不可见字符：先精确，再按后缀匹配。"""
    if name in w:
        return w[name]
    for k, v in w.items():
        if k.endswith(name):
            return v
    raise KeyError(name)


def rope(x, theta=10000.0):
    B, H, T, D = x.shape
    half = D // 2
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Wolf400:
    def __init__(self, ckpt, device="cuda", dtype=torch.float32):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.w = {"".join(ch for ch in k if ch.isprintable() and not ch.isspace()):
                  v.to(device, dtype) for k, v in d["m"].items()}
        self.n_layer, self.n_head, self.d = d["cfg"][0], d["cfg"][1], d["cfg"][2]
        self.device = device
        tops = [v for k, v in self.w.items() if not k.startswith("b.")]
        self.lnf = next(v for v in tops if v.dim() == 1)          # 最终 RMSNorm 权重
        two = [v for v in tops if v.dim() == 2]
        self.emb, self.head = two[0], two[1]                      # 词嵌入、输出层

    @torch.no_grad()
    def logits(self, ids, theta=10000.0):
        w, T = self.w, len(ids)
        x = self.emb[torch.tensor(ids, device=self.device)][None]
        dh = self.d // self.n_head
        mask = torch.triu(torch.ones(T, T, device=self.device, dtype=torch.bool), 1)
        for i in range(self.n_layer):
            p = f"b.{i}."
            h = rms(x, w[p + "n1.w"])
            q, k, v = (h @ w[p + "a.qkv.weight"].T).split(self.d, dim=-1)
            q = q.view(1, T, self.n_head, dh).transpose(1, 2)
            k = k.view(1, T, self.n_head, dh).transpose(1, 2)
            v = v.view(1, T, self.n_head, dh).transpose(1, 2)
            q, k = rope(q, theta), rope(k, theta)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            a = torch.softmax(s.masked_fill(mask, float("-1e9")), dim=-1) @ v
            x = x + a.transpose(1, 2).reshape(1, T, self.d) @ w[p + "a.o.weight"].T
            h = rms(x, w[p + "n2.w"])
            g = h @ w[p + "f.w1.weight"].T
            u = h @ w[p + "f.w3.weight"].T
            x = x + (F.silu(g) * u) @ w[p + "f.w2.weight"].T
        return rms(x, self.lnf) @ self.head.T

    def generate(self, ids, n=24, theta=10000.0, temp=0.8, window=64):
        ids = list(ids)
        for _ in range(n):
            lg = self.logits(ids[-window:], theta)[0, -1] / temp
            ids.append(int(torch.multinomial(torch.softmax(lg, -1), 1)))
        return ids


if __name__ == "__main__":
    m = Wolf400("/data/wolf400M.pt")
    ids = [1, 2, 3]
    for th in (10000.0, 500000.0):
        out = m.generate(ids, n=12, theta=th, temp=0.7)
        lg = m.logits(out[-8:], th)[0, -1].float()
        p = torch.softmax(lg, -1)
        print(f"theta={th:.0f} ids={out} entropy={float(-(p * p.clamp_min(1e-9).log()).sum()):.2f}")
