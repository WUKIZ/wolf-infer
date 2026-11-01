"""400M 模型的 torch 推理接口 —— 让本地助手用上真训出来的那个脑子。

为什么单独一个文件：`core/model.py` 是 numpy 手写实现，读不了 `.pt`，
而且是 4 层 128 维的配置。这里直接复刻训练脚本的架构（20 层 1280 维），
用 torch 加载 `wolf400M_old.pt`。

能力边界：这是纯预训练模型，只会续写，不会一问一答。
`chat()` 存在只是为了让上层接口统一；真要对话得等 SFT。

用法：
    from local400 import Local400
    m = Local400()                      # 自动找 ckpt 和 vocab
    print(m.generate("人工智能的未来是", max_new=60))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = Path("/data/wolf400M_old.pt")
DEFAULT_VOCAB = Path("/data/vocab.json")
FALLBACK_VOCAB = ROOT / "vocab.json"


class Attn(nn.Module):
    def __init__(self, e: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(e, 3 * e, bias=False)
        self.o = nn.Linear(e, e, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [y.view(b, t, self.h, c // self.h).transpose(1, 2)
                   for y in self.qkv(x).split(c, 2)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(b, t, c))


class Blk(nn.Module):
    def __init__(self, e: int, h: int) -> None:
        super().__init__()
        self.l1 = nn.LayerNorm(e)
        self.a = Attn(e, h)
        self.l2 = nn.LayerNorm(e)
        self.m = nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(),
                               nn.Linear(4 * e, e))

    def forward(self, x):
        x = x + self.a(self.l1(x))
        return x + self.m(self.l2(x))


class LM(nn.Module):
    def __init__(self, v: int, t: int, l: int, h: int, e: int) -> None:
        super().__init__()
        self.t = nn.Embedding(v, e)
        self.p = nn.Embedding(t, e)
        self.b = nn.ModuleList([Blk(e, h) for _ in range(l)])
        self.n = nn.LayerNorm(e)
        self.h = nn.Linear(e, v, bias=False)
        self.h.weight = self.t.weight          # tied embedding

    def forward(self, idx):
        x = self.t(idx) + self.p(torch.arange(idx.size(1), device=idx.device))
        for blk in self.b:
            x = blk(x)
        return self.h(self.n(x))


class Local400:
    """加载一次，反复用。首次加载 4.5G 权重约 20 秒。"""

    def __init__(self, ckpt: Path | str | None = None,
                 vocab: Path | str | None = None,
                 device: Optional[str] = None) -> None:
        self.ckpt_path = Path(ckpt) if ckpt else DEFAULT_CKPT
        vp = Path(vocab) if vocab else (DEFAULT_VOCAB if DEFAULT_VOCAB.exists()
                                        else FALLBACK_VOCAB)
        self.vocab: List[str] = json.loads(vp.read_text("utf-8"))
        self.w = {c: i for i, c in enumerate(self.vocab)}

        ck = torch.load(self.ckpt_path, map_location="cpu")
        l, h, e = ck["cfg"][:3]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = LM(len(self.vocab), 512, l, h, e)
        self.m.load_state_dict(ck["m"])
        self.m.to(self.dev).eval()
        self.step = int(ck.get("step", 0))
        del ck

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 60,
                 temperature: float = 0.85, top_k: int = 40,
                 rep: float = 1.4, window: int = 48) -> str:
        """rep = 重复惩罚，window = 回顾窗口。

        400M 在 temperature < 0.5 时会陷进复读 —— 不加惩罚它每次都
        挑同一个最高概率字，于是「你是 你选 你是 你选」。rep=1.4 /
        window=48 实测能打断循环，又不至于把正常重复（的、了）罚成乱码。
        """
        ids = [self.w.get(c, 0) for c in (prompt or "")[-400:]]
        if not ids:
            return ""
        x = torch.tensor([ids], device=self.dev)
        for _ in range(max_new):
            logits = self.m(x[:, -512:])[:, -1] / max(0.05, temperature)
            if rep != 1.0:
                for t in set(x[0, -window:].tolist()):
                    v = logits[0, t]
                    logits[0, t] = v / rep if v > 0 else v * rep
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return "".join(self.vocab[i] for i in x[0, len(ids):].tolist())

    def chat(self, text: str, context=None, system=None) -> str:
        """接口对齐 core.remote.RemoteBrain，让上层不用改。

        预训练模型不懂对话格式，这里只做「接着往下写」，
        真正的问答仍该走远端。返回空串表示本地给不出答案。
        """
        if not text or len(text) < 2:
            return ""
        seed = (context or [])[-1][:60] if context else ""
        return self.generate(seed + text, max_new=50)


if __name__ == "__main__":
    m = Local400()
    print(f"step={m.step} device={m.dev} vocab={len(m.vocab)}")
    print(m.generate("人工智能的未来是", max_new=50))
