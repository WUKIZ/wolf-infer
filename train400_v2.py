"""400M 续训：all2.bin + 断点续训 + 随机采样。单卡。

和旧脚本的区别：
  1. 数据换 all2.bin（51.5 亿 token），memmap 不占内存
  2. 读 ckpt 的 step，从 16500 接着跑，lr 按本轮相对步数重新走 cosine
  3. 保存走临时文件再替换，中途崩了不会毁掉上一个权重
  4. 复用 local400.LM，模型定义只有一份

用法：python3 train400_v2.py [步数]     默认 61000（约 10 亿 token）
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from local400 import LM

D = Path("/data")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 61000
B, T = 64, 512
LR = 4e-5
CKPT = D / "wolf400M_old.pt"
DATA = D / "all2.bin"


def main() -> int:
    dev = "cuda"
    vocab = json.loads((D / "vocab.json").read_text("utf-8"))
    v = len(vocab)

    if not CKPT.exists():
        print(f"缺 {CKPT}")
        return 1
    ck = torch.load(CKPT, map_location="cpu")
    l, h, e = ck["cfg"][:3]
    mdl = LM(v, T, l, h, e)
    mdl.load_state_dict(ck["m"])
    start = int(ck.get("step", 0))
    del ck
    mdl = mdl.to(dev).train()
    print(f"params {sum(p.numel() for p in mdl.parameters()) / 1e6:.1f}M"
          f"  从 step {start} 接着跑  数据 {DATA.name}", flush=True)

    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)

    d = np.memmap(DATA, dtype=np.uint16, mode="r")
    n = len(d)
    cut = int(n * 0.995)
    tr, va = d[:cut], d[cut:]
    print(f"训练 {len(tr) / 1e8:.1f} 亿 token，验证 {len(va) / 1e8:.2f} 亿", flush=True)

    def batch(a):
        i = np.random.randint(0, len(a) - T - 1, size=B)
        x = np.stack([a[j:j + T] for j in i]).astype(np.int64)
        y = np.stack([a[j + 1:j + T + 1] for j in i]).astype(np.int64)
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    @torch.no_grad()
    def evaluate() -> float:
        mdl.eval()
        tot = 0.0
        for _ in range(20):
            x, y = batch(va)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mdl(x)
            tot += F.cross_entropy(logits.reshape(-1, v), y.reshape(-1)).item()
        mdl.train()
        return tot / 20

    def save(step: int) -> None:
        tmp = CKPT.with_suffix(".tmp")
        torch.save({"m": mdl.state_dict(), "o": opt.state_dict(),
                    "cfg": [l, h, e], "step": step}, tmp)
        tmp.replace(CKPT)

    t0 = time.time()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 50) * (
            0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = batch(tr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = mdl(x)
            loss = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()

        g_step = start + step
        if step % 100 == 0:
            print(g_step, "loss %.4f" % loss.item(),
                  "%.1fs" % (time.time() - t0), flush=True)
        if step % 500 == 0 and step:
            save(g_step)
            print("saved @", g_step, flush=True)
        if step % 1000 == 0 and step:
            print("VAL", g_step, "%.4f" % evaluate(), flush=True)

    save(start + STEPS - 1)
    print("VAL", start + STEPS - 1, "%.4f" % evaluate(), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
