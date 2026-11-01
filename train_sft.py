"""对话 SFT：教 400M 一问一答。

数据：home/../sft_ids.npy + sft_mask.npy（81.7 万条，1.25 亿 token）
损失：只在 mask=1 的位置算 —— 问句是条件，答案才是要学的
起点：wolf400M_old.pt（预训练权重）
产物：wolf400M_sft.pt

和动作 SFT 的区别：那个学 JSON 判断，这个学自然语言回答。
lr 比预训练低一档（2e-5），SFT 阶段动得越大越容易忘掉预训练学到的东西。

用法：python3 train_sft.py [步数]     默认 30000（约 1.2 亿 token）
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
ROOT = Path(__file__).resolve().parent
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
B, T = 8, 512
LR = 2e-5
BASE = D / "wolf400M_old.pt"
OUT = D / "wolf400M_sft.pt"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids_p = ROOT / "home" / "sft_ids.npy"
    msk_p = ROOT / "home" / "sft_mask.npy"
    # 数据也可能放在 home 根目录（登录节点生成的那份）
    if not ids_p.exists():
        ids_p = D / "sft_ids.npy"
        msk_p = D / "sft_mask.npy"
    if not ids_p.exists():
        print(f"缺 sft_ids.npy（找过 {ids_p}）")
        return 1

    ids = np.load(ids_p)
    msk = np.load(msk_p)
    print(f"数据 {len(ids):,} token，算 loss {int(msk.sum()):,}", flush=True)

    vocab = json.loads((D / "vocab.json").read_text("utf-8"))
    v = len(vocab)

    ck = torch.load(BASE, map_location="cpu")
    l, h, e = ck["cfg"][:3]
    mdl = LM(v, T, l, h, e)
    mdl.load_state_dict(ck["m"])
    print(f"从 wolf400M_old.pt 起步（step {ck.get('step', 0)}）", flush=True)
    del ck
    mdl = mdl.to(dev).train()
    print(f"params {sum(p.numel() for p in mdl.parameters()) / 1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def batch():
        pick = np.random.randint(0, len(ids) - T - 1, size=B)
        x = np.stack([ids[i:i + T] for i in pick]).astype(np.int64)
        y = np.stack([ids[i + 1:i + T + 1] for i in pick]).astype(np.int64)
        m = np.stack([msk[i + 1:i + T + 1] for i in pick]).astype(np.float32)
        to = lambda a: torch.from_numpy(a).to(dev)                   # noqa: E731
        return to(x), to(y), to(m)

    t0 = time.time()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 200) * (
            0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr

        x, y, m = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = mdl(x)
        raw = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                              reduction="none")
        loss = (raw * m.reshape(-1)).sum() / m.sum().clamp(min=1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(step, "loss %.4f" % loss.item(), "%.1fs" % (time.time() - t0),
                  flush=True)
        if step % 1000 == 0 and step:
            tmp = OUT.with_suffix(".tmp")
            torch.save({"m": mdl.state_dict(), "step": step, "cfg": [l, h, e]}, tmp)
            tmp.replace(OUT)
            print(f"saved {OUT.name} @ {step}", flush=True)

    torch.save({"m": mdl.state_dict(), "step": STEPS, "cfg": [l, h, e]}, OUT)
    print(f"完成，写出 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
