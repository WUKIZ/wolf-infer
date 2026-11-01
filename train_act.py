"""动作 SFT：教 400M「看情境出动作」。

数据：home/act_ids.npy + act_mask.npy（由 act_prep.py 生成）
损失：只在 mask=1 的位置算 —— 问句和状态是条件，不是要背的内容
起点：默认从 wolf400M_old.pt 接着训；没有就从随机权重开始
产物：wolf400M_act.pt，不覆盖预训练权重

用法：
    python3 train_act.py [步数]        默认 3000 步
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

from local400 import LM                      # 复用同一份架构定义

D = Path("/data")
ROOT = Path(__file__).resolve().parent
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
B, T = 8, 512
LR = 3e-5
BASE_CKPT = D / "wolf400M_old.pt"
OUT = D / "wolf400M_act.pt"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids_path = ROOT / "home" / "act_ids.npy"
    msk_path = ROOT / "home" / "act_mask.npy"
    if not ids_path.exists():
        print("缺 act_ids.npy —— 先跑 act_prep.py")
        return 1

    ids = np.load(ids_path)
    msk = np.load(msk_path)
    print(f"数据 {len(ids):,} token，其中算 loss {int(msk.sum()):,}", flush=True)

    vocab = json.loads((D / "vocab.json").read_text("utf-8"))
    v = len(vocab)

    # 从预训练权重起步，没有就随机初始化
    if BASE_CKPT.exists():
        ck = torch.load(BASE_CKPT, map_location="cpu")
        l, h, e = ck["cfg"][:3]
        mdl = LM(v, T, l, h, e)
        mdl.load_state_dict(ck["m"])
        print(f"从 wolf400M_old.pt 起步（step {ck.get('step', 0)}）", flush=True)
        del ck
    else:
        l, h, e = 20, 20, 1280
        mdl = LM(v, T, l, h, e)
        print("没有预训练权重，随机初始化", flush=True)

    mdl = mdl.to(dev)
    print(f"params {sum(p.numel() for p in mdl.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def batch():
        n = len(ids)
        pick = np.random.randint(0, n - T - 1, size=B)
        x = np.stack([ids[i:i + T] for i in pick]).astype(np.int64)
        y = np.stack([ids[i + 1:i + T + 1] for i in pick]).astype(np.int64)
        m = np.stack([msk[i + 1:i + T + 1] for i in pick]).astype(np.float32)
        t = lambda a: torch.from_numpy(a).to(dev)                    # noqa: E731
        return t(x), t(y), t(m)

    t0 = time.time()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 100) * (
            0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr

        x, y, m = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = mdl(x)
        loss_raw = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                   reduction="none")
        loss = (loss_raw * m.reshape(-1)).sum() / m.sum().clamp(min=1.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(step, "loss %.4f" % loss.item(), "%.1fs" % (time.time() - t0),
                  flush=True)
        if step % 500 == 0 and step:
            torch.save({"m": mdl.state_dict(), "step": step,
                        "cfg": [l, h, e]}, OUT)
            print(f"saved {OUT.name} @ {step}", flush=True)

    torch.save({"m": mdl.state_dict(), "step": STEPS, "cfg": [l, h, e]}, OUT)
    print(f"完成，写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
