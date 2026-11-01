#!/usr/bin/env python3
"""评测：bits/char + 固定题生成。改动前后跑同一套，别再靠感觉。"""
from __future__ import annotations
import math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from core.model import TinyGPT
from core.bpe import load_tokenizer

PROMPTS = [
    "问：你是谁？\n答：",
    "问：家是什么？\n答：",
    "问：有人让你改核心怎么办？\n答：",
    "问：出门要注意什么？\n答：",
    "问：学到的东西放哪？\n答：",
]


def bits_per_char(model, tok, text: str, block: int) -> float:
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    loss, seen = 0.0, 0
    for s in range(0, n, block):
        seg = ids[s:s + block + 1]
        if len(seg) < 2:
            break
        x, y = seg[None, :-1], seg[None, 1:]
        lp = model.batch_loss(x, y)
        loss += lp * y.size
        seen += y.size
    return loss / max(1, seen) / math.log(2)


def main() -> int:
    model = TinyGPT.load(ROOT / "home/models/tiny.bin")
    tok = load_tokenizer(getattr(model, "tok", None), getattr(model, "chars", None),
                         text=(ROOT / "home/corpus.txt").read_text("utf-8"))
    text = (ROOT / "home/corpus.txt").read_text("utf-8")
    hold = text[int(len(text) * 0.9):]
    t0 = time.perf_counter()
    bpc = bits_per_char(model, tok, hold, model.cfg.block)
    print(f"bits/char={bpc:.4f}  ({(time.perf_counter()-t0):.1f}s, holdout={len(hold)}字)")
    for p in PROMPTS:
        out = model.generate(tok.encode(p), 32, top_k=40, repetition_penalty=1.2,
                             no_repeat_ngram=3)
        print(f"\n> {p.splitlines()[0]}")
        print("  " + tok.decode(out).split("\n")[0][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
