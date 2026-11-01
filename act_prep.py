"""act_train.jsonl → uint16 ids + mask。

只对「答：」之后的部分算 loss —— 问句和状态是条件，不是要学的内容。
和 sft_prep.py 一个思路，区别是这里学的是结构化 JSON，不是自然语言回答。

用法：
    python3 act_prep.py                      # 默认读 home/act_train.jsonl
    python3 act_prep.py 数据.jsonl 词表.json   # 指定输入和词表
输出：home/act_ids.npy、home/act_mask.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "home" / "act_train.jsonl"
VOCAB = (Path(sys.argv[2]) if len(sys.argv) > 2
         else Path("/data/vocab.json"))
OUT_IDS = ROOT / "home" / "act_ids.npy"
OUT_MSK = ROOT / "home" / "act_mask.npy"
SEP = "答："


def load_vocab(path: Path) -> list[str]:
    """优先读给定词表；读不到就从数据里现造一个，保证脚本能自测。"""
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    chars = set()
    for line in SRC.read_text("utf-8").split("\n"):
        chars.update(line)
    chars.discard("\n")
    return sorted(chars)


def main() -> int:
    vocab = load_vocab(VOCAB)
    w = {c: i for i, c in enumerate(vocab)}
    print(f"词表 {len(vocab)} 字，来源 {'文件' if VOCAB.exists() else '数据现造'}")

    ids: list[int] = []
    mask: list[int] = []
    kept = 0
    for block in SRC.read_text("utf-8").split("\n\n"):
        b = block.strip()
        if not b or SEP not in b:
            continue
        cut = b.index(SEP) + len(SEP)
        head, body = b[:cut], b[cut:]
        hi = [w.get(c, 0) for c in head]
        bi = [w.get(c, 0) for c in body] + [0]        # 0 当结束符
        ids += hi + bi
        mask += [0] * len(hi) + [1] * len(bi)
        kept += 1

    np.save(OUT_IDS, np.array(ids, dtype=np.uint16))
    np.save(OUT_MSK, np.array(mask, dtype=np.uint8))
    n_loss = sum(mask)
    print(f"样本 {kept}  token {len(ids)}  算 loss {n_loss} ({n_loss / max(1, len(ids)):.1%})")
    print(f"写出 {OUT_IDS.name} {OUT_MSK.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
