#!/usr/bin/env python3
"""语料构建：把外部纯文本并进 home/corpus.txt。

这是**离线数据准备**，不是运行期的学习路径 —— 运行期学到的必须走门槛
（deposit → leave → _absorb），这里只是把训练集做大，方便反复训练。

    ./bin/py tools/build_corpus.py --source /tmp/sanguo.txt --chars 50000
    ./bin/py tools/build_corpus.py --source a.txt --source b.txt --chars 20000

行为：
  - 去掉 Gutenberg 头尾、CR、连续空行
  - 新块过一遍 instinct.scan，命中就整块拒绝（不把脏东西写进自我）
  - 旧语料默认保留在前，写之前备份成 corpus.prev.txt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import instinct  # noqa: E402


def clean(text: str) -> str:
    i = text.find("*** START OF")
    if i >= 0:
        text = text[i:].split("\n", 1)[1] if "\n" in text[i:] else text[i:]
    j = text.find("*** END OF")
    if j >= 0:
        text = text[:j]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--chars", type=int, default=50000, help="每个来源最多取多少字符")
    p.add_argument("--corpus", default=str(ROOT / "home" / "corpus.txt"))
    p.add_argument("--fresh", action="store_true", help="不保留旧语料")
    a = p.parse_args()

    corpus = Path(a.corpus)
    old = "" if a.fresh or not corpus.exists() else corpus.read_text("utf-8")
    seen = {hashlib.sha256(old.encode()).hexdigest()[:12]} if old else set()

    chunks = []
    for src in a.source:
        raw = Path(src).read_text("utf-8", errors="replace")
        text = clean(raw)[:a.chars]
        if not text:
            print(f"[corpus] {src} 清洗后是空的，跳过", file=sys.stderr)
            continue
        verdict = instinct.scan(text)
        if verdict:
            print(f"[corpus] {src} 没过本能闸：{verdict.category}，整块拒绝", file=sys.stderr)
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()[:12]
        if sha in seen:
            print(f"[corpus] {src} 已在语料里，跳过", file=sys.stderr)
            continue
        seen.add(sha)
        chunks.append(text)
        print(f"[corpus] +{len(text)} 字符  {src}")

    if not chunks:
        print("[corpus] 没有新增内容", file=sys.stderr)
        return 1

    merged = (old.rstrip() + "\n\n" if old else "") + "\n\n".join(chunks) + "\n"
    if corpus.exists():
        prev = corpus.with_name(corpus.name.replace(".txt", ".prev.txt"))
        prev.write_text(old, "utf-8")
        print(f"[corpus] 旧语料备份 → {prev.name}")

    tmp = corpus.with_name(corpus.name + ".tmp")
    tmp.write_text(merged, "utf-8")
    os.replace(tmp, corpus)

    chars = len(merged)
    vocab = len(set(merged))
    print(f"[corpus] 总计 {chars} 字符  {len(merged.encode())} 字节  vocab={vocab} → {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
