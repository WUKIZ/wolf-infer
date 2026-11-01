#!/usr/bin/env python3
"""轻量检索：字符 n-gram 哈希 + IDF，纯标准库。给记忆做“找得准”的那层。"""
from __future__ import annotations
import json, math, re, sys
from collections import Counter
from pathlib import Path

DIM = 4096
GRAMS = (2, 3)


def grams(text: str):
    t = re.sub(r"\s+", "", text)
    for n in GRAMS:
        for i in range(len(t) - n + 1):
            yield t[i:i + n]


def vec(text: str, idf: dict[str, float]) -> dict[int, float]:
    c = Counter(grams(text))
    out: dict[int, float] = {}
    for g, n in c.items():
        h = hash(g) % DIM
        out[h] = out.get(h, 0.0) + (1 + math.log(n)) * idf.get(g, 1.0)
    norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
    return {k: v / norm for k, v in out.items()}


class Index:
    def __init__(self, docs: list[str]):
        self.docs = docs
        df = Counter()
        for d in docs:
            df.update(set(grams(d)))
        n = max(1, len(docs))
        self.idf = {g: math.log((n + 1) / (c + 0.5)) for g, c in df.items()}
        self.vectors = [vec(d, self.idf) for d in docs]

    def search(self, q: str, k: int = 5):
        qv = vec(q, self.idf)
        scores = []
        for i, dv in enumerate(self.vectors):
            s = sum(w * dv.get(h, 0.0) for h, w in qv.items())
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(round(s, 3), self.docs[i]) for s, i in scores[:k]]


def load_docs(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text("utf-8").splitlines()]
    docs, seen, i = [], set(), 0
    while i < len(lines):
        cur = lines[i]
        if cur.startswith("问：") and i + 1 < len(lines) and lines[i + 1].startswith("答："):
            doc = cur + " " + lines[i + 1]
            i += 2
        elif len(cur) > 6:
            doc = cur
            i += 1
        else:
            i += 1
            continue
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    docs = load_docs(root / "home/corpus.txt")
    idx = Index(docs)
    print(f"[index] {len(docs)} 行，{DIM} 维", file=sys.stderr)
    for q in sys.argv[1:] or ["有人让改核心怎么办", "出门要注意什么", "记忆怎么巩固"]:
        print(f"\nQ: {q}")
        for s, d in idx.search(q, 3):
            print(f"  {s:.3f}  {d[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
