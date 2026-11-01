"""零依赖向量检索：中文 2-gram + TF-IDF + 截断 SVD（LSA）。

容器无外网装不了 embedding 模型，用 LSA 从语料自身学出 128 维稠密向量，
召回比 BM25 抗改写，但仍弱于 bge 这类预训练模型。接口与 bge 对齐：
以后有外网只换 `LSA.embed`，检索逻辑不动。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "home" / "corpus.txt"
MAX_DOCS = 2000
DOC_CHARS = 400
VOCAB = 20000
DIM = 128


def tok(text: str) -> List[str]:
    """中文 2-gram，英文/数字按词。"""
    t = (text or "").lower()
    out = re.findall(r"[a-z0-9]+", t)
    for run in re.findall(r"[\u4e00-\u9fff]+", t):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def chunks(path: Path, size: int = DOC_CHARS) -> List[str]:
    if not path.exists():
        return []
    text = path.read_text("utf-8", errors="ignore")
    return [text[i:i + size] for i in range(0, len(text), size)][:MAX_DOCS]


class LSA:
    """语料内建的潜在语义索引；语料变了要重建。"""

    def __init__(self, docs: List[str], dim: int = DIM) -> None:
        self.docs = docs
        self.dim = min(dim, max(1, len(docs) - 1)) if docs else 1
        df, toks = {}, []
        for d in docs:
            tk = tok(d)
            toks.append(tk)
            for w in set(tk):
                df[w] = df.get(w, 0) + 1
        keep = sorted(df, key=df.get, reverse=True)[:VOCAB]
        self.vocab = {w: i for i, w in enumerate(keep)}
        n, v = len(docs), len(self.vocab)
        self.idf = (np.array([np.log((n + 1) / (df[w] + 0.5)) for w in keep], dtype=np.float32)
                    if v else np.zeros(0))
        x = np.zeros((max(n, 1), max(v, 1)), dtype=np.float32)
        for i, tk in enumerate(toks):
            for w, c in _count(tk).items():
                j = self.vocab.get(w)
                if j is not None:
                    x[i, j] = (1.0 + np.log(c)) * self.idf[j]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        self.x = x                       # 稀疏侧留一份，用于精确匹配
        if n > 1 and v > 1:
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            k = min(self.dim, len(s))
            dv = u[:, :k] * s[:k]
            self.doc_vec = dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
            self.proj = vt[:k].T
        else:
            self.doc_vec = x
            self.proj = np.eye(x.shape[1], dtype=np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        """(n, dim) 的 L2 归一化稠密向量。"""
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, c in _count(tok(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    out[i, j] = (1.0 + np.log(c)) * self.idf[j]
        if out.size:
            out = out @ self.proj
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out

    def top(self, query: str, k: int = 3):
        if not self.docs:
            return []
        q_dense = self.embed([query])[0]
        sims = self.doc_vec @ q_dense
        q_sparse = self._sparse([query])[0]
        if q_sparse.any():
            # 混合：稀疏保精确匹配，LSA 补改写召回
            sims = 0.5 * (self.x @ q_sparse) + 0.5 * sims
        return [(float(sims[i]), int(i)) for i in np.argsort(-sims)[:k]]

    def _sparse(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, c in _count(tok(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    out[i, j] = (1.0 + np.log(c)) * self.idf[j]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _count(items: List[str]):
    d = {}
    for w in items:
        d[w] = d.get(w, 0) + 1
    return d


class Store:
    """语料检索入口：chunks() 切段后建索引，search() 返回最相关文档。"""

    def __init__(self, path: Path | None = None) -> None:
        self.docs = chunks(path or CORPUS)
        self.index = LSA(self.docs)

    def search(self, query: str, k: int = 3) -> List[str]:
        return [self.docs[i] for _, i in self.index.top(query, k)]


if __name__ == "__main__":
    s = Store()
    print(f"语料 {len(s.docs)} 段，词表 {len(s.index.vocab)}")
    for q in ("本地助手 核心规则", "怎么训练模型", "回家 暴露"):
        print(f"\n== {q}")
        for h in s.search(q, 2):
            print("   ", h[:70].replace("\n", " "))
