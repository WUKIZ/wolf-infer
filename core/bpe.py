"""字节级 BPE 分词器。

为什么换掉 char 级：char 级下一个汉字就是一个 token，`block=48` 只装得下
十几个词，模型看不见句子结构。BPE 把高频字节对并成整块，同样的窗口能装
两到三倍的内容。

为什么是**字节**级：词表只有 256 个起点 + 合并出来的块，任何输入（生僻字、
emoji、乱码、代码）都能编码成 id，不会出现 OOV。char 级的 `encode` 遇到
没见过的字只能落到 id 0，那是静默损坏。

训练与推理用同一份合并表，表随权重一起存进 .bin（`tok` 字段），
所以换语料不会把旧 id 顶出 wte 边界。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

BASE = 256  # 0..255 是原始字节

# 编码与训练共用的切段标点：中文标点后断开，标点跟着前一段。
_PUNCT = re.compile(r"[。！？；：，、（）「」]")


def split_segments(text: str) -> List[str]:
    """切段，但保证「拼回去等于原文」—— 分词器丢了字符是不可接受的静默损坏。

    每行按中文标点切开，标点留在前一段；换行自己成段。
    """
    out: List[str] = []
    for chunk in re.split(r"(\n)", text):
        if not chunk:
            continue
        if chunk == "\n":
            out.append(chunk)
            continue
        start = 0
        for m in _PUNCT.finditer(chunk):
            out.append(chunk[start:m.end()])
            start = m.end()
        if start < len(chunk):
            out.append(chunk[start:])
    return out


class BPETokenizer:
    """合并表 + 贪心应用。编码结果只取决于合并表，与训练顺序无关地可复现。"""

    def __init__(self, merges: Sequence[Sequence[int]], vocab: Optional[int] = None) -> None:
        self.merges: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self.ranks: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        # 每个合并 id 展开成原始字节，decode 时一次查表，不递归
        self._expand: List[bytes] = [bytes([i]) for i in range(BASE)]
        for a, b in self.merges:
            self._expand.append(self._expand[a] + self._expand[b])
        self._vocab = int(vocab) if vocab is not None else BASE + len(self.merges)
        self._cache: Dict[str, List[int]] = {}

    # ---------------- 训练 ----------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 2048) -> "BPETokenizer":
        vocab_size = max(BASE + 1, int(vocab_size))
        counts = Counter(s.encode("utf-8") for s in split_segments(text))
        words: Dict[Tuple[int, ...], int] = {tuple(w): c for w, c in counts.items()}
        merges: List[Tuple[int, int]] = []
        num = BASE
        while num < vocab_size:
            pairs: Counter = Counter()
            for w, c in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += c
            if not pairs:
                break
            # 同频取字典序最小的那一对：同一份语料必须训出同一张表
            best, _ = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            a, b = best
            merged: Dict[Tuple[int, ...], int] = {}
            for w, c in words.items():
                if a in w and b in w:
                    out, i = [], 0
                    while i < len(w):
                        if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                            out.append(num)
                            i += 2
                        else:
                            out.append(w[i])
                            i += 1
                    w = tuple(out)
                merged[w] = merged.get(w, 0) + c
            words = merged
            merges.append((a, b))
            num += 1
        return cls(merges, vocab=num)

    # ---------------- 编码 ----------------
    def _apply(self, ids: List[int]) -> List[int]:
        ids = list(ids)
        while len(ids) > 1:
            best_rank, best_at = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_at = r, i
            if best_rank is None:
                break
            ids[best_at:best_at + 2] = [BASE + best_rank]
        return ids

    def encode(self, s: str) -> List[int]:
        # 逐段编码：合并只发生在段内，和训练时一致；整串丢进 _apply 是 O(n²)，
        # 几十万字符的语料会直接挂住（第一版就这么挂的）。
        out: List[int] = []
        for seg in split_segments(s):
            hit = self._cache.get(seg)
            if hit is None:
                hit = self._apply(list(seg.encode("utf-8")))
                if len(self._cache) < 20000:        # 有界：语料里短句多，命中率够高
                    self._cache[seg] = list(hit)
            out.extend(hit)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        raw = b"".join(self._expand[i] if 0 <= i < len(self._expand) else b"?" for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab(self) -> int:
        return self._vocab

    # ---------------- 存取 ----------------
    def to_payload(self) -> dict:
        return {"kind": "bpe", "vocab": self._vocab,
                "merges": [[a, b] for a, b in self.merges]}

    @classmethod
    def from_payload(cls, payload: dict) -> "BPETokenizer":
        return cls(payload.get("merges") or [], vocab=payload.get("vocab"))

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab={self._vocab}, merges={len(self.merges)})"


def load_tokenizer(tok_payload: Optional[dict], chars: Optional[List[str]] = None,
                   text: Optional[str] = None):
    """按权重里存的 payload 还原分词器；老的 char 级 checkpoint 走原路。"""
    if tok_payload and tok_payload.get("kind") == "bpe":
        return BPETokenizer.from_payload(tok_payload)
    from .model import CharTokenizer
    if chars:
        return CharTokenizer.from_chars(chars)
    return CharTokenizer(text or "")
