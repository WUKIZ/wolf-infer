"""丘脑：注意力闸门。

之前是「搜到什么塞什么」—— 5 条结果、4000 字全进 context，让模型自己
在里面捞。丘脑先捞一遍：按 query 与正文的 bigram 重合度排序，只放前 k 条。

和 memory.recall 用同一套字符 bigram，不做 embedding：便宜、可解释。
真语义要等 embedding 模型，那是另一件事。
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[\s，。！？、；：\u201c\u201d\u2018\u2019（）【】…—·,.!?;:\"'()\[\]]+")


def _bg(text: str) -> set[str]:
    t = _PUNCT.sub("", text.lower())
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}


class Thalamus:
    def __init__(self, k: int = 4, floor: float = 0.04) -> None:
        self.k = k
        self.floor = floor
        self.passed = 0
        self.dropped = 0

    def gate(self, query: str, items: list[str]) -> list[str]:
        """重合一排序，取前 k。一条都不挨边就原样给前 k —— 空手更糟。"""
        if not items:
            return []
        q = _bg(query or "")
        if not q:
            self.passed += min(self.k, len(items))
            return items[:self.k]

        scored = []
        for it in items:
            if not it:
                continue
            overlap = q & _bg(it)
            if not overlap:
                continue
            scored.append((len(overlap) / max(1, len(q)), it))

        if not scored:
            self.dropped += max(0, len(items) - self.k)
            self.passed += min(self.k, len(items))
            return items[:self.k]

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [t for s, t in scored if s >= self.floor][:self.k]
        if not keep:
            keep = [t for _, t in scored[:self.k]]

        self.dropped += max(0, len(items) - len(keep))
        self.passed += len(keep)
        return keep

    def stats(self) -> dict[str, int]:
        return {"passed": self.passed, "dropped": self.dropped}
