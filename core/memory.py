"""分层记忆：像脑子那样记，而不是把每句话都塞进语料。

    工作记忆  home/state/short_term.json   有界、会衰减、重复才留
       ↓ 重复出现（count≥2）或一次就很重
    门外      threshold/*.json            还没成为"我"
       ↓ 离家那一刻
    语义记忆  home/live facts + corpus     也就是"本身"

一次性寒暄、短问句、重复内容 → 丢弃，只在 wolf.log 留一行 memory.discard。
核心（canon.py）不在这条链上：学习只能碰到知识层，碰核心的输入直接记
core.write_refused，不落盘。

recall() 是"运用"那一半：回答前先按字面重合度把相关记忆捞出来，交给推理层，
而不是指望 560k 参数的模型自己想起来。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from . import canon, instinct
from core.atomic import atomic_write

FILLER = {
    "你好", "您好", "hi", "hello", "hey", "在吗", "在么", "在不在", "谢谢", "多谢",
    "早", "早安", "晚安", "拜拜", "再见", "好的", "好", "嗯", "哦", "啊", "哈",
    "哈哈", "ok", "okay", "test", "测试",
}

# 分层门槛：低于 WORKING 直接丢，高于 SEMANTIC 直接进门外
WORKING = 0.25
SEMANTIC = 0.55
PROMOTE_COUNT = 2


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _bigrams(text: str) -> set:
    t = "".join(text.split())
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _similar(x: str, y: str, cut: float = 0.6) -> bool:
    """模式分离：两条记忆太像就是同一条。

    纯 Jaccard 对短句太严 ——「昨天说要去公园」和「昨天说过要去公园」
    只差一个字，Jaccard 就掉到 0.625。所以先看长度：差超过 30% 直接
    判不同（「我喜欢喝咖啡」vs「我喜欢喝茶」长度接近但语义不同，靠 0.6
    的门槛挡住），长度接近才比 bigram 重合。
    """
    if abs(len(x) - len(y)) > max(2, int(len(x) * 0.3)):
        return False
    return _jaccard(_bigrams(x), _bigrams(y)) > cut


@dataclass
class Entry:
    text: str
    source: str
    kind: str
    salience: float
    hits: List[str]
    count: int = 1
    first: float = field(default_factory=time.time)
    last: float = field(default_factory=time.time)

    @property
    def sha(self) -> str:
        return _sha(self.text)

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "source": self.source, "kind": self.kind,
                "salience": round(self.salience, 3), "hits": self.hits,
                "count": self.count, "first": self.first, "last": self.last}


class MemoryStore:
    def __init__(self, home, ttl_s: float = 86400.0, max_working: int = 50) -> None:
        self.home = home
        self.ttl_s = ttl_s
        self.max_working = max_working
        self.discarded = 0
        self.refused = 0
        self.promoted = 0
        self.quarantined = 0
        self._path = home.state_dir / "short_term.json"

    # ---------- 工作记忆存取 ----------
    def _load(self) -> List[Entry]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text("utf-8"))
        except (OSError, ValueError):
            return []
        out: List[Entry] = []
        for d in raw.get("entries", []):
            try:
                out.append(Entry(**{k: d[k] for k in
                                    ("text", "source", "kind", "salience", "hits",
                                     "count", "first", "last")}))
            except (KeyError, TypeError):
                continue
        return out

    def _save(self, entries: List[Entry]) -> None:
        entries = sorted(entries, key=lambda e: (e.salience, e.last), reverse=True)
        entries = entries[:self.max_working]
        atomic_write(self._path, json.dumps(
            {"entries": [e.as_dict() for e in entries]}, ensure_ascii=False, indent=2))

    # ---------- 打分 ----------
    def score(self, text: str, source: str = "", kind: str = "chat") -> Dict[str, Any]:
        t = (text or "").strip()
        if not t:
            return {"salience": 0.0, "tier": "discard", "hits": [], "why": "empty"}
        low = t.lower()
        if low in FILLER or len(t) <= 4:
            return {"salience": 0.05, "tier": "discard", "hits": [], "why": "filler"}

        hits = canon.themes_hit(t)
        s = min(0.45, 0.18 * len(hits))          # 与核心相关，才值得留
        s += min(0.20, len(t) / 400.0)           # 信息量
        s += 0.15 * (len(set(t)) / len(t))       # 字符多样性，复读机拿不到分
        if kind == "knowledge":
            s += 0.20
        if source.startswith("http"):
            s += 0.10
        if len(t) < 30 and ("？" in t or "?" in t):
            s -= 0.15                            # 短问句不是知识
        s = max(0.0, min(1.0, s))

        if s >= SEMANTIC:
            tier = "semantic"
        elif s >= WORKING:
            tier = "working"
        else:
            tier = "discard"
        return {"salience": round(s, 3), "tier": tier, "hits": hits, "why": "scored"}

    # ---------- 记 ----------
    def remember(self, text: str, source: str = "", kind: str = "chat") -> Dict[str, Any]:
        t = (text or "").strip()
        if canon.looks_like_core_write(t):
            self.refused += 1
            self.home.log("core.write_refused", f"{source or '-'}\t{t[:80]}")
            return {"tier": "core_refused", "salience": 0.0, "hits": [],
                    "why": "学习不能写核心"}

        bad = instinct.quarantines(t)
        if bad is not None:
            # 坏观念没到 BLOCK 也不能当知识存 —— 存进去就会被下次训练学走
            self.quarantined += 1
            self.home.log("memory.quarantine",
                          f"{bad.category}:{bad.level.name}\t{t[:60]}")
            return {"tier": "quarantine", "salience": 0.0, "hits": [],
                    "why": f"instinct:{bad.category}"}

        verdict = self.score(t, source, kind)
        if verdict["tier"] == "discard":
            self.discarded += 1
            self.home.log("memory.discard", f"{verdict['why']}\t{t[:60]}")
            return verdict

        if verdict["tier"] == "semantic":
            self._deposit(t, source, verdict)
            return verdict

        entries = self._load()
        now = time.time()
        for e in entries:
            # 完全相同，或相似度 >0.85，都当作同一条 —— 后者就是模式分离
            if e.text == t or _similar(e.text, t):
                e.text = t
                e.count += 1
                e.last = now
                e.salience = max(e.salience, verdict["salience"])
                if e.count >= PROMOTE_COUNT:
                    entries.remove(e)
                    self._save(entries)
                    self._deposit(t, source, verdict)
                    self.promoted += 1
                    verdict = dict(verdict, tier="semantic", why="promoted")
                    return verdict
                self._save(entries)
                return dict(verdict, tier="working", why="seen_again")

        entries.append(Entry(text=t, source=source, kind=kind,
                             salience=verdict["salience"], hits=verdict["hits"]))
        self._save(entries)
        return verdict

    def _deposit(self, text: str, source: str, verdict: Dict[str, Any]) -> None:
        self.home.deposit({
            "kind": "knowledge", "source": source or "memory",
            "sha": _sha(text), "text": text, "at": time.time(),
            "salience": verdict["salience"], "core_hits": verdict["hits"],
        })

    # ---------- 巩固 / 遗忘 ----------
    def consolidate(self) -> Dict[str, int]:
        now = time.time()
        entries = self._load()
        keep, dropped = [], 0
        for e in entries:
            if now - e.last > self.ttl_s:
                dropped += 1
                continue
            keep.append(e)
        merged = self._merge_similar(keep)
        self._save(keep)
        if dropped:
            self.home.log("memory.forget", f"ttl={dropped}")
        if merged:
            self.home.log("memory.merge", f"merged={merged}")
        return {"kept": len(keep), "forgot": dropped, "merged": merged}

    def _merge_similar(self, entries: List[Entry], th: float = 0.33) -> int:
        """把向量相近的短期记忆并成一条：保留更长的那句，合并计次并提升显著度。"""
        vs = _vector_store()
        if vs is None or len(entries) < 2:
            return 0
        try:
            vecs = vs.index.embed([e.text for e in entries])
        except (ValueError, MemoryError):
            return 0
        keep: List[Entry] = []
        idx: List[int] = []
        merged = 0
        for i, e in enumerate(entries):
            hit = -1
            for j, kj in enumerate(idx):
                if float(vecs[i] @ vecs[kj]) >= th:
                    hit = j
                    break
            if hit < 0:
                keep.append(e)
                idx.append(i)
                continue
            t = keep[hit]
            merged += 1
            t.count += e.count
            t.salience = min(1.0, max(t.salience, e.salience) + 0.05)
            if len(e.text) > len(t.text):
                t.text = e.text
        entries[:] = keep
        return merged

    # ---------- 运用 ----------
    def recall(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        q = _bigrams(query or "")
        if not q:
            return []
        pool: List[Dict[str, Any]] = []
        for e in self._load():
            pool.append({"text": e.text, "source": e.source, "tier": "working",
                         "salience": e.salience})
        for item in self.home.live.get("facts", []):
            pool.append({"text": str(item.get("text", "")),
                         "source": str(item.get("source", "home")), "tier": "semantic",
                         "salience": float(item.get("salience", 0.6))})
        tdir = self.home.threshold_dir
        if tdir.exists():
            for f in tdir.glob("*.json"):
                try:
                    d = json.loads(f.read_text("utf-8"))
                except (OSError, ValueError):
                    continue
                pool.append({"text": str(d.get("text", "")),
                             "source": str(d.get("source", "threshold")),
                             "tier": "threshold",
                             "salience": float(d.get("salience", 0.5))})

        vs = _vector_store()
        sims = None
        if vs is not None and pool:
            try:
                vecs = vs.index.embed([i["text"] for i in pool])
                sims = vecs @ vs.index.embed([query])[0]
            except (ValueError, MemoryError):
                sims = None
        scored = []
        for n, item in enumerate(pool):
            if not item["text"]:
                continue
            overlap = q & _bigrams(item["text"])
            kw = len(overlap) / max(1, len(q))
            # 向量通道：关键词为 0 时也能召回改写过的记忆
            vs_score = max(float(sims[n]), 0.0) if sims is not None else 0.0
            if kw <= 0.0 and vs_score < 0.45:
                continue
            score = 0.7 * kw + 0.3 * vs_score
            scored.append(dict(item, score=round(score, 3),
                               overlap=len(overlap)))
        scored.sort(key=lambda x: (x["score"], x["salience"]), reverse=True)
        return scored[:k]

    # ---------- 体检 ----------
    def stats(self) -> Dict[str, int]:
        return {"working": len(self._load()), "discarded": self.discarded,
                "refused": self.refused, "promoted": self.promoted,
                "quarantined": self.quarantined,
                "facts": len(self.home.live.get("facts", []))}


_VS_CACHE = []


def _vector_store():
    """懒加载语料级 LSA 索引；加载失败返回 None，关键词召回照旧可用。"""
    if not _VS_CACHE:
        try:
            from vector import Store as _S
            _VS_CACHE.append(_S())
        except (ImportError, OSError, ValueError):
            _VS_CACHE.append(None)
    return _VS_CACHE[0]
