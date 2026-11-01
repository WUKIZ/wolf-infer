"""小脑：习惯自动化。

同一类句子问过三次、三次都是同一个动作，那就固化成习惯 —— 别再问模型。
省的是每次 1–2 秒和一次请求的钱，不是精度。

阈值 0.9 比记忆（0.6）严得多，因为动作判错的代价比多存一条记忆高：
多存一条只是占地方，判错一次就是白出门一趟。

只在「完全不敢确定」时才复用：相似度 ≥0.9 才动手，0.6–0.9 照旧走模型。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_P = re.compile(r"[\s，。！？、；：,.!?;:]+")


def _bg(t: str) -> set[str]:
    s = _P.sub("", t.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def _like(a: str, b: str) -> float:
    x, y = _bg(a), _bg(b)
    return len(x & y) / max(1, len(x | y))


class Cerebellum:
    def __init__(self, path: Path | str, need: int = 3,
                 cut: float = 0.85, ans_cut: float = 0.95) -> None:
        self.p = Path(path)
        self.need = need
        self.cut = cut
        self.ans_cut = ans_cut          # 回答比动作更严：答错比选错动作贵
        self.h: list[dict] = []
        self.a: list[dict] = []          # 回答习惯：{q, ans, n, at}
        self.hits = 0
        self.ans_hits = 0
        if self.p.exists():
            try:
                loaded = json.loads(self.p.read_text("utf-8"))
                if isinstance(loaded, dict):
                    self.h = [x for x in loaded.get("acts", []) if isinstance(x, dict)]
                    self.a = [x for x in loaded.get("ans", []) if isinstance(x, dict)]
                elif isinstance(loaded, list):      # 旧格式兼容
                    self.h = [x for x in loaded if isinstance(x, dict)]
            except (OSError, ValueError):
                pass

    def _save(self) -> None:
        try:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.p.with_name(f".{self.p.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"acts": self.h[-200:], "ans": self.a[-200:]},
                                      ensure_ascii=False), "utf-8")
            os.replace(tmp, self.p)
        except OSError:
            pass

    def learn(self, text: str, action: str) -> None:
        """记一次「这句话 → 这个动作」。够了就固化。"""
        if not text or not action:
            return
        for item in self.h:
            if _like(item["text"], text) >= self.cut:
                if item["action"] == action:
                    item["n"] += 1
                else:
                    item["n"] = 0            # 动作变了，重新数
                    item["action"] = action
                item["text"] = text[:80]
                self._save()
                return
        self.h.append({"text": text[:80], "action": action, "n": 1})
        self._save()

    def recall(self, text: str) -> str | None:
        """够熟就返回固化动作，不够熟返回 None（交回模型）。"""
        if not text:
            return None
        best, top = None, 0.0
        for item in self.h:
            s = _like(item["text"], text)
            if s > top:
                best, top = item, s
        if best and top >= self.cut and best["n"] >= self.need:
            self.hits += 1
            return str(best["action"])
        return None

    def stats(self) -> dict[str, int]:
        return {"habits": sum(1 for i in self.h if i["n"] >= self.need),
                "hits": self.hits, "ans": len(self.a), "ans_hits": self.ans_hits}

    # ---------- 回答固化 ----------
    def learn_answer(self, q: str, ans: str, ttl: float = 7 * 86400) -> None:
        """记住「这类问题 → 这个回答」。

        只该在回答确实稳定时调用（比如自我介绍、固定说明）。带时效的、
        依赖当下的，学进来就是埋雷 —— 七天后自动不算数，是最后一道保险。
        """
        if not q or not ans:
            return
        import time
        now = time.time()
        for item in self.a:
            if _like(item["q"], q) >= self.ans_cut:
                item["q"] = q[:120]
                item["ans"] = ans[:600]
                item["n"] += 1
                item["at"] = now
                self._save()
                return
        self.a.append({"q": q[:120], "ans": ans[:600], "n": 1, "at": now})
        self._save()

    def recall_answer(self, q: str, ttl: float = 7 * 86400) -> str | None:
        """够像、够熟、没过期才复用，否则 None。"""
        import time
        now = time.time()
        best, top = None, 0.0
        for item in self.a:
            if now - float(item.get("at", 0)) > ttl:
                continue
            s = _like(item["q"], q)
            if s > top:
                best, top = item, s
        if best and top >= self.ans_cut and best["n"] >= 2:
            self.ans_hits += 1
            return str(best["ans"])
        return None
