"""远端回答的精确缓存：同样的 (模型+system+问题) 只付一次钱。

不叫「语义缓存」：那要 embedding 模型，手机上不划算。精确命中已经能挡掉
脚本重跑、重复提问、定时任务这些大头，比压缩提示词省得更直接。
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

TTL_S = 6 * 3600


class ReplyCache:
    def __init__(self, path: Path, ttl_s: float = TTL_S) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, at REAL, v TEXT)")
        self.db.commit()

    @staticmethod
    def key(model: str, system: str, text: str) -> str:
        blob = f"{model}\x00{system}\x00{text}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, k: str) -> str | None:
        row = self.db.execute("SELECT at, v FROM r WHERE k = ?", (k,)).fetchone()
        if not row:
            return None
        at, value = row
        if time.time() - at > self.ttl_s:
            self.db.execute("DELETE FROM r WHERE k = ?", (k,))
            self.db.commit()
            return None
        return value

    def put(self, k: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO r (k, at, v) VALUES (?, ?, ?)",
                        (k, time.time(), value))
        self.db.commit()

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM r").fetchone()[0]
        return {"entries": n, "path": str(self.path), "ttl_s": self.ttl_s}
