"""家外记忆：门外学到的，进门不带走，但不销毁。

「进家前把学到的留在家外面」—— 不是删除，是寄存。家外有个柜子，出门
时还能取。这样既保持家里干净，又不丢外面学到的东西，而且外面学的东西
永远不会污染家园记忆。

柜子有上限：满了丢最老的。否则跑几个月就成第二份垃圾场。
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Outside:
    def __init__(self, path: Path | str, cap: int = 200) -> None:
        self.p = Path(path)
        self.cap = cap
        self.items: list[dict] = []
        if self.p.exists():
            try:
                loaded = json.loads(self.p.read_text("utf-8"))
                if isinstance(loaded, list):
                    self.items = [x for x in loaded if isinstance(x, dict)]
            except (OSError, ValueError):
                self.items = []

    def _save(self) -> None:
        try:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.p.with_name(f".{self.p.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.items[-self.cap:], ensure_ascii=False),
                           "utf-8")
            os.replace(tmp, self.p)
        except OSError:
            pass

    def put(self, texts: list[str]) -> int:
        """进门那一刻调用。返回收了几条。"""
        import time
        n = 0
        for t in texts:
            t = (t or "").strip()[:200]
            if not t:
                continue
            self.items.append({"text": t, "at": time.time()})
            n += 1
        if n:
            self.items = self.items[-self.cap:]
            self._save()
        return n

    def take(self, n: int = 5) -> list[str]:
        """出门时调用：最近学到的先取。"""
        return [i["text"] for i in self.items[-n:]]

    def line(self) -> str:
        return f"门外柜={len(self.items)}条" if self.items else ""

    def stats(self) -> dict[str, int]:
        return {"outside": len(self.items), "cap": self.cap}
