"""L2 自我：知道自己是谁、在干什么、最近发生过什么。

不是记忆 —— 记忆存事实，自我存处境。记忆可以很长，自我只有几行。
落 home/state/self.json，每轮读一次，答完写一次。

learned 是「门外学到的」：回家时由 enter_home() 清空归档，
对上「进家之前把学到的留在家外面」这条。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT: Dict[str, Any] = {"goal": "", "recent": [], "learned": [], "home": 0}


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)


class Self:
    """一次会话里共用一份。多建一份不会崩，但状态会各写各的。"""

    def __init__(self, path: Path | str | None = None, keep: int = 3) -> None:
        self.path = Path(path) if path else None
        self.keep = keep
        self.s: Dict[str, Any] = {
            k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT.items()
        }
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text("utf-8"))
                if isinstance(loaded, dict):
                    self.s.update({k: loaded[k] for k in DEFAULT if k in loaded})
            except (OSError, ValueError):
                pass  # 状态坏了用默认值，不拖垮启动

    def _save(self) -> None:
        if not self.path:
            return
        try:
            _atomic(self.path, json.dumps(self.s, ensure_ascii=False))
        except OSError:
            pass

    def line(self) -> str:
        """压成一行拼进 system。没内容就返回空串，不占 token。"""
        bits = []
        if self.s["goal"]:
            bits.append(f"目标={self.s['goal']}")
        if self.s["recent"]:
            bits.append("刚才=" + "；".join(self.s["recent"][-2:]))
        if self.s["learned"]:
            bits.append(f"门外学到 {len(self.s['learned'])} 条")
        return ("当前：" + "，".join(bits)) if bits else ""

    def note(self, text: str) -> None:
        t = (text or "").strip()[:60]
        if not t:
            return
        self.s["recent"] = (self.s["recent"] + [t])[-self.keep:]
        self._save()

    def set_goal(self, text: str) -> None:
        self.s["goal"] = (text or "").strip()[:80]
        self._save()

    def learn(self, text: str) -> None:
        t = (text or "").strip()[:200]
        if not t:
            return
        self.s["learned"] = (self.s["learned"] + [t])[-50:]
        self._save()

    def enter_home(self) -> int:
        """回家：门外学的清空，记一次归家。返回清掉几条。"""
        n = len(self.s["learned"])
        self.s["learned"] = []
        self.s["home"] = int(self.s.get("home", 0)) + 1
        self._save()
        return n
