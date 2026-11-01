"""情绪状态：不是装饰，是采样参数的来源。

三个量，各有各的来路：
  energy  被用得多就掉，闲着也掉，睡够才回。掉到底就是"累了"，回答变短。
  trust   被夸、被耐心对待会涨，被凶会掉。涨了话多点，掉了话少点。
  arousal 被问、被刺激会涨，安静下来会退。涨了温度升高，回答更发散。

状态落 home/state/emotion.json，重启接着上次的情绪，不是每次从零开始。
不碰核心：核心是规则，情绪是状态，两件事。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT = {"energy": 0.7, "trust": 0.5, "arousal": 0.3, "mood": "calm"}

# 事件 → 三个量的变化。加事件就在这里加一行，别在调用处写 if。
DELTA = {
    "praise":    {"trust": +0.05, "energy": +0.03, "arousal": +0.00},
    "cold":      {"trust": -0.05, "energy": +0.00, "arousal": -0.05},
    "question":  {"trust": +0.00, "energy": -0.01, "arousal": +0.05},
    "idle":      {"trust": +0.00, "energy": -0.02, "arousal": -0.05},
    "rest":      {"trust": +0.00, "energy": +0.15, "arousal": -0.10},
}


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Emotion:
    """一次对话里共用一份，别每次新建——那会丢状态。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.s: Dict[str, Any] = dict(DEFAULT)
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.s.update({k: loaded[k] for k in DEFAULT if k in loaded})
            except (OSError, ValueError):
                pass  # 状态坏了就用默认值，不让它拖垮启动

    def _save(self) -> None:
        if self.path:
            try:
                _atomic(self.path, json.dumps(self.s, ensure_ascii=False))
            except OSError:
                pass  # 存不下不影响这一轮对话

    def touch(self, event: str) -> Dict[str, Any]:
        """喂一个事件，返回新状态。未知事件按 idle 处理，不抛异常。"""
        delta = DELTA.get(event, DELTA["idle"])
        for key, dv in delta.items():
            self.s[key] = round(min(1.0, max(0.0, self.s[key] + dv)), 4)

        if self.s["arousal"] > 0.6:
            self.s["mood"] = "alert"
        elif self.s["energy"] < 0.3:
            self.s["mood"] = "tired"
        elif self.s["trust"] < 0.3:
            self.s["mood"] = "wary"
        else:
            self.s["mood"] = "calm"

        self._save()
        return dict(self.s)

    def sampling(self) -> Dict[str, float]:
        """给推理层的采样参数。温度跟 arousal 走，长度跟 energy 走。"""
        return {
            "temperature": round(0.6 + 0.6 * self.s["arousal"], 2),
            "max_tokens": int(80 + 120 * self.s["energy"]),
        }

    def policy(self) -> Dict[str, Any]:
        """情绪决定行为，不只是采样参数 —— 这一层原来缺着。

        arousal 高 → 更愿意出门查；trust 低 → 不主动开口；
        energy 低 → 回答短，省电也省 token。
        """
        return {
            "search": self.s["arousal"] >= 0.45,
            "speak": self.s["trust"] >= 0.3 and self.s["energy"] >= 0.2,
            "max_tokens": int(60 + 260 * self.s["energy"]),
        }

    def line(self) -> str:
        """一行状态描述，塞进 system prompt。不解释、不拟人、不加戏。"""
        return (
            f"当前情绪 {self.s['mood']}：能量 {self.s['energy']:.0%}，"
            f"信任 {self.s['trust']:.0%}，唤醒 {self.s['arousal']:.0%}"
        )
