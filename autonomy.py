"""自主触发：没人说话时，自己挑个主题出门看一眼，回来写进日志。

不碰 core。节奏由 COOLDOWN 控制，默认 6 小时一次 —— 太密就成了扫描器，
会被目标站当爬虫标记。主题按小时取模轮换，不用随机数，行为可复现。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import researcher

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "home" / "state" / "autonomy.json"
JOURNAL = ROOT / "home" / "state" / "journal.md"
COOLDOWN = 6 * 3600

TOPICS = [
    "人工智能 最新进展",
    "开源大模型 发布",
    "国产芯片 突破",
    "超算 算力 新闻",
]


def _load() -> dict:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(STATE)


def due(now: float | None = None) -> bool:
    return (now or time.time()) - float(_load().get("last", 0)) >= COOLDOWN


def tick(force: bool = False) -> str:
    """出门看一眼，把取回的东西写进 journal。冷却期内返回空串。"""
    now = time.time()
    if not force and not due(now):
        return ""
    topic = TOPICS[int(now // 3600) % len(TOPICS)]
    parts = researcher.research(topic, budget=2, chars=3000)

    d = _load()
    d.update({"last": now, "topic": topic, "got": len(parts)})
    _save(d)

    if not parts:
        return f"[autonomy] {topic} 没抓到东西"

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} 自主取回 · {topic}\n")
        for p in parts:
            f.write(f"- {p[:300]}\n")
    return f"[autonomy] {topic} → {len(parts)} 段，已写 journal"


if __name__ == "__main__":
    if "--watch" in sys.argv:
        # 常驻：每 10 分钟看一次是不是到点了，到了才出门
        while True:
            line = tick()
            if line:
                print(line, flush=True)
            time.sleep(600)
    else:
        print(tick(force="--force" in sys.argv))
