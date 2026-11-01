#!/usr/bin/env python3
"""目标循环：每 tick 自己挑一件事做，做完记账，冷却期内不重复。"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.home import Home
from core.memory import MemoryStore

STATE = ROOT / "home/state/agenda.json"
JOURNAL = ROOT / "home/state/journal.md"
COOLDOWN = 3600.0


def load() -> dict:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(STATE)


def candidates(root: Path, mem: MemoryStore) -> list[tuple[int, str, str]]:
    out = []
    corpus = root / "home/corpus.txt"
    chars = len(corpus.read_text("utf-8")) if corpus.exists() else 0
    if chars < 1_000_000:
        out.append((50, "grow_corpus", f"语料 {chars:,} 字，目标 100 万"))
    stats = mem.stats()
    if stats.get("working", 0) >= 2:
        out.append((40, "consolidate", f"工作记忆 {stats['working']} 条待巩固"))
    model = root / "home/models/tiny.bin"
    if model.exists() and time.time() - model.stat().st_mtime > 7 * 86400:
        out.append((30, "eval", "模型超过 7 天没评测"))
    if (root / "home/state/supply.json").exists():
        out.append((20, "supply_check", "检查各家额度"))
    out.append((10, "reflect", "没事，写一笔"))
    return out


def act(name: str, why: str, mem: MemoryStore) -> str:
    if name == "consolidate":
        r = mem.consolidate()
        return f"巩固：{r}"
    if name == "grow_corpus":
        return f"记下待办：{why}"
    if name == "eval":
        return "记下待办：跑 tools/eval.py"
    if name == "supply_check":
        return "记下待办：跑 --supply"
    stats = mem.stats()
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} 无事。记忆 {stats}\n")
    return f"反思已写：{stats}"


def tick(root: Path, mem: MemoryStore) -> str:
    state = load()
    now = time.time()
    for score, name, why in sorted(candidates(root, mem), reverse=True):
        last = float(state.get(name, 0))
        if now - last < COOLDOWN:
            continue
        m = act(name, why, mem)
        state[name] = now
        save(state)
        line = f"[{time.strftime('%H:%M:%S')}] {name} (score={score}) → {m}"
        print(line)
        return line
    print("[loop] 都在冷却，什么都不做")
    return ""


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=1)
    p.add_argument("--every", type=float, default=0.0)
    a = p.parse_args()
    home = Home(ROOT)
    mem = MemoryStore(home)
    for _ in range(a.ticks):
        tick(ROOT, mem)
        if a.every:
            time.sleep(a.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
