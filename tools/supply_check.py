#!/usr/bin/env python3
"""逐个试通 config.brain.providers：只发一条最小请求，不回显 key。

    ./bin/py tools/supply_check.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.remote import RemoteBrain  # noqa: E402


def main() -> int:
    cfg = json.loads((ROOT / "config.json").read_text("utf-8"))
    providers = cfg.get("brain", {}).get("providers") or []
    paths = [Path(p).expanduser() if Path(p).expanduser().is_absolute()
             else (ROOT / p).resolve()
             for p in cfg.get("brain", {}).get("secrets_paths", [])]
    if not providers:
        print("[check] config.brain.providers 是空的")
        return 2

    bad = 0
    skipped = 0
    for p in providers:
        brain = RemoteBrain(p, secrets_paths=paths, key_name=p.get("key", "deepseek"))
        name = p.get("name", "?")
        if not brain.available():
            print(f"{name:12s} 没 key（跳过）")
            skipped += 1
            continue
        t0 = time.perf_counter()
        ans = brain.chat("只回复两个字：在线", system="只回复两个字：在线")
        ms = (time.perf_counter() - t0) * 1000
        if ans:
            print(f"{name:12s} ok  {brain.last_status}  {ms:.0f}ms  {brain.last_tokens} tokens")
        else:
            print(f"{name:12s} 失败 status={brain.last_status}  {ms:.0f}ms")
            bad += 1
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
