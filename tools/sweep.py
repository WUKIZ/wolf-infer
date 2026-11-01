#!/usr/bin/env python3
"""扫自家院子：找出很久没动过的东西。默认只报告，不删。

    ./bin/py tools/sweep.py --days 14            # 看看
    ./bin/py tools/sweep.py --days 14 --delete --yes   # 真删

只扫 home/、excursion/、logs/ 三个地方；core/、权重、语料、manifest、secrets 永不碰。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["home", "excursion", "logs"]
PROTECT = ("core/", "home/models/", "home/corpus", "home/state/manifest.json",
           "home/state/secrets.json", "docs/")


def stale(days: float, now: float | None = None):
    now = now or time.time()
    cutoff = now - days * 86400
    found = []
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if any(rel.startswith(p) for p in PROTECT):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                found.append((rel, st.st_size, st.st_mtime))
    found.sort(key=lambda x: x[1], reverse=True)
    return found


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=14)
    p.add_argument("--delete", action="store_true")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args()

    found = stale(a.days)
    total = sum(size for _, size, _ in found)
    for rel, size, mtime in found[:40]:
        print(f"{size/1024:9.1f} KB  {time.strftime('%m-%d %H:%M', time.localtime(mtime))}  {rel}")
    print(f"[sweep] {len(found)} 个文件，共 {total/1048576:.2f} MB，{a.days:g} 天没动过")

    if not a.delete:
        return 0
    if not a.yes:
        print("[sweep] 真要删就加 --yes；默认只报告", flush=True)
        return 2
    freed = 0
    for rel, size, _ in found:
        try:
            (ROOT / rel).unlink()
            freed += size
        except OSError as exc:
            print(f"[sweep] 删不掉 {rel}: {exc}", flush=True)
    print(f"[sweep] 已删 {freed/1048576:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
