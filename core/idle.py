"""闲置普查：自家院子里，哪些东西很久没动、哪些算力在空转。

只读 + 可选清理，不碰 core/、权重、语料、manifest、secrets。
"""
from __future__ import annotations

import os

from core.meminfo import available_mb
import time
from pathlib import Path
from typing import Dict, List

PROTECT = ("core/", "home/models/", "home/corpus", "home/state/manifest.json",
           "home/state/secrets.json", "docs/")
SCAN_DIRS = ("home", "excursion", "logs")


def _mem_available_mb() -> float:
    return available_mb()


def stale_files(root: Path, days: float = 14.0) -> List[Dict[str, object]]:
    cutoff = time.time() - days * 86400
    out = []
    for name in SCAN_DIRS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if any(rel.startswith(p) for p in PROTECT):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                out.append({"path": rel, "mb": st.st_size / 1048576, "mtime": st.st_mtime})
    out.sort(key=lambda x: x["mb"], reverse=True)
    return out


def survey(root: Path, days: float = 14.0) -> Dict[str, object]:
    stale = stale_files(root, days)
    try:
        load1, load5, _ = os.getloadavg()
    except OSError:
        load1 = load5 = 0.0
    cores = os.cpu_count() or 1
    return {
        "stale_files": len(stale),
        "stale_mb": round(sum(float(x["mb"]) for x in stale), 2),
        "top_stale": stale[:10],
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "mem_available_mb": round(_mem_available_mb(), 0),
        "cores": cores,
        "idle_cores": max(0, round(cores - load1, 2)),
    }


def reclaim(root: Path, days: float = 14.0) -> Dict[str, object]:
    freed = 0.0
    deleted = 0
    for item in stale_files(root, days):
        try:
            (root / str(item["path"])).unlink()
            freed += float(item["mb"])
            deleted += 1
        except OSError:
            continue
    return {"deleted": deleted, "freed_mb": round(freed, 2)}
