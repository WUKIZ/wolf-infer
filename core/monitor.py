"""监测：访问量、字节数、数据变化。

不做抓包 —— 那要 root，也没必要。在应用层记三个东西就够了：
  - 每次抓取：url、成功与否、字节、拒绝原因
  - 每次归家：原因、耗时
  - 每次落盘/并入：条数

写 JSONL（logs/monitor.jsonl），读的时候聚合。数据变化用快照对比：
home/state/watch_baseline.json 存上一次 `--watch` 时的文件尺寸与条数。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from core.atomic import atomic_write


class Monitor:
    def __init__(self, home, max_bytes: int = 2_000_000) -> None:
        self.home = home
        self.path = home.root / "logs" / "monitor.jsonl"
        self.baseline_path = home.state_dir / "watch_baseline.json"
        self.max_bytes = max_bytes

    # ---------- 记 ----------
    def record(self, kind: str, **fields: Any) -> None:
        row = {"t": time.time(), "kind": kind, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                os.replace(self.path, self.path.with_suffix(".jsonl.1"))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass          # 监测坏了不该拖垮主流程

    def events(self, limit: int = 5000) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            for line in self.path.read_text("utf-8").splitlines()[-limit:]:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            return []
        return out

    # ---------- 聚合 ----------
    def window(self, seconds: float = 3600.0) -> Dict[str, Any]:
        now = time.time()
        ev = [e for e in self.events() if now - float(e.get("t", 0)) <= seconds]
        fetches = [e for e in ev if e.get("kind") == "fetch"]
        ok = [e for e in fetches if e.get("ok")]
        return {
            "window_s": seconds,
            "fetches": len(fetches),
            "fetches_ok": len(ok),
            "fails": len(fetches) - len(ok),
            "bytes": sum(int(e.get("bytes") or 0) for e in fetches),
            "rejected": [e.get("rejected") for e in fetches if e.get("rejected")][:5],
            "last_fetch_age_s": round(now - max((e["t"] for e in fetches), default=now), 1),
        }

    def fs_stats(self) -> Dict[str, Any]:
        h = self.home
        def size(p: Path) -> int:
            try:
                return p.stat().st_size
            except OSError:
                return 0
        st = h.state_dir / "short_term.json"
        working = 0
        if st.exists():
            try:
                working = len(json.loads(st.read_text("utf-8")).get("entries", []))
            except (OSError, ValueError):
                working = 0
        return {
            "facts": len(h.live.get("facts", [])),
            "threshold": len(list(h.threshold_dir.glob("*.json"))) if h.threshold_dir.exists() else 0,
            "corpus_bytes": size(h.root / "home" / "corpus.txt"),
            "model_bytes": size(h.root / "home" / "models" / "tiny.bin"),
            "working": working,
            "log_bytes": sum(size(f) for f in (h.root / "logs").glob("*")),
        }

    def deltas(self) -> Dict[str, int]:
        cur = self.fs_stats()
        prev: Dict[str, Any] = {}
        if self.baseline_path.exists():
            try:
                prev = json.loads(self.baseline_path.read_text("utf-8"))
            except (OSError, ValueError):
                prev = {}
        return {k: int(cur[k]) - int(prev.get(k, cur[k])) for k in cur}

    def mark(self) -> None:
        atomic_write(self.baseline_path,
                json.dumps(self.fs_stats(), ensure_ascii=False, indent=2))

    # ---------- 一次看全 ----------
    def report(self, hours: float = 1.0, mark: bool = False) -> Dict[str, Any]:
        rep = {"now": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "window": self.window(hours * 3600),
               "fs": self.fs_stats(),
               "delta": self.deltas()}
        if mark:
            self.mark()
        return rep
