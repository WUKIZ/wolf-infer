"""痕迹清理与静默窗口 —— 只作用于本机自己的东西。

做三件事：
  1. wipe     excursion/ 下的临时文件、home 里的 *.tmp 残片：覆写一遍再删
  2. scrub    日志里的原文片段替换成 sha 摘要（审计还在，内容没了）
  3. quiet    归家后一段时间内不再发出任何请求（静默窗口）

不做的事写清楚：这里没有无线电干扰、没有伪造流量、不碰别人的设备。
"蜂窝干扰"（jam 基站/手机信号）是违法的，会压掉紧急呼叫，这条不在这个项目里做。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from core.atomic import atomic_write

SENSITIVE_EVENTS = {"memory.discard", "memory.quarantine", "core.write_refused"}


class Trail:
    def __init__(self, home, clock=time.time) -> None:
        self.home = home
        self.clock = clock
        self.quiet_path = home.state_dir / "quiet_until.json"

    # ---------- 覆写后删除 ----------
    @staticmethod
    def _shred(path: Path, passes: int = 1) -> bool:
        try:
            size = path.stat().st_size
            with path.open("r+b") as fh:
                fh.write(b"\x00" * size)
                fh.flush()
                os.fsync(fh.fileno())
            path.unlink()
            return True
        except OSError:
            return False

    def wipe(self, ttl_s: float = 0.0) -> int:
        """清理 excursion/ 与 home/ 下的临时残片。门槛、记忆、语料一概不碰。

        注意：SSD 的磨损均衡会让"覆写"不保证物理擦除 —— 这是本机卫生，不是
        取证级擦除，别把它当安全擦除用。
        """
        cutoff = self.clock() - ttl_s
        targets = []
        exc = self.home.root / "excursion"
        if exc.exists():
            targets += [f for f in exc.rglob("*") if f.is_file()]
        for pattern in ("home/state/*.tmp", "home/models/*.tmp", "home/*.tmp"):
            targets += list(self.home.root.glob(pattern))
        n = 0
        for f in targets:
            try:
                if ttl_s > 0 and f.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            n += 1 if self._shred(f) else 0
        if n:
            self.home.log("trail.wipe", f"files={n}")
        return n

    # ---------- 日志脱敏 ----------
    def scrub(self, events: Optional[Iterable[str]] = None) -> int:
        """把敏感事件里的原文换成 sha。行数与时间戳保留，审计链不断。"""
        wanted = set(events or SENSITIVE_EVENTS)
        log = self.home.root / "logs" / "wolf.log"
        if not log.exists():
            return 0
        out, changed = [], 0
        for line in log.read_text("utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[1] in wanted:
                # 事件名之后的所有内容都算原文（detail 里可能还带 tab）
                raw = "\t".join(parts[2:])
                digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
                parts = parts[:2] + [f"sha:{digest}"]
                changed += 1
            out.append("\t".join(parts))
        if changed:
            atomic_write(log, "\n".join(out) + "\n")
        return changed

    # ---------- 静默窗口 ----------
    def quiet(self, seconds: float) -> float:
        until = self.clock() + max(0.0, seconds)
        atomic_write(self.quiet_path, json.dumps({"until": until}, ensure_ascii=False))
        return until

    def quiet_left(self) -> float:
        if not self.quiet_path.exists():
            return 0.0
        try:
            until = float(json.loads(self.quiet_path.read_text("utf-8")).get("until", 0))
        except (OSError, ValueError, TypeError):
            return 0.0
        return max(0.0, until - self.clock())

    def clear_quiet(self) -> None:
        try:
            self.quiet_path.unlink()
        except OSError:
            pass
