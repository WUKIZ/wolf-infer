"""被发现前一秒回家：把活动态收干净，不留尾巴。

一次 conceal 的顺序是固定的，因为每一步都在减少暴露面：
  1. 落盘回家（记忆先安全，再谈别的）
  2. 退租（外租的机器上不该继续跑着我们的进程）
  3. 扫掉 excursion/ 里过期的临时文件
  4. 心跳改 home（别的设备立刻看到"它回家了"）
  5. 记录耗时 —— 超过 lead_time 就是清理太重，下次要减负

幂等：已经在家的再 conceal 一次，只是重做 2-4，结果一样。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional



class Concealer:
    def __init__(
        self,
        home,
        cfg: Optional[Dict[str, Any]] = None,
        broker=None,
        store=None,
        trail=None,
        clock=time.monotonic,
    ) -> None:
        cfg = cfg or {}
        self.home = home
        self.broker = broker
        self.store = store
        self.trail = trail
        self.clock = clock
        self.lead_time_s = float(cfg.get("conceal_lead_time_s", 1.5))
        self.tmp_ttl_s = float(cfg.get("tmp_ttl_s", 3600.0))
        self.quiet_s = float(cfg.get("quiet_s", 300.0))
        self.events = 0

    # ---------- 清理 ----------
    def _sweep(self) -> int:
        """只动 excursion/ 下的过期临时文件；家、门槛、主机一概不碰。"""
        root: Path = self.home.root / "excursion"
        if not root.exists():
            return 0
        cutoff = time.time() - self.tmp_ttl_s
        n = 0
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
            except OSError:
                pass
        return n

    # ---------- 回家 ----------
    def conceal(self, why: str, quiet: bool = True) -> Dict[str, Any]:
        t0 = self.clock()
        steps = []

        if self.home.state != HOME:
            self.home.enter()
            steps.append("state->home")

        if self.broker is not None:
            released = self.broker.release_all()
            if released:
                steps.append(f"leases={released}")

        swept = self._sweep()
        if swept:
            steps.append(f"swept={swept}")

        if self.trail is not None:
            wiped = self.trail.wipe(0.0)
            if wiped:
                steps.append(f"wiped={wiped}")
            if quiet:
                # 只有"被看见"导致的归家才需要静默；正常停机不该把下次出门也堵住
                self.trail.quiet(self.quiet_s)
                steps.append(f"quiet={self.quiet_s:.0f}s")
            else:
                self.trail.clear_quiet()
                steps.append("quiet_cleared")

        if self.store is not None:
            self.store.heartbeat("home")
            steps.append("heartbeat")

        ms = (self.clock() - t0) * 1000.0
        slow = ms > self.lead_time_s * 1000.0
        self.events += 1
        self.home.log("conceal", f"{why}\tms={ms:.2f}\tslow={int(slow)}\t{','.join(steps) or '-'}")
        return {"why": why, "ms": round(ms, 2), "slow": slow, "steps": steps}
