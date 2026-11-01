"""找算力：家里的电只够启动与推理，训练先报价，再决定租不租。

Transport 是插口 —— 接真算力时实现 request(seconds) -> dict|None：
    {"provider": "runpod", "host": "HOST", "rate_per_hour": 0.34,
     "expires_at": 1792610000.0}
没有 transport 就按 config 的 fallback_local 在本机跑（本机慢，但不撒谎）。

账本写在 home/state/lease_ledger.json：按天累计小时数，超过 max_lease_hours_per_day
就不许再租 —— 这条是硬预算，不是建议。
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.atomic import atomic_write


def _today(clock=time.time) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(clock()))


@dataclass
class Quote:
    steps: int
    per_step_s: float
    local_seconds: float
    budget_s: float
    needs_lease: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps, "per_step_s": round(self.per_step_s, 4),
            "local_seconds": round(self.local_seconds, 1),
            "budget_s": self.budget_s, "needs_lease": self.needs_lease,
            "reason": self.reason,
        }


@dataclass
class Lease:
    lid: str
    provider: str
    started: float
    expires: float
    seconds: float
    rate_per_hour: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"lid": self.lid, "provider": self.provider,
                "started": self.started, "expires": self.expires,
                "seconds": self.seconds, "rate_per_hour": self.rate_per_hour}

    def expired(self, clock=time.time) -> bool:
        return clock() >= self.expires


class LeaseBroker:
    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        ledger_path: Optional[Path] = None,
        transport: Optional[Any] = None,
        clock=time.time,
    ) -> None:
        cfg = cfg or {}
        self.max_hours = float(cfg.get("max_lease_hours_per_day", 3.0))
        self.budget_s = float(cfg.get("budget_s", 180.0))
        self.fallback_local = bool(cfg.get("fallback_local", True))
        self.transport = transport
        self.clock = clock
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self._leases: Dict[str, Lease] = {}

    # ---------- 报价 ----------
    def quote(self, per_step_s: float, steps: int) -> Quote:
        total = max(per_step_s, 1e-4) * max(steps, 0)
        needs = total > self.budget_s
        reason = "over_budget" if needs else "local_ok"
        return Quote(steps=steps, per_step_s=per_step_s, local_seconds=total,
                     budget_s=self.budget_s, needs_lease=needs, reason=reason)

    # ---------- 账本 ----------
    def _load(self) -> Dict[str, float]:
        if not self.ledger_path or not self.ledger_path.exists():
            return {}
        try:
            data = json.loads(self.ledger_path.read_text("utf-8"))
            return {str(k): float(v) for k, v in (data.get("hours") or {}).items()}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def _save(self, hours: Dict[str, float]) -> None:
        if not self.ledger_path:
            return
        # 只留最近 14 天，账本别无限长
        keys = sorted(hours)[-14:]
        atomic_write(self.ledger_path, json.dumps(
            {"hours": {k: round(hours[k], 4) for k in keys}}, ensure_ascii=False, indent=2))

    def spent_today(self) -> float:
        return self._load().get(_today(self.clock), 0.0)

    def remaining_hours(self) -> float:
        return max(0.0, self.max_hours - self.spent_today())

    def charge(self, hours: float) -> None:
        if hours <= 0:
            return
        book = self._load()
        day = _today(self.clock)
        book[day] = book.get(day, 0.0) + hours
        self._save(book)

    # ---------- 租约 ----------
    def acquire(self, seconds: float) -> Optional[Lease]:
        if self.transport is None:
            return None
        need_h = max(seconds, 0.0) / 3600.0
        if need_h > self.remaining_hours():
            return None                       # 预算硬顶，超了就回本地跑
        try:
            grant = self.transport.request(seconds)
        except Exception as exc:                     # noqa: BLE001 - 供应商挂了不该拖垮整机
            logging.warning('额度申请失败，按拒绝处理：%s', exc)
            return None
        if not grant:
            return None
        now = self.clock()
        lease = Lease(
            lid=uuid.uuid4().hex[:8],
            provider=str(grant.get("provider", "unknown")),
            started=now,
            expires=float(grant.get("expires_at", now + seconds)),
            seconds=seconds,
            rate_per_hour=float(grant.get("rate_per_hour", 0.0)),
        )
        self._leases[lease.lid] = lease
        return lease

    def release(self, lease: Lease, used_s: Optional[float] = None) -> float:
        """退租并记账。used_s 缺省按实际墙钟算，别按申请量算。"""
        self._leases.pop(lease.lid, None)
        used = lease.seconds if used_s is None else used_s
        used = max(0.0, min(used, lease.seconds))
        self.charge(used / 3600.0)
        return used

    def release_all(self) -> int:
        n = 0
        for lease in list(self._leases.values()):
            self.release(lease, self.clock() - lease.started)
            n += 1
        return n

    def tick(self) -> int:
        """回收过期租约，返回回收数量。"""
        n = 0
        for lease in list(self._leases.values()):
            if lease.expired(self.clock):
                self.release(lease, self.clock() - lease.started)
                n += 1
        return n

    def active(self) -> List[Lease]:
        return list(self._leases.values())
