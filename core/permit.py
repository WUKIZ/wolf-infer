"""权限：给"能自己出门"装上可数的笼头。

不是哲学问题，是可数的预算 —— 全部按天/按小时落盘，超了就拒绝，不是警告：
  网络：域名白名单（空=任意公网）、每小时请求数、每天字节数
  训练：每小时/每天的步数

拒绝的理由会带在返回值里，调用方照实记 rejected，不静默放行。
开关 `permissions.enabled=false` 一刀切停掉所有对外动作。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from core.atomic import atomic_write




DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    # 空列表 = 不限制域名；填后缀带点，如 ".deepseek.com"
    "net": {"allow_hosts": [], "max_requests_per_hour": 60,
            "max_bytes_per_day": 50 * 1024 * 1024},
    "train": {"max_steps_per_hour": 60, "max_steps_per_day": 300},
}

class Permit:
    def __init__(self, path: Path, cfg: Optional[Dict[str, Any]] = None,
                 clock=time.time) -> None:
        self.path = Path(path)
        self.cfg = dict(DEFAULTS)
        for section, values in (cfg or {}).items():
            if isinstance(values, dict) and isinstance(self.cfg.get(section), dict):
                self.cfg[section] = {**self.cfg[section], **values}
            else:
                self.cfg[section] = values
        self.clock = clock

    # ---------- 账 ----------
    def _keys(self) -> Tuple[str, str]:
        t = self.clock()
        return time.strftime("%Y-%m-%d", time.localtime(t)), time.strftime("%Y-%m-%dT%H", time.localtime(t))

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        atomic_write(self.path, json.dumps(data, ensure_ascii=False, indent=2))

    def _bucket(self, data: Dict[str, Any]) -> Dict[str, Any]:
        day, hour = self._keys()
        if data.get("day") != day:
            data = {"day": day, "hour": hour, "day_counts": {}, "hour_counts": {}}
        elif data.get("hour") != hour:
            data["hour"] = hour
            data["hour_counts"] = {}
        data.setdefault("day_counts", {})
        data.setdefault("hour_counts", {})
        return data

    # ---------- 判定 ----------
    def check(self, cap: str, n: float = 1, **facts: Any) -> Tuple[bool, str]:
        if not self.cfg.get("enabled", True):
            return False, "disabled"
        data = self._bucket(self._load())
        day = data["day_counts"]
        hour = data["hour_counts"]

        if cap == "net.request":
            allowed = self.cfg["net"].get("allow_hosts") or []
            host = str(facts.get("host", ""))
            if allowed and host not in allowed and not any(host.endswith("." + h) for h in allowed):
                return False, f"host_not_allowed:{host}"
            if hour.get("net.request", 0) + 1 > self.cfg["net"]["max_requests_per_hour"]:
                return False, "requests_per_hour"
            return True, "ok"

        if cap == "net.bytes":
            if day.get("net.bytes", 0) + n > self.cfg["net"]["max_bytes_per_day"]:
                return False, "bytes_per_day"
            return True, "ok"

        if cap == "train.steps":
            if hour.get("train.steps", 0) + n > self.cfg["train"]["max_steps_per_hour"]:
                return False, "steps_per_hour"
            if day.get("train.steps", 0) + n > self.cfg["train"]["max_steps_per_day"]:
                return False, "steps_per_day"
            return True, "ok"

        return False, f"unknown_cap:{cap}"

    def spend(self, cap: str, n: float = 1, **facts: Any) -> Tuple[bool, str]:
        ok, why = self.check(cap, n, **facts)
        if not ok:
            return False, why
        data = self._bucket(self._load())
        data["day_counts"][cap] = data["day_counts"].get(cap, 0) + n
        data["hour_counts"][cap] = data["hour_counts"].get(cap, 0) + n
        self._save(data)
        return True, "ok"

    # ---------- 体检 ----------
    def snapshot(self) -> Dict[str, Any]:
        data = self._bucket(self._load())
        return {
            "enabled": bool(self.cfg.get("enabled", True)),
            "day": data["day"],
            "used": {k: round(v, 2) for k, v in data["day_counts"].items()},
            "hour": {k: round(v, 2) for k, v in data["hour_counts"].items()},
            "limits": {"net": self.cfg["net"], "train": self.cfg["train"]},
        }
