"""供给层：在"你自己开通的"免费额度之间自动调度。

边界（写死，没有开关）：
- 只用 config 里列出的 provider 和 secrets 里对应的 key；不扫描、不抓取、不猜 key
- 不注册账号、不伪造设备、不绕验证码/风控/地区限制
- 429 一律指数退避冷却，401/403 冷却一天；不换 IP、不重放

能做的"想办法"都在这里：多供应商轮换、冷却、日额度、失败降级。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .remote import RemoteBrain


class Supply:
    def __init__(self, providers: List[Dict[str, Any]], state_path: Path,
                 permit=None, trail=None, monitor=None,
                 secrets_paths: Optional[List[Path]] = None) -> None:
        self.state_path = Path(state_path)
        try:
            self.state = json.loads(self.state_path.read_text("utf-8"))
        except (OSError, ValueError):
            self.state = {}
        self.brains = []
        for cfg in providers:
            brain = RemoteBrain(cfg, permit=permit, trail=trail, monitor=monitor,
                                secrets_paths=secrets_paths,
                                key_name=str(cfg.get("key", "deepseek")))
            self.brains.append((cfg, brain))
        self.per_min_default = int(providers[0].get("per_min_tokens", 0) or 0) if providers else 0

    # ---------- 状态 ----------
    def _slot(self, name: str, now: float | None = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        today = time.strftime("%Y-%m-%d")
        st = self.state.setdefault(name, {})
        if st.get("day") != today:
            st.update({"day": today, "spent": 0})
        if now - float(st.get("win_start", 0) or 0) >= 60:
            st.update({"win_start": now, "win_tokens": 0})
        return st

    def _save(self) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.state, ensure_ascii=False), "utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    def _usable(self, cfg: Dict[str, Any], brain: RemoteBrain, now: float) -> bool:
        if not brain.available():
            return False
        st = self._slot(str(cfg["name"]))
        if st.get("until", 0) > now:
            return False
        per_min = int(cfg.get("per_min_tokens", 0) or 0)
        if per_min and st.get("win_tokens", 0) >= per_min:
            return False
        budget = int(cfg.get("daily_tokens", 0) or 0)
        return not (budget and st.get("spent", 0) >= budget)

    # ---------- 对外 ----------
    def available(self) -> bool:
        now = time.time()
        return any(self._usable(c, b, now) for c, b in self.brains)

    def chat(self, text: str, context: Optional[List[str]] = None,
             system: Optional[str] = None) -> Optional[str]:
        now = time.time()
        for cfg, brain in self.brains:
            if not self._usable(cfg, brain, now):
                continue
            answer = brain.chat(text, context=context, system=system)
            st = self._slot(str(cfg["name"]))
            if answer:
                st["spent"] = st.get("spent", 0) + int(brain.last_tokens or 0)
                st["win_tokens"] = st.get("win_tokens", 0) + int(brain.last_tokens or 0)
                st["fails"] = 0
                self._save()
                return answer
            code = brain.last_status
            if code == 429:
                wait = min(3600, 60 * (2 ** min(st.get("fails", 0), 5)))
                st["fails"] = st.get("fails", 0) + 1
            elif code in (401, 403):
                wait = 86400
            else:
                wait = 300
            st["until"] = now + wait
            self._save()
        return None

    def status(self) -> List[Dict[str, Any]]:
        now = time.time()
        out = []
        for cfg, brain in self.brains:
            st = self._slot(str(cfg["name"]))
            out.append({"name": cfg["name"], "key": brain.available(),
                        "spent": st.get("spent", 0),
                        "budget": cfg.get("daily_tokens", 0),
                        "per_min": cfg.get("per_min_tokens", 0),
                        "win_tokens": st.get("win_tokens", 0),
                        "cooling_s": max(0, int(st.get("until", 0) - now)),
                        "last": brain.last_status})
        return out
