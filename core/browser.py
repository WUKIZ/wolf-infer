"""浏览器客户端：她自己的窗口，走的是她本来就有的那条路。

应用（core/webapp.py）跑在 127.0.0.1 上；这个客户端让本地助手（或测试）用
HTTP 调它，而不是直接碰 urllib。好处是所有出口都过同一套闸：
URL 边界 → 权限预算 → 本能闸 → 静默窗口。

    b = Browser("http://127.0.0.1:8088", token)
    b.fetch("https://example.com")
    b.submit("https://example.com/search", {"q": "本地助手"})
    b.ask("你好")
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional


class Browser:
    def __init__(self, base: str, token: str, timeout: float = 30.0) -> None:
        self.base = base.rstrip("/")
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_state(cls, path) -> "Browser | None":
        """从 home/state/web.json 读应用地址与 token；应用没开就返回 None。"""
        try:
            data = json.loads(open(path, encoding="utf-8").read())
            return cls(str(data["url"]), str(data["token"]))
        except (OSError, ValueError, KeyError):
            return None

    def _call(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(payload or {}, ensure_ascii=False).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json", "X-Wolf-Token": self.token})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:            # noqa: BLE001 - 应用没开就返回错误，不抛
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def status(self) -> Dict[str, Any]:
        return self._call("/api/status")

    def fetch(self, url: str) -> Dict[str, Any]:
        return self._call("/api/fetch", {"url": url, "method": "GET"})

    def submit(self, url: str, fields: Dict[str, str]) -> Dict[str, Any]:
        return self._call("/api/fetch", {"url": url, "method": "POST", "data": fields})

    def ask(self, text: str) -> Dict[str, Any]:
        return self._call("/api/ask", {"text": text})
