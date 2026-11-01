"""工具网关：可信查询 → 签名计划 → 工具只能按计划调；外部返回一律打不可信标签。"""
from __future__ import annotations
import re
from typing import Any
from .labels import Tagged, U
from .plan import PLAN, PlanError
from .flow import enforce
from .kernel import CAGE

_URL = re.compile(r"https?://([^/\s]+)(/[^\s]*)?")


class Gateway:
    def __init__(self):
        self.token: str | None = None

    def begin(self, query: str) -> str:
        """query 必须来自可信输入（用户本人或核心规则）。"""
        q = query if isinstance(query, str) else str(query)
        steps: list[tuple[str, str]] = []
        m2 = re.search(r"(?:搜索|搜一下|查一下|search)\s*(.+)", q)
        if m2:
            steps.append(("search", m2.group(1).strip()))
        m = _URL.search(q)
        if m:
            host = m.group(1)
            steps.append(("fetch", f"{host}/*"))
        if not steps:
            steps.append(("reply", "*"))
        self.token = PLAN.approve(steps, ttl=900)
        CAGE.log("plan", ",".join(f"{t}:{p}" for t, p in steps))
        return self.token

    def _need(self, tool: str, target: str) -> None:
        if self.token is None:
            raise PlanError("no plan")
        PLAN.permit(self.token, tool, target)

    def search(self, query: str):
        self._need("search", query)
        from core.search import search as _search
        raw = _search(query)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        tagged = []
        for r in results or []:
            item = dict(r)
            item["text"] = Tagged(str(item.get("text", "")), U)
            item["url"] = str(item.get("url", ""))
            tagged.append(item)
        return tagged

    def fetch(self, url: str):
        m = _URL.match(url)
        target = f"{m.group(1)}{m.group(2) or '/'}" if m else url
        self._need("fetch", target)
        from core.browser import Browser
        raw = Browser().fetch(url)
        if isinstance(raw, dict):
            raw = dict(raw)
            for k in ("text", "title"):
                if k in raw:
                    raw[k] = Tagged(str(raw[k]), U)
        return raw

    def accept_remote(self, text: str) -> Tagged:
        return Tagged(str(text), U)


GATE = Gateway()
