"""搜索框架：一个入口，多后端；先把 Bing 接上，别的按同一接口加。

只取“搜索结果页”，不抓目标站内容；点结果由浏览器打开，和人在浏览器里点一样。
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import Dict, List

UA = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

_BLOCK = re.compile(r'<li class="b_algo".*?</li>', re.S)
_TITLE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIP = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return html.unescape(_TAG.sub("", s)).replace("\u200b", "").strip()


def parse_bing(page: str, limit: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for block in _BLOCK.findall(page):
        m = _TITLE.search(block)
        if not m:
            continue
        url = html.unescape(m.group(1))
        if url.startswith("/") or "bing.com" in url:
            continue
        title = _clean(m.group(2))
        s = _SNIP.search(block)
        snippet = _clean(s.group(1)) if s else ""
        if not title or url in seen:
            continue
        seen.add(url)
        out.append({"title": title, "url": url, "snippet": snippet[:220]})
        if len(out) >= limit:
            break
    return out


def bing(query: str, limit: int = 8, timeout: float = 15.0) -> List[Dict[str, str]]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "setlang": "zh-Hans"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read(2_000_000).decode("utf-8", "replace")
    return parse_bing(page, limit)


BACKENDS = {"bing": bing}


def search(query: str, limit: int = 8, backend: str = "bing") -> Dict[str, object]:
    fn = BACKENDS.get(backend)
    if fn is None:
        return {"query": query, "backend": backend, "error": "unknown_backend", "results": []}
    try:
        rows = fn(query, limit)
    except Exception as exc:            # noqa: BLE001
        return {"query": query, "backend": backend, "error": type(exc).__name__, "results": []}
    return {"query": query, "backend": backend, "count": len(rows), "results": rows}
