from __future__ import annotations
import logging
"""爬虫：从一个种子出发，同站、限深、限量地取页面。

礼貌三条，缺一不可：
  1. 先读 robots.txt，Disallow 的一律不碰
  2. 只跟同站链接（种子所在的 host）；不跨站扩散
  3. 每页之间停一下，别把人家站点当压测目标

产出是 Harvest 列表（文本已清洗、已过本能闸），落盘由调用方决定 ——
爬虫不管记忆，也不管训练。
"""

import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Set

from .learn import Harvest

LINK = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.I)
SKIP_EXT = (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4",
            ".mp3", ".css", ".js", ".ico", ".gz", ".tar", ".xml")


class Crawler:
    def __init__(self, learner, permit=None, max_pages: int = 5, max_depth: int = 1,
                 same_host: bool = True, delay_s: float = 0.3,
                 clock=time.monotonic) -> None:
        self.learner = learner
        self.permit = permit
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.same_host = same_host
        self.delay_s = delay_s
        self.clock = clock

    # ---------- robots ----------
    def _robots(self, origin: str) -> List[str]:
        url = origin.rstrip("/") + "/robots.txt"
        if self.permit is not None:
            ok, _ = self.permit.check(
                "net.request", host=urllib.parse.urlparse(url).hostname or "")
            if not ok:
                return []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.learner.ua})
            with urllib.request.urlopen(req, timeout=self.learner.timeout) as r:
                text = r.read(64_000).decode("utf-8", "replace")
        except (OSError, ValueError) as exc:            # noqa: BLE001 - 没有 robots 就按全允许
            logging.warning('robots 取不到，按全允许处理：%s', exc)
            return []
        disallow: List[str] = []
        applies = False
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                applies = value == "*" or value.lower() in self.learner.ua.lower()
            elif key == "disallow" and applies and value:
                disallow.append(value)
        return disallow

    @staticmethod
    def _blocked(path: str, disallow: List[str]) -> bool:
        return any(path.startswith(d) for d in disallow)

    # ---------- 链接 ----------
    @staticmethod
    def _links(html: str, base: str, host: str, same_host: bool) -> List[str]:
        out: List[str] = []
        for raw in LINK.findall(html):
            raw = raw.strip()
            if raw.startswith(("mailto:", "javascript:", "tel:")):
                continue
            url = urllib.parse.urljoin(base, raw)
            u = urllib.parse.urlparse(url)
            if u.scheme not in ("http", "https"):
                continue
            if u.path.lower().endswith(SKIP_EXT):
                continue
            if same_host and u.hostname != host:
                continue
            out.append(urllib.parse.urlunparse((u.scheme, u.netloc, u.path, "", u.query, "")))
        return out

    # ---------- 爬 ----------
    def crawl(self, seeds: List[str]) -> List[Harvest]:
        pages: List[Harvest] = []
        seen: Set[str] = set()
        robots: Dict[str, List[str]] = {}

        for seed in seeds:
            if len(pages) >= self.max_pages:
                break
            u0 = urllib.parse.urlparse(seed)
            if u0.scheme not in ("http", "https") or not u0.hostname:
                pages.append(Harvest(source=seed, rejected="url:scheme"))
                continue
            base_host = u0.hostname
            origin = f"{u0.scheme}://{u0.netloc}"
            if origin not in robots:
                robots[origin] = self._robots(origin)

            queue = [(seed, 0)]
            while queue and len(pages) < self.max_pages:
                url, depth = queue.pop(0)
                if url in seen:
                    continue
                seen.add(url)
                path = urllib.parse.urlparse(url).path or "/"
                if self._blocked(path, robots.get(origin, [])):
                    pages.append(Harvest(source=url, rejected="robots"))
                    continue

                h = self.learner.fetch(url, keep_html=True)
                pages.append(h)
                if not h.ok:
                    continue
                if depth < self.max_depth:
                    for link in self._links(h.raw, url, base_host, self.same_host):
                        if link not in seen:
                            queue.append((link, depth + 1))
                if self.delay_s:
                    time.sleep(self.delay_s)
        return pages
