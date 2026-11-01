"""多引擎泛搜：必应 RSS 最省，搜狗补中文长尾，都挂了返回空。

不碰 core：core.search.bing 是旧后端，这里是它的备份与升级。
必应 RSS 一次 3.8KB 拿 10 条；搜狗页面 560KB 但能出中文长尾结果。
百度、神马会返回验证页，不接 —— 那不是能被程序解决的问题。
"""
from __future__ import annotations

import re
import urllib.parse as p
import urllib.request as u
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
TMO = 8

# 垃圾站强信号：只认「免费/破解/全集」这种明确的词，不按题材一刀切 ——
# 用户搜「电影推荐」时豆瓣是好结果，不该被题材规则误杀。
_JUNK_TITLE = re.compile(
    r"(免费|破解|绿色版|无限|私服|在线看|全集|未删减).{0,8}"      # 「免费xxx」
    r"(电影|电视剧|动漫|小说|软件|游戏|下载|观看)|"
    r"(破解|绿色|免费)版", re.I)
_JUNK_DOMAIN = ("downza", "cr173", "52pk", "3dmgame", "yxdown",
                "pc6.com", "ddooo", "xiazaizhijia")


def _junk(h: dict) -> bool:
    """一条结果是否该丢。宁可少丢，不可错杀。"""
    title = str(h.get("title") or "")
    url = str(h.get("url") or "").lower()
    if title and _JUNK_TITLE.search(title):
        return True
    if any(d in url for d in _JUNK_DOMAIN):
        return True
    return False


def fetch(url: str, timeout: int = TMO) -> bytes:
    return u.urlopen(u.Request(url, headers=UA), timeout=timeout).read()


def bing_rss(q: str, n: int = 6) -> list[dict]:
    url = "https://www.bing.com/search?" + p.urlencode({"q": q, "format": "rss"})
    root = ET.fromstring(fetch(url))
    out: list[dict] = []
    for it in root.iter("item"):
        out.append({"title": it.findtext("title", ""),
                    "snippet": (it.findtext("description", "") or "")[:150],
                    "url": it.findtext("link", "")})
        if len(out) >= n:
            break
    return out


def sogou(q: str, n: int = 6) -> list[dict]:
    url = "https://www.sogou.com/web?query=" + p.quote(q)
    html = fetch(url, 12).decode("utf-8", "ignore")
    out: list[dict] = []
    for m in re.finditer(r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not title:
            continue
        out.append({"title": title, "snippet": "",
                    "url": "https://www.sogou.com" + m.group(1)})
        if len(out) >= n:
            break
    return out


def search(q: str, n: int = 6) -> list[dict]:
    """两个引擎都问，结果合并去重。

    必应最近常返回整页垃圾站，只靠它一个会把有效结果全过滤掉，
    所以不再「一个挂了换下一个」，而是两家一起问。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for fn in (bing_rss, sogou):
        try:
            for h in fn(q, n):
                u = h.get("url") or ""
                if u and u not in seen and not _junk(h):
                    seen.add(u)
                    out.append(h)
        except Exception:                 # noqa: BLE001 —— 单个引擎挂了不影响另一个
            continue
    return out[:n * 2]
