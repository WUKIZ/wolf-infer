"""从搜索结果到可读正文：搜 → 抓 → 清洗 → 截断。

只把正文交给模型，不交 HTML。一次研究最多取 budget 页，
每页截 chars/budget 字，塞进 context 后总长可控。
JS 渲染的站（头条文章、部分公众号）抓出来是空壳，直接跳过。
"""
from __future__ import annotations

import re

import search_multi

_MIN_BODY = 300          # 低于这个字数当作没抓到
_BLOCK = re.compile(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", re.S)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")

# 可信来源排在前面。搜索结果的排序里塞满 SEO 站，不排一下会抓到影视聚合页。
_TRUSTED = ("news.qq.com", "sina.com.cn", "sohu.com", "163.com", "gov.cn",
            "github.com", "csdn.net", "36kr.com", "jiemian.com", "thepaper.cn",
            "ithome.com", "cnbeta.com", "leiphone.com", "oschina.net",
            "cnblogs.com", "zhihu.com", "arxiv.org")


def _rank(url: str) -> int:
    return 0 if any(h in url for h in _TRUSTED) else 1


def _relevant(hit: dict, q: str) -> bool:
    """标题或摘要里得出现查询词，否则就是 SEO 垃圾站。

    「国产芯片 突破」这种查询，垃圾站的标题只有「国产视频」，
    不含「国产芯片」也不含「突破」，一条都过不了。
    """
    text = (hit.get("title", "") + " " + hit.get("snippet", "")).lower()
    words = [w.lower() for w in q.split() if len(w) >= 2]
    return any(w in text for w in words)


def text_of(url: str, timeout: int = 10) -> str:
    raw = search_multi.fetch(url, timeout)
    for enc in ("utf-8", "gbk"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        html = raw.decode("utf-8", "ignore")
    html = _BLOCK.sub(" ", html)
    return _SPACE.sub(" ", _TAG.sub(" ", html)).strip()


def _collect(hits: list[dict], q: str, budget: int, chars: int) -> list[str]:
    out: list[str] = []
    per = max(400, chars // budget)
    for h in hits:
        if len(out) >= budget:
            break
        if not _relevant(h, q):           # SEO 垃圾站，标题都对不上
            continue
        try:
            body = text_of(h["url"])
        except Exception:                 # noqa: BLE001 —— 单个站抓不动就换下一个
            continue
        if len(body) < _MIN_BODY:
            continue
        out.append(f"【{h['title']}】{body[:per]}")
    return out


_FILLER = ("今天", "昨天", "明天", "最近", "现在", "有什么", "怎么样",
           "是什么", "为什么", "帮我", "请问", "一下", "吗", "呢", "啊")


def _clean(q: str) -> str:
    """去掉口语虚词。搜索不认「今天有什么X」，只认「X」。"""
    t = q
    for f in _FILLER:
        t = t.replace(f, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t or q


def research(q: str, budget: int = 3, chars: int = 4000) -> list[str]:
    """先按原词搜，不行就化简再搜。

    实测：「今天有什么科技新闻」一条都搜不到；
    「科技新闻」能出，去掉「今天/有什么」就够。
    """
    base = _clean(q)
    head = base.rsplit(" ", 1)[0]
    variants = [q, q + " 新闻", base + " 新闻", base, head + " 新闻"]
    seen: set[str] = set()
    for query in variants:
        if query in seen:
            continue
        seen.add(query)
        hits = search_multi.search(query, n=budget + 6)
        hits.sort(key=lambda h: _rank(h.get("url", "")))   # 可信站先抓
        got = _collect(hits, query, budget, chars)
        if got:
            return got
    return []
