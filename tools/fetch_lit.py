#!/usr/bin/env python3
"""抓公版现代中文（鲁迅/朱自清/胡适）当语料，补掉模板语料的重复感。

    ./bin/py tools/fetch_lit.py --chars 90000 --out home/lit.txt

走 zh.wikisource 的 parse 接口（raw 是重定向壳，extracts 不支持），
拿正文 HTML、剥标签、丢掉页眉页脚那几行，再过一遍本能闸。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import instinct  # noqa: E402
from core.learn import clean_html  # noqa: E402

EXPAND = {"駱駝祥子", "呼蘭河傳", "生死場", "啼笑因緣", "吶喊", "彷徨",
          "故事新編", "朝花夕拾", "沉淪"}


def subpages(title: str, limit: int = 80) -> list[str]:
    url = "https://zh.wikisource.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "allpages", "apprefix": title + "/",
         "aplimit": str(limit), "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": "lone-wolf/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [p["title"] for p in data.get("query", {}).get("allpages", [])]


TITLES = [
    # 鲁迅
    "狂人日記", "孔乙己", "藥", "故鄉", "阿Q正傳", "祝福", "在酒樓上",
    "傷逝", "孤獨者", "明天", "一件小事", "風波", "端午節", "白光",
    "兔和貓", "鴨的喜劇", "社戲", "弟兄", "離婚", "幸福的家庭", "肥皂",
    "長明燈", "示眾", "高老夫子", "補天", "奔月", "理水", "采薇", "鑄劍",
    "出關", "非攻", "起死", "吶喊", "彷徨", "故事新編", "朝花夕拾",
    # 朱自清
    "背影", "荷塘月色", "匆匆", "綠", "槳聲燈影裡的秦淮河", "兒女",
    "溫州的蹤跡", "阿河", "白種人——上帝的驕子", "春", "威尼斯", "冬天",
    # 郁達夫
    "沉淪", "春風沉醉的晚上", "薄奠", "故都的秋", "一個人在途上",
    # 老舍
    "駱駝祥子", "月牙兒", "我這一輩子", "茶館", "濟南的冬天", "趵突泉",
    "想北平", "北京的春節",
    # 蕭紅
    "呼蘭河傳", "生死場", "小城三月", "回憶魯迅先生",
    # 徐志摩
    "再別康橋", "偶然", "沙揚娜拉", "我所知道的康橋",
    # 胡適
    "差不多先生傳", "文學改良芻議", "容忍與自由",
    # 梁啟超
    "少年中國說", "敬業與樂業",
    # 張恨水
    "啼笑因緣",
]

JUNK = re.compile(r"(本作品收錄於|姊妹[计計]划|作者：|公有领域|公有領域|Public domain|"
                  r"PD-old|←|→|校對|来源：|來源：|版本信息|本作品|版權|版权)")
DATE_ONLY = re.compile(r"^[\s\d年月日.,，。、·—-]{1,24}$")


def fetch(title: str, timeout: float = 20.0) -> str:
    url = "https://zh.wikisource.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "parse", "page": title, "prop": "text", "variant": "zh-hans",
         "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": "lone-wolf/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    html = data["parse"]["text"]
    text = clean_html(html, limit=200_000)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or JUNK.search(s) or DATE_ONLY.match(s):
            continue
        lines.append(s)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chars", type=int, default=90000)
    p.add_argument("--out", default=str(ROOT / "home" / "lit.txt"))
    a = p.parse_args()

    parts, total = [], 0
    for title in TITLES:
        if total >= a.chars:
            break
        targets = [title]
        if title in EXPAND:
            try:
                subs = subpages(title, 40)
                if subs:
                    targets = subs
            except Exception:          # noqa: BLE001
                pass
        for t in targets:
            if total >= a.chars:
                break
            try:
                text = fetch(t)
            except Exception as exc:        # noqa: BLE001
                print(f"[lit] {t} 失败 {type(exc).__name__}", file=sys.stderr)
                continue
            if instinct.quarantines(text) is not None:
                print(f"[lit] {t} 没过本能闸，跳过", file=sys.stderr)
                continue
            parts.append(text)
            total += len(text)
            print(f"[lit] {t} +{len(text)} 字符")

    body = "\n\n".join(parts)[:a.chars]
    Path(a.out).write_text(body, "utf-8")
    print(f"[lit] 总计 {len(body)} 字符 → {a.out}")
    print("[lit] 样例:", body[:120].replace("\n", " / "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
