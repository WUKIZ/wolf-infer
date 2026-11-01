"""质量门：长度、中文占比、HTML、英文比例、重复度、域相似度。

域参考集取同源语料开头 256 KB 连续采样：原实现按 400 个 60 字窗口稀疏采样，
20 MB 上覆盖率约 0.1%，同源文本的 3-gram 也撞不上，导致恒判 off_domain。
"""
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[\u4e00-\u9fff]")
WS = re.compile(r"\s+")
HTML = re.compile(r"<\s*(div|p|span|script|style|a\b|br|html|body)", re.I)
WORD = re.compile(r"[A-Za-z]{2,}")


def _grams(s: str, n: int = 3) -> set[str]:
    t = WS.sub("", s)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


@lru_cache(maxsize=1)
def _corpus_grams(chars: int = 262144) -> set[str]:
    p = ROOT / "home" / "corpus.txt"
    if not p.exists():
        return set()
    with p.open(encoding="utf-8") as fh:
        return _grams(fh.read(chars))


def check(text: str, base: set[str] | None = None, strict_domain: bool = True) -> tuple[bool, str]:
    t = (text or "").strip()
    if len(t) < 400:
        return False, "too_short"
    dense = len(WS.sub("", t))
    if len(CJK.findall(t)) / max(1, dense) < 0.5:
        return False, "not_chinese"
    if HTML.search(t):
        return False, "html"
    if len(WORD.findall(t)) > len(t) / 20:
        return False, "too_much_english"
    g = _grams(t)
    if not g:
        return False, "empty"
    if len(g) / max(1, len(t)) < 0.25:
        return False, "repetitive"
    if not strict_domain:
        return True, "ok"
    ref = base if base is not None else _corpus_grams()
    if ref:
        overlap = len(g & ref) / max(1, len(g))
        if overlap < 0.12:
            return False, f"off_domain:{overlap:.2f}"
    return True, "ok"


def accept(text: str, base: set[str] | None = None) -> bool:
    return check(text, base)[0]
