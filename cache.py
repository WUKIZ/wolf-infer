"""磁盘缓存：同一句话不重复花 token，同一个词不重复打网站。

diskcache 5.6.3 —— SQLite 后端，进程安全，TTL 到点自动失效。
缓存落在 home/state/cache/，跟其它状态一起在工作区，重启不丢。

命中率靠两件事：归一化（大小写、全半角、标点、空白都不算差异），
以及调用方给的 scope —— 情绪、上下文这类每次都变的东西不进 key。
"""
from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent

try:
    from diskcache import Cache
    _STORE: Any = Cache(str(ROOT / "home" / "state" / "cache"))
    _PERSIST = True
except ImportError:                       # 手机没装 diskcache 也能跑：退化成进程内缓存
    class _Memory:
        def __init__(self) -> None:
            self.d: dict = {}

        def get(self, k, default=None):
            return self.d.get(k, default)

        def set(self, k, v, expire=None):
            self.d[k] = v
            return v

        def __len__(self):
            return len(self.d)

    _STORE = _Memory()
    _PERSIST = False
_NOISE = re.compile(r"[\s\W_]+")


def norm(text: str) -> str:
    """NFKC → 小写 → 去空白和标点。换个标点问同一句话，算同一句。"""
    return _NOISE.sub("", unicodedata.normalize("NFKC", text or "").lower())


def key(*parts: Any) -> str:
    """把参数压成 16 字节摘要。字符串先归一化，其它按类型分段。"""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        s = norm(p) if isinstance(p, str) else repr(p)
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get(k: str) -> Any:
    return _STORE.get(k)


def put(k: str, value: Any, ttl: int) -> Any:
    _STORE.set(k, value, expire=ttl)
    return value


def memo(ttl: int) -> Callable:
    """给纯函数加缓存：参数进 key，返回值落盘。"""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrap(*a: Any, **kw: Any) -> Any:
            k = key(fn.__name__, a, tuple(sorted(kw.items())))
            hit = _STORE.get(k)
            if hit is not None:
                return hit
            return _STORE.set(k, fn(*a, **kw), expire=ttl)
        return wrap
    return deco


def stats() -> str:
    d = ROOT / "home" / "state" / "cache"
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    return f"cache {len(_STORE)} 项 / {size / 1048576:.2f}MB"
