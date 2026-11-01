#!/usr/bin/env python3
r"""中文显示配置：终端输出编码 + matplotlib 字体。

终端本身显示中文靠的是 Termux 的字体文件（~/.termux/font.ttf），Python 管不到；
这里只管两件 Python 能管的事：
  1. stdout/stderr 强制 UTF-8（locale 是 C 的时候，中文会变成 \u 转义或直接报错）
  2. matplotlib 找到并注册一个真正含 CJK 字形的字体，再用它的**真实家族名**设 rcParams
     —— 家族名不能靠猜，"Noto Sans CJK SC" 和 "Noto Sans Mono CJK SC" 是两个字体。

用法：
    from cnfont import setup
    setup()                      # 只配编码
    setup(matplotlib_cjk=True)   # 再配图表字体，返回使用的字体路径
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 常见落点：Termux 的 fontconfig 目录、用户目录、Debian 系系统目录
CANDIDATES = (
    "{prefix}/share/fonts/TTF/NotoSansMonoCJKsc-Regular.otf",
    "{prefix}/share/fonts/TTF/NotoSansCJKsc-Regular.otf",
    "{prefix}/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "{prefix}/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/system/fonts/NotoSansCJK-Regular.ttc",
)


def _encoding() -> str:
    """把标准流钉在 UTF-8 上。已经对了就什么都不做。"""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc not in ("utf-8", "utf8") and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return (getattr(sys.stdout, "encoding", "") or "").lower()


def find_cjk_font() -> Path | None:
    prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
    for tmpl in CANDIDATES:
        p = Path(tmpl.format(prefix=prefix))
        if p.is_file() and p.stat().st_size > 1_000_000:   # 含 CJK 的字体都很大，小文件是错的
            return p
    return None


def _has_cjk(path: Path) -> bool:
    try:
        from matplotlib.ft2font import FT2Font
        face = FT2Font(str(path))
        return face.get_char_index(0x5B64) != 0      # 「孤」
    except Exception:                                # noqa: BLE001
        return False


def setup(matplotlib_cjk: bool = False) -> dict:
    info = {"encoding": _encoding(), "font": None, "family": None}
    if not matplotlib_cjk:
        return info
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        info["error"] = "matplotlib 未安装：pkg install python-matplotlib"
        return info
    matplotlib.use("Agg", force=False)               # 无显示环境也能出图

    path = find_cjk_font()
    if path is None:
        info["error"] = "没找到含 CJK 的字体文件，先跑 setup-cn.sh"
        return info
    if not _has_cjk(path):
        info["error"] = f"{path.name} 里没有中文字形（文件不对）"
        return info

    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [family] + [
        f for f in matplotlib.rcParams["font.sans-serif"] if f != family]
    matplotlib.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免方块
    info.update(font=str(path), family=family)
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(setup(matplotlib_cjk=True), ensure_ascii=False))
