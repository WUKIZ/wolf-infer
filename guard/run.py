"""入口：先验封存，再上笼，最后跑主程序。缺一个都不启动。"""
from __future__ import annotations
import os, runpy, sys
from pathlib import Path
from core.integrity import Guard
from .runtime import install

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    Guard(root).verify()          # 笼依赖封存：核心被动过，笼不启动
    os.environ["WOLF_CAGE"] = "1"
    if not install():
        print("[cage] 装上失败", file=sys.stderr)
        return 2
    sys.argv = [str(root / "main.py")] + sys.argv[1:]
    runpy.run_path(str(root / "main.py"), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
