#!/usr/bin/env python3
"""换 key：写到同步目录之外，权限 0600，顺手清掉旧的在家副本。

    ./bin/py tools/rotate_key.py --key sk-新的

不会回显 key。换完不用重启 —— core/remote.py 按 mtime 热重载。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--name", default="deepseek", help="secrets.json 里的字段名，如 groq")
    a = p.parse_args()
    key = a.key.strip()
    if len(key) < 20 or any(c.isspace() for c in key):
        print("[key] 格式不对：长度应 ≥ 20 且不含空白", file=sys.stderr)
        return 2

    target = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
                  ) / "wolf" / "secrets.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    try:
        data = json.loads(target.read_text("utf-8"))
    except (OSError, ValueError):
        data = {}
    data[a.name] = key
    target.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    os.chmod(target, 0o600)

    legacy = ROOT / "home" / "state" / "secrets.json"
    if legacy.exists():
        legacy.unlink()
        print(f"[key] 已删除同步目录里的旧副本：{legacy}")

    print(f"[key] 已写入 {target}（0600），无需重启")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
