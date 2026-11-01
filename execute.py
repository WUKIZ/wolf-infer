"""代码执行沙盒：让本地助手能写、能跑、能看结果。

三层限制，缺一层都不安全：
  1. 语法级 —— AST 扫描，拒绝 import os/socket/subprocess/shutil 这类能逃逸的模块
  2. 进程级 —— 超时 5 秒、输出截断 8KB、工作目录锁死在 home/sandbox/
  3. 环境级 —— 只传白名单环境变量，DeepSeek key 之类绝不带进子进程

这不是虚拟机。它挡的是「模型写错/被哄骗写出危险代码」，
不是「专业攻击者」。真要隔离得靠容器，沙盒环境里没有。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOX = ROOT / "home" / "sandbox"

BANNED = {"os", "subprocess", "shutil", "socket", "ctypes", "multiprocessing",
          "pty", "signal", "sys", "importlib", "builtins"}

# 环境变量白名单：只给跑 Python 必需的，其余一律不带
_KEEP = ("PATH", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH", "LANG", "LC_ALL",
         "SSL_CERT_FILE", "HOME", "TMPDIR")


def check(code: str) -> str | None:
    """语法级检查。返回错误描述，通过返回 None。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"语法错误：{exc.msg}（第 {exc.lineno} 行）"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED:
                    return f"禁止 import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BANNED:
                return f"禁止 from {node.module} import ..."
        elif isinstance(node, ast.Call):
            fn = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if fn in {"eval", "exec", "compile", "__import__"}:
                return f"禁止调用 {fn}"
    return None


def run(code: str, timeout: int = 5) -> dict:
    """跑一段 Python，返回 {ok, out, err}。任何异常都不往外抛。"""
    err = check(code)
    if err:
        return {"ok": False, "out": "", "err": err}

    BOX.mkdir(parents=True, exist_ok=True)
    script = BOX / "_run.py"
    script.write_text(code, "utf-8")

    env = {k: v for k, v in os.environ.items() if k in _KEEP}
    try:
        p = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BOX), env=env, timeout=timeout,
            capture_output=True, text=True,
        )
        return {"ok": p.returncode == 0,
                "out": (p.stdout or "")[:8000],
                "err": (p.stderr or "")[:2000]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "err": f"超时 {timeout}s，进程已杀"}
    except Exception as exc:                      # noqa: BLE001 —— 跑不动也不该拖垮对话
        return {"ok": False, "out": "", "err": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    for demo in ("print(2**10)", "import os\nprint(os.getcwd())",
                 "for i in range(10**9): pass"):
        r = run(demo, timeout=2)
        print(repr(demo[:30]), "→", r)
