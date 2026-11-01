"""给 core/remote.py 接上精确缓存。幂等：已经打过的会跳过。

改了三处：
  1. 顶部 import ReplyCache
  2. __init__ 里建 self.cache
  3. chat() 里查缓存 / 存缓存
锚点找不到就报错退出，不写半成品。
"""
from __future__ import annotations

import pathlib

P = pathlib.Path("core/remote.py")
s = P.read_text("utf-8")

if "llmcache" in s:
    print("已经打过补丁，跳过")
    raise SystemExit(0)

A_IMPORT = "from typing import Any, Dict, List, Optional"
A_INIT = "        self.clock = clock\n"
A_MSG = '        messages = [{"role": "system", "content": system or SYSTEM}]'
A_RET = "        return content or None"

missing = [name for name, anchor in
           (("import", A_IMPORT), ("init", A_INIT), ("messages", A_MSG), ("return", A_RET))
           if anchor not in s]
if missing:
    print("锚点缺失:", ", ".join(missing), "；未写入")
    raise SystemExit(1)

s = s.replace(A_IMPORT, A_IMPORT + "\n\nfrom .llmcache import ReplyCache", 1)
s = s.replace(
    A_INIT,
    A_INIT + "        self.cache = ReplyCache(\n"
             "            Path(__file__).resolve().parent.parent / \"home\" / \"state\" / \"llm_cache.db\")\n",
    1)
s = s.replace(
    A_MSG,
    "        sys_text = system or SYSTEM\n"
    "        ck = self.cache.key(self.model, sys_text, text)\n"
    "        cached = self.cache.get(ck)\n"
    "        if cached:\n"
    "            return cached\n"
    "        messages = [{\"role\": \"system\", \"content\": sys_text}]",
    1)
s = s.replace(
    A_RET,
    "        if content:\n"
    "            self.cache.put(ck, content)\n"
    + A_RET,
    1)

P.write_text(s, "utf-8")
print("patched core/remote.py")
