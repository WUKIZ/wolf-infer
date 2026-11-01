"""远端大脑：DeepSeek（OpenAI 兼容接口）。

手机只当"家"：本地 0.7M 模型负责反射，真正的推理走这里。
  POST https://api.deepseek.com/chat/completions
  model: deepseek-chat

三条纪律：
  1. key 只从环境变量 DEEPSEEK_API_KEY 或 home/state/secrets.json（0600）读，
     不进代码、不进日志、不打印
  2. 每次请求过 permit（请求数/字节）和静默窗口；被拒就返回 None，由调用方回落本地
  3. 任何异常都只返回 None —— 远端挂了不该拖垮整机
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

def open_v4(req, timeout: float):
    """只在这次请求里把 DNS 锁到 IPv4。

    安卓的 4G/5G 网络常给 api.deepseek.com 解析出 IPv6，而这条路会被中间设备
    RST（实测 [Errno 104]，同一时刻 IPv4 是 200）。作用域限定在一次调用，
    不污染进程里别的 socket。
    """
    import socket

    orig = socket.getaddrinfo
    socket.getaddrinfo = lambda h, p, f=0, *a, **k: orig(h, p, socket.AF_INET, *a, **k)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    finally:
        socket.getaddrinfo = orig


# 人格与规则从本地私有文件读取（不入库）；缺失时用最小占位
_PRIV = Path(os.environ.get('WOLF_HOME', 'home/private')) / 'system_prompt.py'
SYSTEM = ("你是一个本地助手。回答直接、简短，不确定就说不确定。",)
if _PRIV.exists():
    exec(compile(_PRIV.read_text(encoding='utf-8'), str(_PRIV), 'exec'))

DEFAULT_SECRET = Path(
    os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
) / "wolf" / "secrets.json"


class RemoteBrain:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 permit=None, trail=None, monitor=None,
                 secrets_paths: Optional[List[Path]] = None,
                 key_name: str = "deepseek",
                 clock=time.time) -> None:
        cfg = cfg or {}
        self.base = str(cfg.get("base_url", "https://api.deepseek.com")).rstrip("/")
        self.model = str(cfg.get("model", "deepseek-chat"))
        self.timeout = float(cfg.get("timeout_s", 30))
        self.max_tokens = int(cfg.get("max_tokens", 220))
        self.temperature = float(cfg.get("temperature", 0.7))
        # 记忆摘录要不要发给远端：默认不发。"学到的先留在家门外"也包括别把家里的话送出去。
        self.send_context = bool(cfg.get("send_context", False))
        self.permit = permit
        self.trail = trail
        self.monitor = monitor
        self.clock = clock
        self.secrets_paths = [Path(p) for p in (secrets_paths or [DEFAULT_SECRET])]
        self.key_name = key_name
        self.last_status = 0
        self.last_tokens = 0
        self._key = ""
        self._refresh(force=True)

    # ---------- key ----------
    def _refresh(self, force: bool = False) -> None:   # force 参数保留，恒读最新
        """环境变量优先；文件每次都读 —— 文件只有几十字节，热重载比缓存值钱。"""
        env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if env:
            self._key = env
            return
        self._key = ""
        for path in self.secrets_paths:
            try:
                data = json.loads(path.read_text("utf-8"))
                key = str(data.get(self.key_name) or "").strip()
            except (OSError, ValueError):
                continue
            if key:
                self._key = key
                return

    def available(self) -> bool:
        self._refresh()
        return bool(self._key)

    # ---------- 说话 ----------
    def _local(self, text: str) -> Optional[str]:
        """远端不可用时的本地兜底（走 serve.py）。

        只在技术性失败时启用：permit 拒绝、静默窗口属于策略，不在此列。
        """
        try:
            from core.moe_client import generate
            out = generate(text, max_tokens=min(self.max_tokens, 128)).strip()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if self.monitor is not None:
                self.monitor.record("remote.local_failed", error=str(exc)[:80])
            return None
        if self.monitor is not None:
            self.monitor.record("remote.local_fallback", chars=len(out))
        return out or None

    def chat(self, text: str, context: Optional[List[str]] = None,
             system: Optional[str] = None) -> Optional[str]:
        """远端优先；远端技术性失败时降级到本地推理服务。"""
        if not self.available():
            return self._local(text)
        if self.trail is not None and self.trail.quiet_left() > 0:
            return None
        if self.permit is not None:
            ok, _ = self.permit.spend("net.request", host="api.deepseek.com")
            if not ok:
                return None

        messages = [{"role": "system", "content": system or SYSTEM}]
        if context and self.send_context:
            messages.append({"role": "system",
                             "content": "相关记忆：" + "；".join(c[:80] for c in context[:2])})
        messages.append({"role": "user", "content": text})

        payload = json.dumps({
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stream": False,
        }).encode("utf-8", "ignore")
        req = urllib.request.Request(
            self.base + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"})
        try:
            with open_vi(req, self.timeout) as resp:
                raw = resp.read(1 << 20)
        except urllib.error.HTTPError as exc:
            self.last_status = exc.code
            if exc.code in (401, 403) and self.monitor is not None:
                self.monitor.record("remote.auth_failed", code=exc.code)
            return self._local(text)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            self.last_status = 0
            return self._local(text)

        if self.permit is not None:
            self.permit.spend("net.bytes", len(raw))

        try:
            data = json.loads(raw.decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            return self._local(text)
        if self.monitor is not None:
            self.monitor.record("remote", model=self.model,
                                tokens=usage.get("total_tokens", 0))
        self.last_status = 200
        self.last_tokens = int(usage.get("total_tokens", 0) or 0)
        return content or None
