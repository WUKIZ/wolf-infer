"""本地浏览器应用：她有她的窗口，你有你的。

只绑 127.0.0.1 的小 HTTP 服务：
  GET  /              人看的控制台（状态、URL、表单、返回内容、跟她说句话）
  GET  /api/status    机器读的状态
  GET  /api/log?n=50  最近的抓取/归家事件
  POST /api/fetch     {url, method, data} → 抓取或提交表单，返回清洗后的文本
  POST /api/ask       {text} → 走认知栈，返回回答

所有出口都过 Learner：URL 边界、权限预算、本能闸、静默窗口。这个应用不是
后门，只是把她本来就有的那条路开了一扇窗。

安全：只绑 loopback；/api/* 要 X-Wolf-Token（页面里带），跨站读不到。
"""
from __future__ import annotations
import logging

import json
import secrets
import threading
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .search import search as web_search

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0f1115">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<title>本地助手 · 窗口</title>
<style>
 body{background:#0f1115;color:#d7dce5;font:14px/1.6 ui-monospace,Menlo,monospace;margin:0;padding:24px}
 h1{font-size:15px;font-weight:600;color:#9fb3c8;margin:0 0 12px}
 .bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
 input,textarea,select,button{background:#171b22;color:#d7dce5;border:1px solid #2a3140;border-radius:6px;padding:8px;font:inherit}
 input{flex:1;min-width:280px} textarea{width:100%;height:54px}
 button{cursor:pointer} button:hover{border-color:#4a5568}
 pre{background:#12161d;border:1px solid #222a36;border-radius:8px;padding:12px;white-space:pre-wrap;word-break:break-word;min-height:80px}
 #stat{color:#7f8c9f;margin-bottom:16px} .k{color:#8fb3a8}
 .row{display:flex;gap:8px;margin:12px 0}
 @media(max-width:560px){body{padding:14px} input{min-width:0;width:100%} .bar{flex-direction:column}}
</style></head><body>
<h1>本地助手 · 窗口 <span id="stat"></span></h1>
<form class="bar" action="/search" method="get" target="_blank">
  <input name="q" placeholder="搜索（Bing）—— 回车在新标签打开">
  <input type="hidden" name="token" value="__TOKEN__">
  <button>搜</button>
</form>
<div class="bar">
  <input id="url" placeholder="https://example.com" value="https://example.com">
  <select id="method"><option>GET</option><option>POST</option></select>
  <button onclick="go()">抓取 / 提交</button>
</div>
<textarea id="body" placeholder="POST 表单：q=本地助手&amp;page=1"></textarea>
<pre id="out">等待操作。</pre>
<div class="row">
  <input id="ask" placeholder="跟她说句话（你好 / 你是谁 / 主机在哪）">
  <button onclick="talk()">说</button>
</div>
<pre id="ans"></pre>
<script>
const TOKEN = "__TOKEN__";
async function api(path, payload){
  const r = await fetch(path, {method:"POST",
    headers:{"Content-Type":"application/json","X-Wolf-Token":TOKEN},
    body:JSON.stringify(payload||{})});
  return await r.json();
}
document.addEventListener("DOMContentLoaded",()=>{if("serviceWorker" in navigator){navigator.serviceWorker.register("/sw.js").catch(()=>{});}});
async function refresh(){
  const s = await (await fetch("/api/status",{headers:{"X-Wolf-Token":TOKEN}})).json();
  document.getElementById("stat").textContent =
    `态=${s.state} 记忆=${s.facts} 门槛=${s.threshold} 模型=${s.model} 权限=${s.permit_used}`;
}
async function go(){
  const url = document.getElementById("url").value.trim();
  const method = document.getElementById("method").value;
  const raw = document.getElementById("body").value.trim();
  const data = {};
  if (method === "POST" && raw) for (const p of raw.split("&")) { const [k,v]=p.split("="); if(k) data[k]=decodeURIComponent(v||""); }
  document.getElementById("out").textContent = "…";
  const r = await api("/api/fetch", {url, method, data});
  document.getElementById("out").textContent =
    r.ok ? `${r.url}  ${r.bytes}B  memory=${r.memory}\\n\\n${r.text}` : `拒绝：${r.rejected}`;
  refresh();
}
async function talk(){
  const text = document.getElementById("ask").value.trim();
  if(!text) return;
  document.getElementById("ans").textContent = "…";
  const r = await api("/api/ask", {text});
  document.getElementById("ans").textContent = r.answer || r.error || "";
}
refresh(); setInterval(refresh, 5000);
</script></body></html>
"""


class WebApp:
    def __init__(self, rt, host: str = "127.0.0.1", port: int = 8088) -> None:
        self.rt = rt
        self.host = host
        self.port = int(port)
        self.token = secrets.token_hex(8)
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- 状态 ----------
    def status(self) -> Dict[str, Any]:
        rt = self.rt
        mem = rt.memory.stats() if getattr(rt, "memory", None) is not None else {}
        try:
            model = f"{rt.cfg['model']['n_layers']}L/{rt.cfg['model']['hidden']}d"
        except Exception as exc:            # noqa: BLE001
            logging.warning('模型元信息缺失，回退占位名：%s', exc)
            model = "?"
        used = {}
        if getattr(rt, "permit", None) is not None:
            snap = rt.permit.snapshot()
            used = snap.get("used", {})
        return {
            "state": rt.home.state,
            "facts": len(rt.home.live.get("facts", [])),
            "threshold": len(list(rt.home.threshold_dir.glob("*.json"))),
            "working": mem.get("working", 0),
            "model": model,
            "permit_used": used,
            "at": __import__("time").strftime("%H:%M:%S"),
        }

    # ---------- 动作 ----------
    def do_fetch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = str(payload.get("url") or "").strip()
        method = str(payload.get("method") or "GET").upper()
        data = payload.get("data") or {}
        if not url:
            return {"ok": False, "rejected": "empty_url"}
        learner = self.rt.learner
        if method == "POST":
            h = learner.post(url, data if isinstance(data, dict) else {})
        else:
            h = learner.request(url, method="GET")
        out = {"ok": h.ok, "url": h.source, "bytes": h.bytes_in,
               "rejected": h.rejected, "text": h.text[:4000]}
        if h.ok and getattr(self.rt, "memory", None) is not None:
            verdict = self.rt.memory.remember(h.text, source=h.source, kind="knowledge")
            out["memory"] = verdict["tier"]
        return out

    def do_ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = str(payload.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty"}
        sig = self.rt.cortex.run(text)
        return {"ok": True, "answer": sig.response, "vetoed": sig.vetoed,
                "by": sig.veto_by, "social": sig.meta.get("social"), "trace": sig.trace}

    def log(self, n: int = 50) -> Dict[str, Any]:
        mon = getattr(self.rt, "monitor", None)
        if mon is None:
            return {"events": []}
        return {"events": mon.events()[-n:]}

    # ---------- 服务 ----------
    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "wolf-window/0.1"

            def log_message(self, fmt, *args):      # 安静点，别刷屏
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj: Dict[str, Any]) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

            def _authed(self) -> bool:
                return self.headers.get("X-Wolf-Token") == app.token

            def do_GET(self):                       # noqa: N802
                u = urlparse(self.path)
                if u.path == "/manifest.webmanifest":
                    q = parse_qs(u.query)
                    token = q.get("token", [app.token])[0]
                    body = json.dumps({
                        "name": "本地助手", "short_name": "本地助手",
                        "start_url": f"/?token={token}&src=pwa",
                        "display": "standalone", "orientation": "portrait",
                        "background_color": "#0f1115", "theme_color": "#0f1115",
                        "icons": [{"src": "/icon.svg", "sizes": "any",
                                   "type": "image/svg+xml", "purpose": "any"}],
                    }, ensure_ascii=False)
                    self._send(200, body.encode(), "application/manifest+json")
                    return
                if u.path == "/icon.svg":
                    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 192 192'>"
                           "<rect width='192' height='192' rx='40' fill='#0f1115'/>"
                           "<circle cx='96' cy='96' r='58' fill='none' stroke='#c96f3a' stroke-width='6'/>"
                           "<text x='96' y='122' font-size='78' text-anchor='middle' fill='#e6e9ef'"
                           " font-family='monospace'>狼</text></svg>")
                    self._send(200, svg.encode(), "image/svg+xml")
                    return
                if u.path == "/sw.js":
                    sw = ("const C='wolf-v1';"
                          "self.addEventListener('install',e=>{e.waitUntil(caches.open(C)"
                          ".then(c=>c.addAll(['/icon.svg','/manifest.webmanifest'])))});"
                          "self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;"
                          "e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)))})")
                    self._send(200, sw.encode(), "application/javascript")
                    return
                if u.path == "/":
                    html = PAGE.replace("__TOKEN__", app.token)
                    self._send(200, html.encode(), "text/html; charset=utf-8")
                    return
                if u.path == "/search":
                    q = parse_qs(u.query)
                    if q.get("token", [""])[0] != app.token:
                        self._json(403, {"ok": False, "error": "bad_token"})
                        return
                    query = (q.get("q", [""])[0] or "").strip()
                    data = web_search(query, 10) if query else {"results": [], "error": "empty"}
                    rows = "".join(
                        f'<li><a href="{html_escape(r["url"], quote=True)}" target="_blank" '
                        f'rel="noopener noreferrer">{html_escape(r["title"])}</a>'
                        f'<div class="sn">{html_escape(r["snippet"][:200])}</div></li>'
                        for r in data.get("results", []))
                    err = html_escape(str(data.get("error") or ""))
                    page = ("<!doctype html><meta charset='utf-8'>"
                            "<title>搜索</title><style>body{background:#0f1115;color:#d7dce5;"
                            "font:14px/1.6 ui-monospace,monospace;padding:24px}a{color:#8fb3a8}"
                            ".sn{color:#7f8c9f;font-size:12px;margin:2px 0 12px}li{margin:10px 0}</style>"
                            f"<h3>搜索：{html_escape(query)}</h3>"
                            f"<p class='sn'>{err or str(len(data.get('results', []))) + ' 条结果'}"
                            " · 点标题在新标签打开</p><ol>" + rows + "</ol>")
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                    return
                if not self._authed():
                    self._json(403, {"ok": False, "error": "bad_token"})
                    return
                if u.path == "/api/status":
                    self._json(200, app.status())
                elif u.path == "/api/log":
                    q = parse_qs(u.query)
                    self._json(200, app.log(int(q.get("n", ["50"])[0])))
                else:
                    self._json(404, {"ok": False, "error": "not_found"})

            def do_POST(self):                      # noqa: N802
                if not self._authed():
                    self._json(403, {"ok": False, "error": "bad_token"})
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("payload")
                except (ValueError, json.JSONDecodeError):
                    self._json(400, {"ok": False, "error": "bad_json"})
                    return
                path = urlparse(self.path).path
                if path == "/api/fetch":
                    self._json(200, app.do_fetch(payload))
                elif path == "/api/ask":
                    self._json(200, app.do_ask(payload))
                else:
                    self._json(404, {"ok": False, "error": "not_found"})

        return Handler

    def start_background(self) -> int:
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        # 把地址和 token 留给本地助手自己读 —— 这就是"她能打开并操控"的那扇窗
        state = self.rt.home.state_dir / "web.json"
        try:
            state.write_text(json.dumps(
                {"url": f"http://{self.host}:{self.port}", "token": self.token,
                 "started": __import__("time").time()}, ensure_ascii=False), "utf-8")
        except OSError:
            pass
        return self.port

    def serve_forever(self) -> None:
        self.start_background()
        print(f"[web] http://{self.host}:{self.port}/?token={self.token}", flush=True)
        try:
            while True:
                self._thread.join(0.5)      # type: ignore[union-attr]
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        try:
            (self.rt.home.state_dir / "web.json").unlink()
        except OSError:
            pass
