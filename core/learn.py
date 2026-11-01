"""出门学习。学到的东西不许直接进家。

一轮完整外出：
  离家 → 抓取 → 过本能闸 → 落在门槛外 → 评估算力 → 训练 → 回家

算力策略：
  本地只做启动与推理；训练先估成本，超出预算就"租"（此处是租约抽象，
  接真实算力时实现 LeaseBroker.request 即可）。
"""
from __future__ import annotations

import hashlib
import html
import http.client
import ipaddress
import json
import os

from core.meminfo import available_mb
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from . import instinct
from .home import Home

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[ \t\r\f\v]+")


def clean_html(raw: str, limit: int = 4000) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = TAG.sub(" ", text)
    text = html.unescape(text)
    text = WS.sub(" ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()[:limit]


def url_allowed(url: str) -> Optional[str]:
    """只许 http/https，且不许指向本机/内网/链路本地。

    没有这道检查，curriculum 里写一个 file:///etc/passwd 就能把本机文件读进语料；
    写 http://127.0.0.1:8080 就能拿它当内网探针。config.json 不在守卫清单里，
    所以这条必须在代码里挡，而不是指望配置不被改。
    """
    try:
        u = urllib.parse.urlparse(url)
    except ValueError:
        return "parse"
    if u.scheme not in ("http", "https"):
        return f"scheme:{u.scheme or '-'}"
    host = (u.hostname or "").strip().lower()
    if not host:
        return "host:empty"
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return "host:local"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # 域名：先解析一遍，落点是内网就拒（TOCTOU 仍在，见文档残余风险）
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return None          # 解析不了就交给 fetch 报 net 错误，不在这里假装成功
        for info in infos:
            try:
                resolved = ipaddress.ip_address(info[4][0])
            except ValueError:
                continue
            if (resolved.is_private or resolved.is_loopback
                    or resolved.is_link_local or resolved.is_reserved):
                return f"dns:{resolved}"
        return None
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return f"ip:{ip}"
    return None


@dataclass
class Harvest:
    source: str
    text: str = ""
    ok: bool = False
    rejected: str = ""
    bytes_in: int = 0
    raw: str = ""          # 原始 HTML，只有爬虫要跟链接时才填


def _eval_loss(model, batch) -> float:
    """只前向的验证损失。出门学一轮也要有验收，不然就是在盲改权重。"""
    import numpy as np

    x, y = batch[:, :-1], batch[:, 1:]
    logits, _ = model.forward(x)
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    b, t, v = logits.shape
    return float(-logp.reshape(-1, v)[np.arange(b * t), y.reshape(-1)].mean())


class Learner:
    def __init__(self, home: Home, timeout: float = 12.0, ua: str = "lone-wolf/0.1",
                 max_corpus_bytes: int = 1_000_000, monitor=None, trail=None,
                 permit=None) -> None:
        self.home = home
        self.timeout = timeout
        self.ua = ua
        self.max_corpus_bytes = max_corpus_bytes
        self.monitor = monitor
        self.trail = trail
        self.permit = permit

    # ---------- 出门取知识 ----------
    def fetch(self, url: str, limit: int = 4000, keep_html: bool = False) -> Harvest:
        return self.request(url, method="GET", limit=limit, keep_html=keep_html)

    def post(self, url: str, fields: dict, limit: int = 4000,
             keep_html: bool = False) -> Harvest:
        return self.request(url, method="POST", data=fields, limit=limit, keep_html=keep_html)

    def request(self, url: str, method: str = "GET", data: dict | None = None,
                limit: int = 4000, keep_html: bool = False) -> Harvest:
        h = Harvest(source=url)
        if self.trail is not None and self.trail.quiet_left() > 0:
            h.rejected = f"quiet:{self.trail.quiet_left():.0f}s"
            self._note(url, h)
            return h
        bad = url_allowed(url)
        if bad is not None:
            h.rejected = f"url:{bad}"
            self._note(url, h)
            return h
        if self.permit is not None:
            ok, why = self.permit.spend(
                "net.request", host=urllib.parse.urlparse(url).hostname or "")
            if not ok:
                h.rejected = f"permit:{why}"
                self._note(url, h)
                return h
        body = urllib.parse.urlencode(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers={"User-Agent": self.ua})
        if body is not None:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(1 << 20)
        # http.client.IncompleteRead 不是 OSError 的子类，漏了它整轮外出会直接崩
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                OSError, ValueError) as exc:
            h.rejected = f"net:{type(exc).__name__}"
            self._note(url, h)
            return h

        h.bytes_in = len(raw)
        if self.permit is not None:
            ok, why = self.permit.spend("net.bytes", len(raw))
            if not ok:
                h.rejected = f"permit:{why}"
                self._note(url, h)
                return h
        decoded = raw.decode("utf-8", "replace")
        if keep_html:
            h.raw = decoded[:512_000]
        text = clean_html(decoded, limit)

        verdict = instinct.quarantines(text)
        if verdict is not None:
            h.rejected = f"instinct:{verdict.category}:{verdict.level.name}"
            self._note(url, h)
            return h
        h.text, h.ok = text, True
        self._note(url, h)
        return h

    def _note(self, url: str, h: "Harvest") -> None:
        if self.monitor is not None:
            self.monitor.record("fetch", url=url, ok=h.ok,
                                bytes=h.bytes_in, rejected=h.rejected)

    def harvest(self, urls: Iterable[str]) -> List[Harvest]:
        out: List[Harvest] = []
        for url in urls:
            h = self.fetch(url)
            out.append(h)
            if h.ok:
                # 门外的暂存点：此物尚未成为"我"
                self.home.deposit({
                    "kind": "knowledge",
                    "source": h.source,
                    "sha": hashlib.sha256(h.text.encode()).hexdigest()[:12],
                    "text": h.text,
                    "at": time.time(),
                })
        return out

    # ---------- 算力 ----------
    def local_capacity(self) -> dict:
        mem_kb = int(available_mb() * 1024)
        return {"cores": os.cpu_count() or 1, "mem_mb": mem_kb // 1024}

    def estimate(self, steps: int, batch: int, block: int, dims: int,
                 vocab: int = 88, n_layer: int = 2, n_head: int = 4) -> dict:
        """别用纸面 FLOPs 估 —— 这个尺寸下 numpy 的开销比算术本身大。

        实测这一步：跑 3 步真模型，取中位数当单步成本，再外推。
        """
        import numpy as np

        from .model import Config, TinyGPT

        cfg = Config(vocab=vocab, block=block, n_layer=n_layer,
                     n_head=n_head, n_embd=dims)
        probe = TinyGPT(cfg, np.random.default_rng(0))
        arr = np.random.default_rng(0).integers(0, vocab, (batch, block + 1))

        probe.step(arr)  # 预热：第一笔要付内存分配与导入的账
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            probe.step(arr)
            times.append(time.perf_counter() - t0)
        per_step = sorted(times)[1]

        cap = self.local_capacity()
        total = per_step * steps
        return {
            "per_step_s": round(per_step, 4),
            "local_seconds": round(total, 1),
            "capacity": cap,
            "needs_lease": total > 180,
        }

    # ---------- 训练 ----------
    def train(self, corpus_path: Path, steps: int = 240, block: int = 32,
              n_layer: int = 2, n_embd: int = 64, batch: int = 8,
              lr: float = 3e-4, allow_reshape: bool = False) -> dict:
        import numpy as np

        from .bpe import load_tokenizer

        raw = corpus_path.read_text("utf-8")
        if len(raw) > self.max_corpus_bytes:
            # 只截本轮训练用的窗口，绝不改写语料文件 —— 之前这里会把
            # corpus.txt 直接裁掉，一次误调用就丢了 400KB 语料。
            print(f"[learn] 语料 {len(raw)}B > {self.max_corpus_bytes}B，本轮只用尾部",
                  file=sys.stderr)
            raw = raw[-self.max_corpus_bytes:]
        text = raw
        out = self.home.memory_dir.parent / "models" / "tiny.bin"

        # 分词器必须跟现役权重一致：权重是 BPE 的就不能拿 char 级去编，
        # 否则 id 全错位，形状检查还会一直 skip，等于永远学不进去。
        prev_model = None
        if out.exists():
            try:
                prev_model = TinyGPT.load(out)
            except Exception as exc:      # noqa: BLE001 - 权重坏了不该挡住这一轮
                print(f"[train] 旧权重载入失败：{type(exc).__name__}", file=sys.stderr)
        tok = load_tokenizer(getattr(prev_model, "tok", None),
                             getattr(prev_model, "chars", None), text=text)
        ids = tok.encode(text)
        if len(ids) < block + 2:
            raise ValueError("语料太短，凑不够一个窗口")

        out = self.home.memory_dir.parent / "models" / "tiny.bin"
        # 形状/词表变了就不能偷偷覆盖：那等于把之前学到的全扔掉。
        # 实测过：守护进程用默认 64 维去训 80 维的权重，一次就把好模型换成了噪声。
        if out.exists() and not allow_reshape:
            prev_cfg = prev_model.cfg if prev_model is not None else None
            if prev_cfg is not None:
                have = (prev_cfg.n_embd, prev_cfg.n_layer, prev_cfg.block, prev_cfg.vocab)
                want = (n_embd, n_layer, block, tok.vocab)
                if have != want:
                    return {"steps": 0, "skipped": f"shape_mismatch:{have}->{want}",
                            "saved": str(out)}

        plan = self.estimate(steps, batch, block, n_embd, vocab=tok.vocab,
                             n_layer=n_layer)
        # 240 步在这台机器上约 23s 且零输出 —— 看着像死机，其实在算。
        # 每 10% 打一行到 stderr，长跑有呼吸。
        cpu_only = plan["needs_lease"]
        if cpu_only:
            print(f"[train] 预估 {plan['local_seconds']}s 超出本地预算，租约不可用 → 本机跑",
                  file=sys.stderr)
        cfg = Config(vocab=tok.vocab, block=block, n_layer=n_layer, n_head=4, n_embd=n_embd)
        model, warm = None, False
        if out.exists():
            # 续训而不是重训：每次出门都从零开始等于把之前学到的全丢掉。
            # 只有分词器和形状都对得上才敢接，否则 id 会整体错位。
            prev = prev_model
            if prev is not None:
                c = prev.cfg
                same_shape = (c.vocab, c.block, c.n_embd, c.n_layer) == (
                    tok.vocab, block, n_embd, n_layer)
                same_tok = (getattr(tok, "to_payload", None) is None
                            or getattr(prev, "tok", None) == tok.to_payload())
                if same_shape and same_tok:
                    model, warm = prev, True
        if model is None:
            model = TinyGPT(cfg)

        # 续训只认一件事：验证损失有没有降。没降就不落盘 —— 实测 3e-3 的学习率
        # 在收敛后的权重上跑 20 步，能把 val 从 1.33 抬到 1.50，等于白学。
        val_batch = None
        val_before = None
        if warm and len(ids) > block + 3:
            # 和 tools/train.py 同一套口径：中段 10% 做 val。两边口径不一致的话，
            # 这边「变好了」在那边可能是变差，验收就成了自说自话。
            n_val = max(block + 2, int(len(ids) * 0.1))
            vstart = max(0, len(ids) // 2 - n_val // 2)
            val_ids = ids[vstart:vstart + n_val]
            vrng = np.random.default_rng(20261022)
            vix = vrng.integers(0, max(1, len(val_ids) - block - 1), 16)
            val_batch = np.array([val_ids[i:i + block + 1] for i in vix])
            val_before = _eval_loss(model, val_batch)
        rng = np.random.default_rng(int(time.time()))

        t0 = time.perf_counter()
        losses = []
        tick = max(1, steps // 10)
        for i in range(steps):
            ix = rng.integers(0, len(ids) - block - 1, batch)
            batch_arr = np.array([ids[i:i + block + 1] for i in ix])
            losses.append(model.step(batch_arr, lr=lr, weight_decay=0.01))
            if (i + 1) % tick == 0 or i + 1 == steps:
                print(f"[train] {i + 1}/{steps} loss={losses[-1]:.4f} "
                      f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)

        val_after = _eval_loss(model, val_batch) if val_batch is not None else None
        if val_before is not None and val_after is not None and val_after >= val_before:
            # 学了没变好：留旧权重，只报数。宁可这一轮白跑，也别把好模型换掉。
            return {
                "steps": steps, "warm_start": warm, "kept_prev": True,
                "val_before": round(val_before, 4), "val_after": round(val_after, 4),
                "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
                "seconds": round(time.perf_counter() - t0, 1),
                "est_seconds": round(plan["local_seconds"], 1), "saved": None,
            }

        out = self.home.memory_dir.parent / "models" / "tiny.bin"
        # 训练到一半被 kill 不该留下半个权重文件（load 会炸，整机退化）
        tmp = out.with_name(out.name + ".tmp")
        payload = getattr(model, "tok", None)
        if payload is None and hasattr(tok, "to_payload"):
            payload = tok.to_payload()
        model.save(tmp, chars=getattr(tok, "chars", []), tok=payload)
        os.replace(tmp, out)
        return {
            "steps": steps,
            "warm_start": warm,
            "val_before": round(val_before, 4) if val_before is not None else None,
            "val_after": round(val_after, 4) if val_after is not None else None,
            "loss_first": round(losses[0], 4),
            "loss_last": round(losses[-1], 4),
            "seconds": round(time.perf_counter() - t0, 1),
            "est_seconds": round(plan["local_seconds"], 1),
            "saved": str(out),
        }

    # ---------- 一整轮 ----------
    def cycle(self, urls: List[str], steps: int = 240, extra_text: str = "") -> dict:
        self.home.boot()
        self.home.leave()

        harvests = self.harvest(urls)
        from guard.quality import check as _quality

        good = []
        for h in harvests:
            if not h.ok:
                continue
            okq, why = _quality(h.text)
            if okq:
                good.append(h.text)
            else:
                print(f"[learn] 拒收 {h.source}: {why}", file=sys.stderr)

        corpus = self.home.root / "home" / "corpus.txt"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        with corpus.open("a", encoding="utf-8") as fh:
            for t in good:
                fh.write("\n" + t + "\n")
            if extra_text:
                fh.write("\n" + extra_text + "\n")

        if corpus.stat().st_size > self.max_corpus_bytes:
            # 24 小时跑下去语料会一直涨；留尾部，训练窗口只用近期的
            data = corpus.read_text("utf-8")
            tmp = corpus.with_name(corpus.name + ".tmp")
            tmp.write_text(data[-self.max_corpus_bytes:], "utf-8")
            os.replace(tmp, corpus)
            print(f"[learn] 语料超过 {self.max_corpus_bytes}B，保留尾部", file=sys.stderr)

        result = {
            "fetched": len(harvests),
            "accepted": len(good),
            "rejected": [f"{h.source}:{h.rejected}" for h in harvests if not h.ok],
        }
        if len(good) or extra_text:
            result |= self.train(corpus, steps=steps)
        self.home.enter()
        result["state"] = self.home.state
        return result
