"""自主循环：出门 → 觅食 → 过闸 → 落到门外 → 该训练就训练 → 观察 → 回家。

一步只做一件事，每步都留 event —— 循环不需要日志去猜它在干什么。

    event = {"t", "state", "kind": "forage"|"conceal"|"train"|"idle", ...}

出门顺序和原文对齐：
  - 家门只承担启动与推理，训练按报价决定本地还是外租
  - 带回来的东西放门槛，不进家；离家那一刻才回归本身
  - 观察层判定要回家，就先 conceal，再谈别的
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional



class Wolf:
    def __init__(
        self,
        rt,
        learner,
        store,
        broker,
        concealer,
        curriculum: Optional[List[str]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        browser=None,
    ) -> None:
        self.rt = rt
        self.learner = learner
        self.browser = browser
        self.store = store
        self.broker = broker
        self.concealer = concealer
        self.curriculum = list(curriculum or [])
        cfg = cfg or {}
        self.interval_s = float(cfg.get("interval_s", 15.0))
        self.train_every_bytes = int(cfg.get("train_every_bytes", 4096))
        self.train_steps = int(cfg.get("train_steps", 40))
        self.train_lr = float(cfg.get("train_lr", 3e-4))
        self.train_min_interval_s = float(cfg.get("train_min_interval_s", 1800))
        self.crawl_pages = int(cfg.get("crawl_pages", 1))
        self.crawl_depth = int(cfg.get("crawl_depth", 1))
        self.crawl_delay = float(cfg.get("crawl_delay_s", 0.3))
        self.max_steps = int(cfg.get("max_steps", 0))   # 0 = 一直跑
        self.clock = clock
        self._cursor = 0
        self._pending_bytes = 0
        self._last_train = 0.0
        self.events: List[Dict[str, Any]] = []

    # ---------- 一步 ----------
    def step(self) -> Dict[str, Any]:
        rt = self.rt
        event: Dict[str, Any] = {"t": time.time(), "state": rt.home.state, "kind": "idle"}

        # 1. 观察：先判要不要回家 —— 比觅食优先
        go_home, why = rt.obs.tick()
        event["exposure"] = rt.obs.history[-1].score if rt.obs.history else 0.0
        if go_home:
            event |= {"kind": "conceal", "detail": self.concealer.conceal(why)}
            rt.obs.rebaseline()
            self._record(event)
            return event

        # 观察层没信号不等于安全：出门超过硬上限也必须回家
        max_out = float(rt.cfg.get("observer", {}).get("max_excursion_s", 900.0))
        if rt.home.overdue(max_out):
            event |= {"kind": "conceal",
                      "detail": self.concealer.conceal("max_excursion")}
            rt.obs.rebaseline()
            self._record(event)
            return event

        # 2. 在家里待够了就出门
        if rt.home.state != EXCURSION:
            min_dwell = float(rt.cfg.get("observer", {}).get("min_dwell_s", 3.0))
            if not rt.home.can_leave(min_dwell):
                self._record(event)
                return event
            rt.home.leave()

        # 3. 觅食：一次取一个题目；crawl_pages>1 时改成同站小爬
        if self.curriculum:
            url = self.curriculum[self._cursor % len(self.curriculum)]
            self._cursor += 1
            if self.crawl_pages > 1:
                event |= self._crawl(url)
            else:
                h = self._fetch(url)
                event |= {"kind": "forage", "url": url, "ok": h.ok,
                          "rejected": h.rejected, "bytes": h.bytes_in}
                if h.ok:
                    self._ingest(h, event)

        # 4. 攒够了就训练；训练前先报价，超预算才去租
        if self._pending_bytes >= self.train_every_bytes and self.train_steps > 0:
            event |= self._train()

        self._record(event)
        return event

    def _param_count(self) -> int:
        m = self.rt.cfg.get("model", {})
        d = int(m.get("hidden", 80))
        layers = int(m.get("n_layers", 2))
        return 12 * layers * d * d + 2 * 4096 * d      # 词表按 4096 上限估

    def _fetch(self, url: str):
        """有窗口就走窗口：应用没开时才直连。"""
        from .learn import Harvest

        if self.browser is not None:
            r = self.browser.fetch(url)
            if "error" not in r:
                return Harvest(source=url, ok=bool(r.get("ok")),
                               text=str(r.get("text") or ""),
                               rejected=str(r.get("rejected") or ""),
                               bytes_in=int(r.get("bytes") or 0))
        return self.learner.fetch(url)

    def _ingest(self, h, event: Dict[str, Any]) -> None:
        """一页内容落两处：共用主机（门外）+ 记忆闸（本机门槛）。"""
        rt = self.rt
        sha = hashlib.sha256(h.text.encode()).hexdigest()[:12]
        self.store.put(f"facts/{sha}", {
            "sha": sha, "source": h.source, "text": h.text,
            "at": time.time(), "device": self.store.device,
        })
        mem = getattr(rt, "memory", None)
        if mem is not None:
            verdict = mem.remember(h.text, source=h.source, kind="knowledge")
            event["memory"] = verdict["tier"]
            if verdict["tier"] != "discard":
                self._pending_bytes += len(h.text)
        else:
            rt.home.deposit({"kind": "knowledge", "source": h.source,
                             "sha": sha, "text": h.text, "at": time.time()})
            self._pending_bytes += len(h.text)

    def _crawl(self, url: str) -> Dict[str, Any]:
        from .crawl import Crawler

        crawler = Crawler(self.learner, permit=getattr(self.rt, "permit", None),
                          max_pages=self.crawl_pages, max_depth=self.crawl_depth,
                          delay_s=self.crawl_delay)
        pages = crawler.crawl([url])
        event: Dict[str, Any] = {"kind": "crawl", "url": url, "pages": len(pages),
                                 "ok": sum(1 for p in pages if p.ok),
                                 "rejected": [p.rejected for p in pages if not p.ok][:5]}
        for p in pages:
            if p.ok:
                self._ingest(p, event)
        return event

    def _train(self) -> Dict[str, Any]:
        rt = self.rt
        from . import mem

        corpus = rt.home.root / "home" / "corpus.txt"
        if not corpus.exists():
            return {"kind": "train", "skipped": "no_corpus"}
        cap = float(rt.cfg.get("limits", {}).get("rss_mb", 700))
        est = mem.estimate_training_mb(
            self._param_count(), 6, int(rt.cfg.get("model", {}).get("context", 48)),
            int(rt.cfg.get("model", {}).get("hidden", 80)),
            int(rt.cfg.get("model", {}).get("n_layers", 2)))
        ok, why = mem.check(est, cap)
        if not ok:
            return {"kind": "train", "skipped": f"memory_cap:{why}"}
        if self._last_train and time.time() - self._last_train < self.train_min_interval_s:
            # 别每轮出门都重写一遍权重：5MB 的写放大 + 小步训练会把模型带偏
            return {"kind": "train", "skipped": "too_soon",
                    "next_in_s": round(self.train_min_interval_s - (time.time() - self._last_train))}
        permit = getattr(rt, "permit", None)
        if permit is not None:
            ok, why = permit.spend("train.steps", self.train_steps)
            if not ok:
                return {"kind": "train", "skipped": f"permit:{why}"}
        plan = self.learner.estimate(self.train_steps, 8, 32, 64, vocab=128, n_layer=2)
        quote = self.broker.quote(plan["per_step_s"], self.train_steps)
        lease = None
        if quote.needs_lease:
            lease = self.broker.acquire(quote.local_seconds)
        try:
            result = self.learner.train(
                corpus, steps=self.train_steps, lr=self.train_lr,
                block=int(rt.cfg.get("model", {}).get("context", 48)),
                n_layer=int(rt.cfg.get("model", {}).get("n_layers", 2)),
                n_embd=int(rt.cfg.get("model", {}).get("hidden", 80)),
            )
        finally:
            if lease is not None:
                self.broker.release(lease, quote.local_seconds)
        self._pending_bytes = 0
        self._last_train = time.time()
        return {"kind": "train", "lease": bool(lease), "quote": quote.as_dict(),
                "result": {k: result[k] for k in ("steps", "loss_first", "loss_last", "seconds")}}

    def _record(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        self.events = self.events[-200:]

    # ---------- 循环 ----------
    def run(self, steps: int = 0, interval: Optional[float] = None) -> List[Dict[str, Any]]:
        limit = steps or self.max_steps
        gap = self.interval_s if interval is None else interval
        done = 0
        out: List[Dict[str, Any]] = []
        idle = 0
        try:
            while limit <= 0 or done < limit:
                ev = self.step()
                out.append(ev)
                done += 1
                if limit > 0 and done >= limit:
                    break
                # 空转识别：这一步没干活就逐步拉长睡眠，最长 8 倍；有活立刻恢复。
                idle = min(idle + 1, 3) if ev.get("skipped") else 0
                # 每 60 步做一次记忆巩固（TTL 遗忘 + 向量语义合并）；
            # 原来 consolidate 只由 tools/loop.py 手动触发，daemon 里永远不会自动发生
            self._ticks = getattr(self, '_ticks', 0) + 1
            if self._ticks % 60 == 0:
                _mem = getattr(self, 'memory', None)
                if _mem is not None:
                    _mem.consolidate()
            time.sleep(max(0.0, gap * (2 ** idle)))
        except KeyboardInterrupt:
            # 手动打断也算回家的一种，别把状态挂在门外
            self.concealer.conceal("keyboard_interrupt")
        return out
