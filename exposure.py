"""暴露度：算「外面有多少人看见我了」。

本地助手的规则是「被发现之前回家」。要判断「快被发现」，先得能量出来自己
在外面留了多少痕迹 —— 现在完全没有这个数，只有一句口号。

量法只看三件可测的事，不猜：

    次数     每发一次外部请求，别人那边就多一条访问日志
    域名     摊得越宽，越容易被同一批日志拼起来
    独特性   反复查同一类东西是研究，四处扫是新面孔

三项各自归一后取最大，不取和 —— 任意一项顶到 1.0 就该回家，不需要另外
两项陪着。衰减按小时算，睡一夜等于洗掉一部分。

落 home/state/exposure.json。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# 峰值：50 次请求、25 个域名、40 个不同 query，任何一项到顶就是 1.0
MAX_CALLS = 50
MAX_DOMAINS = 25
MAX_UNIQUE = 40
DECAY_PER_HOUR = 0.06      # 每小时掉 0.06，一天掉一半多


class Exposure:
    def __init__(self, path: Path | str, home_at: float = 0.75) -> None:
        self.p = Path(path)
        self.home_at = home_at
        self.s = {"calls": 0, "domains": [], "queries": [],
                  "touched": time.time()}

        if self.p.exists():
            try:
                loaded = json.loads(self.p.read_text("utf-8"))
                if isinstance(loaded, dict):
                    self.s.update({k: loaded[k] for k in self.s if k in loaded})
            except (OSError, ValueError):
                pass

        self._decay()

    def _decay(self) -> None:
        """按经过的小时数往回收。关机再开也一样算 —— 时间在流逝。"""
        hours = max(0.0, (time.time() - float(self.s["touched"])) / 3600.0)
        if hours < 0.01:
            return
        drop = hours * DECAY_PER_HOUR
        self.s["calls"] = max(0.0, float(self.s["calls"]) - drop * MAX_CALLS)
        self.s["domains"] = self.s["domains"][-int(len(self.s["domains"]) *
                                                   max(0.0, 1 - drop)):]
        self.s["queries"] = self.s["queries"][-int(len(self.s["queries"]) *
                                                   max(0.0, 1 - drop)):]
        self.s["touched"] = time.time()

    def _save(self) -> None:
        try:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.p.with_name(f".{self.p.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.s, ensure_ascii=False), "utf-8")
            os.replace(tmp, self.p)
        except OSError:
            pass

    def touch(self, domain: str = "", query: str = "") -> float:
        """外出一次。返回更新后的暴露度。"""
        self.s["calls"] = float(self.s["calls"]) + 1
        if domain and domain not in self.s["domains"]:
            self.s["domains"].append(domain[:60])
            self.s["domains"] = self.s["domains"][-60:]
        if query and query not in self.s["queries"]:
            self.s["queries"].append(query[:60])
            self.s["queries"] = self.s["queries"][-80:]
        self.s["touched"] = time.time()
        self._save()
        return self.level()

    def level(self) -> float:
        """0 到 1。取三项最大值 —— 任一维度顶到顶就是暴露。"""
        c = top = min(1.0, float(self.s["calls"]) / MAX_CALLS)
        d = min(1.0, len(self.s["domains"]) / MAX_DOMAINS)
        u = min(1.0, len(self.s["queries"]) / MAX_UNIQUE)
        top = max(c, d, u)
        return round(top, 3)

    def should_home(self) -> bool:
        return self.level() >= self.home_at

    def enter_home(self) -> float:
        """回家：次数与独特性清零，域名留着当底账（知道去过哪）。"""
        before = self.level()
        self.s["calls"] = 0
        self.s["queries"] = []
        self.s["touched"] = time.time()
        self._save()
        return before

    def line(self) -> str:
        if not (self.s["calls"] or self.s["domains"] or self.s["queries"]):
            return ""
        return (f"暴露={self.level():.2f} 次数={int(self.s['calls'])} "
                f"域名={len(self.s['domains'])}")
