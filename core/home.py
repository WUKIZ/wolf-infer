"""家的三态：home（基线，静止） → threshold（门槛外，暂存） → excursion（外出，活动）。

规则：
1. 家只承担启动那一点电力和算力，重活在外租算力里干。
2. 外出学到的东西不直接进家 —— 先落在 threshold（门外），过检才算数。
3. 再次离家时，门槛上的东西"回归本身"，并入活动记忆。
4. 随时可回家。回家 = 卸下活动态，落盘冻结。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import instinct, rules
from core.atomic import atomic_write

HOME, THRESHOLD, EXCURSION = "home", "threshold", "excursion"


@dataclass
class Home:
    root: Path
    state: str = HOME
    boot_ms: float = 0.0
    live: Dict[str, Any] = field(default_factory=dict)
    entered_at: float = 0.0   # time.monotonic()，用来算在家待了多久
    left_at: float = 0.0
    first_departure: bool = True   # 刚开机那次出门不该被 min_dwell 挡住

    # ---------- 路径 ----------
    @property
    def state_dir(self) -> Path:
        return self.root / "home" / "state"

    @property
    def memory_dir(self) -> Path:
        return self.root / "home" / "memory"

    @property
    def seed_dir(self) -> Path:
        return self.root / "home" / "seed"

    @property
    def threshold_dir(self) -> Path:
        return self.root / "threshold"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "wolf.log"

    @property
    def corpus_path(self) -> Path:
        return self.root / "home" / "corpus.txt"

    def log(self, event: str, detail: str = "") -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > 2_000_000:
                os.replace(self.log_path, self.log_path.with_suffix(".log.1"))
        except OSError:
            pass
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{event}\t{detail}\n")

    def _ensure(self) -> None:
        for d in (self.state_dir, self.memory_dir, self.seed_dir, self.threshold_dir,
                  self.root / "excursion", self.log_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """读坏了就返回 None，绝不把异常抛到 boot 外面。"""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            self.log("home.corrupt", f"{path.name}:{type(exc).__name__}")
            return None
        return data if isinstance(data, dict) else None

    # ---------- 启动：只花家的一点电 ----------
    def boot(self) -> float:
        self._ensure()
        t0 = time.perf_counter()
        idx = self.memory_dir / "index.json"
        live = self._read_json(idx)
        if live is None:
            live = self._read_json(self.memory_dir / "index.json.bak")
            if live is not None:
                self.log("home.repair", "index.json 损坏，取 .bak")
                atomic_write(idx, json.dumps(live, ensure_ascii=False, indent=2))
        if live is None:
            live = self._read_json(self.seed_dir / "index.json")
            if live is not None:
                self.log("home.repair", "index.json 损坏，回滚 seed")
        if live is None:
            live = {"facts": [], "born": time.time()}
            self.log("home.repair", "无可读记忆，起空白基线")
        live.setdefault("facts", [])
        self.live = live
        self.state = HOME
        self.first_departure = True
        self.entered_at = time.monotonic()
        self.boot_ms = (time.perf_counter() - t0) * 1000
        return self.boot_ms

    def can_leave(self, min_dwell_s: float = 0.0) -> bool:
        """刚起来那次直接出门；回家待过一段再出门，避免观察层抖动来回穿门。"""
        if self.state == EXCURSION:
            return True
        if self.first_departure:
            return True
        return (time.monotonic() - self.entered_at) >= min_dwell_s

    def overdue(self, max_excursion_s: float) -> bool:
        """出门多久必须回家 —— 观察层没信号不等于安全。"""
        if self.state != EXCURSION or max_excursion_s <= 0:
            return False
        return (time.monotonic() - self.left_at) > max_excursion_s

    # ---------- 离家 ----------
    def leave(self) -> int:
        """门槛上的知识在离家这一刻回归本身。返回并入条数。"""
        if self.state == EXCURSION:      # 已经在外面了，别把门槛再收割一遍
            self.log("home.leave", "already_outside")
            return 0
        rules.verify_untouched("home.leave")
        staged = self._drain_threshold()
        known = {f.get("sha") for f in self.live.get("facts", []) if f.get("sha")}
        accepted: List[Dict[str, Any]] = []
        for item in staged:
            text = json.dumps(item, ensure_ascii=False)
            if instinct.quarantines(text) is not None:  # 门外带来的脏东西，留在门外
                self.log("threshold.reject", item.get("source", "?"))
                continue
            if item.get("sha") and item["sha"] in known:  # 同一份知识不并两次
                self.log("threshold.dedupe", item["sha"])
                continue
            if item.get("sha"):
                known.add(item["sha"])
            accepted.append(item)
        self.live.setdefault("facts", []).extend(accepted)
        self.state = EXCURSION
        self.left_at = time.monotonic()
        self.first_departure = False
        # 「出了家学到的会回归本身」：并进记忆的同时也进语料，
        # 否则下次训练用的还是旧语料，等于没学。
        self._absorb(accepted)
        # 出门期间被 kill 也不丢刚并进来的知识
        atomic_write(self.memory_dir / "index.json",
                      json.dumps(self.live, ensure_ascii=False, indent=2))
        self.log("home.leave", f"bind={len(accepted)} staged={len(staged)}")
        return len(accepted)

    def _absorb(self, accepted: List[Dict[str, Any]]) -> None:
        """把刚并进来的东西写进语料 —— 记忆和语料必须是同一份自我。"""
        texts = [str(i.get("text") or i.get("learned") or "").strip() for i in accepted]
        texts = [t for t in texts if t]
        if not texts:
            return
        p = self.corpus_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for t in texts:
                fh.write("\n" + t + "\n")

    # ---------- 回家 ----------
    def enter(self) -> Dict[str, Any]:
        self._ensure()
        payload = {
            "facts": self.live.get("facts", []),
            "closed_at": time.time(),
            "state": HOME,
        }
        idx = self.memory_dir / "index.json"
        bak = self.memory_dir / "index.json.bak"
        blob = json.dumps(self.live, ensure_ascii=False, indent=2)
        # 先落镜像再落正本：任何一刻被杀，至少有一份完整可读。
        atomic_write(bak, blob)
        atomic_write(idx, blob)
        atomic_write(self.state_dir / "last_home.json",
                      json.dumps(payload, ensure_ascii=False, indent=2))
        self.state = HOME
        self.entered_at = time.monotonic()
        self.log("home.enter", f"facts={len(self.live.get('facts', []))}")
        return payload

    # ---------- 带回的东西：先放门外 ----------
    def deposit(self, item: Dict[str, Any]) -> Path:
        """外出时学到的东西，写在门槛上，不进家。"""
        self._ensure()
        h = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        # 纳秒做前缀：同一毫秒内两条 deposit 不会互相覆盖
        path = self.threshold_dir / f"{time.time_ns()}_{h}.json"
        atomic_write(path, json.dumps(item, ensure_ascii=False, indent=2))
        return path

    def _drain_threshold(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.threshold_dir.exists():
            return out
        for p in sorted(self.threshold_dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding='utf-8')))
            except (OSError, ValueError) as exc:
                logging.warning('阈值文件损坏，保留待查 %s：%s', p, exc)
                continue
            p.unlink(missing_ok=True)
        return out

    # ---------- 兜底 ----------
    def restore_seed(self) -> None:
        """底层被改 → 回滚到出厂态。"""
        seed_idx = self.seed_dir / "index.json"
        self.live = self._read_json(seed_idx) or {"facts": [], "born": time.time()}
        self.live.setdefault("facts", [])
        self.enter()
        self.state = HOME
