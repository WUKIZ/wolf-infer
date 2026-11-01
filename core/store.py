"""共用主机：本机和别的设备指向同一个目录（Syncthing / rclone mount / sshfs 都行）。

冲突策略 —— 每台设备只往自己的命名空间追加文件，永不覆盖别人的：
    shared/devices/<device>/<key>.<lamport:08d>.<sha>.json

同步工具因此永远看不到"两边改了同一个文件"，只会看到各设备各自新增的小文件，
不会产生 .sync-conflict 副本。合并按 (lamport, ts, device) 全序取最新，
删除写 tombstone 而不是 unlink —— 否则下一次从别的设备同步回来会"复活"。

存的都是外面带回来的东西和跨设备心跳；家本身（home/）不在这里，家是本机的基线。
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.atomic import atomic_write


def _check_key(key: str) -> str:
    """键只能是相对路径片段：挡住 ../ 和绝对路径，别让人把文件写到主机外面。"""
    if not key or key.startswith("/") or ".." in Path(key).parts:
        raise ValueError(f"bad key: {key!r}")
    return key


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class Record:
    key: str
    device: str
    lamport: int
    ts: float
    sha: str
    payload: Dict[str, Any]
    tombstone: bool = False

    @property
    def order(self) -> tuple:
        # 全序：先逻辑时钟，再墙钟，最后设备名 —— 三台机器算出同一个赢家
        return (self.lamport, self.ts, self.device)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "device": self.device, "lamport": self.lamport,
            "ts": self.ts, "sha": self.sha, "payload": self.payload,
            "tombstone": self.tombstone,
        }


class SharedStore:
    def __init__(self, root: Path, device: Optional[str] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = (device or socket.gethostname().split(".")[0] or "device").strip() or "device"
        self.dir = self.root / "devices" / self.device
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = self._load_clock()
        self._seen_max = 0

    # ---------- 逻辑时钟 ----------
    @property
    def clock_path(self) -> Path:
        return self.dir / "clock.json"

    def _load_clock(self) -> int:
        try:
            return int(json.loads(self.clock_path.read_text("utf-8")).get("lamport", 0))
        except (OSError, ValueError, TypeError):
            return 0

    def _bump(self) -> int:
        # 本机时钟要压过见过的所有记录，跨设备才知道谁更新
        self._clock = max(self._clock, self._seen_max) + 1
        atomic_write(self.clock_path, json.dumps({"lamport": self._clock, "device": self.device}))
        return self._clock

    # ---------- 写 ----------
    def put(self, key: str, payload: Dict[str, Any], tombstone: bool = False) -> Path:
        key = _check_key(key)
        lam = self._bump()
        rec = Record(key=key, device=self.device, lamport=lam, ts=time.time(),
                     sha=_sha(payload), payload=payload, tombstone=tombstone)
        name = f"{key.replace('/', '__')}.{lam:08d}.{rec.sha}.json"
        path = self.dir / name
        atomic_write(path, json.dumps(rec.as_dict(), ensure_ascii=False, sort_keys=True))
        return path

    def delete(self, key: str) -> Path:
        return self.put(key, {}, tombstone=True)

    # ---------- 读 ----------
    def records(self) -> List[Record]:
        out: List[Record] = []
        base = self.root / "devices"
        if not base.exists():
            return out
        for f in sorted(base.glob("*/*.json")):
            if f.name == "clock.json":
                continue
            try:
                d = json.loads(f.read_text("utf-8"))
                rec = Record(
                    key=str(d["key"]), device=str(d["device"]), lamport=int(d["lamport"]),
                    ts=float(d["ts"]), sha=str(d["sha"]),
                    payload=d.get("payload") or {}, tombstone=bool(d.get("tombstone")),
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue          # 同步到一半的文件跳过，不算故障
            self._seen_max = max(self._seen_max, rec.lamport)
            out.append(rec)
        return out

    def merged(self) -> Dict[str, Record]:
        best: Dict[str, Record] = {}
        for rec in self.records():
            cur = best.get(rec.key)
            if cur is None or rec.order > cur.order:
                best[rec.key] = rec
        return best

    def get(self, key: str) -> Optional[Record]:
        rec = self.merged().get(key)
        return None if rec is None or rec.tombstone else rec

    def keys(self, prefix: str = "") -> List[str]:
        return sorted(k for k, r in self.merged().items()
                      if not r.tombstone and k.startswith(prefix))

    def facts(self) -> List[Dict[str, Any]]:
        return [r.payload for r in self.merged().values()
                if not r.tombstone and r.key.startswith("facts/")]

    # ---------- 心跳：谁还活着 ----------
    def heartbeat(self, state: str = "home") -> Path:
        return self.put("_hb", {"state": state, "at": time.time(),
                                "pid": os.getpid(), "device": self.device})

    def devices(self, ttl_s: float = 90.0) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        out: Dict[str, Dict[str, Any]] = {}
        for rec in self.records():
            if rec.key != "_hb":
                continue
            cur = out.get(rec.device)
            if cur is None or rec.order > cur["_order"]:
                age = now - float(rec.payload.get("at", 0.0))
                out[rec.device] = {"state": rec.payload.get("state", "?"),
                                   "age_s": round(age, 1),
                                   "alive": age <= ttl_s,
                                   "_order": rec.order}
        for v in out.values():
            v.pop("_order", None)
        return out

    def prune(self, keep_days: float = 30.0) -> int:
        """清掉自己命名空间下的老追加文件，别人的东西不动。"""
        cutoff = time.time() - keep_days * 86400
        n = 0
        for f in self.dir.glob("*.json"):
            if f.name == "clock.json":
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
            except OSError:
                pass
        return n
