"""底层守卫。底层代码被删、被改、被追加 —— 直接停机并回滚出厂态。

三道锁：
  1. core/ 每个文件的 sha256 写进 manifest
  2. manifest 自己带 seal，且**存两份**（正本 + 镜像）—— 删一份也算篡改
  3. canon.digest()（规则不变核的指纹）一起封进去 —— 改核心文字和改核心代码
     一样会被发现，不用等下一次 --canon 才看出来

任何一项不符 → TamperDetected → main 回滚 seed、停进程。
"""
from __future__ import annotations

import hashlib
import os
import hmac
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import canon
from .rules import TamperDetected

GUARDED = ("rules.py", "instinct.py", "layers.py", "home.py", "integrity.py",
           "observer.py", "learn.py", "model.py", "bpe.py",
           "store.py", "compute.py", "conceal.py", "daemon.py",
           "canon.py", "memory.py", "social.py",
           "monitor.py", "trail.py", "permit.py", "crawl.py",
           "browser.py", "webapp.py", "mem.py", "remote.py", "supply.py",
           "idle.py", "sentry.py", "search.py")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _seal(files: Dict[str, str], canon_digest: str) -> str:
    blob = json.dumps({"files": files, "canon": canon_digest},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class Guard:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.core_dir = root / "core"
        self.manifest_path = root / "home" / "state" / "manifest.json"
        self.mirror_path = root / "home" / "state" / "manifest.mirror.json"

    # ---------- 采集 ----------
    def _collect(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in GUARDED:
            p = self.core_dir / name
            if p.exists():
                out[name] = _sha(p)
        return out

    # ---------- 写 ----------
    def _write(self, files: Dict[str, str], reason: str) -> Path:
        digest = canon.digest()
        key = os.environ.get('WOLF_GUARD_KEY', '')
        seal = _seal(files, digest)
        doc = {'files': files, 'canon': digest, 'seal': seal, 'reason': reason,
               'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'signed': bool(key)}
        if key:
            doc['hmac'] = hmac.new(key.encode(), seal.encode(),
                                   hashlib.sha256).hexdigest()
        blob = json.dumps(doc, indent=2, ensure_ascii=False)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(blob, 'utf-8')
        self.mirror_path.write_text(blob, 'utf-8')

        audit = self.root / 'logs' / 'guard.log'
        audit.parent.mkdir(parents=True, exist_ok=True)
        prev = 'genesis'
        if audit.exists():
            lines = [x for x in audit.read_text(encoding='utf-8').splitlines() if x.strip()]
            if lines:
                prev = hashlib.sha256(lines[-1].encode()).hexdigest()[:16]
        entry = (f'{time.strftime("%Y-%m-%dT%H:%M:%S")}\t{reason}\t{seal}\t'
                 + ' '.join(f'{k}:{v[:12]}' for k, v in sorted(files.items()))
                 + f'\tcanon:{digest}\tprev:{prev}')
        with audit.open('a', encoding='utf-8') as fh:
            fh.write(entry + '\n')
        return self.manifest_path

    def freeze(self) -> Path:
        """出厂时调用一次：写死哈希。之后任何写入都会被察觉。"""
        return self._write(self._collect(), "factory")

    def refreeze(self, reason: str) -> Path:
        """开发期唯一合法改底层的方式。必须由人显式调用，并留下审计行。"""
        banned = {'reseal', 'fix', 'update', 'test', 'x', 'ok'}
        if not reason or len(reason.strip()) < 8 or reason.strip().lower() in banned:
            raise ValueError("refreeze 需要写明理由，审计要留痕")
        return self._write(self._collect(), reason.strip())

    # ---------- 校验 ----------
    def _read(self, path: Path) -> Optional[dict]:
        try:
            data = json.loads(path.read_text("utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def verify(self) -> List[str]:
        if not self.manifest_path.exists():
            raise TamperDetected("manifest_missing")
        data = self._read(self.manifest_path)
        if data is None:
            raise TamperDetected("manifest_unreadable")

        files = data.get("files") or {}
        canon_digest = str(data.get("canon") or "")
        if data.get('signed'):
            key = os.environ.get('WOLF_GUARD_KEY', '')
            expect = hmac.new(key.encode(), str(data.get('seal')).encode(),
                              hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expect, str(data.get('hmac', ''))):
                raise TamperDetected('manifest_hmac_invalid')
        if data.get("seal") != _seal(files, canon_digest):
            raise TamperDetected("manifest_seal_broken")
        if canon_digest != canon.digest():
            raise TamperDetected("canon_drift")

        # 镜像：删掉一份、或者两份不一致，都按篡改处理
        if not self.mirror_path.exists():
            raise TamperDetected("manifest_mirror_missing")
        mirror = self._read(self.mirror_path)
        if not mirror or mirror.get("files") != files or str(mirror.get("canon")) != canon_digest:
            raise TamperDetected("manifest_mirror_mismatch")

        problems: List[str] = []
        now = self._collect()
        for name, digest in files.items():
            if name not in now:
                problems.append(f"deleted:{name}")
            elif now[name] != digest:
                problems.append(f"modified:{name}")
        for name in now:
            if name not in files:
                problems.append(f"injected:{name}")

        if problems:
            raise TamperDetected(",".join(problems))
        return []

    # ---------- 一次看全 ----------
    def attest(self) -> Dict[str, object]:
        self.verify()
        files = self._collect()
        rules = files.get("rules.py", "")
        return {
            "files": len(files),
            "canon": canon.digest(),
            "rules": rules[:12],
            "seal": _seal(files, canon.digest())[:12],
            "model_bytes": (self.root / "home" / "models" / "tiny.bin").stat().st_size
            if (self.root / "home" / "models" / "tiny.bin").exists() else 0,
            "ok": True,
        }
