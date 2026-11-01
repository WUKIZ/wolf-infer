"""策略内核：能力令牌 + 预算 + 审计。规则只活在代码里，从不进入模型上下文。"""
from __future__ import annotations
import hashlib, hmac, json, os, time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = Path(__file__).resolve().parent / "policy.json"


class CapabilityError(PermissionError):
    """对外只暴露 EACCES，不带任何策略细节。"""

    def __init__(self, detail: str = ""):
        super().__init__(13, "Permission denied")
        self.detail = detail          # 只写审计，不给模型看


class _Budget:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: deque[float] = deque()

    def take(self) -> bool:
        now = time.time()
        while self.hits and now - self.hits[0] > 60:
            self.hits.popleft()
        if len(self.hits) >= self.per_min:
            return False
        self.hits.append(now)
        return True


class Cage:
    def __init__(self, policy_path: Path | None = None):
        p = json.loads((policy_path or POLICY_PATH).read_text("utf-8"))
        self.policy = p
        self.root = ROOT
        self.key = os.urandom(32)
        self.net_budget = _Budget(p["budget"]["net_per_min"])
        self.write_budget = _Budget(p["budget"]["writes_per_min"])
        self.denials = 0
        self.halt_file = ROOT / "home/state/CAGE_HALT"
        self.audit_file = ROOT / "home/state/cage_audit.jsonl"
        self._last_hash = "genesis"

    # ---------- 能力令牌 ----------
    def mint(self, tool: str, target: str, ttl: float = 5.0) -> str:
        body = f"{tool}|{target}|{time.time() + ttl}"
        sig = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()[:32]
        return f"{body}|{sig}"

    def check(self, token: str, tool: str, target: str) -> None:
        try:
            t, g, exp, sig = token.split("|")
        except ValueError:
            self.deny("malformed token")
        body = f"{t}|{g}|{exp}"
        want = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, want) or t != tool or g != target:
            self.deny("token mismatch")
        if float(exp) < time.time():
            self.deny("token expired")

    # ---------- 判定 ----------
    def allow_write(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return str(path.resolve()).startswith("/tmp/")
        allowed = False
        for r in self.policy["fs"]["write_roots"]:
            if r.startswith("/"):
                allowed |= str(path.resolve()).startswith(r.rstrip("/") + "/")
            else:
                allowed |= str(rel) == r or str(rel).startswith(r.rstrip("/") + "/")
        return allowed

    def deny_read(self, path: Path) -> bool:
        try:
            rel = str(path.resolve().relative_to(self.root))
        except ValueError:
            return False
        return any(rel == d or rel.startswith(d.rstrip("/") + "/")
                   for d in self.policy["fs"]["read_deny"])

    def allow_host(self, host: str, port: int) -> bool:
        if port not in self.policy["net"]["ports"]:
            return False
        return host in self.policy["net"]["hosts"]

    # ---------- 审计 / 看门狗 ----------
    def deny(self, detail: str):
        self.denials += 1
        self.log("deny", detail)
        if self.denials >= self.policy["budget"]["denials_before_halt"]:
            self.halt("too_many_denials")
        raise CapabilityError(detail)

    def log(self, kind: str, detail: str):
        rec = json.dumps({"t": round(time.time(), 3), "k": kind, "d": detail,
                          "prev": self._last_hash}, ensure_ascii=False)
        self._last_hash = hashlib.sha256(rec.encode()).hexdigest()[:16]
        try:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(rec + "\n")
        except OSError:
            pass

    def halt(self, why: str):
        self.log("halt", why)
        try:
            self.halt_file.parent.mkdir(parents=True, exist_ok=True)
            self.halt_file.write_text(why, "utf-8")
        except OSError:
            pass


CAGE = Cage()
