"""计划门：工具调用只能发生在已签名的计划里。计划由可信输入生成，网页改不动它。"""
from __future__ import annotations
import fnmatch, hashlib, hmac, json, time
from .kernel import CAGE, CapabilityError



def _deny(detail: str):
    try:
        CAGE.deny(detail)
    except CapabilityError:
        pass
    raise PlanError(detail)

class PlanError(PermissionError):
    def __init__(self, detail: str = ""):
        super().__init__(13, "Permission denied")
        self.detail = detail


class PlanGate:
    def __init__(self, key: bytes | None = None):
        self.key = key or CAGE.key

    def approve(self, steps: list[tuple[str, str]], ttl: float = 600.0) -> str:
        body = json.dumps({"s": steps, "exp": time.time() + ttl}, ensure_ascii=False)
        sig = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()[:32]
        return f"{body}|{sig}"

    def permit(self, token: str, tool: str, target: str) -> None:
        try:
            body, sig = token.rsplit("|", 1)
            doc = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            _deny("plan:malformed")
        want = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, want):
            _deny("plan:sig")
        if doc["exp"] < time.time():
            _deny("plan:expired")
        for t, pat in doc["s"]:
            if t == tool and fnmatch.fnmatch(target, pat):
                return
        _deny(f"plan:not_allowed:{tool}")


PLAN = PlanGate()
