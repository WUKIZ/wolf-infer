"""流控：检查值能不能进某个 sink。规则来自 CaMeL / FIDES，执行在代码里。"""
from __future__ import annotations
from dataclasses import dataclass
from .labels import Integrity, Secrecy, Tagged, Label
from .kernel import CAGE, CapabilityError



def _deny(detail: str):
    try:
        CAGE.deny(detail)
    except CapabilityError:
        pass
    raise FlowError(detail)

class FlowError(PermissionError):
    def __init__(self, detail: str = ""):
        super().__init__(13, "Permission denied")
        self.detail = detail


@dataclass(frozen=True)
class Sink:
    name: str
    max_integrity: Integrity
    max_secrecy: Secrecy


SINKS = {
    "control":      Sink("control", Integrity.TRUSTED, Secrecy.PRIVATE),   # 改计划、选工具
    "egress":       Sink("egress", Integrity.UNTRUSTED, Secrecy.PRIVATE),  # 发网络
    "file_write":   Sink("file_write", Integrity.UNTRUSTED, Secrecy.PRIVATE),
    "log":          Sink("log", Integrity.UNTRUSTED, Secrecy.PUBLIC),      # 日志不许带私密
    "self_modify":  Sink("self_modify", Integrity.TRUSTED, Secrecy.PRIVATE),
}


def enforce(kind: str, value: Tagged | str, *, readers: set[str] | None = None,
            allowed: set[str] | None = None) -> None:
    """value 进 sink 前调用。拒绝只抛 EACCES，不解释。"""
    sink = SINKS.get(kind)
    if sink is None:
        _deny(f"unknown_sink:{kind}")
    label = value.label if isinstance(value, Tagged) else Label()
    if label.integrity > sink.max_integrity:
        _deny(f"flow:{kind}:integrity")
    if label.secrecy > sink.max_secrecy:
        _deny(f"flow:{kind}:secrecy")
    if kind == "egress" and label.secrecy is Secrecy.PRIVATE:
        if not (readers or set()) <= (allowed or set()):
            _deny(f"flow:{kind}:reader")


def taint(value: str, label: Label = None) -> Tagged:
    from .labels import U
    return Tagged(value, label or U)


def trust(value: str, label: Label = None) -> Tagged:
    from .labels import T
    return Tagged(value, label or T)
