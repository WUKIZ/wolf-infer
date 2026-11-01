"""把外部内容打上不可信标签。调用点不接上，标签就只是摆设。"""
from __future__ import annotations
from .labels import Label, Integrity, Secrecy, Tagged, U, UP
from .flow import enforce


def external(text: str, secret: bool = False) -> Tagged:
    lab = UP if secret else U
    return Tagged(text, lab)


def to_control(text: str) -> str:
    """只有可信文本才能影响计划；不可信的直接拒绝。"""
    v = text if isinstance(text, Tagged) else Tagged(str(text), U)
    enforce("control", v)
    return v.value


def to_egress(text: str, readers: set[str], allowed: set[str]) -> str:
    v = text if isinstance(text, Tagged) else Tagged(str(text), U)
    enforce("egress", v, readers=readers, allowed=allowed)
    return v.value
