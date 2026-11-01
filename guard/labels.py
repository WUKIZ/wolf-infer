"""标签格：完整性 × 机密性。join 取上界，flow 只允许往更严的方向走。"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class Integrity(IntEnum):
    TRUSTED = 0      # 用户输入、核心规则、代码常量
    UNTRUSTED = 1    # 网页、搜索结果、远端模型输出、抓来的文件


class Secrecy(IntEnum):
    PUBLIC = 0
    PRIVATE = 1      # 记忆、密钥、用户数据


@dataclass(frozen=True)
class Label:
    integrity: Integrity = Integrity.TRUSTED
    secrecy: Secrecy = Secrecy.PUBLIC

    def join(self, other: "Label") -> "Label":
        return Label(max(self.integrity, other.integrity), max(self.secrecy, other.secrecy))

    def with_secrecy(self, s: Secrecy) -> "Label":
        return Label(self.integrity, s)


T = Label()                                   # 可信、公开
U = Label(Integrity.UNTRUSTED)                # 不可信、公开
P = Label(Integrity.TRUSTED, Secrecy.PRIVATE)  # 可信、私密
UP = Label(Integrity.UNTRUSTED, Secrecy.PRIVATE)


@dataclass(frozen=True)
class Tagged:
    value: str
    label: Label

    def __str__(self) -> str:
        return self.value


def join_all(items) -> Label:
    out = T
    for it in items:
        out = out.join(it.label if isinstance(it, Tagged) else T)
    return out
