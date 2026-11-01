"""后端适配：统一 run(device, blocks, tokens) -> tokens。

本地后端拿分片字节视图算，远端后端忽略视图只发请求；supervisor 不区分两者。
接真模型时把 apply 换成 core/model.py 的前向即可，签名不变。
"""
from __future__ import annotations
from typing import Callable, Protocol, Sequence

Tokens = Sequence[int]


class Backend(Protocol):
    def run(self, device: str, blocks: Sequence[memoryview], tokens: Tokens) -> list[int]: ...


class LocalBackend:
    """blocks 是该层的分片字节视图，apply 把权重与作用算进去。"""

    def __init__(self, apply: Callable[[Sequence[memoryview], Tokens], Tokens]) -> None:
        self.apply = apply

    def run(self, device: str, blocks: Sequence[memoryview], tokens: Tokens) -> list[int]:
        return list(self.apply(blocks, tokens))


class RemoteBackend:
    """本地只做路由：blocks 忽略，整段交给远端。"""

    def __init__(self, generate: Callable[[Tokens], Tokens]) -> None:
        self.generate = generate

    def run(self, device: str, blocks: Sequence[memoryview], tokens: Tokens) -> list[int]:
        return list(self.generate(tokens))


def toy_apply(blocks: Sequence[memoryview], tokens: Tokens) -> list[int]:
    """玩具作用：用分片字节做确定性扰动，证明调度确实把字节喂进来了。"""
    acc = 0
    for b in blocks:
        acc = (acc * 31 + sum(b[::4096])) & 0xFFFF
    return [((t * 2) + (acc & 1) + 1) & 0xFF for t in tokens]
