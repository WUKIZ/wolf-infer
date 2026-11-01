"""流水线切分：按字节把有序分片切成 ≤len(devices) 段连续层段，一段一台设备。

顺序不动（流水线前提），段间只传激活。二分容量再顺序装填，O(n log total)。
1 卡与 1000 卡走同一函数——「万亿级」和「4 亿级」的差别只在这里的参数。
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass

try:
    from .shard import ModelSpec, Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shard import ModelSpec, Shard


@dataclass(frozen=True, slots=True)
class Stage:
    device: str
    shard_indices: tuple[int, ...]
    layer_from: int
    layer_to: int
    nbytes: int


def _needed(sizes: list[int], cap: int) -> float:
    n, cur = 1, 0
    for s in sizes:
        if s > cap:
            return float("inf")
        if cur and cur + s > cap:
            n, cur = n + 1, s
        else:
            cur += s
    return n


def plan(spec: ModelSpec, devices: list[str]) -> tuple[Stage, ...]:
    if not devices:
        raise ValueError("devices 不能为空")
    ordered: list[Shard] = sorted(spec.shards, key=lambda s: s.index)
    sizes = [s.nbytes for s in ordered]
    lo, hi = max(sizes), sum(sizes)
    while lo < hi:
        mid = (lo + hi) // 2
        if _needed(sizes, mid) <= len(devices):
            hi = mid
        else:
            lo = mid + 1
    chunks: list[tuple[list[Shard], int]] = []
    buf: list[Shard] = []
    acc = 0
    for s in ordered:
        if acc and acc + s.nbytes > lo:
            chunks.append((buf, acc))
            buf, acc = [], 0
        buf.append(s)
        acc += s.nbytes
    chunks.append((buf, acc))
    return tuple(
        Stage(dev, tuple(x.index for x in group), group[0].layer_from, group[-1].layer_to, nb)
        for dev, (group, nb) in zip(devices, chunks)
    )
