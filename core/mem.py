"""内存闸：700MB 是硬顶，不是目标。

训练时每个参数要三份 float32：权重 + Adam 一阶 + 二阶 = 12 字节。
再加激活值和解释器底座，就是估算值。估不准也没关系 —— 每次评估都量一次
真实 RSS，超了立刻停，不赌。
"""
from __future__ import annotations

import resource
from typing import Tuple

MB = 1 << 20


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def estimate_training_mb(params: int, batch: int, block: int,
                         dims: int, layers: int) -> float:
    state = params * 12                                  # w + m + v
    act = batch * block * dims * 4 * (8 + 3 * layers)    # 前向缓存 + 反向临时
    return (state + act) / MB + 120                      # 120MB 底座（python+numpy）


def check(est_mb: float, cap_mb: float) -> Tuple[bool, str]:
    if est_mb > cap_mb:
        return False, f"estimate={est_mb:.0f}MB>cap={cap_mb:.0f}MB"
    return True, f"estimate={est_mb:.0f}MB"
