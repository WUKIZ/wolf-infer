"""基底核：给动作投最后一票。

模型把话判成 search / reply / silent 是硬判 —— 判成 search 就一定出门。
基底核在它后面加一道软闸：如果这条动作在多巴胺里已经跌得很惨，就降级
而不是照做。

不清零、不硬拦，只让「总是失败的策略」慢慢失去出场机会。这是和「规则
拦截」的区别：规则是你写的，这条是它自己摔出来的。

阈值 0.35：α=0.15 时，连续 3 次失败把 Q 从 0.5 压到约 0.36。低于这个
数说明这条路真走不通，不是偶尔倒霉。
"""
from __future__ import annotations

from typing import Optional


class Basal:
    def __init__(self, floor: float = 0.35) -> None:
        self.floor = floor
        self.downgraded = 0

    def confirm(self, action: Optional[str], dopa) -> Optional[str]:
        """search 口碑崩了就降成 reply；其余原样放行。"""
        if action != "search":
            return action
        if dopa.expect("search") < self.floor:
            self.downgraded += 1
            return "reply"
        return action

    def stats(self) -> dict[str, int]:
        return {"downgraded": self.downgraded}
