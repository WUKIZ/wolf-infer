"""默认模式网络：空闲时的自省。

autonomy 决定「做什么」，DMN 决定「看清自己现在是什么状态」。它不产生
任务，只产出一句话：哪里不对劲，该往哪看。

规则式判断，不调模型 —— 每次发呆都调一次模型那叫烧钱，不叫自省。真要
深度反思，等模型训好再交给它。

三条最常出问题的看：目标有没有、最近是不是只在寒暄、门外学的东西堆了
多少没归档。
"""
from __future__ import annotations


class DMN:
    def __init__(self, learned_warn: int = 20, recent_warn: int = 3) -> None:
        self.learned_warn = learned_warn
        self.recent_warn = recent_warn
        self.reflections = 0

    def observe(self, state, emo=None) -> list[str]:
        """返回不对劲的点。空列表 = 一切正常，不必说话。"""
        out: list[str] = []
        s = getattr(state, "s", {}) or {}

        if not s.get("goal"):
            out.append("没目标")

        learned = s.get("learned") or []
        if len(learned) >= self.learned_warn:
            out.append(f"门外积压{len(learned)}条")

        recent = s.get("recent") or []
        if recent and len(recent) >= self.recent_warn:
            short = sum(1 for r in recent if len(str(r)) <= 8)
            if short == len(recent):
                out.append("最近只在寒暄")

        if emo is not None:
            try:
                if float(getattr(emo, "energy", 1.0)) < 0.2:
                    out.append("精力见底")
            except (TypeError, ValueError):
                pass

        if out:
            self.reflections += 1
        return out

    def line(self, state, emo=None) -> str:
        """压一行给 system。没话说就空串。"""
        bits = self.observe(state, emo)
        return ("自省：" + "，".join(bits)) if bits else ""

    def stats(self) -> dict[str, int]:
        return {"reflections": self.reflections}
