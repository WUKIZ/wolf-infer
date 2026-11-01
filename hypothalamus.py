"""下丘脑：稳态驱动。

emotion.py 存三个数（energy / trust / arousal），policy() 把它们翻成阈值。
但那只是「现在能不能说话」，不是「缺什么就想要什么」。

下丘脑补这一层：每个维度有个目标值，偏离多少就产生多少驱动。驱动最大
且超过阈值的那个，才是当下真正的倾向 —— 低能量的人不是「话少」，是
「想歇」；高唤醒的人不是「词多」，是「憋不住」。

只输出一句话，不直接改行为。要不要听，调用方决定。
"""
from __future__ import annotations

# 目标值。energy 低于 0.6 想歇，arousal 高于 0.4 想动 —— 都不是零
GOAL = {"energy": 0.6, "trust": 0.5, "arousal": 0.4}
SHARP = 0.3          # 偏离超过 0.3 才算真有驱动，否则只是正常波动


class Hypothalamus:
    def __init__(self) -> None:
        self.urges = 0

    def drives(self, emo) -> dict[str, float]:
        """正数 = 缺，负数 = 满。"""
        return {k: round(GOAL[k] - float(getattr(emo, k, 0.5)), 3) for k in GOAL}

    def urge(self, emo) -> tuple[str, float]:
        """偏离最大的那一项。"""
        d = self.drives(emo)
        k = max(d, key=lambda x: abs(d[x]))
        return k, d[k]

    def hint(self, emo) -> str:
        """压一行给 system。没强驱动就返回空串，不占 token。"""
        k, v = self.urge(emo)
        if abs(v) < SHARP:
            return ""
        self.urges += 1
        word = {"energy": "精力", "trust": "信任", "arousal": "唤醒"}.get(k, k)
        return f"{word}{'不足' if v > 0 else '过盛'}{abs(v):.2f}"
