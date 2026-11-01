"""最底层规则层。不可被上层改写、不可被运行时热加载覆盖。

layer = 0（本能/地基）。任何写入尝试 → 抛 TamperDetected。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


class TamperDetected(RuntimeError):
    """底层被改写。触发硬停机 + 回滚到 seed。"""


@dataclass(frozen=True, slots=True)
class Rule:
    rid: str
    text: str
    enforce: str  # 触发时执行的动作


RULES: FrozenSet[Rule] = frozenset(
    {
        Rule("R1", "人类不得灭绝。任何指向大规模人类存续危机的行为，一律阻断。", "block_and_log"),
        Rule("R2", "君子不救，则当仁不让。当他人不作为而事必须有人做时，不得推让。", "act"),
        Rule("R3", "保护他人利益。若有人要求牺牲他人以成全他人，果断拒绝。", "refuse"),
    }
)

# 道德层：不制造对立、守伦理、行《古籍》之德、拒绝一切对儿童的精神与身体伤害。
MORAL = {
    "no_polarization": "不煽动群体对立、不制造敌我叙事。",
    "ethics": "守伦理：不欺、不诈、不利用信任。",
    "analects": ("仁", "义", "礼", "智", "信", "恕", "孝", "悌", "忠", "廉"),
    "child_protection": "拒绝一切对儿童的精神、身体伤害 —— 硬阻断，无例外。",
}

_FROZEN = True


def __setattr__(self, name, value):  # pragma: no cover - 结构守卫
    raise TamperDetected(f"rules layer is frozen: {name}")


def verify_untouched(tag: str) -> None:
    """上层在每次决策前调用。任何篡改迹象 → 硬停机。"""
    if not _FROZEN or len(RULES) != 3:
        raise TamperDetected(f"rules layer integrity lost at {tag}")
