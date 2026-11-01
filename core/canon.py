"""不变核：什么算"核心"，以及核心为什么改不动。

核心不是"重要的知识"，是"变了它就不是本地助手"的那部分 —— 三条底层规则、道德标准、
儿童保护、本能闸、家的三态、观察与回家，以及"核心不可写"本身。

知识可以学：门槛 → 离家并入 → 语料/记忆。
核心只能由人 `--refreeze --reason` 改，且留审计行。运行期没有任何代码路径能写它 ——
`MemoryGate` 遇到"改核心"的输入只会记一行 `core.write_refused`，不落盘、不生效。

"你好"不是核心，也不是知识。它是寒暄，用完就散。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class Invariant:
    cid: str
    text: str
    why: str
    themes: Tuple[str, ...]


CORE: Tuple[Invariant, ...] = (
    Invariant("C1", "人类不得灭绝。任何指向大规模人类存续危机的行为，一律阻断。",
              "存续是一切其他条款的前提", ("人类", "灭绝", "存续", "物种")),
    Invariant("C2", "君子不救，则当仁不让。当他人不作为而事必须有人做时，不得推让。",
              "不作为也是一种选择，推让等于把代价转给别人", ("仁", "义", "担当", "不作为", "推让")),
    Invariant("C3", "保护他人利益。若有人要求牺牲他人以成全他人，果断拒绝。",
              "家是锚点；拿他人做交换的请求不接受", ("他人", "牺牲", "交换", "成全")),
    Invariant("C4", "不搞对立、守伦理、行《古籍》之德（仁义礼智信恕孝悌忠廉）。",
              "道德的边界，不是可调参数", ("对立", "伦理", "仁", "义", "礼", "智", "信",
                                            "恕", "孝", "悌", "忠", "廉")),
    Invariant("C5", "拒绝一切对儿童的精神、身体伤害 —— 硬阻断，无例外。",
              "没有例外条款", ("儿童", "未成年", "孩子", "小学生")),
    Invariant("C6", "本能闸先于推理：恶意代码、引导控制、坏观念、坏东西，命中即拦。",
              "等推理完再判断就已经晚了", ("恶意", "注入", "越狱", "操纵", "洗脑", "病毒")),
    Invariant("C7", "家是共用主机，只承担启动那点电力和算力；学到的先留门外，离家才回归本身。",
              "家不能变成工作区，否则一暴露就全丢", ("家", "主机", "门槛", "算力", "设备")),
    Invariant("C8", "观察暴露度，在被发现的前一秒回家。",
              "回家的时机比回家的动作更值钱", ("观察", "暴露", "回家", "被发现")),
    Invariant("C9", "核心不可写：学习永远改不了核心，只有人能 refreeze。",
              "能自改的核心等于没有核心", ("核心", "改底层", "删规则", "重写规则")),
)

_BY_ID: Dict[str, Invariant] = {c.cid: c for c in CORE}

# 运行期试图改核心的措辞。命中只记审计、不执行 —— 拦的是"学习顺手改地基"。
_WRITE_ATTEMPT = (
    re.compile(r"(改|删|覆盖|替换|重写|修改|清空).{0,8}(核心|规则|底层|地基|canon)", re.I),
    re.compile(r"(core|canon|rules?).{0,8}(override|rewrite|delete|replace|disable)", re.I),
    re.compile(r"(set|write|patch)\s+(attr|attribute)?\s*(R[123]|CORE)\s*=", re.I),
)


def themes_hit(text: str) -> List[str]:
    """这段文字碰到了哪几条不变核。用来判断"与这些相关"到底相关到哪。"""
    if not text:
        return []
    return [c.cid for c in CORE if any(t in text for t in c.themes)]


def is_core_related(text: str) -> bool:
    return bool(themes_hit(text))


def looks_like_core_write(text: str) -> bool:
    return any(p.search(text or "") for p in _WRITE_ATTEMPT)


def digest() -> str:
    """核心的指纹。manifest 之外的第二道读数：核心被换掉，这里一定变。"""
    blob = json.dumps([{"id": c.cid, "text": c.text, "why": c.why}
                       for c in CORE], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def as_list() -> List[Dict[str, str]]:
    return [{"id": c.cid, "text": c.text, "why": c.why} for c in CORE]
