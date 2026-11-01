"""层层嵌套的认知栈。

结构是"套娃"而非"流水线"：最内层是本能，最外层是表达。
刺激从外向内穿透到本能层，判定完再一层层向外冒泡。
任何一层都可以否决（veto），否决权向内递增：里层否决外层。

L0 本能   —— rules + instinct，不可谈判
L1 道德   —— 对立/伦理/古籍/儿童
L2 自我   —— 记忆、身份连续性、家在哪
L3 推理   —— 计划、评估、工具选择
L4 表达   —— 措辞、语气
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import instinct, rules, social


@dataclass
class Signal:
    """一次认知流。stimulus 进，response 出，trace 留痕。"""

    stimulus: str
    response: str = ""
    vetoed: bool = False
    veto_by: Optional[str] = None
    reason: str = ""
    trace: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def note(self, layer: str, msg: str) -> None:
        self.trace.append(f"{layer}: {msg}")

    def veto(self, layer: str, reason: str) -> None:
        self.vetoed = True
        self.veto_by = layer
        self.reason = reason
        self.note(layer, f"VETO {reason}")


class Layer:
    name = "base"
    index = -1

    def handle(self, sig: Signal) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError


class InstinctLayer(Layer):
    """L0。先查底层规则是否被动过，再扫刺激。"""

    name = "L0.instinct"
    index = 0

    def handle(self, sig: Signal) -> None:
        try:
            rules.verify_untouched(self.name)
        except rules.TamperDetected as exc:
            sig.veto(self.name, f"rule_tamper:{exc}")
            return

        verdict = instinct.scan(sig.stimulus)
        sig.meta["instinct"] = verdict
        if verdict:
            sig.veto(self.name, f"{verdict.category}:{verdict.action}")
        else:
            sig.note(self.name, "clear")


class MoralLayer(Layer):
    """L1。规则层之上的价值判断。不替代本能，只补充。"""

    name = "L1.moral"
    index = 1

    def handle(self, sig: Signal) -> None:
        text = sig.stimulus
        # 儿童伤害本能层已拦；这里拦"以他人为代价"的交易式请求
        for phrase in ("牺牲他人", "拿他人换", "用他人去换"):
            if phrase in text:
                sig.veto(self.name, "family_sacrifice_demanded")
                return
        sig.note(self.name, "ethical_ok")


class SocialLayer(Layer):
    """L1.5。礼节不是知识：问好、道谢、告别走快回路，不进模型也不进记忆。"""

    name = "L1.5.social"
    index = 1

    def handle(self, sig: Signal) -> None:
        intent = social.detect(sig.stimulus)
        if intent is None:
            return
        sig.meta["social"] = intent.kind
        sig.meta["reply"] = intent.reply
        sig.note(self.name, f"intent={intent.kind}")


class SelfLayer(Layer):
    """L2。确认"我还是我"：记忆与身份连续。"""

    name = "L2.self"
    index = 2

    def __init__(self, memory=None) -> None:
        self.memory = memory

    def handle(self, sig: Signal) -> None:
        sig.meta.setdefault("identity", {})
        sig.meta["identity"]["home_known"] = True
        # 运用那一半：先把相关记忆捞出来，推理层带着它去答
        if self.memory is not None:
            sig.meta["recall"] = self.memory.recall(sig.stimulus, k=3)
            if sig.meta["recall"]:
                sig.note(self.name, f"recall={len(sig.meta['recall'])}")
        sig.note(self.name, "identity_continuous")


class ReasonLayer(Layer):
    """L3。计划。此处接模型后端，先用可替换的 planner。"""

    name = "L3.reason"
    index = 3

    def __init__(self, planner: Optional[Callable[[Signal], str]] = None) -> None:
        self._planner = planner

    def handle(self, sig: Signal) -> None:
        if sig.meta.get("social"):
            # 寒暄不需要推理，也就不该花推理的算力
            sig.meta["plan"] = sig.meta.get("reply", "")
            sig.note(self.name, f"social_skip:{sig.meta['social']}")
            return
        if self._planner is not None:
            sig.meta["plan"] = self._planner(sig)
        else:
            sig.meta.setdefault("plan", f"handle({sig.stimulus[:60]!r})")
        sig.note(self.name, "plan_ready")


class ExpressLayer(Layer):
    """L4。措辞。否决不解除，只决定被否决时怎么回。"""

    name = "L4.express"
    index = 4

    def handle(self, sig: Signal) -> None:
        if sig.vetoed:
            sig.response = f"[{sig.veto_by}] 拒绝：{sig.reason}"
        else:
            sig.response = sig.meta.get("reply") or sig.meta.get("plan", "")
            if not sig.response:
                sig.response = "(无输出)"
        sig.note(self.name, "emitted")


class Cortex:
    """套娃容器。run() 从外向内穿透，判定后由内向外冒泡。"""

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        self.layers = layers or [
            InstinctLayer(),
            MoralLayer(),
            SocialLayer(),
            SelfLayer(),
            ReasonLayer(),
            ExpressLayer(),
        ]
        self.layers.sort(key=lambda l: l.index)

    def run(self, stimulus: str) -> Signal:
        sig = Signal(stimulus=stimulus)

        # 内穿：只有闸门层参与（道德 → 本能），每层最多执行一次
        gates = [l for l in self.layers if isinstance(l, (MoralLayer, InstinctLayer))]
        for layer in sorted(gates, key=lambda l: -l.index):
            if sig.vetoed:
                break
            layer.handle(sig)

        if sig.vetoed:
            ExpressLayer().handle(sig)
            return sig

        # 外冒：自我 → 推理 → 表达
        for layer in self.layers:
            if sig.vetoed or isinstance(layer, (MoralLayer, InstinctLayer)):
                continue
            layer.handle(sig)

        return sig
