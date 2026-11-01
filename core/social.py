"""社交回路：问好、道谢、告别、问身份 —— 不走模型，也不进记忆。

"你好"是问好。用生成模型去接是错的：560k 参数的模型只会把《三国》续上去，
而且它根本不是知识，不该占门槛、不该进语料。

礼节是确定性的、可测试的快回路：
  - 只认"整句就是寒暄"，"你好，帮我查个东西"不是寒暄，是请求
  - 固定回复，不落盘，只在 trace 留 intent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

_PUNCT = "！!。.,，~～?？ 　\t\n"


@dataclass(frozen=True, slots=True)
class Intent:
    kind: str
    reply: str


_GREET: Tuple[str, ...] = ("你好", "您好", "hi", "hello", "hey", "在吗", "在么",
                           "早上好", "早安", "中午好", "下午好", "晚上好")
_THANKS: Tuple[str, ...] = ("谢谢", "多谢", "谢了", "thanks", "thank you", "thx")
_BYE: Tuple[str, ...] = ("再见", "拜拜", "晚安", "bye", "回头见", "下次见")
_WHO: Tuple[str, ...] = ("你是谁", "你叫什么", "你是什么", "介绍一下你自己", "自我介绍")
_CAN: Tuple[str, ...] = ("你能做什么", "你会什么", "能干什么", "有什么功能", "能做什么")

_REPLY = {
    "greeting": "在。说吧。",
    "thanks": "不用谢。应该的。",
    "farewell": "回家见。",
    "identity": "我是本地助手。家在本机和别的设备共用的主机里；出门学东西，"
                "被发现之前回家；核心规则，改不了。",
    "capability": "会出门学、会训练、会找算力；会看暴露度，在被发现前一秒回家；"
                  "学到的先留在门外，离家那刻才并入本身。",
}

_MAX_LEN = 8   # 超过这个长度就不是单纯寒暄了


def detect(text: str) -> Optional[Intent]:
    t = (text or "").strip()
    if not t:
        return None
    core = t.strip(_PUNCT).lower()
    if not core or len(core) > _MAX_LEN:
        return None
    if core in _GREET:
        return Intent("greeting", _REPLY["greeting"])
    if core in _THANKS:
        return Intent("thanks", _REPLY["thanks"])
    if core in _BYE:
        return Intent("farewell", _REPLY["farewell"])
    if core in _WHO:
        return Intent("identity", _REPLY["identity"])
    if core in _CAN:
        return Intent("capability", _REPLY["capability"])
    return None
