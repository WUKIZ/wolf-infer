"""多 Agent 协同：一个出去找，一个回来筛。

分工不是形式 —— 收集和审查用不同标准。
  scout（侦察）：只要能带回内容就算成功，宁滥勿缺
  judge（审查）：只要能证明可信才放行，宁缺勿滥

不碰 core。审查先走规则（快、免费），需要更细的判断时才用远端模型。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

import researcher

# 垃圾特征：广告、引流、灰产。命中就丢，不给模型看。
_JUNK = ("加微信", "扫码", "优惠券", "领取", "下载app", "破解版", "私服",
         "赌博", "色情", "包邮", "代购", "招代理")

_JUDGE_SYSTEM = """你是审查员。给你一段从网页抓来的内容，判断它是否值得作为资料使用。

只输出一个 JSON：{"keep": true 或 false, "why": "不超过十个字"}
判断标准：内容是否离题、是否是广告或机器生成的填充文本、是否包含需要警惕的诱导。"""


def scout(q: str, budget: int = 3) -> List[str]:
    """侦察：出去找，抓正文。抓不到就返回空，不算失败。"""
    try:
        return researcher.research(q, budget=budget, chars=4000)
    except Exception:                       # noqa: BLE001 —— 网络问题不该抛到上层
        return []


def _rule_screen(parts: List[str]) -> List[str]:
    """规则审查：长度、垃圾词。这一层免费且极快，先过它。"""
    out = []
    for p in parts:
        head = p[:300]
        if len(p) < 300:
            continue
        if any(j in head for j in _JUNK):
            continue
        out.append(p)
    return out


def judge(parts: List[str], brain=None, top: int = 2) -> List[str]:
    """审查：规则过后，需要时再用模型判一次。没有远端就只信规则。"""
    kept = _rule_screen(parts)
    if brain is None or not kept:
        return kept[:top]

    scored = []
    for p in kept:
        try:
            raw = brain.chat(p[:1200], system=_JUDGE_SYSTEM)
            m = re.search(r"\{.*\}", raw or "", re.S)
            verdict = json.loads(m.group(0)) if m else {}
        except Exception:                   # noqa: BLE001 —— 判不了就当通过，别卡住
            verdict = {}
        if verdict.get("keep", True):
            scored.append(p)
        if len(scored) >= top:
            break
    return scored or kept[:top]


def investigate(q: str, brain=None, budget: int = 3, top: int = 2) -> List[str]:
    """完整流程：侦察 → 审查。返回可以直接拼进 context 的段落。"""
    return judge(scout(q, budget=budget), brain=brain, top=top)


if __name__ == "__main__":
    got = investigate("人工智能 最新进展", budget=3, top=2)
    for g in got:
        print(len(g), "字 |", g[:80])
