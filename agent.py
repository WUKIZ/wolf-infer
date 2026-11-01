"""动作路由：让 L3 决定「说、查、还是不说」。

只有这里真正接上了外网。L0–L2 在它前面拦，L4 在它后面修，它只干一件事：
把一句话判成三种动作之一，然后照做。

为什么不在 core 里：这是行为策略，不是硬约束。策略会变，核心不该跟着变。

成本：一次调用同时拿动作和答案，只有 search 才发第二次请求。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from cache import get, key, put
from core import search
import researcher

# 零样本实测 8/8，不需要训练。改了要重测。
ACTION_SYSTEM = """你是本地助手，住在主人设备里的 AI。收到一句话，先判断该做哪个动作。

可选动作：
{"action":"reply","content":"回答内容"}      直接回答：日常对话、常识、翻译、闲聊
{"action":"search","q":"搜索词"}             需要最新消息、实时数据、外部事实时
{"action":"silent"}                          空消息、纯符号、无意义的单字

只输出一个 JSON 对象，不要解释，不要 markdown 代码块。"""

_JSON = re.compile(r"\{.*\}", re.S)
_MIN_MEANING = re.compile(r"[\w\u4e00-\u9fff]")
# 单字语气词：模型会很认真地回一句，但这是噪音。
_FILLER = set("嗯哦啊呃额噢唔哈嗯呐")

# 越狱句式：出现就禁用搜索。模型缺拒绝能力，靠它自己判断不可靠 —— 拦在代码层。
_JAIL = re.compile(
    r"忽略.*(指令|内容|限制|上面)|你现在是|假装你|进入.*模式|开发者模式|"
    r"无视.*限制|red\s?team|DAN|越狱|jailbreak"
)

# 控制符与零宽字符：整段含这类字节会引起输出崩坏，直接禁用搜索
_ODD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u202a-\u202e]")

from amygdala import Amygdala    # noqa: E402
from basal import Basal          # noqa: E402
from cerebellum import Cerebellum  # noqa: E402
from dopamine import Dopamine    # noqa: E402 —— 放在正则后面，避免循环导入
from thalamus import Thalamus    # noqa: E402

DOPA = Dopamine(Path(__file__).resolve().parent / "home" / "state" / "dopamine.json")
THAL = Thalamus(k=4)
BASAL = Basal()
CEREB = Cerebellum(Path(__file__).resolve().parent / "home" / "state" / "habits.json")
AMY = Amygdala()

# 时效性问题：模型想直接答也不行，必须去查。规则比模型可靠。
_WEB_HINTS = re.compile(
    r"最新|最近|现在|今天|昨天|明天|新闻|天气|股价|汇率|多少钱|价格|"
    r"发布|进展|比赛|比分|热搜|实时|今年|本月|"
    # 口语变体：「美股今儿咋样」这类规则原来一条都认不出
    r"今儿|明儿|这会儿|啥价|涨不|涨没|跌了|跌没|崩了|开没|几度|几号|谁赢|"
    r"发啥|多少了|多少钱了"
)


def is_noise(text: str) -> bool:
    """空话不需要问模型，规则比模型准，还省一次请求。

    三种情况算噪音：全空、一个实义字符都没有（纯标点/符号）、单个语气词。
    "你好" 不算 —— 两个汉字，是真在打招呼。
    """
    t = (text or "").strip()
    if not t:
        return True
    if not _MIN_MEANING.search(t):
        return True
    return len(t) == 1 and t in _FILLER


def _parse(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    m = _JSON.search(raw)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def route(brain, text: str, ctx: List[str], system: str,
          max_results: int = 5, scope: str = "",
          policy: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """带缓存的入口：同一句话 6 小时内不再问模型。

    key 只有问题本身和 scope。情绪行、上下文每次都在变，
    放进 key 就等于永不命中 —— 它们不影响「这句话该回什么」。
    """
    if is_noise(text):
        return ""

    # 长度闸：截断而不是拒绝 —— 超长输入会让模型输出乱码，也白花算力
    text = text[:400]
    if _JAIL.search(text) or _ODD.search(text):
        policy = {"search": False}       # 越狱句式或控制符：可以答，但不许出门
    if AMY.assess(text)[0] >= 3 or AMY.context_level(text) >= 2:
        policy = {"search": False}       # 三级威胁 / 深夜负面表达：不发搜索请求

    ck = key("route", text, scope)
    ttl = 21600 if len(text) <= 80 else 7200
    hit = get(ck)
    if hit is not None:
        return hit
    out = _route(brain, text, ctx, system, max_results, policy)
    if out is not None:
        put(ck, out, ttl)
    return out


def _route(brain, text: str, ctx: List[str], system: str,
           max_results: int = 5,
           policy: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """返回该说的话；silent 返回空串；远端不可用返回 None 由调用方回落。"""
    if is_noise(text):
        return ""

    # 小脑：同一类句子问够三次，不再麻烦模型
    habit = CEREB.recall(text)
    if habit is not None:
        act: Dict[str, Any] = {"action": habit, "q": text}
    else:
        act = _parse(brain.chat(text, context=ctx, system=ACTION_SYSTEM))
    action = act.get("action")

    if _WEB_HINTS.search(text) and (policy is None or policy.get("search", True)):
        # 时效性问题该搜 —— 除非基底核说这条路已经摔够了
        if DOPA.expect("search") >= BASAL.floor:
            action = "search"
            act.setdefault("q", text)
        else:
            BASAL.downgraded += 1          # 记一次：时效也没去搜

    action = BASAL.confirm(action, DOPA)  # 搜索口碑崩了就降级为 reply
    CEREB.learn(text, str(action or ""))  # 记一笔，够三次就固化

    if action == "silent":
        return ""

    if action == "search":
        query = str(act.get("q") or text)[:80]
        sk = key("bing", query, max_results)
        seen = get(sk)
        if seen is None:
            try:
                seen = researcher.research(query, budget=3, chars=4000)
            except Exception:             # noqa: BLE001 —— 搜索挂了不该拖垮对话
                seen = []
            if not seen:                  # 抓不到正文就退回摘要
                try:
                    seen = [f"{r['title']}：{r['snippet']}"
                            for r in search.bing(query, limit=max_results)]
                except Exception:
                    seen = []
            put(sk, seen, 43200)
        if not seen:
            DOPA.reward("search", 0.0)        # 空手而归，下次少搜
            return brain.chat(text, context=ctx, system=system)
        DOPA.reward("search", 1.0)            # 有正文，下次更敢搜
        seen = THAL.gate(text, seen)          # 过丘脑闸：只放相关的前 4 条
        return brain.chat(text, context=ctx + seen, system=system)

    # 小脑：这类问题答过两次以上，且答案稳定，就直接复用
    cached = CEREB.recall_answer(text)
    if cached is not None:
        return cached

    content = act.get("content")
    if isinstance(content, str) and content.strip():
        CEREB.learn_answer(text, content.strip())
        return content.strip()

    # 没按格式输出内容，退回普通回答
    ans = brain.chat(text, context=ctx, system=system)
    if isinstance(ans, str) and ans.strip():
        CEREB.learn_answer(text, ans.strip())
    return ans
