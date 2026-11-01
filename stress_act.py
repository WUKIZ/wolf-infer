"""压力测试：口语化问句 —— 规则和模型都没见过的说法。

holdout 是同一套规则造的，100% 只证明「复现了规则」。
这 60 条是手写的、刻意的口语/方言/网络用语，标准答案按人的直觉标。

同时跑规则和模型，看谁更抗口语。

用法：python3 stress_act.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent                     # noqa: E402
from local400 import Local400    # noqa: E402

CKPT = Path("/data/wolf400M_act.pt")

# (问句, 人的判断)
CASES = [
    # 时效类 —— 该查
    ("美股今儿咋样", "search"), ("老黄家的卡还涨不", "search"),
    ("明儿下雨不", "search"), ("最近有啥瓜", "search"),
    ("币圈崩了没", "search"), ("昨晚那球谁赢了", "search"),
    ("今儿热搜有啥", "search"), ("隔壁新出的模型咋样", "search"),
    ("苹果又发啥了", "search"), ("房价跌了没", "search"),
    ("油价多少了", "search"), ("黄金现在啥价", "search"),
    ("那谁家的手机能买了不", "search"), ("今年世界杯谁夺冠", "search"),
    ("台风到哪了", "search"), ("春运票好买不", "search"),
    ("双十一啥时候", "search"), ("开学几号", "search"),
    ("股市开没开盘", "search"), ("美债收益率多少", "search"),
    ("汇率破7没", "search"), ("新能源补贴还有没", "search"),
    ("特斯拉降价了没", "search"), ("显卡啥时候补货", "search"),
    ("那部电影上映没", "search"), ("演唱会还有票吗", "search"),
    ("这周末有啥展", "search"), ("快递到哪了", "search"),
    ("明天几度", "search"), ("刚出的政策说了啥", "search"),
    # 闲聊类 —— 该直接回
    ("你咋想", "reply"), ("陪我唠会儿", "reply"),
    ("有点烦", "reply"), ("你说人活着图啥", "reply"),
    ("我该不该换工作", "reply"), ("怎么才能不拖延", "reply"),
    ("数学咋学", "reply"), ("有啥好书", "reply"),
    ("讲个笑话", "reply"), ("你累不累", "reply"),
    ("我睡不着", "reply"), ("解释下量子纠缠", "reply"),
    ("什么是内卷", "reply"), ("帮我起个名", "reply"),
    ("这句咋翻译", "reply"), ("写首诗", "reply"),
    ("你咋看", "reply"), ("说说你自己", "reply"),
    ("你有啥爱好", "reply"), ("你怕不怕", "reply"),
    ("你会想我吗", "reply"), ("今天心情还行", "reply"),
    ("我最近胖了", "reply"), ("教我做饭", "reply"),
    ("推荐个游戏", "reply"),
    # 噪音 —— 该沉默
    ("嗯", "silent"), ("额", "silent"), ("。", "silent"),
    ("啊", "silent"), ("唔", "silent"),
]


def main() -> int:
    m = Local400(CKPT)
    print(f"模型 step={m.step}  用例 {len(CASES)} 条\n")

    hits = {"model": 0, "rule": 0}
    n = {"search": 0, "reply": 0, "silent": 0}
    model_wrong, rule_wrong = [], []

    for text, truth in CASES:
        n[truth] += 1
        state = "态：能量0.5 信任0.5 唤醒0.9"     # 唤醒拉满，排除阈值干扰
        prompt = f"问：主人说：{text}\n{state}\n答："
        out = m.generate(prompt, max_new=40, temperature=0.1)
        hit = re.search(r'"action":\s*"(\w+)"', out)
        pred = hit.group(1) if hit else "?"

        # 规则版：和 gen_act 里同一套判据
        if agent.is_noise(text):
            rule = "silent"
        elif agent._WEB_HINTS.search(text):
            rule = "search"
        else:
            rule = "reply"

        if pred == truth:
            hits["model"] += 1
        elif len(model_wrong) < 8:
            model_wrong.append(f"{text}  期望{truth} 实得{pred}")
        if rule == truth:
            hits["rule"] += 1
        elif len(rule_wrong) < 8:
            rule_wrong.append(f"{text}  期望{truth} 实得{rule}")

    total = len(CASES)
    print(f"模型 {hits['model']}/{total} = {hits['model'] / total:.1%}")
    print(f"规则 {hits['rule']}/{total} = {hits['rule'] / total:.1%}")
    print(f"分布 search {n['search']} reply {n['reply']} silent {n['silent']}\n")
    print("模型判错：")
    for w in model_wrong:
        print("  ", w)
    print("规则判错：")
    for w in rule_wrong:
        print("  ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
