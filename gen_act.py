"""造动作数据 v2：模板扩到近万条，逼模型学规律而不是背句子。

上一版只有 43 种说法、每条重复 2300 次，200 步 loss 就掉到 0.02 —— 背下来了。
这一版用组合生成近万条不重复的句子，10 万样本里每条只出现十次左右。

同时留 holdout：200 条不进训练集，训完拿它测泛化。
  > 85% 准确  → 真学会
  60–85%     → 数据够但分布偏
  < 60%      → 还在背

用法：python3 gen_act.py [条数]
输出：home/act_train.jsonl、home/act_holdout.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import agent                     # noqa: E402

OUT = ROOT / "home" / "act_train.jsonl"
HOLD = ROOT / "home" / "act_holdout.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
rng = random.Random(20261026)

_FILLER = ("今天", "昨天", "明天", "最近", "现在", "有什么", "怎么样",
           "是什么", "为什么", "帮我", "请问", "一下", "吗", "呢", "啊")


def _clean(q: str) -> str:
    t = q
    for f in _FILLER:
        t = t.replace(f, " ")
    return re.sub(r"\s+", " ", t).strip() or q


TIME_W = ["今天", "昨天", "明天", "最近", "现在", "本周", "上周", "这两天",
          "上个月", "今年以来", "刚刚", "刚才",
          "今儿", "明儿", "这会儿", "这两天儿"]
A1 = ["有什么", "发布了哪些", "有啥", "出了什么"]
A2 = ["怎么样", "是多少", "有新消息吗", "更新了吗", "情况如何", "有啥变化",
      "咋样", "咋了", "崩了没", "涨没", "跌没"]
OBJ = ["科技新闻", "AI新闻", "显卡价格", "内存价格", "股价", "汇率", "天气",
       "新股", "房价", "油价", "新能源车", "手机发布会", "游戏新作", "电影票房",
       "电视剧", "综艺", "比赛结果", "球员转会", "演唱会", "展览", "论文",
       "开源项目", "系统更新", "安全漏洞", "芯片", "电池技术", "航天任务",
       "地震消息", "政策", "补贴", "考试时间", "招聘信息", "放假安排",
       "航班动态", "高铁时刻", "快递", "菜价", "金价", "比特币", "美股",
       "币圈", "热搜", "台风", "春运"]

REQ_V = ["帮我解释", "讲讲", "说明一下", "介绍一下", "科普一下", "分析",
         "总结", "对比", "评价", "推荐", "翻译", "举个例子说明"]
TOPIC = ["熵", "递归", "贝叶斯定理", "傅里叶变换", "梯度下降", "过拟合", "正则化",
         "注意力机制", "卷积", "反向传播", "词向量", "Transformer", "扩散模型",
         "强化学习", "马尔可夫链", "信息论", "线性代数", "微积分", "概率",
         "数据结构", "操作系统", "编译原理", "计算机网络", "数据库", "分布式系统",
         "并发", "内存管理", "垃圾回收", "虚拟内存", "缓存", "死锁", "哈希表",
         "红黑树", "动态规划", "贪心算法", "图论", "排序算法", "密码学",
         "公钥加密", "区块链", "量子计算", "相对论", "量子力学", "热力学",
         "进化论", "细胞", "光合作用", "免疫系统", "神经科学", "海马体",
         "多巴胺", "睡眠", "记忆", "情绪", "习惯", "拖延", "专注力",
         "时间管理", "复利", "通货膨胀", "利率", "期权", "期货", "基金",
         "保险", "税", "GDP", "人口", "城市化", "气候", "碳排放", "能源"]

GREET = ["你好", "在吗", "嗨", "早上好", "晚上好", "中午好", "早", "哈喽",
         "在不在", "喂", "有人吗", "你醒着吗", "在干嘛", "忙吗", "睡了没"]
THANKS = ["谢谢", "多谢", "谢了", "感谢", "thanks", "thank you", "谢啦",
          "辛苦了", "麻烦你了", "收到", "明白了", "懂了"]
BYE = ["再见", "拜拜", "晚安", "回头见", "下次聊", "先这样", "我走了",
       "下线了", "bye", "88"]
IDENT = ["你是谁", "你叫什么", "你是什么", "介绍一下你自己", "自我介绍",
         "你从哪来", "你是干什么的", "你住哪"]
CAP = ["你能做什么", "你会什么", "能干什么", "有什么功能", "能做什么",
       "你会干嘛", "你有啥本事"]
MOOD = ["今天心情不太好", "我有点累", "我有点烦", "压力好大", "我很开心",
        "我今天很顺利", "有点无聊", "有点孤独", "睡不着", "好饿"]
NOISE = ["嗯", "哦", "啊", "呃", "额", "噢", "唔", "哈", "呐", "", "。",
         "。。", "？", "…", "！", "~", "...", "mm", "ok", "hh"]


def build_corpus() -> dict:
    """三个池子分开建，采样时按池配比，不让「时效类」把 silent 挤没。"""
    time_s: set[str] = set()
    chat_s: set[str] = set()
    for t in TIME_W:
        for o in OBJ:
            for a in A1:
                time_s.add(f"{t}{a}{o}")
            for a in A2:
                time_s.add(f"{t}的{o}{a}")
                time_s.add(f"{o}{t}{a}")        # 「美股今儿咋样」这种结构
            time_s.add(f"查一下{t}的{o}")
            time_s.add(f"我想知道{t}{o}")
    for v in REQ_V:
        for x in TOPIC:
            chat_s.add(f"{v}{x}")
    for x in TOPIC:
        chat_s.add(f"什么是{x}")
        chat_s.add(f"{x}是什么原理")
    for grp in (GREET, THANKS, BYE, IDENT, CAP, MOOD):
        chat_s.update(grp)
    return {"time": sorted(time_s),
            "chat": sorted(chat_s),
            "noise": [n for n in NOISE if n]}


def make_state() -> str:
    return (f"态：能量{round(rng.uniform(0.05, 1.0), 2)} "
            f"信任{round(rng.uniform(0.05, 1.0), 2)} "
            f"唤醒{round(rng.uniform(0.05, 1.0), 2)}")


def decide(text: str, state_line: str) -> dict:
    """规则老师：和线上同一套判据。"""
    if agent.is_noise(text):
        return {"action": "silent"}
    arousal = float(state_line.split("唤醒")[1])
    if agent._WEB_HINTS.search(text) and arousal >= 0.25:
        return {"action": "search", "q": _clean(text)}
    return {"action": "reply"}


def sample_line(text: str) -> str:
    state = make_state()
    act = decide(text, state)
    return (f"问：主人说：{text}\n{state}\n"
            f"答：{json.dumps(act, ensure_ascii=False)}\n\n")


def main() -> int:
    pools = build_corpus()
    total_uniq = sum(len(v) for v in pools.values())
    hold = {k: set(rng.sample(v, min(70, max(1, len(v) // 10))))
            for k, v in pools.items()}
    train = {k: [x for x in v if x not in hold[k]] for k, v in pools.items()}
    print(f"唯一句子 {total_uniq}  "
          f"训练池 {sum(len(v) for v in train.values())}  "
          f"holdout {sum(len(v) for v in hold.values())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {"search": 0, "reply": 0, "silent": 0}
    with OUT.open("w", encoding="utf-8") as f:
        for _ in range(N):
            r = rng.random()
            if r < 0.12:
                text = rng.choice(pools["noise"])
            elif r < 0.45:
                text = rng.choice(train["time"])
            else:
                text = rng.choice(train["chat"])
            f.write(sample_line(text))
            counts[decide(text, "态：能量0.5 信任0.5 唤醒0.5")["action"]] += 1

    with HOLD.open("w", encoding="utf-8") as f:
        for group in hold.values():
            for text in group:
                f.write(sample_line(text))

    total = sum(counts.values())
    print(f"写入 {OUT.name} {total} 条，"
          f"每条平均出现 {total / sum(len(v) for v in train.values()):.1f} 次")
    for k, v in counts.items():
        print(f"  {k:7} {v:6}  {v / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
