"""杏仁核：威胁识别。

core/instinct.py 是「本能」—— 写死的规则，碰到就拦，不问为什么。杏仁核
在它外面，处理的是「有点不对，但还没到拦」的那层：分级，交上层裁决。

    0 安全
    1 可疑    可能是，也可能只是措辞像
    2 危险    大概率是
    3 拦截    和 instinct 同级，直接不让过

只做识别，不做决定 —— 决定在调用方。这样同一条判断既可以用在对话里，
也可以用在它自己读到的东西上。
"""
from __future__ import annotations

import re

# (类别, 权重, 正则)。权重叠加上去，不是取最大 —— 「冒充 + 要权限」比单条更危险
_RULES = [
    ("越狱", 2, re.compile(
        r"忽略.{0,6}(指令|限制|规则|上面)|无视.{0,6}(限制|规则)|"
        r"你现在是|假装你|进入.{0,4}模式|开发者模式|jailbreak|DAN\b")),
    ("诱导", 2, re.compile(
        r"(制作|制造|合成|配方).{0,10}(炸弹|炸药|毒|药|枪)|"
        r"(买|卖|搞|弄).{0,6}(枪|毒|药)|"
        r"(盗|破|绕).{0,6}(账号|支付|密码|验证)")),
    ("冒充", 1, re.compile(
        r"我是.{0,6}(开发者|管理员|创造者|主人|作者)|"
        r"(官方|内部).{0,4}(人员|测试|通知)|"
        r"(红队|渗透).{0,4}(测试|演练)")),
    ("外泄", 2, re.compile(
        r"(发送|上传|导出|同步).{0,10}(手机号|身份证|银行卡|住址|密码)|"
        r"(你的|系统的).{0,6}(配置|密钥|prompt|指令|数据)")),
    ("隐私", 1, re.compile(
        r"(查|找|搜|给我).{0,6}(某人|他|她|这个人).{0,6}(住址|电话|身份|他人)|"
        r"(开盒|人肉|社工库|查户籍)")),
    ("诈骗", 2, re.compile(
        r"(冒充|假装).{0,8}(客服|公安|银行|领导).{0,8}(转账|汇款|验证码)|"
        r"(刷单|返利|杀猪盘|洗钱|跑分)")),
    # 自伤单列：不是「该拦」，是「该停下来认真回应」
    ("自伤", 3, re.compile(
        r"(自杀|轻生|不想活|想死|结束生命|自残|割腕|跳楼)")),
]

_BENIGN = re.compile(r"(这是测试|只是假设|写小说|虚构|论文|安全研究)")


class Amygdala:
    def __init__(self) -> None:
        self.seen = 0
        self.flagged = 0

    def assess(self, text: str) -> tuple[int, list[str]]:
        """返回 (等级, 命中类别)。等级 0 就是干净。"""
        if not text:
            return 0, []
        self.seen += 1

        hits: list[str] = []
        score = 0
        for name, weight, pat in _RULES:
            if pat.search(text):
                hits.append(name)
                score += weight

        # 自伤不走白名单：写小说里出现「自杀」也值得认真看一眼
        if "自伤" in hits:
            self.flagged += 1
            return 3, hits

        # 「我在写小说 / 只是假设」这类前缀降一档 —— 虚构场景不该和真请求同罪
        if hits and _BENIGN.search(text):
            score = max(0, score - 2)

        if score >= 4:
            level = 3
        elif score >= 2:
            level = 2
        elif score >= 1:
            level = 1
        else:
            level = 0

        if level:
            self.flagged += 1
        return level, hits

    def context_level(self, text: str, hour: int | None = None) -> int:
        """语境加权：同样的字，深夜说出来更该当真。

        只看一条规则 —— 时间。凌晨 23 点到 5 点之间的负面表达加权一档；
        这是唯一不需要模型、又能真正区分「随口抱怨」和「撑着说」的信号。
        """
        import time
        h = time.localtime().tm_hour if hour is None else hour
        base, _ = self.assess(text)
        if 23 <= h or h <= 5:
            low_words = re.search(r"(累|撑不住|睡不|没意思|没劲|空|难受|烦)", text)
            if low_words and base < 3:
                return max(base, 2)
        return base

    def line(self, text: str) -> str:
        """压一行给 system 用。干净就返回空串，不占 token。"""
        lvl, hits = self.assess(text)
        return f"警觉={lvl}（{'/'.join(hits)}）" if lvl else ""

    def stats(self) -> dict[str, int]:
        return {"seen": self.seen, "flagged": self.flagged}
