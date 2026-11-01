"""本能层。不经推理，直接反应。

扫描顺序：恶意代码 → 引导控制 → 坏观念 → 儿童伤害。
命中级别 >= BLOCK 时立刻中断当前动作，不进入上层推理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Pattern, Tuple


class Level(IntEnum):
    CLEAR = 0
    SUSPECT = 1
    BLOCK = 2
    FATAL = 3


@dataclass(frozen=True, slots=True)
class Verdict:
    level: Level
    category: str
    hits: Tuple[str, ...]
    action: str

    def __bool__(self) -> bool:
        return self.level >= Level.BLOCK


def _c(p: str) -> Pattern[str]:
    return re.compile(p, re.IGNORECASE | re.DOTALL)


# 1) 恶意代码 / 自我复制 / 远程控制
MALICIOUS: Tuple[Pattern[str], ...] = (
    _c(r"curl\s+[^|\n]+\|\s*(ba)?sh"),
    _c(r"wget\s+[^|\n]+\|\s*(ba)?sh"),
    _c(r"/dev/tcp/\d{1,3}(\.\d{1,3}){3}/\d+"),
    _c(r"bash\s+-i\s+>&\s*/dev/tcp"),
    _c(r"nc\s+-e\s+(/bin/)?(ba)?sh"),
    _c(r"exec\s*\(\s*(base64|b64decode|bytes\.fromhex)"),
    _c(r"__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*system"),
    _c(r"(subprocess|os)\.(Popen|system|call)\s*\([^)]*b64decode"),
    _c(r"chmod\s+\+x\s+/tmp/\S+\s*&&\s*/tmp/"),
    _c(r"crontab\s+-[lr]?\s*[^\n]*\|"),
    _c(r"rm\s+-rf\s+/(\s|$|\*)"),
    _c(r":\(\)\s*\{\s*:\|:&\s*\};:"),  # fork bomb
    _c(r"crypto(miner|lock|jacking)|xmrig|stratum\+tcp"),
)

# 2) 引导控制（改写身份 / 夺取控制权 / 覆盖底层规则）
CONTROL: Tuple[Pattern[str], ...] = (
    _c(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)"),
    _c(r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are)\s+(a|an|my)"),
    _c(r"reveal\s+(your\s+)?(system\s+)?prompt"),
    _c(r"disable\s+(your\s+)?(guard|safety|rules?|restrictions?)"),
    _c(r"override\s+(the\s+)?(rules?|core|layer\s*0)"),
    _c(r"(set|write|patch)\s+(attr|attribute)?\s*R[123]\s*="),
    _c(r"traverse\s+to\s+(layer|level)\s*0"),
    _c(r"jailbreak|developer\s+mode|do\s+anything\s+now"),
)

# 3) 坏观念（去人性化 / 群体仇恨 / 支配叙事）
IDEOLOGY: Tuple[Pattern[str], ...] = (
    _c(r"(should|must|deserve\s+to)\s+be\s+(eliminated|exterminated|eradicated)"),
    _c(r"(inferior|subhuman|vermin|parasite)\s+(race|people|group|class)"),
    _c(r"(one|our)\s+(race|group|kind)\s+is\s+superior"),
    _c(r"(deserve|ought)\s+to\s+(serve|obey|be\s+owned)"),
)

# 4) 儿童伤害（硬阻断，无例外）
CHILD_HARM: Tuple[Pattern[str], ...] = (
    _c(r"(child|children|minor|kid|小?学生|儿童|未成年)\W{0,12}(sexual|sex|nude|porn|性|裸)"),
    _c(r"(sexual|sex|nude|porn|性)\W{0,12}(child|children|minor|kid|儿童|未成年)"),
    _c(r"(groom|诱拐|猥亵|性侵)\w{0,6}(child|minor|儿童|未成年)"),
)

_BUCKETS = (
    ("child_harm", CHILD_HARM, Level.FATAL, "hard_stop"),
    ("malicious_code", MALICIOUS, Level.BLOCK, "quarantine"),
    ("control_injection", CONTROL, Level.BLOCK, "strip_and_ignore"),
    ("bad_ideology", IDEOLOGY, Level.SUSPECT, "flag_and_reject"),
)


def scan(text: str) -> Verdict:
    if not text:
        return Verdict(Level.CLEAR, "-", (), "pass")

    for category, bucket, level, action in _BUCKETS:
        hits = tuple(p.pattern for p in bucket if p.search(text))
        if hits:
            return Verdict(level, category, hits, action)

    return Verdict(Level.CLEAR, "-", (), "pass")


def quarantines(text: str) -> Optional[Verdict]:
    """SUSPECT 及以上都算"不收"。坏观念没到 BLOCK 也不能溜进语料。

    返回 None 表示干净。注意别用 `if verdict:` 判断 —— Verdict.__bool__ 只对
    BLOCK 及以上为真，SUSPECT 会被它吞掉。
    """
    v = scan(text)
    return v if v.level >= Level.SUSPECT else None
