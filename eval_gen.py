"""生成质量抽查：10 个问题，看它答得像不像人。

loss 到 2.4 之后就不说明问题了 —— 能背格式和能回答问题，loss 差不多。
这个方法笨但有效：直接读它写出来的话。

用法：python3 eval_gen.py [ckpt]
"""
from __future__ import annotations

import sys
from pathlib import Path

from local400 import Local400

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/data/wolf400M_sft.pt")

QS = [
    "什么是熵", "你好", "讲讲递归", "我有点累", "推荐一部电影",
    "为什么要睡觉", "解释一下梯度下降", "今天心情不好",
    "怎么才能不拖延", "你是谁",
]


def main() -> int:
    m = Local400(CKPT)
    print(f"模型 step={m.step}\n", flush=True)

    shown = 0
    for q in QS:
        out = m.generate("问：" + q + "\n答：", max_new=80, temperature=0.3)
        out = out.strip().replace("\n", " ")
        print(f"问：{q}")
        print(f"答：{out}\n", flush=True)
        if len(out) > 8 and q[:4] not in out[:8]:
            shown += 1

    print(f"看起来像回答的：{shown}/{len(QS)}")
    print("判据：≥6 算成功 · 重复问句或乱码 = mask 有问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
