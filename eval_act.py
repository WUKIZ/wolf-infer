"""holdout 评测：只看 action 判对没有，不看 q —— q 是从问句抄的，不算能力。

loss 低不代表学会：输出是固定 JSON 骨架，背熟 `{"action": "` 就能压到 0.01。
真正的问题是「换一句没见过的问法，它还判得对吗」，只有这个脚本能回答。

用法：python3 eval_act.py [ckpt路径]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from local400 import Local400

ROOT = Path(__file__).resolve().parent
HOLD = ROOT / "home" / "act_holdout.jsonl"
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/wolf400M_act.pt")


def main() -> int:
    if not HOLD.exists():
        print(f"缺 {HOLD} —— 先跑 gen_act.py")
        return 1
    if not CKPT.exists():
        print(f"缺 {CKPT} —— 先跑 train_act.py")
        return 1

    m = Local400(CKPT)
    print(f"模型 step={m.step} device={m.dev}", flush=True)

    ok = n = 0
    wrong: list[str] = []
    for block in HOLD.read_text("utf-8").split("\n\n"):
        if "答：" not in block:
            continue
        prompt, gold = block.split("答：", 1)
        out = m.generate(prompt + "答：", max_new=40, temperature=0.1)
        hit = re.search(r'"action":\s*"(\w+)"', out)
        try:
            truth = json.loads(gold.strip())["action"]
        except (ValueError, KeyError):
            continue
        n += 1
        if hit and hit.group(1) == truth:
            ok += 1
        elif len(wrong) < 5:
            wrong.append(f"期望 {truth} 实得 {(hit.group(1) if hit else '?')} ← {prompt.strip()[-30:]}")

    print(f"\n准确率 {ok}/{n} = {ok / max(1, n):.1%}")
    for w in wrong:
        print("  ", w)
    print("\n判据：>80% 真学会 · 60–80% 半成 · <60% 只会背")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
