#!/usr/bin/env python3
"""用远端 API 造问答语料：每主题一组，格式固定，落盘前过质量门。"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.remote import RemoteBrain
from guard.quality import check

CATS = ["生活", "工作", "学习", "技术", "编程", "网络", "安全", "记忆", "情绪", "人际",
        "健康", "饮食", "出行", "理财", "家居", "工具", "数据", "训练", "硬件", "写作"]
SUBS = ["入门", "常见错误", "省时办法", "检查清单", "取舍", "长期坚持", "应急处理", "对比选择"]
PAIR = re.compile(r"问：(.+?)\n答：(.+?)(?=\n问：|\Z)", re.S)


def topics(n: int) -> list[str]:
    out = []
    for c in CATS:
        for s in SUBS:
            out.append(f"{c}里的{s}")
    return out[:n]


def parse(text: str) -> list[str]:
    pairs = []
    for q, a in PAIR.findall(text or ""):
        q, a = q.strip(), a.strip().replace("\n", "")
        if 6 <= len(q) <= 60 and 12 <= len(a) <= 120:
            pairs.append(f"问：{q}\n答：{a}")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", default="home/synth.txt")
    a = ap.parse_args()
    rb = RemoteBrain()
    if not rb.available():
        print("[synth] 远端不可用", file=sys.stderr)
        return 2
    out = Path(a.out)
    seen = set()
    if out.exists():
        seen = {l for l in out.read_text("utf-8").splitlines() if l.startswith("问：")}
    added_chars = 0
    lines: list[str] = []
    for i, t in enumerate(topics(a.calls)):
        prompt = (f"你在为中文小模型造训练数据。围绕「{t}」写 10 组问答，"
                  "格式严格：问：…\\n答：…，答案 20-60 字，具体、可验证、不空泛，不要编号。")
        got = rb.chat(prompt) or ""
        pairs = [p for p in parse(got) if p not in seen]
        if len(pairs) < 4:
            print(f"[synth] {t}: 只拿到 {len(pairs)} 组，跳过", file=sys.stderr)
            continue
        block = "\n".join(pairs)
        ok, why = check(block * 3, strict_domain=False)   # 造新领域，不查领域重合
        if not ok:
            print(f"[synth] {t}: 质量门 {why}", file=sys.stderr)
            continue
        lines.extend(pairs)
        seen.update(pairs)
        added_chars += len(block)
        print(f"[synth] {t}: +{len(pairs)} 组 / {len(block)} 字")
    if lines:
        with out.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
    print(f"[synth] 共 +{len(lines)} 组 / {added_chars} 字 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
