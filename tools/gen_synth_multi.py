#!/usr/bin/env python3
"""手机端合成语料：自带质量检查，不依赖 guard/。"""
from __future__ import annotations
import os
import argparse, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.remote import RemoteBrain

CATS = ["生活", "工作", "学习", "技术", "编程", "网络", "安全", "记忆", "情绪", "人际",
        "健康", "饮食", "出行", "理财", "家居", "工具", "数据", "训练", "硬件", "写作"]
SUBS = ["入门", "常见错误", "省时办法", "检查清单", "取舍", "长期坚持", "应急处理", "对比选择"]
MODS = ["新手", "预算有限", "时间很紧", "一个人做", "团队协作", "在家里", "在办公室", "长期", "短期", "失败过一次"]
PAIR = re.compile(r"问：(.+?)\n答：(.+?)(?=\n问：|\Z)", re.S)


def topics(n: int, offset: int = 0) -> list[str]:
    pool = [f"{c}里的{s}（{m}）" for c in CATS for s in SUBS for m in MODS]
    return pool[offset:offset + n]


def parse(text: str) -> list[str]:
    out = []
    for q, a in PAIR.findall(text or ""):
        q, a = q.strip(), a.strip().replace("\n", "")
        if 6 <= len(q) <= 60 and 12 <= len(a) <= 120:
            out.append(f"问：{q}\n答：{a}")
    return out


def ok_text(block: str) -> bool:
    chars = re.sub(r"\s+", "", block)
    cjk = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
    return len(chars) >= 400 and cjk / max(1, len(chars)) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--out", default="home/synth.txt")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--key", default="")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    if a.key:
        os.environ["DEEPSEEK_API_KEY"] = a.key
    rb = RemoteBrain({"base_url": a.base_url, "model": a.model,
                      "max_tokens": a.max_tokens, "temperature": a.temperature,
                      "timeout_s": a.timeout})
    if not rb.available():
        print("[synth] 远端不可用", file=sys.stderr); return 2
    out = Path(a.out)
    seen = set()
    if out.exists():
        seen = {l for l in out.read_text("utf-8").splitlines() if l.startswith("问：")}
    lines, chars = [], 0
    for t in topics(a.calls, a.offset):
        prompt = (f"你在为中文小模型造训练数据。围绕「{t}」写 10 组问答，"
                  "格式严格：问：…\\n答：…，答案 20-60 字，具体、可验证、不空泛，不要编号。")
        raw = ""
        for attempt in range(3):
            raw = rb.chat(prompt) or ""
            if raw:
                break
            time.sleep(2 + attempt * 3)
        pairs = [p for p in parse(raw) if p not in seen]
        if len(pairs) < 4:
            continue
        block = "\n".join(pairs)
        if not ok_text(block * 3):
            continue
        lines.extend(pairs); seen.update(pairs); chars += len(block)
        print(f"[synth] {t}: +{len(pairs)}")
    if lines:
        with out.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
    print(f"[synth] +{len(lines)} 组 / {chars} 字 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
