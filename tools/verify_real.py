#!/usr/bin/env python3
"""真实性验证：从六个角度证明"这不是壳"。

不是读自己的日志自证 —— 每条都独立复算：
  1. 权重真的在盘上：绕开 TinyGPT，直接用 numpy 读 checkpoint，核对形状与数值范围
  2. 前向是真的：用权重独立复算 head 层 logits，和 model.forward 对得上
  3. 训练是真的：固定一批数据跑 20 步，损失必须下降、参数必须变化
  4. 生成是真的：同一提示两次采样必须不同（不是查表）
  5. 守卫是真的：用 hashlib 独立重算 core/ 哈希，和 manifest 比对
  6. 记忆是真的：写进门槛的文件必须能在磁盘上读到原文，离家必须让语料变大

    ./bin/py tools/verify_real.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.home import Home  # noqa: E402
from core.memory import MemoryStore  # noqa: E402
from core.model import CharTokenizer, TinyGPT  # noqa: E402

ROWS = []


def row(name: str, ok: bool | None, evidence: str) -> None:
    ROWS.append((name, "PASS" if ok is True else ("INFO" if ok is None else "FAIL"), evidence))


def check_checkpoint() -> None:
    path = ROOT / "home" / "models" / "tiny.bin"
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    params = [k for k in z.files if k.startswith("p_")]
    sizes = int(sum(z[k].size for k in params))
    nonzero = float(np.abs(z["p_head"]).mean())
    row("权重在盘上（独立读取）", len(params) > 10 and sizes > 100_000,
        f"{len(params)} 张量 / {sizes:,} 参数 / vocab={cfg['vocab']} / mean|head|={nonzero:.4f}")


def check_forward() -> None:
    m = TinyGPT.load(ROOT / "home" / "models" / "tiny.bin")
    tok = CharTokenizer.from_chars(m.chars)
    ids = np.array([tok.encode("話說天下大勢")[:8]])
    logits, cache = m.forward(ids)
    # 独立复算最后一层：xn @ head
    xn = cache["xn"]
    manual = xn.reshape(-1, m.cfg.n_embd) @ m.params["head"]
    delta = float(np.abs(manual - logits.reshape(-1, m.cfg.vocab)).max())
    row("前向可独立复算", delta < 1e-6, f"max|Δ|={delta:.2e}")


def check_training() -> None:
    m = TinyGPT.load(ROOT / "home" / "models" / "tiny.bin")
    tok = CharTokenizer.from_chars(m.chars)
    text = (ROOT / "home" / "corpus.txt").read_text("utf-8")[:4000]
    ids = np.array([tok.encode(text)[:49] for _ in range(2)])
    before = m.params["head"].copy()

    def loss() -> float:
        lg, _ = m.forward(ids[:, :-1])
        z = lg - lg.max(-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        b, t, v = lg.shape
        y = ids[:, 1:]
        return float(-lp.reshape(-1, v)[np.arange(b * t), y.reshape(-1)].mean())

    l0 = loss()
    for _ in range(20):
        m.step(ids, lr=1e-3, weight_decay=0.01)
    l1 = loss()
    moved = float(np.abs(m.params["head"] - before).max())
    row("训练真的更新了权重", l1 < l0 and moved > 0,
        f"loss {l0:.4f}→{l1:.4f}  max|ΔW|={moved:.2e}")


def check_sampling() -> None:
    m = TinyGPT.load(ROOT / "home" / "models" / "tiny.bin")
    tok = CharTokenizer.from_chars(m.chars)
    ids = tok.encode("話說")
    a = tok.decode(m.generate(ids, 24))
    b = tok.decode(m.generate(ids, 24))
    row("生成是采样不是查表", a != b and len(a) == 24, f"{a[:12]}… ≠ {b[:12]}…")


def check_guard() -> None:
    manifest = json.loads((ROOT / "home" / "state" / "manifest.json").read_text("utf-8"))
    bad = []
    for name, digest in manifest["files"].items():
        p = ROOT / "core" / name
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != digest:
            bad.append(name)
    row("守卫哈希可独立复算", not bad, "全部一致" if not bad else "不一致：" + "、".join(bad))


def check_memory() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="wolf-real-"))
    h = Home(tmp)
    h._ensure()
    (h.seed_dir / "index.json").write_text(json.dumps({"facts": [], "born": 0}), "utf-8")
    h.boot()
    h.leave()
    text = "家是共用主机，只承担启动的电力和算力，学到的先留在门外。"
    verdict = MemoryStore(h).remember(text, source="https://example.com", kind="knowledge")
    files = list(h.threshold_dir.glob("*.json"))
    on_disk = files and text in files[0].read_text("utf-8")
    corpus_before = h.corpus_path.stat().st_size if h.corpus_path.exists() else 0
    h.enter()
    h.leave()
    corpus_after = h.corpus_path.stat().st_size if h.corpus_path.exists() else 0
    row("记忆真的落盘并进语料", bool(on_disk) and corpus_after > corpus_before,
        f"tier={verdict['tier']} 门槛文件={len(files)} 语料 {corpus_before}→{corpus_after}B")


def check_no_stub() -> None:
    import main as entry
    planner, info = entry.build_planner(ROOT / "home" / "models" / "tiny.bin")
    row("不是回显壳", planner is not None and info.startswith("on:"), info)


def main() -> int:
    for fn in (check_checkpoint, check_forward, check_training, check_sampling,
               check_guard, check_memory, check_no_stub):
        try:
            fn()
        except Exception as exc:                     # noqa: BLE001
            row(fn.__name__, False, f"{type(exc).__name__}: {exc}")
    w = max(len(r[0]) for r in ROWS) + 2
    print(f"{'检查项':<{w}} {'结论':<5} 证据")
    print("-" * (w + 72))
    for name, verdict, evidence in ROWS:
        print(f"{name:<{w}} {verdict:<5} {evidence}")
    bad = sum(1 for _, v, _ in ROWS if v == "FAIL")
    print(f"\n{len(ROWS)} 项：{len(ROWS) - bad} 通过，{bad} 失败")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
