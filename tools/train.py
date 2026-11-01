#!/usr/bin/env python3
"""正经训练入口：留出验证集，按验证损失存最好的那份。

和 Learner.train 的分工：
  - Learner.train 是"出门学一轮"的快速适应，跑几十步就回；
  - tools/train.py 是可以反复跑的离线训练，有 held-out、有权重衰减、有早停。

训练损失的下降不算进步，验证损失才算。语料只有十几 KB，模型很快就会开始背，
所以这里每次评估都可能把 best 覆盖掉 —— 最后落盘的是 val 最低的那一份。

    ./bin/py tools/train.py --steps 400 --wd 0.05
    ./bin/py tools/train.py --layers 3 --dim 96 --block 48 --steps 150 --wd 0.05
    ./bin/py tools/train.py --eval-only --model home/models/tiny.bin
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import mem  # noqa: E402
from core.bpe import BPETokenizer, load_tokenizer  # noqa: E402
from core.model import CharTokenizer, Config, TinyGPT  # noqa: E402

LOCK = ROOT / "home" / "state" / "train.lock"


def acquire_lock() -> bool:
    """同一时间只许一个训练进程 —— 两个进程抢同一个 tmp 会互相删文件。"""
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text("utf-8").strip())
            os.kill(pid, 0)
            return False
        except (OSError, ValueError):
            pass
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), "utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK.unlink()
    except OSError:
        pass


def eval_loss(model: TinyGPT, batch: np.ndarray) -> float:
    """不更新参数的前向损失 —— 拿它当 early-stop 的依据。"""
    x, y = batch[:, :-1], batch[:, 1:]
    logits, _ = model.forward(x)
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    b, t, v = logits.shape
    return float(-logp.reshape(-1, v)[np.arange(b * t), y.reshape(-1)].mean())


def build_val(ids: list[int], block: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, max(1, len(ids) - block - 1), n)
    return np.array([ids[i:i + block + 1] for i in ix])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "home" / "corpus.txt"))
    p.add_argument("--model", default=str(ROOT / "home" / "models" / "tiny.bin"))
    p.add_argument("--out", default="")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr-final", type=float, default=None,
                   help="余弦退火终点，缺省 lr*0.1")
    p.add_argument("--warmup", type=int, default=0,
                   help="前 N 步线性升温，避免大模型一开始就把权重打飞")
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=40)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="到点自己停（留出写盘余量），0=不设")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--cold", action="store_true", help="不续训，强制从头")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    p.add_argument("--vocab", type=int, default=2048,
                   help="bpe 的目标词表大小（含 256 个字节底座）")
    p.add_argument("--tok-cache", default="",
                   help="合并表缓存：存在就直接读，省掉每次重训 BPE")
    p.add_argument("--report", default=str(ROOT / "home" / "models" / "train_report.json"))
    a = p.parse_args()

    if not a.eval_only and not acquire_lock():
        print("[train] 已有训练进程在跑（home/state/train.lock），退出", file=sys.stderr)
        return 4
    try:
        return _run(a)
    finally:
        if not a.eval_only:
            release_lock()


def _split(ids, block: int, val_frac: float, seed: int):
    """验证集取中段，别取尾巴 —— 语料是拼起来的，尾巴往往和主体不同分布。"""
    n_val = max(block + 2, int(len(ids) * val_frac))
    if len(ids) < n_val + block + 2:
        return None
    val_start = max(0, len(ids) // 2 - n_val // 2)
    val_ids = ids[val_start:val_start + n_val]
    train_ids = ids[:val_start] + ids[val_start + n_val:]
    return train_ids, val_ids, build_val(val_ids, block, 16, seed)


def build_examples(tok, text: str, block: int):
    """把「问：…/答：…」切成样本：只对答句算损失，问句作为上下文。"""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        q = lines[i].strip()
        if q.startswith("问：") and i + 1 < len(lines) and lines[i + 1].strip().startswith("答："):
            a = lines[i + 1].strip()[2:].strip()
            prompt = f"问：{q[2:].strip()}\n答："
            ip, ia = tok.encode(prompt), tok.encode(a + "\n")
            ids = ip + ia
            mask = [0] * max(0, len(ip) - 1) + [1] * len(ia)
            for s in range(0, max(1, len(ids) - 1), block):
                seg, mk = ids[s:s + block + 1], mask[s:s + block]
                if len(seg) >= 2:
                    out.append((seg, mk))
            i += 2
        else:
            i += 1
    if not out:                      # 没有问答对：退回整段滑窗，全部计损失
        ids = tok.encode(text)
        for s in range(0, max(1, len(ids) - 1), block):
            seg = ids[s:s + block + 1]
            if len(seg) >= 2:
                out.append((seg, [1] * (len(seg) - 1)))
    return out


def make_batch(samples, block: int, batch: int, rng):
    picks = rng.integers(0, len(samples), batch)
    arr = np.zeros((batch, block + 1), dtype=np.int64)
    m = np.zeros((batch, block), dtype=np.float64)
    for k, j in enumerate(picks):
        seg, mk = samples[j]
        n = min(len(seg), block + 1)
        arr[k, :n] = seg[:n]
        m[k, :n - 1] = mk[:n - 1]
    return arr, m


def eval_masked(model, samples, block: int, batch: int = 16, seed: int = 0,
                rounds: int = 4) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(rounds):
        arr, m = make_batch(samples, block, batch, rng)
        total += model.batch_loss(arr[:, :-1], arr[:, 1:], m)
    return total / rounds


def _run(a) -> int:

    corpus = Path(a.corpus)
    text = corpus.read_text("utf-8")
    model_path = Path(a.model)
    out = Path(a.out) if a.out else model_path

    if a.eval_only:
        m = TinyGPT.load(model_path)
        tok = load_tokenizer(getattr(m, "tok", None), getattr(m, "chars", None), text=text)
        split = _split(tok.encode(text), a.block, a.val_frac, a.seed)
        if split is None:
            print(f"[eval] 语料太短，凑不出 {a.block} 的窗口", file=sys.stderr)
            return 2
        _, val_ids, val_batch = split
        if m.cfg.block != a.block or m.cfg.vocab != tok.vocab:
            print(f"[eval] 形状/词表对不上：model={m.cfg} tok={tok.vocab}", file=sys.stderr)
            return 2
        print(json.dumps({"model": str(model_path), "val_loss": round(eval_loss(m, val_batch), 4),
                          "params": int(sum(x.size for x in m.params.values())),
                          "vocab": tok.vocab}, ensure_ascii=False))
        return 0

    # 合并表只在训练段上训：val 段参与的话，就是让模型提前见过验证集。
    n_val_chars = max(a.block * 8, int(len(text) * a.val_frac))
    vs = max(0, len(text) // 2 - n_val_chars // 2)
    train_text = text[:vs] + text[vs + n_val_chars:]
    if a.tokenizer == "bpe":
        tok = None
        if a.tok_cache and Path(a.tok_cache).exists():
            try:
                cached = load_tokenizer(json.loads(Path(a.tok_cache).read_text("utf-8")))
                if cached.vocab == a.vocab:
                    tok = cached
                    print(f"[train] 合并表缓存命中：{a.tok_cache}（vocab={tok.vocab}）")
            except (OSError, ValueError, KeyError):
                tok = None
        if tok is None:
            tok = BPETokenizer.train(train_text, a.vocab)
            if a.tok_cache:
                Path(a.tok_cache).write_text(json.dumps(tok.to_payload()), "utf-8")
        tok_payload = tok.to_payload()
    else:
        tok = CharTokenizer(text)
        tok_payload = None

    train_examples = build_examples(tok, train_text, a.block)
    val_examples = build_examples(tok, text[vs:vs + n_val_chars], a.block)
    if not train_examples or not val_examples:
        print("[train] 语料切不出样本", file=sys.stderr)
        return 2

    cfg = Config(vocab=tok.vocab, block=a.block, n_layer=a.layers,
                 n_head=a.heads, n_embd=a.dim)
    model, warm = None, False
    if model_path.exists() and not a.cold:
        try:
            prev = TinyGPT.load(model_path)
            c = prev.cfg
            same_shape = (c.vocab, c.block, c.n_embd, c.n_layer, c.n_head) == (
                tok.vocab, a.block, a.dim, a.layers, a.heads)
            # 形状对不够，合并表也得对：语料一换，同样 2048 的词表会训出另一张表，
            # id 一错位，续训出来的就是噪声。
            same_tok = tok_payload is None or getattr(prev, "tok", None) == tok_payload
            if same_shape and same_tok:
                model, warm = prev, True
            elif same_shape:
                print("[train] 合并表和旧权重不一致，不续训，从头来", file=sys.stderr)
        except Exception as exc:              # noqa: BLE001
            print(f"[train] 旧权重载入失败，从头训：{type(exc).__name__}", file=sys.stderr)
    if model is None:
        model = TinyGPT(cfg, np.random.default_rng(a.seed))

    params = int(sum(x.size for x in model.params.values()))
    cap = float(os.environ.get("WOLF_RSS_MB", "700"))
    est = mem.estimate_training_mb(params, a.batch, a.block, a.dim, a.layers)
    ok, why = mem.check(est, cap)
    print(f"[train] 内存预估 {est:.0f}MB / 上限 {cap:.0f}MB ({'ok' if ok else '拒绝'})")
    if not ok:
        print(f"[train] 拒绝训练：{why}", file=sys.stderr)
        return 3
    rng = np.random.default_rng(a.seed + 1)

    v = eval_masked(model, val_examples, a.block)
    best_v, best_step = v, 0
    print(f"[train] {a.layers}L/{a.dim}d/{a.block}b  tok={a.tokenizer}/{tok.vocab}  "
          f"params={params:,}  warm={warm}  train={len(train_examples)}  val={len(val_examples)}")
    print(f"{'step':>5} {'train':>7} {'val':>7}  {'best':>7}")
    print(f"{0:>5} {'-':>7} {v:>7.4f}  {best_v:>7.4f} *")

    history = [{"step": 0, "val": round(v, 4)}]
    stale = 0
    running: list[float] = []
    t0 = time.perf_counter()
    lr_final = a.lr_final if a.lr_final is not None else a.lr * 0.1

    for step in range(1, a.steps + 1):
        if a.max_hours and time.perf_counter() - t0 > a.max_hours * 3600:
            print(f"[train] 到 {a.max_hours:g} 小时上限，停在 {step - 1} 步；"
                  f"最好的在 {best_step} 步")
            break
        # 余弦退火（带线性 warmup）：前段大步走，后段小步收
        progress = (step - 1) / max(1, a.steps)
        if a.warmup and step <= a.warmup:
            lr_t = a.lr * step / a.warmup
        else:
            lr_t = lr_final + 0.5 * (a.lr - lr_final) * (
                1 + math.cos(math.pi * progress))
        batch_arr, mask = make_batch(train_examples, a.block, a.batch, rng)
        running.append(model.step(batch_arr, lr=lr_t, weight_decay=a.wd, mask=mask))

        if step % a.eval_every and step != a.steps:
            continue

        v = eval_masked(model, val_examples, a.block)
        mean_train = float(np.mean(running[-a.eval_every:]))
        star = ""
        if mem.rss_mb() > cap:
            print(f"[train] 真实 RSS {mem.rss_mb():.0f}MB 超过上限 {cap:.0f}MB，停在 {step} 步",
                  file=sys.stderr)
            stale = 10 ** 6
        if v < best_v - 1e-4:
            best_v, best_step, stale = v, step, 0
            star = " *"
            tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
            model.save(tmp, chars=getattr(tok, "chars", []), tok=tok_payload)
            tmp.replace(out)
        else:
            stale += 1
        history.append({"step": step, "train": round(mean_train, 4), "val": round(v, 4),
                        "best": round(best_v, 4)})
        print(f"{step:>5} {mean_train:>7.4f} {v:>7.4f}  {best_v:>7.4f}{star}")

        if stale >= a.patience:
            print(f"[train] 验证损失 {a.patience} 次没降，停在第 {step} 步；"
                  f"最好的在 {best_step} 步")
            break

    secs = time.perf_counter() - t0
    report = {
        "config": {"layers": a.layers, "dim": a.dim, "block": a.block, "heads": a.heads,
                   "batch": a.batch, "lr": a.lr, "wd": a.wd, "seed": a.seed},
        "params": params, "warm_start": warm,
        "tokens": {"train": sum(len(s) for s, _ in train_examples),
                   "val": sum(len(s) for s, _ in val_examples), "vocab": tok.vocab},
        "best": {"step": best_step, "val_loss": round(best_v, 4)},
        "seconds": round(secs, 1), "steps_run": history[-1]["step"],
        "saved": str(out), "history": history,
    }
    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"[train] best val={best_v:.4f} @ {best_step} 步  {secs:.1f}s  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
