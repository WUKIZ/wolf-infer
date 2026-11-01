#!/usr/bin/env python3
"""torch 版训练：BPE + TinyGPT，可在 CUDA / DCU(ROCm) 上跑。"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.bpe import BPETokenizer  # noqa: E402


class Block(nn.Module):
    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.h = h
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape
        hd = D // self.h
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.h, hd).transpose(1, 2)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = att.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)))))
        return x


class TinyTorch(nn.Module):
    def __init__(self, vocab, block, layers, dim, heads, dropout=0.0):
        super().__init__()
        self.block, self.vocab = block, vocab
        self.wte = nn.Embedding(vocab, dim)
        self.wpe = nn.Embedding(block, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        mask = torch.ones(T, T, device=idx.device, dtype=torch.bool).tril()
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chars", type=int, default=80_000_000)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tok-chars", type=int, default=12_000_000)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--char-level", action="store_true", help="高频字符表，跳过 BPE")
    ap.add_argument("--amp", action="store_true", help="bf16 自动混合精度")
    ap.add_argument("--resume", default="", help="从 checkpoint 续训（只恢复权重）")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok-out", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch] device={dev} {torch.__version__}", flush=True)

    t0 = time.time()
    text = Path(a.corpus).read_text(encoding="utf-8", errors="replace")[:a.chars]
    print(f"[torch] 语料 {len(text)/1e6:.1f}M 字符", flush=True)
    if a.tokenizer and Path(a.tokenizer).exists():
        payload = json.load(open(a.tokenizer, encoding="utf-8"))
        if payload.get("kind") == "chars":
            from collections import Counter
            chars = payload["chars"]
            index = {c: i + 1 for i, c in enumerate(chars)}
            class _CharTok:
                def __init__(self): self.vocab = len(chars) + 1
                def encode(self, t): return [index.get(c, 0) for c in t]
                def __repr__(self): return f"CharTok(vocab={self.vocab})"
            tok = _CharTok()
        else:
            tok = BPETokenizer.from_payload(payload)
    elif a.char_level:
        from collections import Counter
        cnt = Counter(text[:a.tok_chars])
        chars = [c for c, _ in cnt.most_common(a.vocab - 1)]
        index = {c: i + 1 for i, c in enumerate(chars)}
        class _CharTok:
            def __init__(self): self.vocab = len(chars) + 1
            def encode(self, t): return [index.get(c, 0) for c in t]
            def __repr__(self): return f"CharTok(vocab={self.vocab})"
        tok = _CharTok()
        json.dump({"kind": "chars", "chars": chars}, open(a.tok_out or (a.out + ".tok.json"), "w", encoding="utf-8"), ensure_ascii=False)
    else:
        tok = BPETokenizer.train(text[:a.tok_chars], vocab_size=a.vocab)
    print(f"[torch] 分词器 {tok} ({time.time()-t0:.0f}s)", flush=True)
    tok_out = a.tok_out or (a.out + ".tok.json")
    if not a.char_level:
        json.dump(tok.to_payload(), open(tok_out, "w", encoding="utf-8"), ensure_ascii=False)

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = max(a.block * a.batch * 4, len(ids) // 100)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"[torch] token={len(ids)/1e6:.1f}M train={len(train_ids)/1e6:.1f}M val={len(val_ids)/1e6:.2f}M", flush=True)

    model = TinyTorch(a.vocab, a.block, a.layers, a.dim, a.heads).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[torch] 参数 {nparam/1e6:.2f}M", flush=True)
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["state"])
        print(f"[torch] 续训：{a.resume} (上次 val {ck['cfg'].get('n_param','?')})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    best, bad = 9.9, 0
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_ids, a.block, a.batch, dev)
        if a.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[torch] step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_ids, a.block, a.batch, dev)
                _, vl = model(xv, yv)
            model.train()
            print(f"[torch] EVAL step {step} val {vl.item():.4f}", flush=True)
            if vl.item() < best:
                best, bad = vl.item(), 0
                torch.save({"cfg": vars(a) | {"n_param": nparam},
                            "state": model.state_dict()}, a.out)
                print(f"[torch] saved {a.out} val {best:.4f}", flush=True)
            else:
                bad += 1
        if time.time() - t0 > a.max_hours * 3600:
            print("[torch] 到达时限，停止", flush=True)
            break
    print(f"[torch] done best_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
