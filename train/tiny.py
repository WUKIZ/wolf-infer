"""可训练的 GPT（torch 版），配合 train/loop.py 在单卡上训练。"""
import math

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        a, _ = self.att(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    """词表 151936 / d=1024 / 16 层，约 357M 参数（词嵌入与输出头共享）。"""

    def __init__(self, vocab=151936, d=1024, layers=16, heads=16, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx):
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.emb(idx) + self.pos(pos)
        mask = torch.triu(torch.full((t, t), float('-inf'), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        # 共享头会把 logits 放大 sqrt(d) 倍，必须除回去，否则初始 loss 爆炸
        return self.head(self.ln_f(x)) / math.sqrt(self.d)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    import os
    import time

    from train.loader import Loader
    from train.loop import Trainer

    corpus = os.environ.get('WOLF_CORPUS', '<data>/mix.txt')
    tok = os.environ.get('WOLF_TOKENIZER', '<model>/tokenizer.json')
    steps = int(os.environ.get('WOLF_STEPS', 10))

    model = MiniGPT().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    loader = Loader(corpus, tok, seq_len=256, batch=2)
    trainer = Trainer(model, opt, loader, accum=8, log_every=1)

    t0 = time.perf_counter()
    out = trainer.run(steps)
    dt = time.perf_counter() - t0
    print('参数量 %.1f M | %d 步 %.1f s | %.0f token/s | %d token'
          % (model.n_params() / 1e6, steps, dt, out['tokens'] / dt, out['tokens']))


if __name__ == '__main__':
    main()
