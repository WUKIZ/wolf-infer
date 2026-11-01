"""训练循环：梯度累积 + 裁剪 + 日志。"""
import math
import time

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, opt, loader, accum=4, clip=1.0, log_every=10):
        self.model, self.opt, self.loader = model, opt, loader
        self.accum, self.clip, self.log_every = accum, clip, log_every
        self.step = 0
        self.tokens = 0

    def _loss(self, x, y):
        logits = self.model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    def run(self, steps, on_step=None):
        self.model.train()
        it = iter(self.loader)
        for _ in range(steps):
            total, t0 = 0.0, time.perf_counter()
            for _ in range(self.accum):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    x, y = next(it)
                x = torch.as_tensor(x, device='cuda').long()
                y = torch.as_tensor(y, device='cuda').long()
                loss = self._loss(x, y) / self.accum
                loss.backward()
                total += float(loss)
                self.tokens += x.numel()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            if self.step % self.log_every == 0:
                dt = (time.perf_counter() - t0) * 1000
                print('[train] step %d  loss %.4f  ppl %.2f  %.0f ms'
                      % (self.step, total, math.exp(min(total, 20)), dt), flush=True)
            if on_step is not None:
                on_step(self)
        return {'step': self.step, 'tokens': self.tokens}


if __name__ == '__main__':
    import tempfile
    import os
    from train.loader import Loader

    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, 'c.txt')
    with open(corpus, 'w', encoding='utf-8') as fh:
        fh.write('中国的首都是北京。' * 400)
    tk = '/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json'
    ld = Loader(corpus, tk, seq_len=16, batch=2)

    net = torch.nn.Sequential(torch.nn.Embedding(151936, 64),
                              torch.nn.Linear(64, 151936)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    tr = Trainer(net, opt, ld, accum=2, log_every=1)
    out = tr.run(3)
    print('完成 %(step)d 步，消耗 %(tokens)d token' % out)
