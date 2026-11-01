"""CUDA Graph 捕获的定长批量解码。

约束：所有请求位置相同（同一段 prompt）、形状固定。
自检：python3 -u core/moe_graph.py --check
  1) 本文件的 _rope 与 qwen_backend.rope_at 对拍
  2) 图内一步的 logits 与 eager decode 同一步对拍
"""
import sys
import time

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/core')
import torch

from core.moe_batch import BatchEngine, project
from qwen_backend import rope_at
from qwen_moe import norm32

MODEL = '/data/models/Qwen1.5-MoE-A2.7B'
PROMPT = '中国的首都是'


class GraphDecoder(BatchEngine):
    def __init__(self, batch, max_len, prompt):
        super().__init__()
        self.pin_per_layer = 60
        self.warm()
        self.batch, self.max_len, self.prompt = batch, max_len, prompt
        self.layers = self.cfg['num_hidden_layers']
        self.heads = self.cfg['num_attention_heads']
        self.dh = self.cfg.get('head_dim') or self.cfg['hidden_size'] // self.heads
        self.theta = self.cfg['rope_theta']
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dh, 2, device='cuda').float() / self.dh))
        shape = (batch, self.heads, max_len, self.dh)
        self.k_buf = [torch.zeros(shape, dtype=torch.bfloat16, device='cuda') for _ in range(self.layers)]
        self.v_buf = [torch.zeros(shape, dtype=torch.bfloat16, device='cuda') for _ in range(self.layers)]
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device='cuda')
        self._g_pos = torch.zeros((), dtype=torch.long, device='cuda')
        self._range = torch.arange(max_len, device='cuda')
        self.graph = None
        self.logits_static = None

    def _rope(self, x, cos, sin):
        """rotate_half 约定：x*cos + cat(-x2, x1)*sin。"""
        half = self.dh // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def _cos_sin(self, pos):
        ang = pos.float() * self.inv_freq
        both = torch.cat((ang, ang))
        return both.cos()[None, None, None, :].to(torch.bfloat16), \
               both.sin()[None, None, None, :].to(torch.bfloat16)

    def _forward(self):
        """图内一步：只读 _g_pos，写 k/v 静态缓冲，返回 logits。"""
        cos, sin = self._cos_sin(self._g_pos)
        keep = self._range <= self._g_pos
        x = self._emb[self.tok]
        for L in range(self.layers):
            t = self._dense(L)
            h = norm32(x, t['input_layernorm.weight'], self.eps)
            q = project(t, h, 'q_proj').view(-1, 1, self.heads, self.dh).transpose(1, 2)
            k = project(t, h, 'k_proj').view(-1, 1, self.heads, self.dh).transpose(1, 2)
            v = project(t, h, 'v_proj').view(-1, 1, self.heads, self.dh).transpose(1, 2)
            self.k_buf[L][:, :, self._g_pos] = self._rope(k, cos, sin)[:, :, 0]
            self.v_buf[L][:, :, self._g_pos] = v[:, :, 0]
            att = (self._rope(q, cos, sin) @ self.k_buf[L].transpose(-1, -2)) / (self.dh ** 0.5)
            att = att.masked_fill(~keep, float('-inf'))
            out = torch.softmax(att, dim=-1) @ self.v_buf[L]
            x = x + out.transpose(1, 2).reshape(x.shape) @ t['self_attn.o_proj.weight'].T
            x = self._moe(L, x)
        return norm32(x, self._norm, self.eps) @ self._head.T

    def build(self):
        ids = self.prompt
        cache = {}
        first = int(self.prefill(ids, cache)[0, -1].argmax())
        for L in range(self.layers):
            k, v = cache[L]
            self.k_buf[L][:, :, :k.shape[2]] = k
            self.v_buf[L][:, :, :v.shape[2]] = v
        self._g_pos.fill_(len(ids))
        self.tok.fill_(first)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits_static = self._forward()
        return first

    def step(self, tokens):
        self.tok.copy_(torch.tensor(tokens, device='cuda').view(-1, 1))
        self.graph.replay()
        self._g_pos += 1
        return self.logits_static

    def check(self, first, steps=3):
        """三段对拍：rope 约定 → _forward(eager) vs decode → 图 vs eager。"""
        probe = torch.randn(1, self.heads, 1, self.dh, device='cuda', dtype=torch.bfloat16)
        cos, sin = self._cos_sin(torch.tensor(5, device='cuda'))
        print('check rope    maxdiff %.6f'
              % (self._rope(probe, cos, sin) - rope_at(probe, self.theta, 5)).abs().max().item())

        base = int(self._g_pos)          # 建图后 _g_pos 仍停在 len(prompt)
        self.tok.fill_(first)
        eager_fwd = self._forward().clone()

        caches = [{} for _ in range(self.batch)]
        for c in caches:
            self.prefill(self.prompt, c)
        eager_dec = self.decode([first] * self.batch, caches).reshape(eager_fwd.shape)
        print('check forward maxdiff %.6f  （_forward 与 decode 是否同一套数学）'
              % (eager_fwd - eager_dec).abs().max().item())

        self._g_pos.fill_(base)
        self.tok.fill_(first)
        tok = first
        print('step  pos  graph  eager   maxdiff')
        for step in range(steps):
            pos = int(self._g_pos)
            g = self.step([tok] * self.batch).clone()
            e = self.decode([tok] * self.batch, caches).reshape(g.shape)
            ga, ea = int(g[0].reshape(-1).argmax()), int(e[0].reshape(-1).argmax())
            print('%4d %4d %6d %6d %9.6f' % (step, pos, ga, ea, (g - e).abs().max().item()))
            tok = ea


def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(MODEL + '/tokenizer.json')
    ids = tk.encode(PROMPT).ids
    eng = GraphDecoder(8, 128, ids)
    first = eng.build()
    if '--check' in sys.argv:
        eng.check(first)
    toks = [first] * 8
    t0 = time.perf_counter()
    for _ in range(8):
        out = eng.step(toks)
        toks = [int(r.argmax()) for r in out]
    dt = time.perf_counter() - t0
    print('graph B=8 %.2f tok/s  %.1f ms/步' % (64 / dt, dt / 8 * 1e3))
    print('文本', repr(tk.decode(toks)))


if __name__ == '__main__':
    main()
