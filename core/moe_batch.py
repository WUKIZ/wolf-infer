import os
"""批处理：N 个请求共用一次专家搬运（MoE 逐 token 独立，可拼接）。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from moe_kv import KVEngine
from moe_pcie import D_DIR
from qwen_backend import rope_at
from qwen_moe import norm32


def project(t, x, name):
    """q/k/v 投影：Qwen1.5 带 bias，Qwen3 不带。"""
    y = x @ t['self_attn.%s.weight' % name].T
    bias = t.get('self_attn.%s.bias' % name)
    return y if bias is None else y + bias


class BatchEngine(KVEngine):
    """批处理引擎：只批 MoE 专家搬运，注意力按请求串行（_attn_c）。

    已知边界：真正的批量 attention 未实现（原 _attn_b 有 KV 拼接错误，已删除）。
    因此 batch 吞吐的收益来自"专家一次搬给多请求"，与注意力批量无关。
    """
    def _attn_c(self, L, x, cache):
        t, cfg = self._dense(L), self.cfg
        H = cfg['num_attention_heads']
        dh = cfg.get('head_dim') or cfg['hidden_size'] // H
        B, T, d = x.shape
        h = norm32(x, t['input_layernorm.weight'], self.eps)

        if not hasattr(t, '_qkv_w'):
            t._qkv_w = torch.cat([t['self_attn.q_proj.weight'],
                                  t['self_attn.k_proj.weight'],
                                  t['self_attn.v_proj.weight']], dim=0)
            t._qkv_b = torch.cat([t['self_attn.q_proj.bias'],
                                  t['self_attn.k_proj.bias'],
                                  t['self_attn.v_proj.bias']], dim=0)
        qkv = h @ t._qkv_w.T + t._qkv_b
        q, k, v = (z.view(B, T, H, dh).transpose(1, 2)
                   for z in qkv.split(self.cfg['hidden_size'], dim=-1))
        c = cache.get(L)
        off = 0 if c is None else c[0].shape[2]
        q, k = rope_at(q, cfg['rope_theta'], off), rope_at(k, cfg['rope_theta'], off)
        if c is not None:
            k, v = torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)
        cache[L] = (k, v)
        att = q @ k.transpose(-1, -2) / (dh ** 0.5)
        if T > 1:
            att = att + torch.full((T, T), float('-inf'), device=q.device,
                                   dtype=q.dtype).triu(1)
        w = torch.softmax(att, dim=-1)
        return x + (w @ v).transpose(1, 2).reshape(B, T, d) @ t['self_attn.o_proj.weight'].T

    @torch.no_grad()
    def prefill(self, ids, cache, last_only: bool = False):
        """last_only=True 只对最后一个位置算 head：调用方通常只需要 last logits，
        而 head 是 (T,2048)x(2048,151936)，长序列下它比注意力还贵。"""
        x = self._emb[torch.tensor([ids], device='cuda')]
        x = self._pp_in(x)
        for L in range(*self._pp_bounds()):
            x = self._moe(L, self._attn_c(L, x, cache))
        if not self._pp_out(x):
            return None
        if last_only:
            x = x[:, -1:]
        return norm32(x, self._norm, self.eps) @ self._head.T

    def _rope_uv(self, n, dh, theta, device, dtype):
        """预计算 rope 查表：n 个位置 × dh 维，按 (n, dh, theta) 缓存。"""
        cache = getattr(self, '_rope_cache', None)
        if cache and cache[0] >= n and cache[1] == dh and cache[2] == theta:
            return cache[3], cache[4]
        inv = 1.0 / (theta ** (torch.arange(0, dh, 2, device=device).float() / dh))
        ang = torch.outer(torch.arange(n, device=device).float(), inv)
        ang = torch.cat((ang, ang), dim=-1)
        cos, sin = ang.cos().to(dtype), ang.sin().to(dtype)
        self._rope_cache = (n, dh, theta, cos, sin)
        return cos, sin

    @staticmethod
    def _rot(x, cos, sin, dh):
        half = dh // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    @torch.no_grad()
    def prefill_batch(self, items, last_only: bool = False):
        """等长请求批量 prefill：把每请求 24 层的固定开销摊到 B 条上。

        items = [(ids, cache), ...]，要求所有 ids 长度相同；长度不齐由调用方分桶。
        """
        lens = {len(ids) for ids, _ in items}
        if len(lens) != 1:
            raise ValueError('同批请求必须等长，实际 %s' % sorted(lens))
        x = self._emb[torch.tensor([ids for ids, _ in items], device='cuda')]
        caches = [c for _, c in items]
        x = self._pp_in(x)
        for L in range(*self._pp_bounds()):
            x = self._moe(L, self._attn_pre_b(L, x, caches))
        if not self._pp_out(x):
            return None
        if last_only:
            x = x[:, -1:]
        return norm32(x, self._norm, self.eps) @ self._head.T

    def _attn_pre_b(self, L, x, caches):
        """批量 prefill 一层注意力：与 _attn_c 同算法，K/V 按请求切回各自 cache。"""
        t, cfg = self._dense(L), self.cfg
        H = cfg['num_attention_heads']
        dh = cfg.get('head_dim') or cfg['hidden_size'] // H
        B, T, d = x.shape
        h = norm32(x, t['input_layernorm.weight'], self.eps)
        if not hasattr(t, '_qkv_w'):
            t._qkv_w = torch.cat([t['self_attn.q_proj.weight'],
                                  t['self_attn.k_proj.weight'],
                                  t['self_attn.v_proj.weight']], dim=0)
            t._qkv_b = torch.cat([t['self_attn.q_proj.bias'],
                                  t['self_attn.k_proj.bias'],
                                  t['self_attn.v_proj.bias']], dim=0)
        qkv = h @ t._qkv_w.T + t._qkv_b
        q, k, v = (z.view(B, T, H, dh).transpose(1, 2)
                   for z in qkv.split(cfg['hidden_size'], dim=-1))
        off = 0 if caches[0].get(L) is None else caches[0][L][0].shape[2]
        q, k = rope_at(q, cfg['rope_theta'], off), rope_at(k, cfg['rope_theta'], off)
        for i, c in enumerate(caches):
            ki, vi = k[i:i + 1], v[i:i + 1]
            old = c.get(L)
            if old is not None:
                ki = torch.cat([old[0], ki], dim=2)
                vi = torch.cat([old[1], vi], dim=2)
            c[L] = (ki, vi)
        att = q @ k.transpose(-1, -2) / (dh ** 0.5)
        if T > 1:
            att = att + torch.full((T, T), float('-inf'), device=q.device,
                                   dtype=att.dtype).triu(1)
        w = torch.softmax(att, dim=-1)
        return x + (w @ v).transpose(1, 2).reshape(B, T, d) @ t['self_attn.o_proj.weight'].T

    def _attn_b(self, L, x, caches):
        """批量 decode 的一层注意力。

        KV 放在本引擎的定长缓冲里，容量从 8 起按需翻倍；长度用 Python 整数
        跟踪（state[3]），不读张量 -> 每层没有 .item() 同步，图捕获的前提。
        首层从 caches 里把 prefill 留下的 KV 搬进缓冲，此后 caches 不再逐层
        写回（旧实现每步 768 次 int(pos[i]) + 1536 次 cat，是当时的主要 CPU 开销）。
        """
        t = self._dense(L)
        H = self.cfg['num_attention_heads']
        dh = self.cfg.get('head_dim') or self.cfg['hidden_size'] // H
        theta = self.cfg['rope_theta']
        B = x.shape[0]

        kvb = getattr(self, '_kvb', None)
        if kvb is None:
            kvb = self._kvb = {}
        state = kvb.get(L)
        if state is not None and state[0].shape[0] != B:
            state = None
        if state is None:
            lens = [0 if caches[i].get(L) is None else caches[i][L][0].shape[2]
                    for i in range(B)]
            n = max(lens) if lens else 0
            cap = max(16, (n + 1 + 15) // 16 * 16)
            Kp = torch.zeros(B, H, cap, dh, dtype=x.dtype, device=x.device)
            Vp = torch.zeros(B, H, cap, dh, dtype=x.dtype, device=x.device)
            for i, ln in enumerate(lens):
                if ln:
                    Kp[i, :, :ln] = caches[i][L][0][0]
                    Vp[i, :, :ln] = caches[i][L][1][0]
            state = kvb[L] = [Kp, Vp, torch.tensor(lens, device=x.device), n]

        Kp, Vp, pos, n = state
        if n + 1 > Kp.shape[2]:
            cap = max(Kp.shape[2], (n + 1 + 15) // 16 * 16)
            Kp = torch.cat([Kp, torch.zeros(B, H, cap - Kp.shape[2], dh,
                                            dtype=Kp.dtype, device=Kp.device)], dim=2)
            Vp = torch.cat([Vp, torch.zeros(B, H, cap - Vp.shape[2], dh,
                                            dtype=Vp.dtype, device=Vp.device)], dim=2)
            state[0], state[1] = Kp, Vp

        cap = Kp.shape[2]
        h = norm32(x, t['input_layernorm.weight'], self.eps)
        if not hasattr(t, '_qkv_w'):
            # 三个投影拼成一次 GEMM：少两次 kernel 启动与两次权重读，每层多占 25 MB
            t._qkv_w = torch.cat([t['self_attn.q_proj.weight'],
                                  t['self_attn.k_proj.weight'],
                                  t['self_attn.v_proj.weight']], dim=0)
            t._qkv_b = torch.cat([t['self_attn.q_proj.bias'],
                                  t['self_attn.k_proj.bias'],
                                  t['self_attn.v_proj.bias']], dim=0)
        qkv = h @ t._qkv_w.T + t._qkv_b
        q, k, v = (z.view(B, 1, H, dh).transpose(1, 2)
                   for z in qkv.split(self.cfg['hidden_size'], dim=-1))

        cos, sin = self._rope_uv(cap, dh, theta, q.device, q.dtype)
        cq = cos[pos].view(B, 1, 1, dh)
        sk = sin[pos].view(B, 1, 1, dh)
        q = self._rot(q, cq, sk, dh)
        k = self._rot(k, cq, sk, dh)

        rows = torch.arange(B, device=x.device)
        Kp[rows, :, pos] = k[:, :, 0]
        Vp[rows, :, pos] = v[:, :, 0]
        pos.add_(1)
        state[3] = n + 1
        self._step_len = n + 1
        keep = torch.arange(cap, device=x.device)[None, :] < pos[:, None]

        att = q @ Kp.transpose(-1, -2) / (dh ** 0.5)
        att = att.masked_fill(~keep.view(B, 1, 1, cap), float('-inf'))
        w = torch.softmax(att, dim=-1)
        out = (w @ Vp).transpose(1, 2).reshape(B, 1, -1)
        return x + out @ t['self_attn.o_proj.weight'].T

    # ── 流水并行（PP）钩子：默认 (0, NL) 即原行为 ──────────────
    def _pp_bounds(self):
        return getattr(self, 'pp_range', (0, self.cfg['num_hidden_layers']))

    def _pp_in(self, x):
        rank = getattr(self, 'pp_rank', 0)
        if rank == 0:
            return x
        from core import pp as pp_comm
        buf = pp_comm.recv(tuple(x.shape), torch.float32, rank - 1)   # gloo 只吃 CPU
        return buf.to(device=x.device, dtype=x.dtype)

    def _pp_out(self, x):
        rank, world = getattr(self, 'pp_rank', 0), getattr(self, 'pp_world', 1)
        if rank < world - 1:
            from core import pp as pp_comm
            pp_comm.send(x.float().cpu(), rank + 1)
            return False
        return True

    @torch.no_grad()
    def decode(self, toks, caches):
        """批量解码一步。

        整批 (B,1,d) 张量在 24 层里直接流转；旧实现每层 torch.cat + split(1)
        会把一批拆成 B 个张量再拼回来，24 层合计约 1600 次小算子。
        """
        x = self._emb[torch.tensor(toks, device='cuda')].view(len(toks), 1, -1)
        x = self._pp_in(x)
        for L in range(*self._pp_bounds()):
            if self.cfg.get('profile'):
                _t = time.perf_counter()
                x = self._attn_b(L, x, caches)
                self.dbg_attn = getattr(self, 'dbg_attn', 0.0) + time.perf_counter() - _t
                _t = time.perf_counter()
                x = self._moe(L, x)
                self.dbg_moe = getattr(self, 'dbg_moe', 0.0) + time.perf_counter() - _t
            else:
                x = self._moe(L, self._attn_b(L, x, caches))
            if os.environ.get('WOLF_DIAG') and L % 5 == 0:
                print('  [diag] L%-3d 显存 %.2f GB  pool %d 块 %.2f GB'
                      % (L, torch.cuda.memory_allocated() / 2 ** 30,
                         len(self._pool), sum(t.numel() * t.element_size() for t in self._pool) / 2 ** 30),
                      flush=True)
        if not self._pp_out(x):
            return None
        n = getattr(self, '_step_len', 0)
        for cache in caches:
            cache['len'] = n
        return norm32(x, self._norm, self.eps) @ self._head.T

if __name__ == '__main__':
    import time
    from tokenizers import Tokenizer
    from moe_pcie import D_DIR

    eng = BatchEngine().warm()
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    ids = tk.encode('中国的首都是').ids
    for B in (1, 4, 8):
        caches = [{} for _ in range(B)]
        toks = [int(eng.prefill(ids, c)[0, -1].argmax()) for c in caches]
        seq = []
        start = time.perf_counter()
        for _ in range(8):
            lg = eng.decode(toks, caches)
            toks = [int(x.argmax()) for x in lg]
            seq.append(toks[0])
        dt = time.perf_counter() - start
        print('batch %d: %.2f tok/s total (%.1f per stream)'
              % (B, B * 8 / dt, 8 / dt))
        print('  seq:', tk.decode(seq))


if __name__ == '__main__':
    import time
    from tokenizers import Tokenizer
    from moe_pcie import D_DIR

    eng = BatchEngine().warm()
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    ids = tk.encode('中国的首都是').ids
    for B in (1, 4, 8):
        caches = [{} for _ in range(B)]
        toks = [int(eng.prefill(ids, c)[0, -1].argmax()) for c in caches]
        seq = []
        start = time.perf_counter()
        for _ in range(8):
            lg = eng.decode(toks, caches)
            toks = [int(x.argmax()) for x in lg]
            seq.append(toks[0])
        dt = time.perf_counter() - start
        print('batch %d: %.2f tok/s total (%.1f per stream)'
              % (B, B * 8 / dt, 8 / dt))
        print('  seq:', tk.decode(seq))
