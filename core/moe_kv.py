"""在 pcie 引擎上加 KV cache：decode 只算新 token。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from moe_pcie import PCIeEngine, D_DIR
from qwen_backend import rope_at
from qwen_moe import norm32


class KVEngine(PCIeEngine):
    def _attn(self, L, x):
        t, cfg = self._dense(L), self.cfg
        H = cfg['num_attention_heads']
        dh = cfg.get('head_dim') or cfg['hidden_size'] // H
        B, T, d = x.shape
        h = norm32(x, t['input_layernorm.weight'], self.eps)

        def lin(name):
            y = h @ t['self_attn.%s.weight' % name].T
            b = t.get('self_attn.%s.bias' % name)
            return y if b is None else y + b

        q = lin('q_proj').view(B, T, H, dh).transpose(1, 2)
        k = lin('k_proj').view(B, T, H, dh).transpose(1, 2)
        v = lin('v_proj').view(B, T, H, dh).transpose(1, 2)
        c = self._kv.get(L)
        off = 0 if c is None else c[0].shape[2]
        q, k = rope_at(q, cfg['rope_theta'], off), rope_at(k, cfg['rope_theta'], off)
        if c is not None:
            k, v = torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)
        self._kv[L] = (k, v)
        att = q @ k.transpose(-1, -2) / (dh ** 0.5)
        if T > 1:
            att = att + torch.full((T, T), float('-inf'), device=q.device,
                                   dtype=q.dtype).triu(1)
        w = torch.softmax(att, dim=-1)
        return x + (w @ v).transpose(1, 2).reshape(B, T, d) @ t['self_attn.o_proj.weight'].T

    def logits(self, ids, cache=False):
        if not cache or not getattr(self, '_kv', None):
            self._kv = {}
        step = [ids[-1]] if (cache and self._kv) else ids
        if not hasattr(self, '_emb'):
            self._emb = self._get('model.embed_tokens.weight').to('cuda')
            self._norm = self._get('model.norm.weight').to('cuda')
            self._head = self._get('lm_head.weight').to('cuda')
        x = self._emb[torch.tensor([step], device='cuda')]
        for L in range(self.cfg['num_hidden_layers']):
            x = self._attn(L, x)
            x = self._moe(L, x)
        return norm32(x, self._norm, self.eps) @ self._head.T


if __name__ == '__main__':
    from tokenizers import Tokenizer
    eng = KVEngine().warm()
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    ids = tk.encode('中国的首都是').ids
    eng.logits(ids)
    seq = list(ids)
    t0 = time.perf_counter()
    for _ in range(16):
        seq.append(int(eng.logits(seq, cache=True)[0, -1].argmax()))
    dt = time.perf_counter() - t0
    print('kv %.3f tok/s (%.3f s/token)' % (16 / dt, dt / 16))
    print(tk.decode(seq))
