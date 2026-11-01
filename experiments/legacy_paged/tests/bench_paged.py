"""paged 路径基准：prefill / decode tok/s / evictions-per-token / io_ratio。"""
import json, os, sys, time

sys.path.insert(0, '/data/wolf/core')
import torch
from transformers import AutoTokenizer
from qwen_moe import attention
from qwen_moe_shard import load_dense, paged_moe
from qwen_backend import rms
from pager import Pager
from shard import Shard
from core.tmpdir import TMP

D = '/data/models/Qwen1.5-MoE-A2.7B'
S = TMP + '/moe_shards'
N_DEC = 8
PROMPT = '中国的首都是'

cfg = json.load(open(os.path.join(D, 'config.json')))
head = torch.load(os.path.join(S, 'head.pt'), map_location='cpu')
d, h = cfg['hidden_size'], cfg['moe_intermediate_size']
shards = [Shard(L * 60 + e, os.path.join(S, 'L%02de%02d.bin' % (L, e)), 0,
                3 * d * h * 2, L, L + 1, expert=e)
          for L in range(cfg['num_hidden_layers']) for e in range(cfg['num_experts'])]
by = {(s.layer_from, s.expert): s for s in shards}
ts = {L: load_dense(os.path.join(S, 'L%02d.dense.pt' % L))
      for L in range(cfg['num_hidden_layers'])}
emb = head['model.embed_tokens.weight'].to('cuda', torch.bfloat16)
nrm = head['model.norm.weight'].to('cuda', torch.bfloat16)
lm = head['lm_head.weight'].to('cuda', torch.bfloat16)
pager = Pager(2048 * 1024 * 1024)

io_ns = [0]
_orig = Pager.view
def timed_view(self, shard):
    t0 = time.perf_counter()
    v = _orig(self, shard)
    io_ns[0] += time.perf_counter() - t0
    return v
Pager.view = timed_view


def run(ids):
    x = emb[torch.tensor([ids], device='cuda')]
    for L in range(cfg['num_hidden_layers']):
        x = attention(ts[L], x, cfg)
        x = paged_moe(L, ts[L], x, cfg, pager, by)
    x = rms(x, nrm, cfg['rms_norm_eps'])
    return x @ lm.T


if __name__ == '__main__':
    tk = AutoTokenizer.from_pretrained(D)
    seq = tk.encode(PROMPT)
    run(seq)
    ev0 = pager.stats.evictions
    t0 = time.perf_counter()
    run(seq)
    prefill = time.perf_counter() - t0
    io0 = io_ns[0]
    t0 = time.perf_counter()
    for _ in range(N_DEC):
        seq.append(int(run(seq)[0, -1].argmax()))
    dec = time.perf_counter() - t0
    print('paged prefill %.3fs  decode %.3f tok/s (%.3fs/token)'
          % (prefill, N_DEC / dec, dec / N_DEC))
    print('evictions/token %.1f  io_ratio %.3f'
          % ((pager.stats.evictions - ev0) / (N_DEC + 1),
             (io_ns[0] - io0) / (dec + prefill)))
    print('out:', tk.decode(seq))
    pager.close()
