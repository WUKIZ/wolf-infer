"""一层专家：CPU 存 bf16、按需搬 GPU 算，测每层毫秒。"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import numpy as np, torch
from qwen_moe import attention, mlp, norm32
from qwen_moe_shard import load_dense
from core.tmpdir import TMP

S = TMP + '/moe_shards'
cfg = json.load(open('/data/models/Qwen1.5-MoE-A2.7B/config.json'))
D, H = cfg['hidden_size'], cfg['moe_intermediate_size']
t = load_dense(os.path.join(S, 'L00.dense.pt'))
x = torch.randn(1, 5, D, device='cuda', dtype=torch.bfloat16)


def exp_cpu(L, e):
    b = torch.from_numpy(np.memmap(os.path.join(S, 'L%02de%02d.bin' % (L, e)),
                                   dtype=np.uint16, mode='r')).view(torch.bfloat16)
    n = D * H
    return b[:n].view(H, D), b[n:2 * n].view(H, D), b[2 * n:].view(D, H)


def layer_pcie(L, x):
    h = norm32(x, t['post_attention_layernorm.weight'], cfg['rms_norm_eps']).reshape(-1, D)
    probs = torch.softmax(h @ t['mlp.gate.weight'].T, dim=-1)
    vals, idx = torch.topk(probs, cfg['num_experts_per_tok'], dim=-1)
    out = torch.zeros_like(h)
    for e in sorted({int(i) for i in idx.flatten()}):
        rows, slots = (idx == e).nonzero(as_tuple=True)
        wg, wu, wd = exp_cpu(L, e)
        eo = mlp(h[rows], wg.to('cuda'), wu.to('cuda'), wd.to('cuda'))
        out.index_add_(0, rows, eo * vals[rows, slots][:, None])
    sh = mlp(h, t['mlp.shared_expert.gate_proj.weight'].to('cuda'),
             t['mlp.shared_expert.up_proj.weight'].to('cuda'),
             t['mlp.shared_expert.down_proj.weight'].to('cuda'))
    return out + sh


for _ in range(2):
    y = layer_pcie(0, x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    y = layer_pcie(0, x)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 5 * 1000
print('layer pcie %.1f ms -> 24 layers %.2f s/token (%.1f tok/s)' % (ms, ms * 24 / 1000, 1000 / (ms * 24)))
