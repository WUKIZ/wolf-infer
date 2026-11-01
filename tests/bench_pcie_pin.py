import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import torch
from safetensors import safe_open
from qwen_moe import mlp, norm32
D_DIR = '/data/models/Qwen1.5-MoE-A2.7B'
cfg = json.load(open(os.path.join(D_DIR, 'config.json')))
idx = json.load(open(os.path.join(D_DIR, 'model.safetensors.index.json')))['weight_map']
fs, W = {}, {}
for k in [k for k in idx if k.startswith('model.layers.0.')]:
    fn = idx[k]
    if fn not in fs:
        fs[fn] = safe_open(os.path.join(D_DIR, fn), framework='pt')
    W[k] = fs[fn].get_tensor(k).to(torch.bfloat16)
t = {k[len('model.layers.0.'):]: v for k, v in W.items()}
D, H = cfg['hidden_size'], cfg['moe_intermediate_size']; N = D * H
x = torch.randn(1, 5, D, device='cuda', dtype=torch.bfloat16)
pins = [torch.empty(N, dtype=torch.bfloat16, pin_memory=True) for _ in range(3)]
def run(pinmode):
    h = norm32(x, t['post_attention_layernorm.weight'].to('cuda'), cfg['rms_norm_eps']).reshape(-1, D)
    p = torch.softmax(h @ t['mlp.gate.weight'].to('cuda').T, dim=-1)
    v, i = torch.topk(p, cfg['num_experts_per_tok'], dim=-1); out = torch.zeros_like(h)
    for e in sorted({int(k) for k in i.flatten()}):
        src = [t['mlp.experts.%d.%s.weight' % (e, n)] for n in ('gate_proj', 'up_proj', 'down_proj')]
        if pinmode:
            gs = []
            for pb, sw in zip(pins, src):
                pb.view_as(sw).copy_(sw); gs.append(pb.to('cuda', non_blocking=True).view_as(sw))
            w = (gs[0], gs[1], gs[2])
        else:
            w = (src[0].to('cuda'), src[1].to('cuda'), src[2].to('cuda'))
        r, s = (i == e).nonzero(as_tuple=True)
        out.index_add_(0, r, mlp(h[r], *w) * v[r, s][:, None])
    return out
for m in (False, True):
    run(m); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        run(m)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 5 * 1000
    print('%s %.1f ms/layer -> %.1f tok/s' % ('pin' if m else 'page', ms, 1000 / (ms * 24)))
