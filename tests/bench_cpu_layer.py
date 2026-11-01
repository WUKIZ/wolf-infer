import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import torch
from safetensors import safe_open
from qwen_moe import attention, moe_block
D = '/data/models/Qwen1.5-MoE-A2.7B'
cfg = json.load(open(os.path.join(D, 'config.json')))
idx = json.load(open(os.path.join(D, 'model.safetensors.index.json')))['weight_map']
keep = [k for k in idx if k.startswith('model.layers.0.') or k == 'model.embed_tokens.weight']
fs, w = {}, {}
for k in keep:
    fn = idx[k]
    if fn not in fs:
        fs[fn] = safe_open(os.path.join(D, fn), framework='pt')
    w[k] = fs[fn].get_tensor(k).to(torch.float32)
x = w['model.embed_tokens.weight'][torch.tensor([[40, 100, 200, 300, 500]])]
t = {k[len('model.layers.0.'):]: v for k, v in w.items() if k.startswith('model.layers.0.')}
for _ in range(2):
    y = moe_block(t, attention(t, x, cfg), cfg)
t0 = time.perf_counter()
for _ in range(3):
    y = moe_block(t, attention(t, x, cfg), cfg)
dt = (time.perf_counter() - t0) / 3
print('layer0 cpu %.1f ms -> 24 layers %.2f s/token (threads %d)'
      % (dt * 1000, dt * 24, torch.get_num_threads()))
