"""KV cache 验收：增量解码与稠密 greedy 必须逐个 token 相同，并给出耗时对比。"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import torch
from qwen_backend import load, logits as dense_logits, generate_cached

d = "/data/models/Qwen3-1.7B"
cfg, w = load(d)
w = {k: v.to("cuda", torch.float32) for k, v in w.items()}
ids = [40, 100, 200, 300]
N = 5

t0 = time.time()
cached = generate_cached(cfg, w, ids, n=N)
t_cached = time.time() - t0

seq = list(ids)
t0 = time.time()
for _ in range(N):
    seq.append(int(dense_logits(cfg, w, seq, dtype=torch.float32)[0, -1].argmax()))
t_dense = time.time() - t0

print("cached", cached)
print("dense ", seq)
print(f"time cached {t_cached:.2f}s  dense {t_dense:.2f}s  speedup {t_dense / t_cached:.1f}x")
assert cached == seq, (cached, seq)
print("kv cache ok")
