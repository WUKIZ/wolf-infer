"""64 条请求 prefill：逐条 vs 等长分桶批量，比较总耗时。"""
import sys, time
sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

BASE = ['请解释梯度消失的原因与缓解办法。', '什么是过拟合，如何检测和缓解？',
        '二分查找的时间复杂度为什么是对数级？', '自注意力机制相比循环网络的优势在哪？']
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()
N = 64
idss = [tk.encode(BASE[i % 4] + BASE[(i * 3 + 1) % 4]).ids for i in range(N)]

t0 = time.perf_counter()
for ids in idss:
    eng.prefill(ids, {}, last_only=True)
t_seq = time.perf_counter() - t0
print('逐条   %d 条 %.2f s (%.3f s/条)' % (N, t_seq, t_seq / N))

t0 = time.perf_counter()
buckets = {}
for ids in idss:
    buckets.setdefault(len(ids), []).append(ids)
for n, group in buckets.items():
    for s in range(0, len(group), 16):
        batch = group[s:s + 16]
        eng.prefill_batch([(b, {}) for b in batch], last_only=True)
t_bat = time.perf_counter() - t0
print('批量   %d 条 %.2f s (%.3f s/条)  加速 %.1fx'
      % (N, t_bat, t_bat / N, t_seq / max(t_bat, 1e-6)))
print('分桶', {k: len(v) for k, v in buckets.items()})
