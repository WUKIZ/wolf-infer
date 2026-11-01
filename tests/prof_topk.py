import os
"""对比 top-k 专家数的速度与输出差异；用法: python3 tests/prof_topk.py <k>"""
import statistics
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

top = int(sys.argv[1])
torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
ids = tk.encode('中国的首都是').ids
eng = BatchEngine()
eng.cfg['moe_prob_floor'] = float(os.environ.get('WOLF_FLOOR', '0'))
eng.cfg['num_experts_per_tok'] = top
eng.warm()
print('top =', top)

# 单流贪婪 16 步，用于和 top-4 逐 token 比较（粗判质量变化）
seq = list(ids)
for _ in range(16):
    seq.append(int(eng.logits(seq, cache=True)[0, -1].argmax()))
print('单流 tokens', seq[len(ids):])
print('单流 text  ', repr(tk.decode(seq[len(ids):])))

# B=32 / B=128 稳态耗时
for B in (1, 8):
    caches = [{} for _ in range(B)]
    toks = [int(eng.prefill(ids, c)[0, -1].argmax()) for c in caches]
    seqs = [[t] for t in toks]
    for _ in range(3):
        out = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(out[i].reshape(-1).argmax()))
    times = []
    for _ in range(16):
        t0 = time.perf_counter()
        out = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(out[i].reshape(-1).argmax()))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    mean = statistics.mean(times)
    print('B=%3d mean %5.0f  p99 %5.0f ms/步  %6.1f tok/s'
          % (B, mean, times[int(0.99 * len(times))], B / (mean / 1e3)))
