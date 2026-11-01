"""等长批量 prefill 与逐条 prefill 的末位 logits 对齐检验。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()
P = ['中国的首都是', '水的化学式是']
n_min = min(len(tk.encode(p).ids) for p in P)
ids = [tk.encode(p).ids[:n_min] for p in P]

ref = [eng.prefill(i, {}, last_only=True)[0, -1] for i in ids]
caches = [{}, {}]
lg = eng.prefill_batch([(i, c) for i, c in zip(ids, caches)], last_only=True)
for k in range(len(ids)):
    print('req%d maxdiff %.6f' % (k, float((lg[k, -1] - ref[k]).abs().max())))
