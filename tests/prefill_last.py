"""last_only 与全量 prefill 的末位 logits 必须一致；同时报省了多少时间。"""
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()
ids = tk.encode('梯度消失是指在深层神经网络中，反向传播时梯度逐层相乘。' * 20).ids[:256]

t0 = time.perf_counter()
a = eng.prefill(ids, {})
t_full = time.perf_counter() - t0
t0 = time.perf_counter()
b = eng.prefill(ids, {}, last_only=True)
t_last = time.perf_counter() - t0
print('token %d  全量 %.2f s  仅末位 %.2f s  加速 %.1fx'
      % (len(ids), t_full, t_last, t_full / max(t_last, 1e-6)))
print('末位 logits maxdiff %.6f' % float((a[0, -1] - b[0, -1]).abs().max()))
