"""长上下文：不同长度下的 prefill 与 decode 耗时、显存占用。"""
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
BASE = ('梯度消失是指在深层神经网络中，反向传播时梯度逐层相乘，当每层的导数都小于一时，'
        '梯度会指数式衰减，导致靠近输入的层几乎无法更新。缓解方法包括残差连接、归一化层，'
        '以及更平滑的激活函数。注意力机制允许模型直接访问序列中的任意位置，'
        '从而缓解长距离依赖难以学习的问题。专家混合模型通过路由网络为每个词元选择少数专家，'
        '在保持参数规模的同时控制计算量。') * 40
all_ids = tk.encode(BASE).ids
print('可用词元', len(all_ids))

eng = BatchEngine()
eng.warm()

for n in (1024, 2048, 4096):
    ids = all_ids[:n]
    cache = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    lg = eng.prefill(ids, cache)
    prefill = time.perf_counter() - t0
    tok = int(lg[0, -1].argmax())
    t0 = time.perf_counter()
    for _ in range(8):
        lg = eng.decode([tok], [cache])
        tok = int(lg[0].argmax())
    dec = (time.perf_counter() - t0) / 8
    print('n=%4d  prefill %.2f s  首token %.1f ms  decode %.0f ms/步  峰值显存 %.1f GB'
          % (n, prefill, prefill * 1000, dec * 1000,
             torch.cuda.max_memory_allocated() / 2 ** 30))
    torch.cuda.empty_cache()
