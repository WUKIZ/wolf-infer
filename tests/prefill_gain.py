"""64 条不同 prompt 的 prefill 总耗时：全量 head vs 仅末位。"""
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

BASE = ['请解释梯度消失的原因与缓解办法。', '什么是过拟合，如何检测和缓解？',
        '二分查找的时间复杂度为什么是对数级？', '自注意力机制相比循环网络的优势在哪？',
        '解释一下混合专家模型的路由原理。', '为什么深度学习需要归一化层？',
        '残差连接解决了什么问题？', '卷积和全连接在参数效率上差别多大？']
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()
N = 64
idss = [tk.encode(BASE[i % 8] + BASE[(i * 3 + 1) % 8]).ids for i in range(N)]

for flag, tag in ((False, '全量 head'), (True, '仅末位')):
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    for ids in idss:
        eng.prefill(ids, {}, last_only=flag)
    dt = time.perf_counter() - t0
    print('%-10s 64 条共 %.2f s  (%.3f s/条)  %s'
          % (tag, dt, dt / N, '慢' if not flag else '快 %.1fx' % (0, )))
