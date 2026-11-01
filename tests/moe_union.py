"""同一批下：相同 prompt vs 不同 prompt，比较 moe 每步耗时（验证成本取决于专家并集）。"""
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

SETS = {
    '相同': ['中国的首都是'] * 8,
    '不同': ['中国的首都是', '水的化学式是', '太阳从哪边升起', '一天有多少小时',
             '梯度消失是什么', '过拟合怎么缓解', '二分查找的原理', '什么是自注意力'],
}
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.cfg['profile'] = True
eng.warm()

for tag, prompts in SETS.items():
    eng._kvb = {}
    caches = [{} for _ in prompts]
    toks = [int(eng.prefill(tk.encode(p).ids, c)[0, -1].argmax())
            for p, c in zip(prompts, caches)]
    seqs = [[t] for t in toks]
    eng.dbg_attn = eng.dbg_moe = 0.0
    t0 = time.perf_counter()
    for _ in range(16):
        lg = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(lg[i].argmax()))
    wall = time.perf_counter() - t0
    print('%s prompt: 每步 %.0f ms (attn %.0f / moe %.0f)'
          % (tag, wall / 16 * 1000, eng.dbg_attn / 16 * 1000, eng.dbg_moe / 16 * 1000))
    torch.cuda.empty_cache()
