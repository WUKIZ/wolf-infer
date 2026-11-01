"""对比 floor=0 与 floor=0.05 的生成文本，判断语义是否退化。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

P = ['中国的首都是', '水的化学式是', '太阳从哪边升起', '用一句话解释梯度消失']
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()

for floor in (0.0, 0.05):
    eng.cfg['moe_prob_floor'] = floor
    eng._kvb = {}
    caches = [{} for _ in P]
    toks = [int(eng.prefill(tk.encode(p).ids, c)[0, -1].argmax()) for p, c in zip(P, caches)]
    seqs = [[t] for t in toks]
    for _ in range(15):
        lg = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(lg[i].argmax()))
    print('=== floor', floor)
    for p, s in zip(P, seqs):
        print('  ', p, '->', repr(tk.decode(s)))
eng.cfg['moe_prob_floor'] = 0.0
