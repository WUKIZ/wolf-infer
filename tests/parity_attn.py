"""批量 decode 与单流 decode 的逐 token 对拍。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

PROMPTS = ['中国的首都是', '水的化学式是', '太阳从哪边升起', '一天有多少小时']
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()

single = []
for p in PROMPTS:
    c, out = {}, []
    lg = eng.prefill(tk.encode(p).ids, c)
    tok = int(lg[0, -1].argmax())
    out.append(tok)
    for _ in range(7):
        lg = eng.decode([tok], [c])
        tok = int(lg[0].argmax())
        out.append(tok)
    single.append(out)

eng._kvb = {}
caches = [{} for _ in PROMPTS]
toks = [int(eng.prefill(tk.encode(p).ids, c)[0, -1].argmax()) for p, c in zip(PROMPTS, caches)]
seqs = [[t] for t in toks]
for _ in range(7):
    lg = eng.decode([s[-1] for s in seqs], caches)
    for i, s in enumerate(seqs):
        s.append(int(lg[i].argmax()))
batch = [s for s in seqs]

print('单流', single)
print('批量', batch)
print('一致', single == batch)
