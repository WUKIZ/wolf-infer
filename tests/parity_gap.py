"""对拍：不只比 token 是否相同，还报出分叉点的 top1-top2 差值。"""
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
    c, toks, gaps = {}, [], []
    lg = eng.prefill(tk.encode(p).ids, c)
    top2 = torch.topk(lg[0, -1].float(), 2).values
    gaps.append(float(top2[0] - top2[1]))
    tok = int(lg[0, -1].argmax())
    toks.append(tok)
    for _ in range(7):
        lg = eng.decode([tok], [c])
        top2 = torch.topk(lg.reshape(-1).float(), 2).values
        gaps.append(float(top2[0] - top2[1]))
        tok = int(lg[0].argmax())
        toks.append(tok)
    single.append((toks, gaps))

eng._kvb = {}
caches = [{} for _ in PROMPTS]
toks = [int(eng.prefill(tk.encode(p).ids, c)[0, -1].argmax()) for p, c in zip(PROMPTS, caches)]
seqs = [[t] for t in toks]
for _ in range(7):
    lg = eng.decode([s[-1] for s in seqs], caches)
    for i, s in enumerate(seqs):
        s.append(int(lg[i].argmax()))

for i, (p, (ref, gaps)) in enumerate(zip(PROMPTS, single)):
    got = seqs[i]
    diff = next((k for k in range(min(len(ref), len(got))) if ref[k] != got[k]), None)
    if diff is None:
        print('%-8s 一致' % p)
    else:
        print('%-8s 分叉@%d  top1-top2=%.4f  ref=%s got=%s'
              % (p, diff, gaps[diff], ref[:diff + 2], got[:diff + 2]))
