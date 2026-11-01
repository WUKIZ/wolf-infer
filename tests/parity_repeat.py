"""同一批量重复两次，验证引擎确定性（逐 token 相同）。"""
import sys
sys.path.insert(0, '.')
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

PROMPTS = ['中国的首都是', '水的化学式是', '太阳从哪边升起', '一天有多少小时']
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()

def run():
    eng._kvb = {}
    caches = [{} for _ in PROMPTS]
    toks = [int(eng.prefill(tk.encode(p).ids, c)[0, -1].argmax()) for p, c in zip(PROMPTS, caches)]
    seqs = [[t] for t in toks]
    for _ in range(7):
        lg = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(lg[i].argmax()))
    return seqs

a, b = run(), run()
print('确定性', a == b)
print(a)
