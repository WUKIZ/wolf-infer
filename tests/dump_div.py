"""打印 top-4 与指定 top-k 的分叉处文本，供人工判断「错」还是「不同」。"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer
from eval_set import PROMPTS

GEN = 32
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine(); eng.warm()

def run(top):
    eng.cfg['num_experts_per_tok'] = top
    eng._kvb = {}
    caches = [{} for _ in PROMPTS]
    toks = [int(eng.prefill(tk.encode(q).ids, c)[0, -1].argmax())
            for (_, q), c in zip(PROMPTS, caches)]
    seqs = [[t] for t in toks]
    for _ in range(GEN):
        out = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(out[i].reshape(-1).argmax()))
    return [s[1:] for s in seqs]

ref = run(4); out = run(k)
for i, (cat, q) in enumerate(PROMPTS):
    a, b = ref[i], out[i]
    n = 0
    while n < len(a) and a[n] == b[n]:
        n += 1
    print('---', cat, '|', q[:20], '| 分叉@%d' % n)
    print('  top-4:', repr(tk.decode(a[max(0, n - 4):n + 8])))
    print('  top-%d:' % k, repr(tk.decode(b[max(0, n - 4):n + 8])))
