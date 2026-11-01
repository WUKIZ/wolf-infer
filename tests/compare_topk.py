"""对拍不同专家数的生成质量（批量解码 + 每轮清缓存）。"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer
from eval_set import PROMPTS, chatml

GEN = 32
torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()


def run(top):
    eng.cfg['num_experts_per_tok'] = top
    eng._kvb = {}                      # 批量路径的 KV 缓冲
    if hasattr(eng, '_kv'):
        eng._kv = {}                   # 单流路径的 KV 缓存，必须一起清
    caches = [{} for _ in PROMPTS]
    toks = [int(eng.prefill(tk.encode(chatml(q)).ids, c)[0, -1].argmax())
            for (_, q), c in zip(PROMPTS, caches)]
    seqs = [[t] for t in toks]
    for _ in range(GEN):
        out = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(out[i].reshape(-1).argmax()))
    return [s[1:] for s in seqs]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


ref_top = int(sys.argv[1]) if len(sys.argv) > 1 else 4
others = [int(x) for x in sys.argv[2:]] or [1, 2, 3]

ref = run(ref_top)
print('参考 top-%d 完成，GEN=%d' % (ref_top, GEN))

for k in others:
    out = run(k)
    print('== top-%d ==' % k)
    per_cat = {}
    for i, (cat, q) in enumerate(PROMPTS):
        n = lcp(ref[i], out[i])
        per_cat.setdefault(cat, []).append(n / GEN)
        print('  %-3s 前缀 %2d/%d %s' % (cat, n, GEN, '一致' if n == GEN else '分叉'))
    for cat, ratio in per_cat.items():
        print('  [%s] 平均前缀一致率 %.2f' % (cat, sum(ratio) / len(ratio)))
