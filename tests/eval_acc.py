"""关键词判对的正确率评测：10 条事实 + 4 条算术（算术要求只输出数字）。"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer
from eval_set import chatml

GEN = 128
CASES = [
    ('事实', '中国的首都是哪座城市？', ['北京']),
    ('事实', '《红楼梦》的作者是谁？', ['曹雪芹']),
    ('事实', '地球上面积最大的海洋是哪个？', ['太平洋']),
    ('事实', '光在真空中的速度大约是多少？', ['299', '30万', '30 万']),
    ('事实', '水的化学式是什么？', ['H2O', 'H₂O']),
    ('事实', '珠穆朗玛峰是世界第几高峰？', ['第一', '1']),
    ('事实', '太阳系中最大的行星是哪颗？', ['木星']),
    ('事实', '中国的四大发明中，哪一项与航海关系最密切？', ['指南针']),
    ('事实', '平年一年有多少天？', ['365']),
    ('事实', '人类体细胞中有多少对染色体？', ['23']),

    ('算术', '只输出数字，不要解释：一个班有 42 名学生，其中 3/7 是女生，女生有多少人？', ['18']),
    ('算术', '只输出数字，不要解释：从 1 加到 100 等于多少？', ['5050']),
    ('算术', '只输出数字，不要解释：3 个人 3 天修 3 米路，9 个人 9 天修多少米？', ['27']),
    ('算术', '只输出数字，不要解释：一件商品先涨价 20%，再降价 20%，'
             '最终价格是原价的百分之多少？', ['96']),
]

torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()


def run(top):
    eng.cfg['num_experts_per_tok'] = top
    eng._kvb = {}
    caches = [{} for _ in CASES]
    toks = [int(eng.prefill(tk.encode(chatml(q)).ids, c)[0, -1].argmax())
            for (_, q, _), c in zip(CASES, caches)]
    seqs = [[t] for t in toks]
    for _ in range(GEN):
        out = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(out[i].reshape(-1).argmax()))
    return [tk.decode(s[1:]) for s in seqs]


for top in (4, 3, 2, 1):
    texts = run(top)
    hit = {'事实': 0, '算术': 0}
    for (cat, q, keys), txt in zip(CASES, texts):
        ok = any(k in txt for k in keys)
        hit[cat] += 1 if ok else 0
        if top == 4 or not ok:
            print('  top-%d %-2s %s | %r' % (top, cat, 'OK ' if ok else 'X  ', txt[:40]))
    print('top-%d  事实 %d/10  算术 %d/4'
          % (top, hit['事实'], hit['算术']))
