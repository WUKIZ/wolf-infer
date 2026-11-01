"""候选生成 + 打分择优：top-2 采样 K 个，top-4 用平均对数概率选最优。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

K, GEN, TEMP = 4, 24, 0.8
CASES = [
    ('中国的首都是哪座城市？', ['北京']),
    ('《红楼梦》的作者是谁？', ['曹雪芹']),
    ('地球上面积最大的海洋是哪个？', ['太平洋']),
    ('水的化学式是什么？', ['H2O', 'H₂O']),
    ('太阳系中最大的行星是哪颗？', ['木星']),
    ('中国的四大发明中，哪一项与航海关系最密切？', ['指南针']),
    ('平年一年有多少天？', ['365']),
    ('珠穆朗玛峰是世界第几高峰？', ['第一', '1']),
    ('人类体细胞中有多少对染色体？', ['23']),
    ('光在真空中的速度大约是多少？', ['299', '30万', '30 万']),
]
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()


def chatml(q):
    return '<|im_start|>user\n' + q + '<|im_end|>\n<|im_start|>assistant\n'


ids = [tk.encode(chatml(q)).ids for q, _ in CASES]
eng.cfg['num_experts_per_tok'] = 2
caches = [{} for _ in range(len(CASES) * K)]
seqs = []
for i, pid in enumerate(ids):
    for k in range(K):
        lg = eng.prefill(pid, caches[i * K + k])
        p = torch.softmax(lg[0, -1].float() / TEMP, -1)
        seqs.append([int(torch.multinomial(p, 1))])
for _ in range(GEN - 1):
    lg = eng.decode([s[-1] for s in seqs], caches).reshape(len(seqs), -1)
    p = torch.softmax(lg.float() / TEMP, -1)
    for i, s in enumerate(seqs):
        s.append(int(torch.multinomial(p[i], 1)))
eng.cfg['num_experts_per_tok'] = 4
any_ok = rerank_ok = 0
for i, (q, keys) in enumerate(CASES):
    scores, texts = [], []
    for k in range(K):
        cand = seqs[i * K + k]
        lg = eng.logits(ids[i] + cand)
        lp = torch.log_softmax(lg[0].float(), -1)[len(ids[i]) - 1:len(ids[i]) - 1 + len(cand)]
        scores.append(float(lp[torch.arange(len(cand)), torch.tensor(cand)].mean()))
        texts.append(tk.decode(cand).split('<|im_end|>')[0])
    best = max(range(K), key=lambda k: scores[k])
    hits = [any(kk in t for kk in keys) for t in texts]
    any_ok += 1 if any(hits) else 0
    rerank_ok += 1 if hits[best] else 0
    print('候选命中 %d/%d  择优 %s | %r'
          % (sum(hits), K, 'OK' if hits[best] else 'X', texts[best][:28]))
    torch.cuda.empty_cache()

n = len(CASES)
print('任一候选正确 %d/%d   择优后正确 %d/%d' % (any_ok, n, rerank_ok, n))
