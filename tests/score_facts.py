"""给答案打分：对 20 条常识半句，计算正确答案的平均 NLL（越小越好）。"""
import math
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

CASES = [
    ('中国的首都是', '北京'), ('日本的首都是', '东京'), ('法国的首都是', '巴黎'),
    ('英国的首都是', '伦敦'), ('美国的首都是', '华盛顿'),
    ('古都长安现在的名字是', '西安'), ('长江是中国最长的', '河流'),
    ('水的化学式是', 'H2O'), ('二氧化碳的化学式是', 'CO2'),
    ('太阳从', '东'), ('一年有', '12'), ('一天有', '24'),
    ('一小时有', '60'), ('珠穆朗玛峰是世界第', '一'),
    ('太阳系中最大的行星是', '木星'), ('离太阳最近的行星是', '水星'),
    ('光速约为每秒', '30'), ('《红楼梦》的作者是', '曹雪芹'),
    ('《西游记》的作者是', '吴承恩'), ('中国的四大发明之一是', '指南针'),
]

torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()

for top in (1, 2, 3, 4, 8):
    eng.cfg['num_experts_per_tok'] = top
    total, hits = 0.0, 0
    for prompt, answer in CASES:
        pids = tk.encode(prompt).ids
        aids = tk.encode(answer).ids
        logits = eng.logits(pids + aids)                      # (1, T, V)
        logp = torch.log_softmax(logits[0].float(), dim=-1)[len(pids) - 1:len(pids) - 1 + len(aids)]
        nll = -logp[torch.arange(len(aids)), torch.tensor(aids)].mean().item()
        total += nll
        hits += 1 if int(logp[0].argmax()) == aids[0] else 0
    n = len(CASES)
    print('top-%d  答案平均 NLL %.4f  (PPL %.3f)  首词命中 %d/%d'
          % (top, total / n, math.exp(total / n), hits, n))
    torch.cuda.empty_cache()
