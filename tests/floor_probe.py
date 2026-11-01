"""在不同专家概率下限下测：答案 NLL（质量）与整段前向耗时（速度）。"""
import math
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

CASES = [
    ('中国的首都是', '北京'), ('日本的首都是', '东京'), ('法国的首都是', '巴黎'),
    ('英国的首都是', '伦敦'), ('美国的首都是', '华盛顿'), ('古都长安现在的名字是', '西安'),
    ('水的化学式是', 'H2O'), ('二氧化碳的化学式是', 'CO2'), ('太阳从', '东'),
    ('一年有', '12'), ('一天有', '24'), ('一小时有', '60'),
    ('珠穆朗玛峰是世界第', '一'), ('太阳系中最大的行星是', '木星'),
    ('离太阳最近的行星是', '水星'), ('光速约为每秒', '30'),
    ('《红楼梦》的作者是', '曹雪芹'), ('《西游记》的作者是', '吴承恩'),
    ('中国的四大发明之一是', '指南针'), ('长江是中国最长的', '河流'),
]
torch.set_num_threads(14)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()

for floor in (0.0, 0.02, 0.05, 0.1, 0.15, 0.0):
    eng.cfg['moe_prob_floor'] = floor
    total, hits, t0 = 0.0, 0, time.perf_counter()
    for prompt, answer in CASES:
        pids, aids = tk.encode(prompt).ids, tk.encode(answer).ids
        lg = eng.logits(pids + aids)
        lp = torch.log_softmax(lg[0].float(), dim=-1)[len(pids) - 1:len(pids) - 1 + len(aids)]
        total += -lp[torch.arange(len(aids)), torch.tensor(aids)].mean().item()
        hits += 1 if int(lp[0].argmax()) == aids[0] else 0
    n = len(CASES)
    print('floor %.0e  答案 NLL %.4f (PPL %.3f)  首词 %d/%d  总耗时 %.2f s'
          % (floor, total / n, math.exp(total / n), hits, n, time.perf_counter() - t0))
    torch.cuda.empty_cache()
eng.cfg['moe_prob_floor'] = 0.0
