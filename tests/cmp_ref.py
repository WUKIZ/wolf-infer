"""与改动前保存的参考 logits 逐值对比。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
ref = torch.load('.tmp/ref_logits.pt')
eng = BatchEngine()
eng.warm()
for p, r in ref.items():
    cur = eng.logits(tk.encode(p).ids).float().cpu()
    print('%-12s maxdiff %.4f  top1 %s'
          % (p, float((cur - r).abs().max()),
             bool(int(cur[0, -1].argmax()) == int(r[0, -1].argmax()))))
