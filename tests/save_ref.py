"""未改前的参考 logits，供 moe_block 重写后逐值对比。"""
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.warm()
ref = {}
for p in ['中国的首都是', '水的化学式是', '请解释梯度消失的原因']:
    ref[p] = eng.logits(tk.encode(p).ids).float().cpu()
torch.save(ref, '.tmp/ref_logits.pt')
print('参考已存', {k: tuple(v.shape) for k, v in ref.items()})
