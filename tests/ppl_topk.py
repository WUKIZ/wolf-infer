"""用固定文本的困惑度衡量不同专家数对建模能力的影响（无需标签）。"""
import math
import sys

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
TEXT = (
    '梯度消失是指在深层神经网络中，反向传播时梯度逐层相乘，'
    '当每层的导数都小于一时，梯度会指数式衰减，导致靠近输入的层几乎无法更新。'
    '常见的缓解方法包括残差连接、归一化层、以及使用更平滑的激活函数。'
    '在自然语言处理中，注意力机制允许模型直接访问序列中的任意位置，'
    '从而缓解了长距离依赖难以学习的问题。专家混合模型通过路由网络为每个词元'
    '选择少数几个专家进行计算，在保持参数规模的同时控制计算量。'
) * 4
ids = tk.encode(TEXT).ids[:512]
eng = BatchEngine()
eng.warm()

for top in (4, 3, 2, 1):
    eng.cfg['num_experts_per_tok'] = top
    logits = eng.logits(ids)                       # (1, T, V)
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = torch.tensor(ids[1:], device=logp.device)
    nll = -logp[torch.arange(len(tgt)), tgt].mean().item()
    print('top-%d  PPL %.3f  (NLL %.4f, %d tokens)' % (top, math.exp(nll), nll, len(tgt)))
