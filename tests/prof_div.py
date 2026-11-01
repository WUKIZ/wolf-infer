"""按「专家并集」真实口径测批吞吐：每个请求用不同文本，避免路由退化。"""
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

BASE = [
    '请解释梯度消失的原因与缓解办法。', '什么是过拟合，如何检测和缓解？',
    '二分查找的时间复杂度为什么是对数级？', '自注意力机制相比循环网络的优势在哪？',
    '解释一下混合专家模型的路由原理。', '为什么深度学习需要归一化层？',
    '残差连接解决了什么问题？', '卷积和全连接在参数效率上差别多大？',
    '什么是学习率预热，为什么有用？', '解释一下 KV 缓存的作用。',
    '批处理为什么会提高吞吐？', '显存带宽和算力的瓶颈如何区分？',
    '模型量化会带来什么精度损失？', '流水并行和专家并行各解决什么问题？',
    '为什么大模型推理时首 token 延迟高？', '解释一下投机解码的原理。',
]

def mk(i):
    return BASE[i % len(BASE)] + BASE[(i * 7 + 3) % len(BASE)]

torch.set_num_threads(14)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
eng = BatchEngine()
eng.cfg['profile'] = True
eng.warm()

GEN = 4
for B in (512,):
    prompts = [mk(i) for i in range(B)]
    eng._kvb = {}
    caches = [{} for _ in prompts]
    toks = []
    for _i, (_p, _c) in enumerate(zip(prompts, caches)):
        toks.append(int(eng.prefill(tk.encode(_p).ids, _c, last_only=True)[0, -1].argmax()))
        if (_i + 1) % 32 == 0:
            print('  prefill %d/%d' % (_i + 1, len(prompts)), flush=True)
    seqs = [[t] for t in toks]
    eng.dbg_attn = eng.dbg_moe = 0.0
    t0 = time.perf_counter()
    for _ in range(GEN):
        lg = eng.decode([s[-1] for s in seqs], caches)
        for i, s in enumerate(seqs):
            s.append(int(lg[i].argmax()))
    wall = time.perf_counter() - t0
    print('B=%d 每步 %.0f ms (attn %.0f / moe %.0f) %.1f tok/s'
          % (B, wall / GEN * 1000, eng.dbg_attn / GEN * 1000,
             eng.dbg_moe / GEN * 1000, B * GEN / wall))
    torch.cuda.empty_cache()
