"""新旧路径对拍。ref=稠密真值；test=pcie 路径。
配置：batch=1、greedy、GEN=48、遇 EOS 停止。性能测试是 batch=8，两者不同口径。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import torch
from tokenizers import Tokenizer
from qwen_moe import load_moe, logits as dense_logits
from core.tmpdir import TMP

D_DIR = '/data/models/Qwen1.5-MoE-A2.7B'
OUT = TMP + '/parity.pt'
GEN = 48
EOS = {151643, 151645}
CASES = [
    ('短', '中国的首都是'),
    ('短', '人工智能的未来在于'),
    ('短', '请解释什么是梯度下降，并给出一个例子'),
    ('中', '在一个风和日丽的下午，小镇上的人们正在'),
    ('中', '写一段关于夏天的短文：'),
    ('续写', '海边的风很大，她没有回头，因为'),
    ('续写', '他把最后一块电池装进机器，按下开关，'),
    ('长', '机器学习是人工智能的一个分支。深度学习使用多层神经网络。' * 24),
]


def greedy(fn, ids, n=GEN):
    seq, hit = list(ids), -1
    for i in range(n):
        t = int(fn(seq)[0, -1].argmax())
        seq.append(t)
        if t in EOS:
            hit = i + 1
            break
    return seq[len(ids):], hit


if __name__ == '__main__':
    mode = sys.argv[1]
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    if mode == 'ref':
        cfg, w = load_moe(D_DIR, device='cuda')
        res = {}
        for c, p in CASES:
            ids = tk.encode(p).ids
            g, hit = greedy(lambda s: dense_logits(cfg, w, s), ids)
            res[p] = {'n': len(ids),
                      'logits': dense_logits(cfg, w, ids)[0, -1].float().half().cpu(),
                      'gen': g, 'eos': hit}
        torch.save(res, OUT)
        print('ref saved', len(res))
    else:
        from core.moe_batch import BatchEngine
        eng = BatchEngine().warm()
        ref = torch.load(OUT)
        ok = 0
        for c, p in CASES:
            ids = tk.encode(p).ids
            lg = eng.prefill(ids, {})[0, -1].float().cpu()
            r = ref[p]['logits'].float()
            d = (r - lg).abs()
            gen, hit = greedy(lambda s: eng.prefill(s, {}), ids)
            same = gen == ref[p]['gen']
            ok += same
            print('[%s] n=%d max %.3f top1 %s 生成 %d tok%s %s'
                  % (c, ref[p]['n'], float(d.max()),
                     int(r.argmax()) == int(lg.argmax()), len(gen),
                     ('（EOS@%d）' % hit) if hit > 0 else '（未 EOS）',
                     '逐字一致' if same else '不同'))
            if not same:
                print('   稠密:', tk.decode(ref[p]['gen'])[:60])
                print('   pcie:', tk.decode(gen)[:60])
        print('生成一致 %d/%d；配置 batch=1/greedy/GEN=%d' % (ok, len(CASES), GEN))
