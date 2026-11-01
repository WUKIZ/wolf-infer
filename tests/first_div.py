"""定位首个分叉：dense 与 pcie 分进程，避免共驻。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import torch
from tokenizers import Tokenizer
from core.tmpdir import TMP
D = '/data/models/Qwen1.5-MoE-A2.7B'
OUT = TMP + '/firstdiv.pt'
P = '在一个风和日丽的下午，小镇上的人们正在'
N = 48
if __name__ == '__main__':
    mode = sys.argv[1]
    tk = Tokenizer.from_file(os.path.join(D, 'tokenizer.json'))
    ids = tk.encode(P).ids
    if mode == 'dense':
        from qwen_moe import load_moe, logits as dense_logits
        cfg, w = load_moe(D, device='cuda')
        seq, tops = list(ids), []
        for _ in range(N):
            a = dense_logits(cfg, w, seq)[0, -1].float()
            v, t = a.topk(2)
            tops.append((int(t[0]), float(v[0] - v[1])))
            seq.append(int(t[0]))
        torch.save({'gen': seq[len(ids):], 'tops': tops}, OUT)
        print('dense saved', len(tops))
    else:
        from core.moe_batch import BatchEngine
        eng = BatchEngine().warm()
        d = torch.load(OUT)
        seq = list(ids)
        for i, ref_tok in enumerate(d['gen']):
            b = eng.prefill(seq, {})[0, -1].float().cpu()
            if int(b.argmax()) != ref_tok:
                v = b.topk(2)
                print('首个分叉 @%d 稠密 top1-top2 差 %.4f' % (i, d['tops'][i][1]))
                print(' pcie top2', [(int(x), round(float(y), 3)) for x, y in zip(v.indices, v.values)])
                print(' pcie 该步 top1-top2 差 %.4f' % float(v.values[0] - v.values[1]))
                break
            seq.append(ref_tok)
        else:
            print('%d 步无分叉' % N)
