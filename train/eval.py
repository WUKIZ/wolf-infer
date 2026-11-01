"""验证：验证集 loss / ppl + 少量生成样例。"""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, max_batches=20):
    """跑固定批数，返回平均 loss 与 ppl（口径：按 token 加权）。"""
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = torch.as_tensor(x, device='cuda').long()
        y = torch.as_tensor(y, device='cuda').long()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += float(loss) * y.numel()
        count += y.numel()
    avg = total / max(count, 1)
    return {'loss': avg, 'ppl': math.exp(min(avg, 20)), 'tokens': count}


@torch.no_grad()
def sample(model, tok, prompt, gen=16):
    """贪心生成，返回完整文本（含 prompt）。"""
    model.eval()
    ids = tok.encode(prompt).ids
    for _ in range(gen):
        lg = model(torch.tensor([ids], device='cuda'))[0, -1]
        ids.append(int(lg.argmax()))
    return tok.decode(ids)


if __name__ == '__main__':
    import os
    import tempfile
    from tokenizers import Tokenizer

    from train.loader import Loader

    tk_path = '/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json'
    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, 'c.txt')
    with open(corpus, 'w', encoding='utf-8') as fh:
        fh.write('中国的首都是北京。' * 400)

    net = torch.nn.Sequential(torch.nn.Embedding(151936, 64),
                              torch.nn.Linear(64, 151936)).cuda()
    ld = Loader(corpus, tk_path, seq_len=16, batch=2)
    r = evaluate(net, ld, max_batches=5)
    print('验证 loss %.4f | ppl %.2f | token %d' % (r['loss'], r['ppl'], r['tokens']))
    tok = Tokenizer.from_file(tk_path)
    print('样例:', repr(sample(net, tok, '中国的首都是', gen=8)))
