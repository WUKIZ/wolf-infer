"""bge-small-zh vs LSA 同协议对照（纯 torch，不装 sentence-transformers）。

协议：从语料均匀取 20 段，用每段前 8 字当查询，看能否召回自己。
"""
import json
import sys

sys.path.insert(0, '.')
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

import vector as V

BGE = '/data/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master'


class Bge:
    def __init__(self, path=BGE, maxlen=256):
        cfg = json.load(open(path + '/config.json'))
        self.h = cfg['hidden_size']
        self.nh = cfg['num_attention_heads']
        self.maxlen = maxlen
        self.w = load_file(path + '/model.safetensors')
        self.tk = Tokenizer.from_file(path + '/tokenizer.json')
        self.tk.enable_truncation(maxlen)
        self.tk.enable_padding(pad_id=0, pad_token='[PAD]')

    def _batch(self, texts):
        outs = self.tk.encode_batch(texts)
        ids = torch.tensor([o.ids for o in outs], dtype=torch.long)
        mask = torch.tensor([o.attention_mask for o in outs], dtype=torch.float)
        return ids, mask[:, None, None, :]      # mask: (B,1,1,T)

    def _ln(self, x, p):
        return torch.nn.functional.layer_norm(
            x, (self.h,), self.w[p + '.weight'], self.w[p + '.bias'], 1e-12)

    @torch.no_grad()
    def embed(self, texts, bs=32):
        out = [self._fwd(texts[i:i + bs]) for i in range(0, len(texts), bs)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def _fwd(self, texts):
        ids, mask = self._batch(texts)
        b, t = ids.shape
        h = self.h
        dh = h // self.nh
        x = self.w['embeddings.word_embeddings.weight'][ids]
        x = x + self.w['embeddings.position_embeddings.weight'][:t]
        x = x + self.w['embeddings.token_type_embeddings.weight'][0]
        x = self._ln(x, 'embeddings.LayerNorm')
        nl = sum(1 for k in self.w if k.endswith('.attention.output.LayerNorm.bias'))
        for i in range(nl):
            p = 'encoder.layer.%d.' % i
            q = x @ self.w[p + 'attention.self.query.weight'].T + self.w[p + 'attention.self.query.bias']
            k = x @ self.w[p + 'attention.self.key.weight'].T + self.w[p + 'attention.self.key.bias']
            v = x @ self.w[p + 'attention.self.value.weight'].T + self.w[p + 'attention.self.value.bias']
            q, k, v = (u.view(b, t, self.nh, dh).transpose(1, 2) for u in (q, k, v))
            att = (q @ k.transpose(-1, -2)) / (dh ** 0.5) + (1.0 - mask) * -1e4
            ctx = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, h)
            ctx = ctx @ self.w[p + 'attention.output.dense.weight'].T + self.w[p + 'attention.output.dense.bias']
            x = self._ln(x + ctx, p + 'attention.output.LayerNorm')
            f = torch.nn.functional.gelu(
                x @ self.w[p + 'intermediate.dense.weight'].T + self.w[p + 'intermediate.dense.bias'])
            f = f @ self.w[p + 'output.dense.weight'].T + self.w[p + 'output.dense.bias']
            x = self._ln(x + f, p + 'output.LayerNorm')
        return torch.nn.functional.normalize(x[:, 0], dim=-1)


def main():
    s = V.Store()
    docs = s.docs
    print('语料 %d 段' % len(docs))
    picks = [int(i * len(docs) / 20) for i in range(20)]
    pairs = [(docs[i][20:40], docs[i]) for i in picks if len(docs[i]) >= 40]
    print('查询 %d 条（每段中段 20 字）\n' % len(pairs))

    l1 = l3 = 0
    for q, tgt in pairs:
        hits = s.search(q, k=3)
        hits = [h[0] if isinstance(h, tuple) else h for h in hits]
        l1 += bool(hits) and (hits[0] in tgt or tgt in hits[0])
        l3 += any(h in tgt or tgt in h for h in hits[:3])
    print('LSA  top1 %2d/%d   top3 %2d/%d' % (l1, len(pairs), l3, len(pairs)))

    b = Bge()
    D = b.embed(docs)
    Q = b.embed([q for q, _ in pairs])
    order = (Q @ D.T).argsort(1)[:, ::-1][:, :3]
    b1 = b3 = 0
    for r, (_, tgt) in enumerate(pairs):
        want = docs.index(tgt)
        b1 += int(order[r, 0] == want)
        b3 += int(want in order[r].tolist())
    print('bge  top1 %2d/%d   top3 %2d/%d' % (b1, len(pairs), b3, len(pairs)))


if __name__ == '__main__':
    main()
