"""自查询检验：命中判据改为「内容相同」而非「下标相同」，避免重复段落误判。"""
import random

import vector

s = vector.Store()
docs = s.docs
print('语料 %d 段，词表 %d' % (len(docs), len(s.index.vocab)))
random.seed(0)
hit_idx = {1: 0, 3: 0, 5: 0}
hit_txt = {1: 0, 3: 0, 5: 0}
n = 0
for i in random.sample(range(len(docs)), 20):
    d = docs[i]
    if len(d) < 40:
        continue
    n += 1
    q = d[10:30]
    idx = [j for _, j in s.index.top(q, 5)]
    for k in (1, 3, 5):
        hit_idx[k] += 1 if i in idx[:k] else 0
        hit_txt[k] += 1 if any(q in docs[j] for j in idx[:k]) else 0
print('下标相同 top1/3/5 = %d/%d, %d/%d, %d/%d'
      % (hit_idx[1], n, hit_idx[3], n, hit_idx[5], n))
print('内容命中 top1/3/5 = %d/%d, %d/%d, %d/%d'
      % (hit_txt[1], n, hit_txt[3], n, hit_txt[5], n))
