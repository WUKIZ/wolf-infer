"""打印句子两两相似度，用来校准合并阈值。"""
import sys

sys.path.insert(0, '.')
from core.memory import _vector_store

S = ['海马体负责把短期记忆合并成语义记忆',
     '海马体的作用是把短期记忆合并成语义记忆',
     '海马体负责将短期记忆合并为语义记忆',
     '显卡显存不足时要降低批大小',
     '长时间训练要注意检查点保存']
vs = _vector_store()
vecs = vs.index.embed(S)
for i in range(len(S)):
    row = ' '.join('%.2f' % float(vecs[i] @ vecs[j]) for j in range(len(S)))
    print('%d %-24s %s' % (i, S[i][:12], row))
