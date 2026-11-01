"""recall 端到端：验证向量通道能在「零关键词重叠」时召回。"""
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.memory import MemoryStore


class StubHome:
    def __init__(self):
        self.live = {'facts': [
            {'text': '本地助手的核心规则是不伤害自己人，家门外要收敛暴露度', 'source': 'home', 'salience': 0.9},
            {'text': '训练模型时要给显存预算留出余量，否则中途会被杀掉', 'source': 'home', 'salience': 0.7},
        ]}
        self.threshold_dir = Path('.tmp/thr')
        self.threshold_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path('.tmp/state')
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def log(self, *a, **k):
        pass

    def deposit(self, *a, **k):
        pass

    def __getattr__(self, name):
        # 桩：Home 的其他写接口一律空操作
        return lambda *a, **k: None


m = MemoryStore(StubHome(), ttl_s=3600)
m.remember('海马体负责把短期记忆合并成语义记忆', source='chat')
m.remember('混合专家模型每个词元只激活少数专家', source='chat')

for q in ('不伤害自己人', '显卡会不会爆', '专家是怎么选的', '记忆怎么巩固'):
    hits = m.recall(q, k=2)
    print('== %s' % q)
    for h in hits:
        print('   %.3f  %s' % (h['score'], h['text'][:40]))
