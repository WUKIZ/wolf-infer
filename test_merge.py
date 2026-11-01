"""验证 consolidate 的向量合并：近似句合并、无关句保留。"""
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.memory import MemoryStore


class StubHome:
    def __init__(self):
        self.live = {'facts': []}
        self.threshold_dir = Path('.tmp/thr')
        self.threshold_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path('.tmp/state')
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def log(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


m = MemoryStore(StubHome(), ttl_s=3600)
m.remember('海马体负责把短期记忆合并成语义记忆', source='chat')
m.remember('海马体的作用是把短期记忆合并成语义记忆', source='chat')
m.remember('海马体负责将短期记忆合并为语义记忆', source='chat')
m.remember('显卡显存不足时要降低批大小', source='chat')
print('合并前', len(m._load()))
print('consolidate ->', m.consolidate())
for e in m._load():
    print('  count=%d sal=%.2f %s' % (e.count, e.salience, e.text))
