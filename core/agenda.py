"""任务议程：提出 → 过层 → 执行 → 验收 → 续接。

只做队列与判定，不做执行；执行由 daemon 按 task.kind 路由。
"""
from dataclasses import dataclass

ACTIONS = ('reply', 'search', 'silent')


@dataclass
class Task:
    kind: str
    goal: str
    priority: float = 0.5
    tries: int = 0
    state: str = 'pending'


class Agenda:
    def __init__(self, max_size=16):
        self.max_size = max_size
        self.queue = []

    def propose(self, energy=0.5, arousal=0.5, trust=0.5, idle_steps=0, pending=0):
        """从驱动状态生成候选任务；返回 Task 或 None。"""
        if pending >= self.max_size:
            return None
        if idle_steps < 3:
            return None                     # 不空闲就不主动
        if energy < 0.2:
            kind, goal = 'silent', '能量过低，保持静默'
        elif trust < 0.3:
            kind, goal = 'silent', '信任不足，先观察'
        elif arousal > 0.7:
            kind, goal = 'search', '唤醒度高，主动检索新信息'
        else:
            kind, goal = 'reply', '惯例性整理：回顾最近记忆'
        pri = min(1.0, (arousal + (1 - energy)) / 2)
        return Task(kind=kind, goal=goal, priority=round(pri, 3))

    def admit(self, task, cortex=None):
        """过调度层；cortex(task) 返回 False 即拒。"""
        if task is None:
            return False
        if task.kind not in ACTIONS:
            return False
        if cortex is not None and not cortex(task):
            return False
        self.queue.append(task)
        self.queue.sort(key=lambda t: -t.priority)
        return True

    def next(self):
        for t in self.queue:
            if t.state == 'pending':
                t.state = 'running'
                return t
        return None

    def settle(self, task, ok):
        """结算：成功出队；失败一次降权重试；两次失败出队。"""
        task.tries += 1
        if ok:
            task.state = 'done'
            self.queue.remove(task)
            return 'done'
        if task.tries >= 2:
            task.state = 'failed'
            self.queue.remove(task)
            return 'failed'
        task.state = 'pending'
        task.priority = max(0.1, task.priority - 0.2)
        return 'retry'

    def summary(self):
        return {'pending': sum(1 for t in self.queue if t.state == 'pending'),
                'running': sum(1 for t in self.queue if t.state == 'running'),
                'size': len(self.queue)}


if __name__ == '__main__':
    a = Agenda()
    t = a.propose(energy=0.8, arousal=0.9, trust=0.7, idle_steps=5)
    assert a.admit(t), '提出后被拒'
    got = a.next()
    assert got is t and t.state == 'running'
    assert a.settle(got, ok=False) == 'retry' and t.state == 'pending'
    assert a.settle(a.next(), ok=True) == 'done' and a.summary()['size'] == 0
    assert a.propose(idle_steps=0) is None, '不空闲却提出任务'
    print('agenda 自检通过')
