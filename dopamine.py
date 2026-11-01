"""多巴胺：奖励预测误差。

    Q(动作) += α × (实际收益 − Q(动作))

这是运行时唯一能把经验留在权重之外的地方。搜索搜到有用东西，下一次
就更倾向搜；搜回来全是垃圾，下次就少搜。不靠重训，靠每次落地的一点点。

α=0.15 是经验值：太大（0.5）一次失败就把动作打死，太小（0.01）几十次
才动一格，都学不出东西。

状态落 home/state/dopamine.json，一行 JSON，坏了就当空表 —— 多巴胺丢了
只影响倾向，不影响系统能不能跑。
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Dopamine:
    def __init__(self, path: Path | str, alpha: float = 0.15) -> None:
        self.p = Path(path)
        self.a = alpha
        self.q: dict[str, float] = {}
        if self.p.exists():
            try:
                loaded = json.loads(self.p.read_text("utf-8"))
                if isinstance(loaded, dict):
                    self.q = {k: float(v) for k, v in loaded.items()
                              if isinstance(v, (int, float))}
            except (OSError, ValueError):
                self.q = {}

    def _save(self) -> None:
        try:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.p.with_name(f".{self.p.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.q, ensure_ascii=False), "utf-8")
            os.replace(tmp, self.p)
        except OSError:
            pass          # 写不进去不拖垮对话，只是这轮没学到

    def expect(self, act: str) -> float:
        """对某个动作的预期收益。没见过的按 0.5 起，中性。"""
        return self.q.get(act, 0.5)

    def reward(self, act: str, r: float) -> float:
        """实际收益 r 落下来，返回预测误差 —— 正数是惊喜，负数是失望。"""
        old = self.expect(act)
        self.q[act] = round(old + self.a * (r - old), 4)
        self._save()
        return round(r - old, 4)

    def bias(self, acts: list[str]) -> dict[str, float]:
        """给候选动作打分，调用方拿去加权而不是硬选。"""
        return {a: self.expect(a) for a in acts}

    def line(self) -> str:
        """压成一行进 system，只报偏离中性的那几个。"""
        if not self.q:
            return ""
        bits = [f"{k}={v:.2f}" for k, v in self.q.items()
                if abs(v - 0.5) >= 0.1]
        return ("倾向：" + "，".join(bits)) if bits else ""
