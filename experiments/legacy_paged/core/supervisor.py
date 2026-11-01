"""调度器：分片 + 页表 + 设备 → 一条可运行路径。上层只调 forward()。

run 由调用方注入：run(device, blocks, tokens) -> tokens。
这里只负责：哪层在哪台设备、哪片现在进内存、用完立刻释放。
"""
from __future__ import annotations
import os
import sys

try:
    from .pager import Pager
    from .pipeline import Stage, plan
    from .shard import ModelSpec
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pager import Pager
    from pipeline import Stage, plan
    from shard import ModelSpec


class Supervisor:
    def __init__(self, spec: ModelSpec, devices: list[str], budget_bytes: int, run) -> None:
        self.spec = spec
        self.pager = Pager(budget_bytes)
        self.stages: tuple[Stage, ...] = plan(spec, devices)
        self._by_index = {s.index: s for s in spec.shards}
        # run 可以是单个函数，也可以是 {device: run} —— 按设备混用本地/远端后端
        self.run = run if callable(run) else (lambda d, b, t: run[d](d, b, t))

    def layers_on(self, device: str) -> tuple[int, int]:
        for st in self.stages:
            if st.device == device:
                return st.layer_from, st.layer_to
        raise KeyError(device)

    def forward(self, tokens, lo: int = 0, hi: int | None = None):
        hi = self.spec.n_layers if hi is None else hi
        for st in self.stages:
            if st.layer_to <= lo or st.layer_from >= hi:
                continue
            for i in st.shard_indices:          # 逐片流过，常驻只占一片
                shard = self._by_index[i]
                if isinstance(tokens, dict):    # 层号由调度器指定，后端只负责算
                    tokens["layer"] = shard.layer_from
                block = self.pager.view(shard)
                try:
                    tokens = self.run(st.device, [block], tokens)
                finally:
                    block.release()
        return tokens

    def close(self) -> None:
        self.pager.close()


if __name__ == "__main__":
    import tempfile

    from shard import Shard

    path = os.path.join(tempfile.mkdtemp(), "m.bin")
    SZ = 1_000_000
    with open(path, "wb") as fh:
        fh.write(b"\0" * (64 * SZ))
    spec = ModelSpec("mock-1T", 10**12, 64, 128000,
                     tuple(Shard(i, path, i * SZ, SZ, i, i + 1) for i in range(64)))
    seen: list[str] = []

    def run(device, blocks, tokens):
        seen.append(device)
        return tokens + len(blocks)

    sup = Supervisor(spec, ["dcu0", "dcu1", "dcu2", "dcu3"], 3 * SZ, run)
    out = sup.forward(0)
    print("stages", len(sup.stages), "devices", seen, "out", out)
    assert len(seen) == len(spec.shards) and out == len(spec.shards)
    assert sum(s.nbytes for s in sup.stages) == spec.total_bytes
    sup.close()
    print("ok")
