"""端到端：分片 → 页表 → 4 设备流水 → 后端作用 → 输出。"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from shard import ModelSpec, Shard
from supervisor import Supervisor
from backend import LocalBackend, toy_apply

SZ = 1_000_000
path = os.path.join(tempfile.mkdtemp(), "m.bin")
with open(path, "wb") as fh:
    fh.write(b"\0" * (64 * SZ))

spec = ModelSpec("mock-1T", 10**12, 64, 128000,
                 tuple(Shard(i, path, i * SZ, SZ, i, i + 1) for i in range(64)))
be = LocalBackend(toy_apply)
sup = Supervisor(spec, ["dcu0", "dcu1", "dcu2", "dcu3"], 3 * SZ, be.run)

tokens = [1, 2, 3, 4]
out = sup.forward(tokens)
print("stages", len(sup.stages), "out", out, sup.pager.stats)
assert len(out) == len(tokens)
assert list(out) != tokens                 # 权重确实被喂进作用里了
assert sup.pager.stats.misses == 64        # 64 片各取一次
assert sup.pager.stats.evictions >= 60     # 预算只放 3 片，必须不停换页
sup.close()
print("e2e ok")
