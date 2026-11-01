"""分片 / 换页 / 流水线 三块一次测完。"""
import mmap, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from shard import ModelSpec, Shard
from pager import Pager
from pipeline import plan

n = 0
def ok(name, cond):
    global n
    assert cond, "FAIL " + name
    n += 1

spec = ModelSpec("mock-1T", 10**12, 64, 128000,
                 tuple(Shard(i, "m.bin", i * 250_000_000, 250_000_000, i, i + 1) for i in range(64)))
ok("1T 形状合法", spec.check())
ok("1T 总量 16TB", spec.total_bytes == 16_000_000_000)
ok("缺层被拒", not ModelSpec("bad", 1, 4, 1, (Shard(0, "x", 0, 10, 0, 2),)).check())
ok("叠层被拒", not ModelSpec("dup", 1, 2, 1,
        (Shard(0, "x", 0, 10, 0, 2), Shard(1, "x", 10, 10, 1, 2))).check())

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.bin")
    with open(p, "wb") as fh:
        fh.write(b"\x01\x02" * 524288)
    shards = [Shard(i, p, i * 262144, 262144, i, i + 1) for i in range(4)]
    pg = Pager(budget_bytes=524288)
    for s in shards[:3]:
        mv = pg.view(s)
        ok(f"片{s.index} 长度", len(mv) == 262144)
        ok(f"片{s.index} 内容", bytes(mv[:2]) == b"\x01\x02")
        del mv
    ok("常驻不超预算", pg.resident_bytes() <= pg.budget)
    ok("发生过换出", pg.stats.evictions >= 1)
    try:
        pg.view(Shard(9, p, 0, 1048576, 0, 1)); over = False
    except MemoryError:
        over = True
    ok("超限片被拒", over)
    pg.close()

for k in (1, 4, 16):
    st = plan(spec, [f"dcu{i}" for i in range(k)])
    ok(f"{k}设备 覆盖完整", [i for s in st for i in s.shard_indices] == list(range(64)))
    ok(f"{k}设备 字节守恒", sum(s.nbytes for s in st) == spec.total_bytes)
    ok(f"{k}设备 不超段数", len(st) <= k)
print(f"compute ok ({n} asserts)")
