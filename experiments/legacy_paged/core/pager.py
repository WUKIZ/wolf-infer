"""页表：常驻字节有硬上限，超了按 LRU 换出；单片装不下直接拒绝。

按片映射（偏移向下对齐到页边界），所以 resident_bytes 计的是真实映射量，
不是整个文件的长度。模型比内存大时不崩，靠换页；换不动就 fail fast。
调用方用完立刻 del memoryview，否则 mmap.close() 抛 BufferError。
"""
from __future__ import annotations
import contextlib
import mmap
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass

try:
    from .shard import Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shard import Shard

GRAN = mmap.ALLOCATIONGRANULARITY


@dataclass(slots=True)
class PageStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class Pager:
    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be > 0")
        self.budget = budget_bytes
        self._live: OrderedDict[int, mmap.mmap] = OrderedDict()
        self._fh: dict[int, object] = {}
        self._skip: dict[int, int] = {}
        self._len: dict[int, int] = {}
        self._pins: dict[int, int] = {}
        self.stats = PageStats()

    def resident_bytes(self) -> int:
        return sum(self._len.values())

    @staticmethod
    def _key(shard: Shard) -> tuple[str, int]:
        """分页身份用 (路径, 偏移)：不同文件重复用 index 也不会串内容。"""
        return (shard.path, shard.offset)

    @staticmethod
    def _span(shard: Shard) -> tuple[int, int, int]:
        """(对齐偏移, 映射长度, 片内偏移)。"""
        head = shard.offset - (shard.offset % GRAN)
        return head, shard.offset - head + shard.nbytes, shard.offset - head

    def _acquire(self, shard: Shard) -> mmap.mmap:
        hit = self._live.get(self._key(shard))
        if hit is not None:
            self._live.move_to_end(self._key(shard))
            self.stats.hits += 1
            return hit
        head, length, _ = self._span(shard)
        if length > self.budget:
            raise MemoryError(f"shard {shard.index} {length}B > budget {self.budget}B")
        skipped = 0
        while self._live and self.resident_bytes() + length > self.budget:
            idx, mm = self._live.popitem(last=False)
            if self._pins.get(idx):            # 正被租用，放回并跳到下一个
                self._live[idx] = mm
                skipped += 1
                if skipped >= len(self._live):
                    raise MemoryError("所有常驻分片都在租用中，无法换出")
                continue
            skipped = 0
            mm.close()
            self._fh.pop(idx).close()          # type: ignore[union-attr]
            self._skip.pop(idx, None)
            self._len.pop(idx, None)
            self.stats.evictions += 1
        fh = open(shard.path, "rb")
        mm = mmap.mmap(fh.fileno(), length, offset=head, access=mmap.ACCESS_READ)
        self._fh[self._key(shard)] = fh
        self._live[self._key(shard)] = mm
        self._skip[self._key(shard)] = shard.offset - head
        self._len[self._key(shard)] = length
        self.stats.misses += 1
        return mm

    def view(self, shard: Shard) -> memoryview:
        mm = self._acquire(shard)
        skip = self._skip[self._key(shard)]
        return memoryview(mm)[skip: skip + shard.nbytes]

    @contextlib.contextmanager
    def lease(self, shard: Shard):
        """租用一片：期间不会被换出，退出时自动释放视图。"""
        view = self.view(shard)
        self._pins[self._key(shard)] = self._pins.get(self._key(shard), 0) + 1
        try:
            yield view
        finally:
            self._pins[self._key(shard)] -= 1
            if self._pins[self._key(shard)] <= 0:
                self._pins.pop(shard.index, None)
            view.release()

    def close(self) -> None:
        for mm in self._live.values():
            mm.close()
        for fh in self._fh.values():
            fh.close()
        self._live.clear()
        self._fh.clear()
        self._skip.clear()
        self._len.clear()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        with open(p, "wb") as fh:
            fh.write(b"\x01\x02" * 524288)                   # 1 MB
        shards = [Shard(i, p, i * 262144, 262144, i, i + 1) for i in range(4)]
        pager = Pager(budget_bytes=524288)                   # 只放得下两片
        for s in shards[:3]:
            mv = pager.view(s)
            assert len(mv) == 262144 and bytes(mv[:2]) == b"\x01\x02"
            del mv
        assert pager.resident_bytes() <= pager.budget, pager.resident_bytes()
        assert pager.stats.evictions >= 1
        try:
            pager.view(Shard(9, p, 0, 1048576, 0, 1))
        except MemoryError:
            logging.warning('分片装载超出内存，跳过本次预取')
        else:
            raise AssertionError("超限分片必须拒绝")
        pager.close()
        print("ok", pager.stats, pager.resident_bytes())
