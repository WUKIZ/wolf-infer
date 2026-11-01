"""分片描述：唯一知道权重字节分布的地方。万亿级和 4 亿级只差这里的数字。

Shard  = 一片权重在磁盘上的位置 + 它覆盖的层区间（expert 非空即 MoE 专家片）。
ModelSpec = 总参数 / 层数 / 词表 / 分片表，上层只拿它，不认识具体规模。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Shard:
    index: int
    path: str
    offset: int
    nbytes: int
    layer_from: int
    layer_to: int                 # 左闭右开
    expert: int | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    n_params: int
    n_layers: int
    vocab: int
    shards: tuple[Shard, ...]

    @property
    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self.shards)

    @property
    def bytes_per_param(self) -> float:
        return self.total_bytes / max(1, self.n_params)

    def covering(self, layer: int) -> tuple[Shard, ...]:
        return tuple(s for s in self.shards if s.layer_from <= layer < s.layer_to)

    def check(self) -> bool:
        """每层恰好被分片覆盖一次；缺片或重叠都算不合格。"""
        for layer in range(self.n_layers):
            if len(self.covering(layer)) != 1:
                return False
        return all(s.nbytes > 0 and s.offset >= 0 for s in self.shards)


if __name__ == "__main__":
    dense = ModelSpec("mock-400M", 400_000_000, 12, 6000,
                      tuple(Shard(i, "m.bin", i * 67_000_000, 67_000_000, i, i + 1) for i in range(12)))
    moe = ModelSpec("mock-1T", 1_000_000_000_000, 64, 128_000,
                    tuple(Shard(i, "m.bin", i * 250_000_000, 250_000_000, i, i + 1) for i in range(64)))
    assert dense.check() and moe.check()
    assert moe.total_bytes == 16_000_000_000
    bad = ModelSpec("bad", 1, 4, 1, (Shard(0, "x", 0, 10, 0, 2),))
    assert not bad.check()                      # 层 2、3 没片，必须拒
    print("ok", moe.name, f"{moe.total_bytes/1e12:.1f}TB", moe.bytes_per_param, "B/param")
