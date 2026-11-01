"""通信层：列块切分（按输入维）需要 all-reduce，行块切分不需要。

单进程时 all_sum 是恒等，因此同一份代码在单卡/多卡都能跑。
多卡时把 backend 换成 nccl 即可，其余不改。
"""
import os

import torch

try:
    import torch.distributed as dist
except ImportError:
    dist = None


def init(backend: str = "gloo", rank: int = 0, world: int = 1,
         addr: str = "127.0.0.1", port: int = 29500) -> bool:
    if dist is None or world <= 1:
        return False
    if not dist.is_initialized():
        dist.init_process_group(backend, init_method=f"tcp://{addr}:{port}",
                                rank=rank, world_size=world)
    return True


def shutdown() -> None:
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist is not None and dist.is_initialized() else 0


def all_sum(x):
    """没有分布式时直接返回输入（恒等），保证单机路径不变。"""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def column_partition(weight, world: int, index: int):
    """按输入维（最后一维）切一块：每块算 y 的一个偏和。"""
    cols = weight.shape[1]
    per = (cols + world - 1) // world
    lo = min(index * per, cols)
    hi = min(lo + per, cols)
    return weight[:, lo:hi], lo, hi


if __name__ == "__main__":
    w = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    x = torch.ones(1, 12)
    full = x @ w.T
    total = torch.zeros_like(full)
    for i in range(3):
        wi, lo, hi = column_partition(w, 3, i)
        total += x[:, lo:hi] @ wi.T
    total = all_sum(total)
    diff = float((total - full).abs().max())
    print("column split diff", diff)
    assert diff < 1e-5
    print("comm ok")
