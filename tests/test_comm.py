"""两进程 gloo：列块各算一半，all_reduce 后必须等于整矩阵乘。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import torch
import torch.multiprocessing as mp
from tile_comm import all_sum, column_partition, init, shutdown


def worker(rank, world, port, out):
    init("gloo", rank, world, "127.0.0.1", port)
    w = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    x = torch.ones(1, 12)
    wi, lo, hi = column_partition(w, world, rank)
    total = all_sum(x[:, lo:hi] @ wi.T)
    shutdown()
    out[rank] = total.tolist()


if __name__ == "__main__":
    out = mp.Manager().dict()
    mp.spawn(worker, args=(2, 29511, out), nprocs=2, join=True)
    w = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10
    ref = torch.ones(1, 12) @ w.T
    a, b = torch.tensor(out[0]), torch.tensor(out[1])
    print("proc0", a.tolist())
    print("proc1", b.tolist())
    print("ref  ", ref.tolist())
    assert torch.allclose(a, ref) and torch.allclose(b, ref)
    print("comm 2proc ok")
