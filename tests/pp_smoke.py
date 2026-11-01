"""流水并行管道冒烟：CPU + gloo，两进程各持一半层，与全模型逐值对拍。

    python3 tests/pp_smoke.py
"""
import sys

sys.path.insert(0, '.')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from core import pp

NL, D, B, T = 8, 32, 4, 6
SEED_W, SEED_X = 1234, 7


def build():
    """同样的种子 → 两个进程生成完全相同的权重，便于各取各的层。"""
    torch.manual_seed(SEED_W)
    return [nn.Linear(D, D) for _ in range(NL)]


def run_layers(layers, x):
    with torch.no_grad():
        for layer in layers:
            x = torch.relu(layer(x))
    return x


def reference():
    torch.manual_seed(SEED_X)
    x = torch.randn(B, T, D)
    return run_layers(build(), x)


def worker(rank, world, port, out):
    pp.init(rank, world, port)
    lo, hi = pp.layers_of(rank, world, NL)
    torch.manual_seed(SEED_X)
    x = torch.randn(B, T, D)
    if rank > 0:                                  # 非首段：收上游激活
        x = pp.recv((B, T, D), torch.float32, rank - 1)
    x = run_layers(build()[lo:hi], x)             # 只算自己那段层
    if rank < world - 1:                          # 非末段：把结果往下传
        pp.send(x, rank + 1)
        out.put((rank, lo, hi))
    else:
        torch.save(x, '.tmp/pp_last.pt')
        out.put((rank, lo, hi))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29511, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    got = {}
    while not q.empty():
        r, lo, hi = q.get()
        got[r] = (lo, hi)
    want = reference()
    have = torch.load('.tmp/pp_last.pt', weights_only=True)
    diff = (want - have).abs().max().item()
    print('层切分 rank0=%s rank1=%s' % (got[0][:2], got[1][:2]))
    print('maxdiff %.3e → %s' % (diff, 'PASS' if diff < 1e-6 else 'FAIL'))
