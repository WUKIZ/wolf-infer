"""流水并行骨架：层切分与激活传递。

只提供管道原语，不改 BatchEngine；接线由调用方完成。
后端用 gloo 时可在纯 CPU 上验证（不需要 GPU）。
"""
import os

import torch
import torch.distributed as dist


def init(rank, world, port=29511, backend='gloo'):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(port))
    dist.init_process_group(backend, rank=rank, world_size=world)
    return rank, world


def layers_of(rank, world, n_layers):
    """均分，余数给前面的 rank。返回 [lo, hi)。"""
    per = (n_layers + world - 1) // world
    lo = min(rank * per, n_layers)
    return lo, min(lo + per, n_layers)


def send(x, dst):
    dist.send(x.detach().contiguous(), dst=dst)


def recv(shape, dtype, src, device='cpu'):
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src)
    return buf


def relay(x, rank, world, shape, dtype, device='cpu'):
    """中段：先收（非首段）、再发（非末段）；首段只发，末段只收。"""
    if rank > 0:
        x = recv(shape, dtype, rank - 1, device)
    if rank < world - 1:
        send(x, rank + 1)
    return x
