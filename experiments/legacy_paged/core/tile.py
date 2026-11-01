"""矩阵块：把权重按输出维切成行块，逐块换入，不需要通信就能降低常驻。

行块：y[:, r0:r1] = x @ W[r0:r1].T —— 各块独立，拼起来即完整结果。
列块（输入维切分）才需要 all-reduce，那是跨设备时的事，本文件不做。
"""
import os
import sys

import numpy as np
import torch

try:
    from .pager import Pager
    from .shard import Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pager import Pager
    from shard import Shard


def row_blocks(path: str, rows: int, cols: int, rows_per_block: int,
               elem_bytes: int = 4) -> list[tuple[int, int, int, int]]:
    """返回 (r0, r1, offset, nbytes)：行在内存里连续，所以一块就是一段字节。"""
    out = []
    for r0 in range(0, rows, rows_per_block):
        r1 = min(r0 + rows_per_block, rows)
        out.append((r0, r1, r0 * cols * elem_bytes, (r1 - r0) * cols * elem_bytes))
    return out


def linear_blocks(pager, path: str, blocks, x, tag: int = 0, dtype=torch.float32):
    """x: (B, cols) 张量；逐块读入、算完拼接，返回 (B, rows)。"""
    parts = []
    for i, (r0, r1, off, nb) in enumerate(blocks):
        view = pager.view(Shard(tag + i, path, off, nb, 0, 1))
        try:
            w = torch.frombuffer(bytearray(view), dtype=dtype).reshape(r1 - r0, x.shape[-1]).to(x.device)
            parts.append(x @ w.T)
        finally:
            view.release()
    return torch.cat(parts, dim=-1)


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    rows, cols = 256, 128
    w = rng.normal(0, 0.05, (rows, cols)).astype(np.float32)
    x = rng.normal(0, 1, (1, cols)).astype(np.float32)
    ref = torch.tensor(x) @ torch.tensor(w).T

    d = tempfile.mkdtemp()
    p = os.path.join(d, "w.bin")
    with open(p, "wb") as fh:
        fh.write(w.tobytes())
    blocks = row_blocks(p, rows, cols, 64)
    pg = Pager(budget_bytes=64 * cols * 4)      # 只放得下一块
    got = linear_blocks(pg, p, blocks, torch.tensor(x))
    pg.close()
    diff = float((got - ref).abs().max())
    print("tile diff", diff, "blocks", len(blocks))
    assert diff < 1e-5
    print("tile ok")
