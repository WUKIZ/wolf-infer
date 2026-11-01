"""行块执行的一层必须等于整矩阵计算的同一层。"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import torch
from pager import Pager
from qwen_backend import layer_forward, load
from qwen_tile import layer_forward_tiled, write_tiles

d = "/data/models/Qwen3-1.7B"
cfg, w = load(d)
dev = "cuda"
w = {k: v.to(dev, torch.bfloat16) for k, v in w.items()}

L = 0
p = f"model.layers.{L}."
t = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
tiles = write_tiles(w, L, tempfile.mkdtemp(), rows_per_block=512)

blk = 512 * cfg["hidden_size"] * 2
pager = Pager(budget_bytes=4 * blk)          # 只放两块（4 MB）
x = torch.randn(1, 4, cfg["hidden_size"], device=dev, dtype=torch.bfloat16)
mask = torch.triu(torch.ones(4, 4, device=dev, dtype=torch.bool), 1)

ref = layer_forward(t, x, mask, cfg)
got = layer_forward_tiled(t, tiles, x, cfg, pager)
pager.close()
diff = float((got - ref).abs().max())
print("tiled vs dense", diff, "evictions", pager.stats.evictions)
assert diff < 1e-1, diff
print("qwen tile ok")
