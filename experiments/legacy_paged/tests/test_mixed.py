"""混合后端：同一次 forward 里，前两层走本地磁盘分片，后两层走伪远端。"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from model import Config, TinyGPT
from supervisor import Supervisor
from local_backend import layer_bytes, make_run
import shardfile

cfg = Config(n_embd=32, n_head=4, n_layer=4, block=16, vocab=64)
model = TinyGPT(cfg, rng=np.random.default_rng(0))
per = layer_bytes(cfg.n_embd)
idx = np.array([[1, 5, 9, 3]], dtype=np.int64)
T = idx.shape[1]
x0 = (model.forward(idx) and np.zeros((1, T, cfg.n_embd), np.float32))
mask = np.triu(np.ones((T, T), bool), 1)

spec = shardfile.write(model, tempfile.mkdtemp())
seen = {"local": 0, "remote": 0}

def remote_run(device, blocks, tokens):
    seen["remote"] += 1
    return tokens                      # 远端视为黑箱，不改 hidden state

local = make_run(model)
def local_run(device, blocks, tokens):
    seen["local"] += 1
    return local(device, blocks, tokens)

sup = Supervisor(spec, ["local0", "remote0"], 2 * per,
                 {"local0": local_run, "remote0": remote_run})
out = sup.forward({"x": x0, "mask": mask, "layer": 0})
print("dispatch", seen, "stages", len(sup.stages), "layer", out["layer"])
assert seen["remote"] == 2 and seen["local"] == 2      # 四层各走各的后端
assert out["layer"] == cfg.n_layer - 1
sup.close()
print("mixed backend ok")
