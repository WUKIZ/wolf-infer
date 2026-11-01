"""从磁盘验证：内存权重清零后，靠分片文件恢复到等价结果。"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from model import Config, TinyGPT
from supervisor import Supervisor
from local_backend import layer_bytes, make_run, _ln
import shardfile

cfg = Config(n_embd=32, n_head=4, n_layer=4, block=16, vocab=64)
model = TinyGPT(cfg, rng=np.random.default_rng(0))
per = layer_bytes(cfg.n_embd)


def top(name):
    return next(v for k, v in model.params.items() if k.endswith(name))


idx = np.array([[1, 5, 9, 3]], dtype=np.int64)
T = idx.shape[1]
wte = top("wte").copy()
lnf_g, lnf_b = top("lnf_g").copy(), top("lnf_b").copy()
x0 = (wte[idx] + top("wpe")[:T]).astype(np.float32)
mask = np.triu(np.ones((T, T), bool), 1)
ref = model.forward(idx)[0]

d = tempfile.mkdtemp()
spec = shardfile.write(model, d)
for k in model.params:                    # 清空内存权重，逼它必须从磁盘读
    model.params[k][...] = 0.0

back = shardfile.load_spec(d)
sup = Supervisor(back, ["cpu0"], 2 * per, make_run(model))   # 预算只放两层
out = sup.forward({"x": x0, "mask": mask, "layer": 0})

xn = _ln(out["x"], lnf_g, lnf_b)
logits = xn @ wte.T
diff = float(np.abs(logits - ref).max())
print("from-disk diff", diff, "evictions", sup.pager.stats.evictions, "stages", len(sup.stages))
assert np.allclose(logits, ref, atol=1e-5), diff
assert sup.pager.stats.evictions >= 1
assert sup.pager.resident_bytes() <= sup.pager.budget
sup.close()
print("from-disk == dense ok")
