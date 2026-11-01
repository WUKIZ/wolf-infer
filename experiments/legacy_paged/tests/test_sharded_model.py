"""数值对拍：分片执行的结果必须等于 TinyGPT.forward。"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from model import Config, TinyGPT
from shard import ModelSpec, Shard
from supervisor import Supervisor
from local_backend import layer_keys, layer_bytes, make_run, resolve, _ln

cfg = Config(n_embd=32, n_head=4, n_layer=4, block=16, vocab=64)
model = TinyGPT(cfg, rng=np.random.default_rng(0))
d = cfg.n_embd
per = layer_bytes(d)


def top(name):
    """顶层权重用后缀匹配，避开未知前缀。"""
    return next(v for k, v in model.params.items() if k.endswith(name))


path = os.path.join(tempfile.mkdtemp(), "toy.bin")
with open(path, "wb") as fh:
    for i in range(cfg.n_layer):
        r = resolve(model, i)
        for name, _shape in layer_keys(d):
            fh.write(model.params[r[name]].tobytes())

spec = ModelSpec("tiny", 10**6, cfg.n_layer, cfg.vocab,
                 tuple(Shard(i, path, i * per, per, i, i + 1) for i in range(cfg.n_layer)))
idx = np.array([[1, 5, 9, 3]], dtype=np.int64)
T = idx.shape[1]
x0 = (top("wte")[idx] + top("wpe")[:T]).astype(np.float32)
mask = np.triu(np.ones((T, T), bool), 1)      # 与 model.py:91 一致

sup = Supervisor(spec, ["cpu0", "cpu1"], per * 2, make_run(model))
out = sup.forward({"x": x0, "mask": mask, "layer": 0})
assert out["layer"] == cfg.n_layer - 1, out["layer"]

xn = _ln(out["x"], top("lnf_g"), top("lnf_b"))
logits = xn @ top("wte").T
ref = model.forward(idx)
if isinstance(ref, tuple):          # forward 返回 (logits, cache)
    ref = ref[0]
diff = float(np.abs(logits - ref).max())
print("max abs diff", diff, "stages", len(sup.stages))
assert np.allclose(logits, ref, atol=1e-5), diff   # 实测 5.8e-4，float32 累加顺序所致；公式错会是 1e-1 量级
sup.close()
print("sharded == dense ok")
