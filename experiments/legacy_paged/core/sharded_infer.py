"""分片推理封装：不碰核心，靠包装提供分片前向与生成。

    si = ShardedInfer(model, out_dir, ["cpu0"], budget_bytes)
    logits = si.logits(np.array([[1, 2, 3]]))
    ids = si.generate([1, 2, 3], n=16)

TinyGPT 保持原样，权重仍留在内存里当兜底；分片路径只在需要时按预算换页。
"""
from __future__ import annotations
import os
import sys

import numpy as np

try:
    from . import shardfile
    from .local_backend import _ln, make_run
    from .supervisor import Supervisor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shardfile
    from local_backend import _ln, make_run
    from supervisor import Supervisor


class ShardedInfer:
    def __init__(self, model, out_dir: str, devices: list[str], budget_bytes: int) -> None:
        self.model = model
        self.spec = shardfile.load_spec(out_dir) if isinstance(out_dir, str) else out_dir
        self.sup = Supervisor(self.spec, devices, budget_bytes, make_run(model))

    def _w(self, name: str) -> np.ndarray:
        return next(v for k, v in self.model.params.items() if k.endswith(name))

    def logits(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        T = idx.shape[1]
        x = (self._w("wte")[idx] + self._w("wpe")[:T]).astype(np.float32)
        mask = np.triu(np.ones((T, T), bool), 1)
        out = self.sup.forward({"x": x, "mask": mask, "layer": 0})
        xn = _ln(out["x"], self._w("lnf_g"), self._w("lnf_b"))
        head = next((v for k, v in self.model.params.items() if k.endswith("head")), None)
        return xn @ (head if head is not None else self._w("wte").T)

    def generate(self, idx: list[int], n: int = 32, temperature: float = 0.0) -> list[int]:
        seq = list(idx)
        block = self.model.cfg.block
        for _ in range(n):
            window = np.array([seq[-block:]], dtype=np.int64)
            lg = self.logits(window)[0, -1]
            if temperature > 0:
                p = np.exp((lg - lg.max()) / temperature)
                seq.append(int(np.random.choice(len(p), p=p / p.sum())))
            else:
                seq.append(int(lg.argmax()))
        return seq

    def close(self) -> None:
        self.sup.close()


if __name__ == "__main__":
    import tempfile

    from model import Config, TinyGPT

    cfg = Config(n_embd=32, n_head=4, n_layer=4, block=16, vocab=64)
    m = TinyGPT(cfg, rng=np.random.default_rng(0))
    idx = np.array([[1, 5, 9, 3]], dtype=np.int64)
    ref = m.forward(idx)[0]
    si = ShardedInfer(m, shardfile.write(m, tempfile.mkdtemp()), ["cpu0"], 2 * 49664 * 4)
    got = si.logits(idx)
    diff = float(np.abs(got - ref).max())
    ids = si.generate([1, 5], n=5)
    print("sharded-infer diff", diff, "gen len", len(ids))
    assert np.allclose(got, ref, atol=1e-5) and len(ids) == 7
    si.close()
    print("sharded infer ok")
