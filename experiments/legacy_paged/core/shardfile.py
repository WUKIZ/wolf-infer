"""分片文件：把模型权重按层写成独立文件，并生成可复用的 ModelSpec。

写：write(model, out_dir) -> ModelSpec
读：load_spec(out_dir) -> ModelSpec
布局与 local_backend.layer_keys 一致，所以 Pager 读出字节后可直接 unpack。
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import sys

import numpy as np

try:
    from .local_backend import layer_keys, resolve
    from .shard import ModelSpec, Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from local_backend import layer_keys, resolve
    from shard import ModelSpec, Shard

SPEC = "spec.json"


def write(model, out_dir: str) -> ModelSpec:
    os.makedirs(out_dir, exist_ok=True)
    d = model.cfg.n_embd
    shards: list[Shard] = []
    for i in range(model.cfg.n_layer):
        where = resolve(model, i)
        blocks = [np.asarray(model.params[where[n]], dtype=np.float32).tobytes()
                  for n, _ in layer_keys(d)]
        path = os.path.abspath(os.path.join(out_dir, f"layer_{i:04d}.bin"))
        with open(path, "wb") as fh:
            fh.write(b"".join(blocks))
        shards.append(Shard(i, path, 0, os.path.getsize(path), i, i + 1))
    spec = ModelSpec("sharded", int(sum(v.size for v in model.params.values())),
                     model.cfg.n_layer, model.cfg.vocab, tuple(shards))
    with open(os.path.join(out_dir, SPEC), "w", encoding="utf-8") as fh:
        json.dump({"name": spec.name, "n_params": spec.n_params, "n_layers": spec.n_layers,
                   "vocab": spec.vocab,
                   "shards": [[s.index, os.path.basename(s.path), s.offset, s.nbytes,
                               s.layer_from, s.layer_to] for s in spec.shards]}, fh, indent=1)
    write_sums(out_dir)
    return spec


def load_spec(out_dir: str) -> ModelSpec:
    with open(os.path.join(out_dir, SPEC), encoding="utf-8") as fh:
        raw = json.load(fh)
    shards = tuple(Shard(i, os.path.abspath(os.path.join(out_dir, name)), off, nb, lf, lt)
                   for i, name, off, nb, lf, lt in raw["shards"])
    return ModelSpec(raw["name"], raw["n_params"], raw["n_layers"], raw["vocab"], shards)


SUMS = "shards.sha256"


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(out_dir: str) -> dict[str, str]:
    """给每个分片记一份 sha256；跨平台搬运后据此判断字节是否完整。"""
    sums = {os.path.basename(p): _sha(p)
            for p in sorted(glob.glob(os.path.join(out_dir, "layer_*.bin")))}
    with open(os.path.join(out_dir, SUMS), "w", encoding="utf-8") as fh:
        json.dump(sums, fh, indent=1, sort_keys=True)
    return sums


def verify(out_dir: str) -> list[str]:
    """返回字节对不上的分片名；空列表即全部完好。"""
    with open(os.path.join(out_dir, SUMS), encoding="utf-8") as fh:
        want = json.load(fh)
    bad = []
    for name, digest in want.items():
        path = os.path.join(out_dir, name)
        if not os.path.exists(path) or _sha(path) != digest:
            bad.append(name)
    return bad


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    class C:
        n_embd, n_head, n_layer, block, vocab = 8, 2, 3, 8, 16

    params = {}
    for i in range(C.n_layer):
        for name, shape in layer_keys(C.n_embd):
            params[f"b{i}.{name}"] = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    m = SimpleNamespace(cfg=C(), params=params)

    d = tempfile.mkdtemp()
    spec = write(m, d)
    back = load_spec(d)
    assert back.n_layers == 3 and len(back.shards) == 3
    assert back.total_bytes == spec.total_bytes and back.check()
    print("ok", back.total_bytes, "bytes in", len(back.shards), "shards")
