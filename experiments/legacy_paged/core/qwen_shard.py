"""把 Qwen3 按层落盘并逐层执行：每层 11 个张量，布局沿用权重自己的键名。"""
import os
import sys

import torch
from core.tmpdir import TMP

try:
    from .qwen_backend import layer_forward, rms
    from .shard import ModelSpec, Shard
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_backend import layer_forward, rms
    from shard import ModelSpec, Shard

ORDER = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def write_shards(w: dict, n_layer: int, out_dir: str) -> ModelSpec:
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for i in range(n_layer):
        p = f"model.layers.{i}."
        buf = b"".join(w[p + k].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                      for k in ORDER)
        path = os.path.abspath(os.path.join(out_dir, f"layer_{i:04d}.bin"))
        with open(path, "wb") as fh:
            fh.write(buf)
        shards.append(Shard(i, path, 0, len(buf), i, i + 1))
    return ModelSpec("qwen1.7b", 0, n_layer, 151936, tuple(shards))


def make_run(cfg, shapes, dtype):
    def run(dev, blocks, tokens):
        layer = tokens["layer"]
        flat = torch.frombuffer(bytearray(blocks[0]), dtype=torch.bfloat16)
        off, t = 0, {}
        for k in ORDER:
            shape = tuple(shapes[k])
            num = 1
            for s in shape:
                num *= s
            t[k] = flat[off:off + num].reshape(shape).to(dev, dtype)
            off += num
        tokens["x"] = layer_forward(t, tokens["x"], tokens["mask"], cfg)
        return tokens
    return run


def logits(cfg, w, ids, budget_bytes, device="cuda"):
    from supervisor import Supervisor
    n_layer = cfg["num_hidden_layers"]
    shapes = {k: tuple(w["model.layers.0." + k].shape) for k in ORDER}
    dtype = w["model.embed_tokens.weight"].dtype
    spec = write_shards(w, n_layer, TMP + "/qwen_shards")
    ids_t = torch.tensor([ids], device=device)
    T = ids_t.shape[1]
    x = w["model.embed_tokens.weight"][ids_t]
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
    sup = Supervisor(spec, [device], budget_bytes, make_run(cfg, shapes, dtype))
    out = sup.forward({"x": x, "mask": mask, "layer": 0})
    sup.close()
    xn = rms(out["x"], w["model.norm.weight"], cfg["rms_norm_eps"])
    return xn @ w["model.embed_tokens.weight"].T


if __name__ == "__main__":
    from qwen_backend import load, logits as dense_logits

    d = "/data/models/Qwen3-1.7B"
    cfg, w = load(d)
    w = {k: v.to("cuda", torch.bfloat16) for k, v in w.items()}
    ids = [40, 100, 200, 300]
    ref = dense_logits(cfg, w, ids, dtype=torch.bfloat16)
    got = logits(cfg, w, ids, budget_bytes=2 * 130_000_000)
    print("sharded vs dense", float((got - ref).abs().max()))
    print("qwen shard ok")


def write_shards_int8(w: dict, n_layer: int, out_dir: str):
    """int8 + 每行 absmax scale：体积减半，解包只需一次乘加。"""
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for i in range(n_layer):
        p = f"model.layers.{i}."
        qbuf, sbuf, total = [], [], 0
        for k in ORDER:
            t = w[p + k].detach().cpu().float()
            flat = t.reshape(1, -1) if t.dim() == 1 else t          # 一维按整条一行
            s = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            q = (flat / s[:, None]).round().clamp(-127, 127).to(torch.int8)
            qb, sb = q.numpy().tobytes(), s.to(torch.float16).numpy().tobytes()
            qbuf.append(qb)
            sbuf.append(sb)
            total += len(qb)
        path = os.path.abspath(os.path.join(out_dir, f"layer_{i:04d}.q8"))
        with open(path, "wb") as fh:
            fh.write(b"".join(qbuf))
        with open(path + ".scale", "wb") as fh:
            fh.write(b"".join(sbuf))
        shards.append(Shard(i, path, 0, total, i, i + 1))
    return ModelSpec("qwen1.7b-int8", 0, n_layer, 151936, tuple(shards))


if __name__ == "__main__":
    pass


def make_run_int8(cfg, shapes, dtype, scale_dir):
    def run(dev, blocks, tokens):
        layer = tokens["layer"]
        raw = torch.frombuffer(bytearray(blocks[0]), dtype=torch.int8)
        with open(os.path.join(scale_dir, f"layer_{layer:04d}.q8.scale"), "rb") as fh:
            sraw = fh.read()
        off = soff = 0
        t = {}
        for k in ORDER:
            shape = tuple(shapes[k])
            numel = 1
            for dim in shape:
                numel *= dim
            rows = 1 if len(shape) == 1 else shape[0]
            cols = numel // rows
            n = numel
            q = raw[off:off + n].reshape(rows, cols).to(torch.float32)
            sc = torch.frombuffer(bytearray(sraw[soff:soff + rows * 2]), dtype=torch.float16)
            t[k] = (q * sc.float()[:, None]).reshape(shape).to(dtype).to(dev)
            off += n
            soff += rows * 2
        tokens["x"] = layer_forward(t, tokens["x"], tokens["mask"], cfg)
        return tokens
    return run
