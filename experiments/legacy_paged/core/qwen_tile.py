"""按矩阵行块执行 Qwen3 单层：预算可以小于单个矩阵。

每个矩阵单独一个文件，按输出维切成行块；行块算完直接拼接，不需要通信。
"""
import os
import sys

import torch

try:
    from .pager import Pager
    from .shard import Shard
    from .tile import linear_blocks, row_blocks
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pager import Pager
    from shard import Shard
    from tile import linear_blocks, row_blocks

MATRICES = (
    ("q", "self_attn.q_proj.weight"),
    ("k", "self_attn.k_proj.weight"),
    ("v", "self_attn.v_proj.weight"),
    ("o", "self_attn.o_proj.weight"),
    ("gate", "mlp.gate_proj.weight"),
    ("up", "mlp.up_proj.weight"),
    ("down", "mlp.down_proj.weight"),
)


def write_tiles(w: dict, layer: int, out_dir: str, rows_per_block: int = 512) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for tag, key in MATRICES:
        t = w[f"model.layers.{layer}.{key}"].detach().cpu().contiguous()
        rows, cols = t.shape
        path = os.path.abspath(os.path.join(out_dir, f"L{layer:02d}_{tag}.bin"))
        with open(path, "wb") as fh:
            fh.write(t.view(torch.uint8).numpy().tobytes())
        out[tag] = (path, rows, cols, row_blocks(path, rows, cols, rows_per_block, elem_bytes=2))
    return out


def layer_forward_tiled(t: dict, tiles: dict, x, cfg, pager, base: int = 100):
    """t: 归一化权重；tiles: write_tiles 的结果；x: (B,T,d) bf16。"""
    from qwen_backend import rms, rope

    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    dh, eps = cfg["head_dim"], cfg["rms_norm_eps"]
    B, T, d = x.shape

    def lin(name, inp, tag):
        path, _rows, _cols, blocks = tiles[name]
        return linear_blocks(pager, path, blocks, inp, tag, dtype=torch.bfloat16)

    h = rms(x, t["input_layernorm.weight"], eps)
    q = lin("q", h, base).view(B, T, H, dh).transpose(1, 2)
    k = lin("k", h, base + 200).view(B, T, KV, dh).transpose(1, 2)
    v = lin("v", h, base + 400).view(B, T, KV, dh).transpose(1, 2)
    q = rope(rms(q, t["self_attn.q_norm.weight"], eps), cfg["rope_theta"])
    k = rope(rms(k, t["self_attn.k_norm.weight"], eps), cfg["rope_theta"])
    k = k.repeat_interleave(H // KV, dim=1)
    v = v.repeat_interleave(H // KV, dim=1)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + lin("o", a.transpose(1, 2).reshape(B, T, d), base + 600)
    h = rms(x, t["post_attention_layernorm.weight"], eps)
    g = torch.nn.functional.silu(lin("gate", h, base + 800)) * lin("up", h, base + 1000)
    return x + lin("down", g, base + 1200)
