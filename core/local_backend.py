"""分片后端：一片字节 → 某一层权重 → 只跑这一层。

规范名固定为 ln1_g ln1_b qkv proj ln2_g ln2_b fc fc2，
真实键名按「最后一个 b{layer}. 之后的部分」匹配，前缀是什么都不影响。
一层 = (16·d + 48·d²) 字节（float32）。
"""
from __future__ import annotations
import numpy as np

EPS = 1e-5


def layer_keys(d: int) -> list[tuple[str, tuple[int, ...]]]:
    return [("ln1_g", (d,)), ("ln1_b", (d,)), ("qkv", (d, 3 * d)), ("proj", (d, d)),
            ("ln2_g", (d,)), ("ln2_b", (d,)), ("fc", (d, 4 * d)), ("fc2", (4 * d, d))]


def layer_bytes(d: int) -> int:
    return sum(int(np.prod(s)) for _, s in layer_keys(d)) * 4


def _clean(key: str) -> str:
    """去掉零宽/不可打印字符——真实键名里混了它们，肉眼与正常字符无法区分。"""
    return "".join(ch for ch in key if ch.isprintable() and not ch.isspace())




def _ln(x, g, b):
    """与 model.ln_forward 同形：先算倒数再乘，避免除法带来的末位差异。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    return (x - mu) * inv * g + b


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def unpack(model, layer: int, block) -> None:
    d = model.cfg.n_embd
    flat = np.frombuffer(block, dtype=np.float32)
    need = layer_bytes(d) // 4
    if flat.size != need:
        raise ValueError(f"层 {layer} 需要 {need} 个 float32，拿到 {flat.size}")
    where = resolve(model, layer)
    off = 0
    for name, shape in layer_keys(d):
        n = int(np.prod(shape))
        model.params[where[name]] = flat[off:off + n].reshape(shape).copy()
        off += n


def forward_layer(model, layer: int, x, mask):
    c = model.cfg
    W = {name: model.params[key] for name, key in resolve(model, layer).items()}
    B, T = x.shape[0], x.shape[1]
    d, H = c.n_embd, c.n_head
    dh = d // H
    h1 = _ln(x, W["ln1_g"], W["ln1_b"])
    qkv = h1 @ W["qkv"]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    att = _softmax(np.where(mask, -1e9, q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh))
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    x = x + y @ W["proj"]
    h2 = _ln(x, W["ln2_g"], W["ln2_b"])
    f_pre = h2 @ W["fc"]
    x = x + np.maximum(f_pre, 0) @ W["fc2"]
    return x


def make_run(model):
    def run(device, blocks, tokens):
        layer = tokens["layer"]
        unpack(model, layer, blocks[0])
        tokens["x"] = forward_layer(model, layer, tokens["x"], tokens["mask"])
        # 层号由 Supervisor 逐片指定，这里不自增
        return tokens
    return run


if __name__ == "__main__":
    from types import SimpleNamespace

    class C:
        n_embd, n_head, n_layer, block, vocab = 32, 4, 4, 16, 64

    p = {}
    for i in range(C.n_layer):
        for name, shape in layer_keys(C.n_embd):
            p[f"zz.b{i}.{name}"] = np.zeros(shape, np.float32)      # 故意加前缀
    m = SimpleNamespace(cfg=C(), params=p)
    blk = np.zeros(layer_bytes(32) // 4, dtype=np.float32).tobytes()
    unpack(m, 0, blk)
    y = forward_layer(m, 0, np.zeros((1, 4, 32), np.float32),
                      np.tril(np.ones((1, 1, 4, 4), bool), 1))
    assert y.shape == (1, 4, 32) and len(resolve(m, 0)) == 8
    print("ok", layer_bytes(32), "bytes/layer")


# —— 兜底：真实键名不按 b{layer}. 命名时，按 __init__ 的插入顺序分组 ——
_LAYER_NAMES = ("ln1_g", "ln1_b", "qkv", "proj", "ln2_g", "ln2_b", "fc", "fc2")


def resolve(model, layer: int) -> dict[str, str]:
    """规范名 → 真实键名；裸模型（键名即规范名）走恒等映射。"""
    keys = list(model.params)
    names = [n for n, _ in layer_keys(model.cfg.n_embd)]
    if all(n in model.params for n in names):
        return {n: n for n in names}      # 裸模型：键名与规范名完全一致
    pre = f"b{layer}."
    out: dict[str, str] = {}
    for k in keys:
        c = _clean(k)
        i = c.find(pre)
        if i >= 0:
            out[c[i + len(pre):]] = k
    if len(out) == len(_LAYER_NAMES):
        return out
    start = 4 + layer * len(_LAYER_NAMES)
    if len(keys) < start + len(_LAYER_NAMES):
        raise KeyError(f"层 {layer} 权重键不足: total={len(keys)}")
    return {n: keys[start + j] for j, n in enumerate(_LAYER_NAMES)}
