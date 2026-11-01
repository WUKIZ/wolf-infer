"""模型后端。真实现，不是壳。

numpy 手写的 decoder-only transformer：
  token 嵌入 + 位置嵌入
  N 层 [ LN → 因果自注意力 → 残差 → LN → FFN → 残差 ]
  LN → 词表投影 → logits
训练：交叉熵 + Adam。反向传播逐项手推，含 LayerNorm 的完整梯度。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20261022)


def ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    return xhat * g + b, (xhat, std)


def ln_backward(dy: np.ndarray, cache, g: np.ndarray):
    xhat, std = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dnorm = dy * g
    dx = (dnorm - dnorm.mean(-1, keepdims=True)
          - xhat * (dnorm * xhat).mean(-1, keepdims=True)) / std
    return dx, dg, db


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


@dataclass
class Config:
    vocab: int = 0
    block: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64


class TinyGPT:
    def __init__(self, cfg: Config, rng: Optional[np.random.Generator] = None,
                 dtype=np.float32) -> None:
        self.cfg = cfg
        self.dtype = dtype
        r = rng or RNG
        d, v = cfg.n_embd, cfg.vocab
        f = lambda *shape, scale: r.normal(0, scale, shape).astype(dtype)  # noqa: E731
        resid = 0.02 / math.sqrt(2 * cfg.n_layer)   # nanoGPT：残差投影按层数缩放
        P: Dict[str, np.ndarray] = {
            "wte": f(v, d, scale=0.02),
            "wpe": f(cfg.block, d, scale=0.01),
            "lnf_g": np.ones(d, dtype=dtype),
            "lnf_b": np.zeros(d, dtype=dtype),
        }
        for i in range(cfg.n_layer):
            p = f"h{i}."
            P |= {
                p + "ln1_g": np.ones(d, dtype=dtype), p + "ln1_b": np.zeros(d, dtype=dtype),
                p + "qkv": f(d, 3 * d, scale=0.02),
                p + "proj": f(d, d, scale=resid),
                p + "ln2_g": np.ones(d, dtype=dtype), p + "ln2_b": np.zeros(d, dtype=dtype),
                p + "fc": f(d, 4 * d, scale=0.02),
                p + "fc2": f(4 * d, d, scale=resid),
            }
        self.params = P
        self.m = {k: np.zeros_like(a) for k, a in P.items()}
        self.v = {k: np.zeros_like(a) for k, a in P.items()}
        self.t = 0

    # ---------------- 前向 ----------------
    def forward(self, idx: np.ndarray):
        c, P = self.cfg, self.params
        B, T = idx.shape
        d, H, dh = c.n_embd, c.n_head, c.n_embd // c.n_head
        x = P["wte"][idx] + P["wpe"][:T]
        mask = np.triu(np.ones((T, T), bool), 1)
        cache: Dict[str, object] = {"idx": idx, "layers": []}

        for i in range(c.n_layer):
            p = f"h{i}."
            h1, ln1 = ln_forward(x, P[p + "ln1_g"], P[p + "ln1_b"])
            qkv = h1 @ P[p + "qkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            att = softmax(np.where(mask, np.float32(-1e9),
                                   q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)))
            y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            x = x + y @ P[p + "proj"]

            h2, ln2 = ln_forward(x, P[p + "ln2_g"], P[p + "ln2_b"])
            f_pre = h2 @ P[p + "fc"]
            x = x + np.maximum(f_pre, 0) @ P[p + "fc2"]

            cache["layers"].append({"ln1": ln1, "h1": h1, "q": q, "k": k, "v": v,
                                    "att": att, "ln2": ln2, "h2": h2, "f_pre": f_pre})

        xn, lnf = ln_forward(x, P["lnf_g"], P["lnf_b"])
        cache |= {"x_final": x, "xn": xn, "lnf": lnf}
        return xn @ (P["head"] if "head" in P else P["wte"].T), cache

    # ---------------- 反向 ----------------
    def backward(self, cache, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        c, P = self.cfg, self.params
        B, T = dlogits.shape[:2]
        d, H, dh = c.n_embd, c.n_head, c.n_embd // c.n_head
        g: Dict[str, np.ndarray] = {}

        xn = cache["xn"]
        if "head" in P:                     # 旧权重：独立输出头
            g["head"] = xn.reshape(-1, d).T @ dlogits.reshape(-1, c.vocab)
            dxn = dlogits @ P["head"].T
        else:                               # 新权重：与 wte 绑定
            dxn = dlogits @ P["wte"]

        dx, g["lnf_g"], g["lnf_b"] = ln_backward(dxn, cache["lnf"], P["lnf_g"])

        for i in reversed(range(c.n_layer)):
            p = f"h{i}."
            L = cache["layers"][i]

            # FFN 支路
            f = np.maximum(L["f_pre"], 0)
            g[p + "fc2"] = f.reshape(-1, 4 * d).T @ dx.reshape(-1, d)
            df = (dx @ P[p + "fc2"].T) * (L["f_pre"] > 0)
            g[p + "fc"] = L["h2"].reshape(-1, d).T @ df.reshape(-1, 4 * d)
            dxh2 = df @ P[p + "fc"].T
            dx2, g[p + "ln2_g"], g[p + "ln2_b"] = ln_backward(dxh2, L["ln2"], P[p + "ln2_g"])

            # 注意力支路：x_mid 同时喂给 proj 与 ln2，上游梯度要含 dx2
            dx_att = dx + dx2
            y = (L["att"] @ L["v"]).transpose(0, 2, 1, 3).reshape(B, T, d)
            g[p + "proj"] = y.reshape(-1, d).T @ dx_att.reshape(-1, d)
            dattn = (dx_att @ P[p + "proj"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

            datt = dattn @ L["v"].transpose(0, 1, 3, 2)
            dv = L["att"].transpose(0, 1, 3, 2) @ dattn
            datt = L["att"] * (datt - (datt * L["att"]).sum(-1, keepdims=True))
            dq = datt @ L["k"] / math.sqrt(dh)
            dk = datt.transpose(0, 1, 3, 2) @ L["q"] / math.sqrt(dh)

            dqkv = np.concatenate(
                [dq.transpose(0, 2, 1, 3).reshape(B, T, d),
                 dk.transpose(0, 2, 1, 3).reshape(B, T, d),
                 dv.transpose(0, 2, 1, 3).reshape(B, T, d)], axis=-1)
            g[p + "qkv"] = L["h1"].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
            dxh1 = dqkv @ P[p + "qkv"].T
            dx1, g[p + "ln1_g"], g[p + "ln1_b"] = ln_backward(dxh1, L["ln1"], P[p + "ln1_g"])

            dx = dx + dx1 + dx2

        idx = cache["idx"]
        g["wte"] = np.zeros_like(P["wte"])
        np.add.at(g["wte"], idx.reshape(-1), dx.reshape(-1, d))
        if "head" not in P:                 # 绑定后 embedding 还要吃一份输出头梯度
            g["wte"] += dlogits.reshape(-1, c.vocab).T @ xn.reshape(-1, d)
        g["wpe"] = np.zeros_like(P["wpe"])
        np.add.at(g["wpe"], np.arange(T), dx.sum(0))
        return g

    # ---------------- 训练 ----------------
    def step(self, batch: np.ndarray, lr: float = 3e-3,
             b1: float = 0.9, b2: float = 0.99, eps: float = 1e-8,
             weight_decay: float = 0.0, clip: float = 1.0,
             mask: Optional[np.ndarray] = None) -> float:
        x, y = batch[:, :-1], batch[:, 1:]
        logits, cache = self.forward(x)
        B, T, V = logits.shape
        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        if mask is None:
            mask = np.ones((B, T), dtype=np.float64)
        w = mask.reshape(-1).astype(np.float64)
        denom = max(1.0, float(w.sum()))
        picked = logp.reshape(-1, V)[np.arange(B * T), y.reshape(-1)]
        loss = -float((picked * w).sum() / denom)

        dlogits = np.exp(logp)
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1
        dlogits *= mask[:, :, None]
        dlogits /= denom
        grads = self.backward(cache, dlogits)
        # 梯度裁剪（nanoGPT 做法）：小模型偶发尖峰会把权重打飞，裁到范数 1.0
        total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if total > clip:
            scale = clip / total
            for k in grads:
                grads[k] *= scale

        self.t += 1
        for k, gr in grads.items():
            if k not in self.m:             # 旧权重载入时才补上的 head，状态要现建
                self.m[k] = np.zeros_like(gr)
                self.v[k] = np.zeros_like(gr)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            # 解耦权重衰减（AdamW 那一路）：语料只有十几 KB，不加这个
            # 两百步就把训练集背下来了，验证损失随即回升。
            self.params[k] -= lr * (mhat / (np.sqrt(vhat) + eps)
                                    + weight_decay * self.params[k])
        return loss

    def batch_loss(self, x: np.ndarray, y: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
        """只算损失，不更新。mask=1 的 token 才计入（问句不算）。"""
        logits, _ = self.forward(x)
        B, T, V = logits.shape
        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        if mask is None:
            mask = np.ones((B, T), dtype=np.float64)
        w = mask.reshape(-1).astype(np.float64)
        denom = max(1.0, float(w.sum()))
        picked = logp.reshape(-1, V)[np.arange(B * T), y.reshape(-1)]
        return -float((picked * w).sum() / denom)

    # ---------------- 推理 ----------------
    # ---------------- 增量推理 ----------------
    def init_past(self, cache) -> Dict[str, object]:
        """把一次全量前向的 K/V 收成增量状态；之后每个 token 只算一次。"""
        return {"k": [L["k"] for L in cache["layers"]],
                "v": [L["v"] for L in cache["layers"]],
                "x": cache["x_final"][:, -1:, :],
                "t": cache["x_final"].shape[1]}

    def forward_step(self, token: int, past):
        P, c = self.params, self.cfg
        H = c.n_head
        dh = c.n_embd // H
        pos = int(past["t"])
        x = (P["wte"][[token]] + P["wpe"][[pos]]).reshape(1, 1, -1)
        ks, vs = [], []
        for i in range(c.n_layer):
            p = f"h{i}."
            h1, _ = ln_forward(x, P[p + "ln1_g"], P[p + "ln1_b"])
            q, k, v = np.split(h1 @ P[p + "qkv"], 3, axis=-1)
            q = q.reshape(1, 1, H, dh).transpose(0, 2, 1, 3)
            k = k.reshape(1, 1, H, dh).transpose(0, 2, 1, 3)
            v = v.reshape(1, 1, H, dh).transpose(0, 2, 1, 3)
            K = np.concatenate([past["k"][i], k], axis=2)
            V = np.concatenate([past["v"][i], v], axis=2)
            att = softmax(q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh))
            y = (att @ V).transpose(0, 2, 1, 3).reshape(1, 1, c.n_embd)
            x = x + y @ P[p + "proj"]
            h2, _ = ln_forward(x, P[p + "ln2_g"], P[p + "ln2_b"])
            x = x + np.maximum(h2 @ P[p + "fc"], 0) @ P[p + "fc2"]
            ks.append(K)
            vs.append(V)
        xn, _ = ln_forward(x, P["lnf_g"], P["lnf_b"])
        return (xn @ (P["head"] if "head" in P else P["wte"].T))[0, -1], \
            {"k": ks, "v": vs, "x": x, "t": pos + 1}

    @staticmethod
    def _probs(logits, history, temperature, top_k, repetition_penalty, no_repeat_ngram):
        """温度 → 重复惩罚 → 禁 n-gram → top-k。全部作用在 logits 上。"""
        z = logits.astype(np.float64) / max(temperature, 1e-6)
        if repetition_penalty and repetition_penalty != 1.0:
            for t in set(history[-64:]):
                z[t] = z[t] / repetition_penalty if z[t] > 0 else z[t] * repetition_penalty
        if no_repeat_ngram and no_repeat_ngram > 0 and len(history) >= no_repeat_ngram:
            gram = no_repeat_ngram
            prefix = tuple(history[-(gram - 1):]) if gram > 1 else ()
            banned = {history[i + gram - 1]
                      for i in range(len(history) - gram + 1)
                      if tuple(history[i:i + gram - 1]) == prefix}
            for t in banned:
                z[t] = -1e9
        if top_k and 0 < top_k < z.size:
            z[z < np.sort(z)[-top_k]] = -1e9
        p = np.exp(z - z.max())
        return p / p.sum()

    def generate(self, idx: List[int], n: int, temperature: float = 0.8,
                 top_k: int = 40, repetition_penalty: float = 1.15,
                 no_repeat_ngram: int = 3,
                 rng: Optional[np.random.Generator] = None) -> List[int]:
        """带 KV cache 的采样：每步只前向一个新 token。"""
        r = rng or RNG
        out = list(idx)
        ctx = out[-self.cfg.block:]
        logits, cache = self.forward(np.array([ctx]))
        past = self.init_past(cache)
        logits = logits[0, -1]
        for _ in range(n):
            p = self._probs(logits, out, temperature, top_k,
                            repetition_penalty, no_repeat_ngram)
            tok = int(r.choice(len(p), p=p))
            out.append(tok)
            if past["t"] >= self.cfg.block:
                # 窗口满：用最后 block 个 token 重算一次，位置回到 0..block-1
                lg, cache = self.forward(np.array([out[-self.cfg.block:]]))
                past = self.init_past(cache)
                logits = lg[0, -1]
            else:
                logits, past = self.forward_step(tok, past)
        return out[len(idx):]

    # ---------------- 存取 ----------------
    def save(self, path: Path, chars: Optional[List[str]] = None,
             tok: Optional[dict] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 用文件句柄写，否则 numpy 会自作主张加 .npz 后缀
        with path.open("wb") as fh:
            np.savez_compressed(
                fh,
                cfg=json.dumps(self.cfg.__dict__),
                chars=json.dumps(chars or [], ensure_ascii=False),
                # 分词器随权重走：BPE 的合并表一旦和权重错位，id 全废
                tok=json.dumps(tok or {}, ensure_ascii=False),
                **{f"p_{k}": a for k, a in self.params.items()},
            )

    @classmethod
    def load(cls, path: Path) -> "TinyGPT":
        z = np.load(path, allow_pickle=False)
        cfg = Config(**json.loads(str(z["cfg"])))
        obj = cls(cfg)
        if "p_head" in z.files:            # 旧格式：保留独立输出头
            obj.params["head"] = z["p_head"].astype(obj.dtype)
        for k in list(obj.params):
            if f"p_{k}" in z.files:
                obj.params[k] = z[f"p_{k}"].astype(obj.dtype)
        # 词表随权重一起存：语料再增长也不会把 id 顶出 wte 的边界
        try:
            obj.chars = json.loads(str(z["chars"])) if "chars" in z.files else []
        except (ValueError, TypeError):
            obj.chars = []
        try:
            payload = json.loads(str(z["tok"])) if "tok" in z.files else {}
            obj.tok = payload if isinstance(payload, dict) and payload else None
        except (ValueError, TypeError):
            obj.tok = None
        return obj


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @classmethod
    def from_chars(cls, chars: List[str]) -> "CharTokenizer":
        obj = cls.__new__(cls)
        obj.chars = list(chars)
        obj.stoi = {c: i for i, c in enumerate(obj.chars)}
        obj.itos = {i: c for c, i in obj.stoi.items()}
        return obj

    def encode(self, s: str) -> List[int]:
        # 越界即崩：未知字符一律落到 0，保证 id < vocab
        return [min(self.stoi.get(c, 0), self.vocab - 1) if self.vocab else 0 for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)

    @property
    def vocab(self) -> int:
        return len(self.chars)
