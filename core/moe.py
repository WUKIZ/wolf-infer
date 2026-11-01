"""MoE 前向：top-k 路由 + 加权合并。

判据：k = 专家总数时，结果必须等于「每个专家都算一遍、按 softmax 权重加权」的稠密和。
这条等式成立，路由才算对。
"""
import torch
import torch.nn.functional as F


def expert_ffn(x, w1, w3, w2):
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def route(x, gate, k):
    logits = x @ gate.T
    E = gate.shape[0]
    if k >= E:
        return torch.softmax(logits, dim=-1), torch.arange(E, device=x.device).expand(x.shape[0], E)
    top = torch.topk(logits, k, dim=-1)
    return torch.softmax(top.values, dim=-1), top.indices


def moe_forward(x, experts, gate, k):
    """按专家聚合：同一专家收到的 token 一起算，再按权重散回。"""
    E = gate.shape[0]
    out = torch.zeros_like(x)
    w_all, idx_all = route(x, gate, k)
    for e in range(E):
        hit = idx_all == e
        if not hit.any():
            continue
        rows, slots = hit.nonzero(as_tuple=True)
        h = expert_ffn(x[rows], *experts[e])
        out.index_add_(0, rows, h * w_all[rows, slots][:, None])
    return out


def dense_reference(x, experts, gate):
    p = torch.softmax(x @ gate.T, dim=-1)
    return sum(p[:, e][:, None] * expert_ffn(x, *experts[e]) for e in range(gate.shape[0]))


if __name__ == "__main__":
    torch.manual_seed(0)
    N, d, h, E = 8, 32, 64, 4
    x = torch.randn(N, d)
    gate = torch.randn(E, d)
    experts = [(torch.randn(h, d) * 0.1, torch.randn(h, d) * 0.1, torch.randn(d, h) * 0.1)
               for _ in range(E)]

    ref = dense_reference(x, experts, gate)
    full = moe_forward(x, experts, gate, k=E)
    print("k=E diff", float((full - ref).abs().max()))
    assert torch.allclose(full, ref, atol=1e-5)

    two = moe_forward(x, experts, gate, k=2)
    w2v, idx2 = route(x, gate, 2)
    assert idx2.shape == (N, 2) and w2v.shape == (N, 2)
    print("k=2 权重和", float(w2v.sum(-1).min()), float(w2v.sum(-1).max()))
    print("k=2 shape", tuple(two.shape), "k=E shape", tuple(full.shape))
    print("moe ok")
