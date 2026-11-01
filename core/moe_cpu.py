"""MoE 的 CPU 分支：命中显存走 GPU，未命中在 CPU 算。

判据：k = 专家总数时，必须等于稠密加权和（与 moe.py 同一条等式）。
"""
import torch
import torch.nn.functional as F


def _expert(x, wg, wu, wd):
    return (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T


def moe_cpu(x, gate_w, experts, shared, k):
    """x:(N,d); gate_w:(E,d); experts=[(wg,wu,wd)]; shared=(wg,wu,wd) 或 None。"""
    logits = x @ gate_w.T
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, min(k, logits.shape[-1]), dim=-1)
    out = torch.zeros_like(x)
    for e in range(gate_w.shape[0]):
        rows, slots = (idx == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        out.index_add_(0, rows, _expert(x[rows], *experts[e]) * vals[rows, slots][:, None])
    if shared is not None:
        out = out + _expert(x, *shared)
    return out


def dense_ref(x, gate_w, experts, shared):
    p = torch.softmax(x @ gate_w.T, dim=-1)
    out = sum(p[:, e][:, None] * _expert(x, *experts[e]) for e in range(len(experts)))
    return out + (_expert(x, *shared) if shared is not None else 0)


if __name__ == '__main__':
    torch.manual_seed(0)
    N, d, h, E = 6, 16, 32, 4
    x = torch.randn(N, d)
    gate = torch.randn(E, d)
    ex = [(torch.randn(h, d) * .1, torch.randn(h, d) * .1, torch.randn(d, h) * .1) for _ in range(E)]
    sh = (torch.randn(h, d) * .1, torch.randn(h, d) * .1, torch.randn(d, h) * .1)
    a, b = moe_cpu(x, gate, ex, sh, E), dense_ref(x, gate, ex, sh)
    print('k=E diff', float((a - b).abs().max()))
    assert torch.allclose(a, b, atol=1e-5)
    print('moe_cpu ok')
