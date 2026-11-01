"""专家并行：专家按 expert % world 切分，每层对贡献做 all_reduce 求和。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from qwen_moe import mlp, norm32


def init(backend='nccl'):
    """
        从环境变量初始化进程组，返回 (rank, world)。

        backend 默认 nccl；跨容器或握手卡住时改 gloo，并把 all_reduce 换成
        CPU 张量（慢约 100 倍，仅验正确性）。设备号取 LOCAL_RANK。
"""
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    return rank, world


def moe_block_ep(t, x, cfg, rank, world):
    """只算本卡负责的专家，求和后加共享专家（共享项各卡本地算，无通信）。"""
    eps = cfg['rms_norm_eps']
    top = cfg['num_experts_per_tok']
    b, s, d = x.shape
    h = norm32(x, t['post_attention_layernorm.weight'], eps)
    flat = h.reshape(-1, d)
    probs = torch.softmax(flat @ t['mlp.gate.weight'].T, dim=-1)
    vals, idxs = torch.topk(probs, top, dim=-1)

    out = torch.zeros(flat.shape, device=flat.device, dtype=torch.float32)
    mine = sorted({e for e in idxs.reshape(-1).tolist() if e % world == rank})
    if mine:
        gw, uw, dw = t.experts(mine)
        act = torch.nn.functional.silu(torch.matmul(flat, gw.transpose(1, 2)))
        act = act * torch.matmul(flat, uw.transpose(1, 2))
        eo = torch.matmul(act, dw.transpose(1, 2)).float()
        for k, e in enumerate(mine):
            wk = (vals * (idxs == e)).sum(-1).float()
            out += eo[k] * wk[:, None]

    dist.all_reduce(out)

    sh = mlp(flat, t['mlp.shared_expert.gate_proj.weight'],
             t['mlp.shared_expert.up_proj.weight'],
             t['mlp.shared_expert.down_proj.weight'])
    gate_s = torch.sigmoid(flat @ t['mlp.shared_expert_gate.weight'].T)
    out = out + sh.float() * gate_s
    return x + out.to(x.dtype).reshape(b, s, d)


def attach(engine, rank, world):
    """
        把引擎的 _moe 换成专家并行版本。

        注意力、嵌入、共享专家仍每卡完整计算，所以换来的是单卡显存容量
        （专家减半），不是吞吐：B=32 实测 26.24 tok/s ≈ 单卡 85 的 31%；
        通信仅 0.376 ms/次，差距来自 dense 复制与 14 核被两进程平分。
"""
    def _moe(layer, x):
        """闭包：把 engine 的单层 MoE 调用转到 moe_block_ep，rank/world 已绑定。"""
        return moe_block_ep(engine._dense(layer), x, engine.cfg, rank, world)
    engine._moe = _moe
    return engine
