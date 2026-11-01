"""给定规模与卡数，算出 TP/PP/DP/EP 组合与是否可行（纯 CPU，可离线跑）。"""
GB = 1 << 30
STATE_BYTES = 16          # bf16 p+g + fp32 master + Adam m/v
ACT_BYTES_PER_TOKEN = 34 * 2 * 2   # 层数×隐藏×2（bf16）粗估，随激活重算可再降 4 倍


def plan(n_params, n_gpu, mem_gb, usable=0.85, tp_max=8, ep=1, ckpt=True):
    mem = mem_gb * GB * usable
    act = ACT_BYTES_PER_TOKEN * 4096 * (1 if ckpt else 4) / 4
    per_gpu_state = n_params * STATE_BYTES
    need_shards = -(-per_gpu_state // int(mem - act))          # 向上取整
    ok = need_shards <= n_gpu
    dp = min(n_gpu, need_shards) if ok else n_gpu
    tp = min(tp_max, max(1, n_gpu // 8))
    pp = max(1, n_gpu // (dp * tp))
    return dict(params_B=n_params / 1e9, gpus=n_gpu, state_TB=per_gpu_state / 1e12,
                shards_needed=need_shards, feasible=ok,
                tp=tp, pp=pp, dp=dp, ep=ep)


if __name__ == '__main__':
    for name, p, g, m in [('400M×4卡', 4e8, 4, 64), ('14B×4卡', 1.43e10, 4, 64),
                          ('70B×20卡', 7e10, 20, 64), ('1T×1024卡', 1e12, 1024, 80),
                          ('10T×2300卡', 1e13, 2300, 80)]:
        r = plan(p, g, m)
        print('%-12s %s' % (name, {k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in r.items()}))
