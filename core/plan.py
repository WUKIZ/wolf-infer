"""按硬件参数推导放置策略与速度下界。纯函数、无第三方依赖，CPU 节点可直接跑。"""

GB = 1024 ** 3

# 每 token 必读字节 ≈ 激活参数 × 2 B（bf16）
# 达成率在本项目硬件实测校准：resident 10.8 / pinned 6.32 / stream 1.16 tok/s
RATE = {'resident': 0.05, 'pinned': 0.26, 'stream': 0.50}

# 非专家参数（attention / embed / head / layernorm）约占模型总字节的比例
NON_EXPERT_RATIO = 0.15


def plan(total_mem_gb, vram_gb, cards, params_b, active_b,
         vram_gbps=1200.0, pcie_gbps=130.0, disk_gbps=12.0,
         experts_per_layer=60, tensor_mb=5.77):
    """返回该硬件下的放置策略与速度下界。"""
    model_gb = params_b * 2.0
    per_token_gb = active_b * 2.0
    vram_budget = vram_gb * 0.85
    mem_budget = total_mem_gb * 0.85

    expert_gb = model_gb * (1.0 - NON_EXPERT_RATIO)
    dense_gb = model_gb * NON_EXPERT_RATIO
    pinned_need = expert_gb * PIN_INFLATION + dense_gb   # pinned 实占 ≈ 张量 x 3.4

    if model_gb <= vram_budget:
        mode, pin_per_layer, band = 'resident', experts_per_layer, vram_gbps
    elif pinned_need <= mem_budget:
        mode, pin_per_layer, band = 'pinned', experts_per_layer, pcie_gbps
    else:
        room_gb = max(0.0, mem_budget - dense_gb)
        per_layer_gb = experts_per_layer * 3 * tensor_mb / 1024.0 * PIN_INFLATION
        pin_per_layer = int(min(experts_per_layer, room_gb / per_layer_gb))
        if pin_per_layer:
            mode = 'pinned'
            band = pcie_gbps * (0.4 + 0.6 * pin_per_layer / experts_per_layer)
        else:
            mode, band = 'stream', disk_gbps

    ceiling = band / per_token_gb
    return {'mode': mode, 'pin_per_layer': pin_per_layer,
            'ep': 1 if cards < 2 else min(cards, 8),
            'model_gb': model_gb, 'per_token_gb': per_token_gb,
            'band_gbps': band, 'ceiling_tok_s': ceiling,
            'expect_tok_s': ceiling * RATE[mode]}


def describe(p):
    return ('%-8s 预热=%2d 并行=%d | 模型 %.1f GB 每token %.1f GB 带宽 %.0f GB/s'
            ' | 上限 %5.0f 预期 %5.1f tok/s'
            % (p['mode'], p['pin_per_layer'], p['ep'], p['model_gb'],
               p['per_token_gb'], p['band_gbps'], p['ceiling_tok_s'], p['expect_tok_s']))


RUNTIME_GB = 16.0      # torch + KV + 临时张量预留

# 实测标定（60.75 G 容器）：预热 RSS 峰值 61.2 G、第 14 层被杀，
# 即每个「整层 60 专家」实占约 3.4 GB，而专家本身只有 1.01 GB。
PIN_INFLATION = 3.4


def pin_for_host(layer_count, experts_per_layer, tensor_bytes, host_gb):
    """主机内存能装下多少专家：够就全装，不够按比例。"""
    per_layer_gb = experts_per_layer * 3 * tensor_bytes / GB * PIN_INFLATION
    room_gb = max(0.0, host_gb - RUNTIME_GB)
    max_layers = int(room_gb / per_layer_gb)
    if max_layers >= layer_count:
        return experts_per_layer
    return max(0, int(experts_per_layer * max_layers / layer_count))


def host_budget_gb():
    """容器可用的主机内存上限（GB）：优先 cgroup v2 / v1，退回 /proc/meminfo。"""
    for path in ('/sys/fs/cgroup/memory.max',
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(path, encoding='utf-8') as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw.isdigit():
            value = int(raw)
            if value < (1 << 60):
                return value / GB
    try:
        with open('/proc/meminfo', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return 0.0
