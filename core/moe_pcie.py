"""MoE 引擎：路由/注意力/激活借 qwen_moe，本文件只做数据搬运。

两个正交开关（类属性，不用环境变量）：
  resident       True=专家常驻显存（每层三块连续显存）
  pin_per_layer  每层预热多少专家进 pinned；0=不预热，全部按需读盘
每层结束丢 page cache 并关闭 safetensors 句柄，避免容器内存被 mmap 撑满。
"""
import ctypes
import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open
from qwen_moe import attention, moe_block, norm32

D_DIR = '/data/models/Qwen1.5-MoE-A2.7B'
NAMES = ('gate_proj', 'up_proj', 'down_proj')
PREFIX = 'mlp.experts.'
_LIBC = ctypes.CDLL('libc.so.6', use_errno=True)


def drop_cache(path):
    """丢弃文件 page cache；safetensors 的 mmap 页会计入 cgroup 内存。"""
    try:
        fd = os.open(path, os.O_RDONLY)
        _LIBC.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
        os.close(fd)
    except OSError:
        pass


def parse_expert_key(key):
    """'mlp.experts.3.up_proj.weight' -> (3, 'up_proj')；不合法抛 KeyError。"""
    rest = key[len(PREFIX):]
    idx, _, tail = rest.partition('.')
    name = tail.partition('.')[0]
    if not idx.isdigit() or name not in NAMES:
        raise KeyError('未知专家权重 %r' % key)
    return int(idx), name



def load_hot(path, k):
    """按命中频次取每层 top-k 专家，供热预热使用。"""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    per = {}
    for key, cnt in raw.items():
        layer, idx = (int(x) for x in key.split(','))
        per.setdefault(layer, []).append((cnt, idx))
    return {L: [i for _, i in sorted(v, reverse=True)[:k]] for L, v in per.items()}




class _Layer(dict):
    """层权重视图：非专家项已常驻，专家项转发给引擎按需取。"""

    def __init__(self, engine, layer, base):
        """初始化引擎：读配置、按主机内存选档位（resident/pinned/stream）、预分配缓冲。"""
        super().__init__(base)
        self.engine = engine
        self.layer = layer

    def experts(self, ids):
        """批量取专家权重，返回顺序为 (gate, up, down)。"""
        return self.engine.experts_gpu(self.layer, ids)

    def __getitem__(self, key):
        """按层号取 _Layer 视图并缓存，首次访问时构建。"""
        if key.startswith(PREFIX):
            idx, name = parse_expert_key(key)
            return self.engine._expert(idx, name, self.layer)
        return dict.__getitem__(self, key)


class PCIeEngine:
    """单卡 MoE 引擎：非专家常驻显存，专家按档位预热或按需搬运。"""

    resident = None        # None=按显存自动；True/False=强制
    pin_per_layer = None   # None=按主机内存自动；0=全部按需读盘
    hot = None             # 可选 {层: [专家号]}，给了就按它预热

    def __init__(self, model_dir=D_DIR):
        """初始化引擎：读配置、按主机内存选档位（resident/pinned/stream）、预分配缓冲。"""
        self.dir = model_dir
        with open(os.path.join(model_dir, 'config.json')) as fh:
            self.cfg = json.load(fh)
        cfg = self.cfg
        cfg.setdefault('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
        self.eps = cfg['rms_norm_eps']
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as fh:
            self.idx = json.load(fh)['weight_map']
        d, h = cfg['hidden_size'], cfg['moe_intermediate_size']
        self._shapes = ((h, d), (h, d), (d, h))
        self.ts = {}
        self._slab = {}        # (层, 张量名) -> 本层专家连续块（pinned）
        self._gather_buf = {}  # (张量名, K) -> 复用的 pinned 汇聚缓冲
        self._slab_idx = {}    # 层 -> {专家号: slab 行号}
        self._pool = None
        self._gpu_e = {}
        self._gpu_order = []        # LRU 顺序：末尾最近使用
        self._gpu_bytes = 0
        self._gpu_budget = int(os.environ.get('WOLF_EXPERT_BUDGET_MB', '4096')) * 2 ** 20
        self._files = {}
        self._emb = self._norm = self._head = None
        self.pinned_loads = 0
        self.disk_loads = 0

        if self.pin_per_layer is None:
            from core.plan import GB, host_budget_gb, pin_for_host
            layers = max(int(k.split('.')[2]) for k in self.idx
                         if k.startswith('model.layers.')) + 1
            tensor_bytes = self._shapes[0][0] * self._shapes[0][1] * 2
            host_gb = host_budget_gb()
            self.pin_per_layer = pin_for_host(layers, cfg['num_experts'],
                                              tensor_bytes, host_gb)
            experts_gb = layers * cfg['num_experts'] * 3 * tensor_bytes / GB
            vram_gb = torch.cuda.get_device_properties(0).total_memory / GB
            self.resident = experts_gb * 1.15 <= vram_gb * 0.85
            logging.info('自动档位：pin=%d/%d 驻留=%s（主机 %.1f G，显存 %.1f G）',
                         self.pin_per_layer, cfg['num_experts'],
                         self.resident, host_gb, vram_gb)

    def share_from(self, other):
        """复用另一个已 warm 引擎的权重，避免重复读盘。"""
        self.ts = other.ts
        self._gpu_e = other._gpu_e
        self._emb, self._norm, self._head = other._emb, other._norm, other._head
        return self

    def _get(self, key):
        """取 (层, 专家) 的三块权重（gate/up/down），命中缓存直接返回。"""
        fn = self.idx[key]
        f = self._files.get(fn)
        if f is None:
            f = self._files[fn] = safe_open(os.path.join(self.dir, fn), framework='pt')
        return f.get_tensor(key)

    def _dense(self, layer):
        """把某层的非专家权重（attn/norm/shared）缓存在显存，专家部分不缓存。"""
        view = self.ts.get(layer)
        if view is None:
            p = 'model.layers.%d.' % layer
            base = {k[len(p):]: self._get(k).to('cuda') for k in self.idx
                    if k.startswith(p) and PREFIX not in k}
            view = self.ts[layer] = _Layer(self, layer, base)
        return view

    def _slab_row(self, layer, idx):
        """专家在本层 slab 中的行号；未预热返回 -1。"""
        return self._slab_idx.get(layer, {}).get(idx, -1)

    def _expert(self, idx, name, layer):
        """单张专家权重：命中 slab 走 pinned 拷贝，未命中读盘。"""
        if self.resident:
            key = (layer, name)
            if key in self._gpu_order:
                self._gpu_order.remove(key)
                self._gpu_order.append(key)
            return self._gpu_e[key][idx]
        slot = NAMES.index(name)
        if self._pool is None:
            self._pool = [torch.empty(sh, dtype=torch.bfloat16, device='cuda')
                          for sh in self._shapes]
        row = self._slab_row(layer, idx)
        if row < 0:
            self.disk_loads += 1
            key = 'model.layers.%d.mlp.experts.%d.%s.weight' % (layer, idx, name)
            self._pool[slot].copy_(self._get(key))
        else:
            self.pinned_loads += 1
            self._pool[slot].copy_(self._slab[(layer, name)][row], non_blocking=True)
        return self._pool[slot]

    def experts_gpu(self, layer, ids):
        """
        一次取多专家，返回 (gate, up, down) 已在卡上。

        index_select 的默认输出是普通内存，直接 .to('cuda') 会退化成同步页拷贝；
        这里先写进复用的 pinned 缓冲再异步上卡。实测 moe 段无变化——瓶颈在
        bmm 算力而非搬运，不要在此继续优化。
"""
        rows = [self._slab_row(layer, e) for e in ids]
        sel = torch.tensor(rows)
        out = []
        for name, shape in zip(NAMES, self._shapes):
            buf = self._gather_buf.get((name, len(ids)))
            if buf is None:
                buf = torch.empty((len(ids),) + shape,
                                  dtype=torch.bfloat16, pin_memory=True)
                self._gather_buf[(name, len(ids))] = buf
            slab = self._slab.get((layer, name))
            if slab is not None and all(r >= 0 for r in rows):
                torch.index_select(slab, 0, sel, out=buf)
                self.pinned_loads += len(ids)
            else:
                pattern = 'model.layers.%d.mlp.experts.%d.%s.weight'
                for row, e in enumerate(ids):
                    buf[row] = self._get(pattern % (layer, e, name))
                self.disk_loads += len(ids)
            out.append(buf.to('cuda', non_blocking=True))
        return tuple(out)


    def _cache_gpu_e(self, key, block):
        """写入显存专家块；超出预算时按最久未用淘汰。"""
        if key in self._gpu_e:
            self._gpu_bytes -= self._gpu_e[key].numel() * self._gpu_e[key].element_size()
            self._gpu_order.remove(key)
        self._gpu_e[key] = block
        self._gpu_order.append(key)
        self._gpu_bytes += block.numel() * block.element_size()
        while self._gpu_bytes > self._gpu_budget and len(self._gpu_order) > 1:
            old = self._gpu_order.pop(0)
            ev = self._gpu_e.pop(old, None)
            if ev is not None:
                self._gpu_bytes -= ev.numel() * ev.element_size()
                del ev
        if self._gpu_bytes > self._gpu_budget:
            torch.cuda.empty_cache()

    def _release_layer(self, layer):
        """丢该层所有分片的页缓存并关句柄。"""
        for fn in {self.idx[k] for k in self.idx
                   if k.startswith('model.layers.%d.' % layer)}:
            drop_cache(os.path.join(self.dir, fn))
        self._files.clear()
        gc.collect()

    def _picks(self, layer, n_expert):
        """本层要预热的专家号：hot 优先，否则按档位等距取样。"""
        if self.hot:
            return sorted(set(self.hot.get(layer, [])))[:n_expert]
        wanted = max(0, min(self.pin_per_layer, n_expert))
        if wanted >= n_expert:
            return list(range(n_expert))
        return sorted({min(n_expert - 1, int(i * n_expert / wanted))
                       for i in range(wanted)}) if wanted else []

    def warm(self):
        """装入专家：resident 全量上卡；否则按 picks 预热进 pinned。"""
        n_layer = self.cfg['num_hidden_layers']
        n_expert = self.cfg['num_experts']
        pattern = 'model.layers.%d.mlp.experts.%d.%s.weight'
        for layer in range(n_layer):
            self._dense(layer)
            if self.resident:
                for name, shape in zip(NAMES, self._shapes):
                    block = torch.empty((n_expert,) + shape,
                                        dtype=torch.bfloat16, device='cuda')
                    for idx in range(n_expert):
                        block[idx] = self._get(pattern % (layer, idx, name))
                    self._cache_gpu_e((layer, name), block)
            else:
                picks = self._picks(layer, n_expert)
                self._slab_idx[layer] = {e: r for r, e in enumerate(picks)}
                for name, shape in zip(NAMES, self._shapes):
                    slab = torch.empty((len(picks),) + shape,
                                       dtype=torch.bfloat16, pin_memory=True)
                    for row, idx in enumerate(picks):
                        slab[row] = self._get(pattern % (layer, idx, name))
                    self._slab[(layer, name)] = slab
            self._release_layer(layer)
            logging.info('warm L%d', layer)
        self._emb = self._get('model.embed_tokens.weight').to('cuda')
        self._norm = self._get('model.norm.weight').to('cuda')
        self._head = self._get('lm_head.weight').to('cuda')
        return self

    def _moe(self, layer, x):
        """MoE 层：取该层权重做专家混合，含残差与共享专家。"""
        return moe_block(self._dense(layer), x, self.cfg)

    def logits(self, ids):
        """对整段输入做一次前向，返回 (1, T, vocab) 的 logits，不写 KV 缓存。"""
        if self._emb is None:
            self.warm()
        x = self._emb[torch.tensor([ids], device='cuda')]
        for layer in range(self.cfg['num_hidden_layers']):
            x = attention(self._dense(layer), x, self.cfg)
            x = self._moe(layer, x)
        return norm32(x, self._norm, self.eps) @ self._head.T
