"""B=8 解码一步的耗时拆分：attn / moe / 其余，同时统计专家张量装载量。

用法：python3 -u tests/prof_b8.py [B] [STEPS]
"""
import sys
import time

sys.path.insert(0, 'core')
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

MODEL = '/data/models/Qwen1.5-MoE-A2.7B'
PROMPT = '中国的首都是'
EXPERT_TENSOR_MB = 5.77          # 单个专家张量（1408×2048 bf16）


def main(batch, steps):
    tk = Tokenizer.from_file(MODEL + '/tokenizer.json')
    ids = tk.encode(PROMPT).ids
    eng = BatchEngine()
    eng.pin_per_layer = 60
    eng.warm()

    caches = [{} for _ in range(batch)]
    toks = [int(eng.prefill(ids, c)[0, -1].argmax()) for c in caches]

    spent = {'attn': 0.0, 'moe': 0.0}
    inner_attn, inner_moe = eng._attn_b, eng._moe

    def timed(fn, key):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            spent[key] += time.perf_counter() - t0
            return out
        return wrapper

    eng._attn_b = timed(inner_attn, 'attn')
    eng._moe = timed(inner_moe, 'moe')

    pinned0, disk0 = eng.pinned_loads, eng.disk_loads
    t0 = time.perf_counter()
    for _ in range(steps):
        out = eng.decode(toks, caches)
        toks = [int(row.reshape(-1).argmax()) for row in out]
    total = time.perf_counter() - t0

    other = total - spent['attn'] - spent['moe']
    print('B=%d steps=%d  %.2f tok/s  %.1f ms/步'
          % (batch, steps, batch * steps / total, total / steps * 1e3))
    for key, label in (('attn', 'attn '), ('moe', 'moe  ')):
        print('%s  %6.1f ms/步  %4.0f%%' % (label, spent[key] / steps * 1e3, 100 * spent[key] / total))
    print('other  %6.1f ms/步  %4.0f%%' % (other / steps * 1e3, 100 * other / total))

    pinned = (eng.pinned_loads - pinned0) / steps
    disk = (eng.disk_loads - disk0) / steps
    print('每步专家张量  pinned %.1f  disk %.1f  合计 %.1f  ≈ %.2f GB'
          % (pinned, disk, pinned + disk, (pinned + disk) * EXPERT_TENSOR_MB / 1024))
    print('文本', repr(tk.decode(toks)))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
