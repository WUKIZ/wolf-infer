"""回归台：固定题库 + 三类负载 + 基线快照对比。

  python3 tests/bench.py --tag v1                 跑一遍，存 snapshots/v1.json
  python3 tests/bench.py --compare a.json b.json  判回归（PASS/FAIL）
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, '.')
import torch
from tokenizers import Tokenizer

from core.moe_batch import BatchEngine

MODEL = '/data/models/Qwen1.5-MoE-A2.7B'
SNAP = pathlib.Path('snapshots')

# (提示, 任一关键词命中即算通过) —— 关键词取最短判别形式，不依赖措辞
CASES = [
    ('中国的首都是', ['北京']), ('水的化学式是', ['H2O', 'H₂O']),
    ('The capital of France is', ['Paris']), ('1+1=', ['2']),
    ('太阳从哪个方向升起？', ['东']), ('二进制 1010 是多少？', ['10']),
    ('一年有多少个月？', ['12']), ('地球围绕什么转？', ['太阳']),
    ('Python 里定义函数用哪个关键字？', ['def']),
    ('HTTP 404 表示', ['未找到', 'Not Found', 'not found']),
    ('sizeof(int) 在 64 位 Linux 上通常是', ['4']), ('JSON 全称是', ['JavaScript']),
]
GEN = 8


def flat(t):
    """(..., V) → (N, V)：兼容 (V,) 与 (1, V) 两种输出形状。"""
    return t.reshape(-1, t.shape[-1])


def greedy(eng, tok, prompt, caches, gen=GEN):
    """单条贪心生成，返回 (文本, token 序列)。"""
    ids = tok.encode(prompt).ids
    lg = flat(eng.prefill(ids, caches[0], last_only=True))[0]
    cur = int(lg.argmax())
    out = [cur]
    for _ in range(gen - 1):
        lg = flat(eng.decode([cur], caches))[0]
        cur = int(lg.argmax())
        out.append(cur)
    return tok.decode(out), out


def bench_single(eng, tok):
    """逐条贪心，量单请求速度 + 答对与否。"""
    caches = [{}]
    pass_map, t0 = {}, time.perf_counter()
    for i, (prompt, keys) in enumerate(CASES):
        txt, _ = greedy(eng, tok, prompt, caches)
        caches[0].clear()                     # 每条独立上下文
        cid = 'c%02d' % i
        pass_map[cid] = any(k in txt for k in keys)
        print('[%02d/%d] %-24s %s' % (i + 1, len(CASES), prompt[:22],
                                     'PASS' if pass_map[cid] else 'FAIL ' + repr(txt[:36])),
              flush=True)
    return pass_map, len(CASES) * GEN / (time.perf_counter() - t0)


def bench_div(eng, tok, b=32, gen=GEN):
    """多样性批：每条 prompt 都不同，避免专家并集退化成 4。"""
    base = [p for p, _ in CASES]
    prompts = [base[i % len(base)] + base[(i * 7 + 3) % len(base)] for i in range(b)]
    caches = [{} for _ in prompts]
    eng._kvb = {}
    t0 = time.perf_counter()
    ids_of = [tok.encode(p).ids for p in prompts]
    order = sorted(range(len(prompts)), key=lambda k: len(ids_of[k]))   # 等长才可同批
    toks = [0] * len(prompts)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and len(ids_of[order[j + 1]]) == len(ids_of[order[i]]):
            j += 1
        chunk = order[i:j + 1]
        lg = eng.prefill_batch([(ids_of[k], caches[k]) for k in chunk], last_only=True)
        for row, k in zip(lg.reshape(len(chunk), -1).argmax(-1).tolist(), chunk):
            toks[k] = int(row)
        i = j + 1
    t1 = time.perf_counter()
    for _ in range(gen - 1):
        lg = eng.decode(toks, caches).reshape(len(prompts), -1)
        toks = [int(t) for t in lg.argmax(-1)]
    t2 = time.perf_counter()
    print('  [div] prefill %d 条 %.2f s（%.3f s/条）| decode %d 步 %.0f ms/步'
          % (len(prompts), t1 - t0, (t1 - t0) / len(prompts), gen - 1,
             (t2 - t1) / (gen - 1) * 1000), flush=True)
    return len(prompts) * gen / (t2 - t0)


def bench_prefill(eng, tok, n=64, length=20):
    """等长批 prefill：量批量路径的单条成本（对照组是 1.790 s/条）。"""
    ids = tok.encode('解释一下专家并集为什么会决定 MoE 推理的吞吐。' * 3).ids[:length]
    items = [(ids, {}) for _ in range(n)]
    t0 = time.perf_counter()
    eng.prefill_batch(items, last_only=True)
    return (time.perf_counter() - t0) / n


def compare(old_path, new_path):
    """OLD 与 NEW 两份快照比：正确性回退或吞吐掉 >10% 即 FAIL。"""
    old, new = (json.loads(pathlib.Path(x).read_text()) for x in (old_path, new_path))
    fails = []
    for cid, ok in old['case_pass'].items():
        if ok and not new['case_pass'].get(cid):
            fails.append('用例回退 %s' % cid)
    for k in ('tok_s_single', 'tok_s_div32', 'prefill_s_item'):
        o, n = old[k], new[k]
        if k == 'prefill_s_item':                    # 此项越小越好
            if n > o * 1.1:
                fails.append('%s 变慢 %.0f%%（%.3f→%.3f）' % (k, (n / o - 1) * 100, o, n))
        elif n < o * 0.9:
            fails.append('%s 下降 %.0f%%（%.1f→%.1f）' % (k, (1 - n / o) * 100, o, n))
    print('回归 %s：%d 项' % ('FAIL' if fails else 'PASS', len(fails)))
    for f in fails:
        print('  -', f)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', help='快照名，存 snapshots/<tag>.json')
    ap.add_argument('--compare', nargs=2, metavar=('OLD', 'NEW'))
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)

    torch.set_num_threads(14)
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    eng = BatchEngine()
    eng.warm()
    eng._kvb = {}          # KV 块池：不重置会跨用例累积，越跑越慢
    pass_map, single = bench_single(eng, tok)
    div32 = bench_div(eng, tok)
    pf = bench_prefill(eng, tok)
    snap = {'tag': a.tag or 'run', 'torch': torch.__version__,
            'case_pass': pass_map, 'tok_s_single': round(single, 3),
            'tok_s_div32': round(div32, 3), 'prefill_s_item': round(pf, 4),
            'pass_rate': '%d/%d' % (sum(pass_map.values()), len(pass_map))}
    SNAP.mkdir(exist_ok=True)
    out = SNAP / ('%s.json' % snap['tag'])
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    print('\n通过 %s | 单条 %.2f | 多样性B32 %.1f | prefill %.3f s/条 → %s'
          % (snap['pass_rate'], single, div32, pf, out))


if __name__ == '__main__':
    sys.exit(main())
