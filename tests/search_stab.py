"""搜索稳定性实测：12 条查询 × 3 遍，量命中率、跨轮一致性与结果质量。"""
import sys
import time

sys.path.insert(0, '.')
from search_multi import search

QUERIES = [
    ('事实', '珠穆朗玛峰 海拔 多少米'),
    ('事实', '水的沸点 标准大气压'),
    ('事实', 'Qwen1.5-MoE 参数量'),
    ('时效', '2026年 中国 空间站 最新 任务'),
    ('时效', '今天的 上证指数'),
    ('时效', '最近 AI 芯片 新闻'),
    ('口语', '电脑蓝屏了咋整'),
    ('口语', '咋样才能早起'),
    ('口语', '手机特别卡怎么办'),
    ('英文', 'vLLM PagedAttention paper'),
    ('英文', 'ROCm DCU support'),
    ('英文', 'MoE expert parallelism survey'),
]
ROUNDS = 3


def urls_of(hits):
    return {h.get('url') or h.get('link') or '' for h in hits}


def main():
    total_ok = 0
    jac_sum, empty_runs = 0.0, 0
    for cat, q in QUERIES:
        runs = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            hits = search(q, 6) or []
            dt = time.perf_counter() - t0
            runs.append(urls_of(hits) - {''})
            if not runs[-1]:
                empty_runs += 1
            print('  %-4s %-26s %d 条 %.1f s' % (cat, q[:24], len(runs[-1]), dt), flush=True)
        ok = all(r for r in runs)
        total_ok += ok
        inter = set.intersection(*runs)
        union = set.union(*runs)
        jac = len(inter) / len(union) if union else 0.0
        jac_sum += jac
        print('%s 跨轮 Jaccard %.2f' % ('' if ok else '', jac), flush=True)
    n = len(QUERIES)
    print('\n命中 %d/%d（三轮都有结果）| 空轮次 %d/%d | 平均跨轮一致性 %.2f'
          % (total_ok, n, empty_runs, n * ROUNDS, jac_sum / n))


if __name__ == '__main__':
    main()
