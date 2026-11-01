"""按批测每步耗时分布（P50/P95/P99）与吞吐；OOM 记为失败而不是崩溃退出。"""
import statistics
import sys
import time

sys.path.insert(0, '.')
import torch
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer

torch.set_num_threads(12)
tk = Tokenizer.from_file('/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json')
ids = tk.encode('中国的首都是').ids
eng = BatchEngine()
eng.warm()

for B in (32, 128, 512, 1024):
    try:
        caches = [{} for _ in range(B)]
        toks = [int(eng.prefill(ids, c)[0, -1].argmax()) for c in caches]
        seqs = [[t] for t in toks]
        for _ in range(3):
            out = eng.decode([s[-1] for s in seqs], caches)
            for i, s in enumerate(seqs):
                s.append(int(out[i].reshape(-1).argmax()))
        times = []
        for _ in range(32):
            t0 = time.perf_counter()
            out = eng.decode([s[-1] for s in seqs], caches)
            for i, s in enumerate(seqs):
                s.append(int(out[i].reshape(-1).argmax()))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]
        mean = statistics.mean(times)
        print('B=%4d  mean %5.0f  p50 %5.0f  p95 %5.0f  p99 %5.0f ms/步  %6.1f tok/s'
              % (B, mean, pick(0.5), pick(0.95), pick(0.99), B / (mean / 1e3)))
    except Exception as exc:
        torch.cuda.empty_cache()

        print('B=%4d 失败: %s' % (B, type(exc).__name__))
    finally:
        torch.cuda.empty_cache()
