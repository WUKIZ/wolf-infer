"""EP 容量实证：两卡各持一半专家，all_reduce 求和，与单卡输出对比。

    python3 tests/ep_capacity.py
"""
import os
import sys

sys.path.insert(0, '.')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer

from core import ep
from core.moe_batch import BatchEngine
from core.moe_pcie import D_DIR

PROMPT = '中国的首都是'
GEN = 6
WORLD = 2


def greedy(eng, ids, gen=GEN):
    cache = {}
    lg = eng.prefill(ids, cache, last_only=True)
    cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
    out = [cur]
    for _ in range(gen - 1):
        lg = eng.decode([cur], [cache])
        cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
        out.append(cur)
    return out


def worker(rank, port, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    os.environ['WOLF_NO_PIN'] = '1'
    r, w = ep.init('nccl')
    eng = BatchEngine()
    ep.attach(eng, r, w)
    eng.warm()
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    toks = greedy(eng, tk.encode(PROMPT).ids)
    gb = torch.cuda.memory_allocated() / 2 ** 30
    q.put((r, tk.decode(toks), round(gb, 2)))
    dist.barrier()
    dist.destroy_process_group()


def main():
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 29544, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    got = {}
    while not q.empty():
        r, txt, gb = q.get()
        got[r] = (txt, gb)
    if len(got) == WORLD:
        for r in range(WORLD):
            print('rank%d  显存 %5.2f GB  %r' % (r, got[r][1], got[r][0]))
        print('两卡一致 ', 'PASS' if got[0][0] == got[1][0] else 'FAIL')
    else:
        print('进程异常退出，只收到', len(got), '个结果')


if __name__ == '__main__':
    main()
