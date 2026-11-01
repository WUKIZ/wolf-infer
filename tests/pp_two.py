"""PP 两卡验证：rank0 跑前半层，rank1 跑后半层 + head，与单卡逐字对比。

    python3 tests/pp_two.py
"""
import os
import sys

sys.path.insert(0, '.')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer

from core import pp
from core.moe_batch import BatchEngine
from core.moe_pcie import D_DIR

PROMPT = '中国的首都是'
GEN = 6
WORLD = 2


def greedy_ref(eng, ids, gen=GEN):
    """单卡参考路径。"""
    cache = {}
    lg = eng.prefill(ids, cache, last_only=True)
    cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
    out = [cur]
    for _ in range(gen - 1):
        lg = eng.decode([cur], [cache])
        cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
        out.append(cur)
    return out


def worker(rank, port, out):
    os.environ['WOLF_NO_PIN'] = '1'
    torch.cuda.set_device(rank)
    pp.init(rank, WORLD, port)
    eng = BatchEngine()
    eng.pp_rank, eng.pp_world = rank, WORLD
    eng.pp_range = pp.layers_of(rank, WORLD, eng.cfg['num_hidden_layers'])
    eng.warm()
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    ids = tk.encode(PROMPT).ids
    cache = {}
    lg = eng.prefill(ids, cache, last_only=True)
    last = WORLD - 1
    if rank == last:
        toks = [int(lg.reshape(-1, lg.shape[-1])[0].argmax())]
        pp.send(torch.tensor(toks, dtype=torch.long), 0)
    else:
        toks = pp.recv((1,), torch.long, last).tolist()
    for _ in range(GEN - 1):
        lg = eng.decode(toks[-1:], [cache])
        if rank == last:
            toks.append(int(lg.reshape(-1, lg.shape[-1])[0].argmax()))
            pp.send(torch.tensor(toks[-1:], dtype=torch.long), 0)
        else:
            toks += pp.recv((1,), torch.long, last).tolist()
    out.put((rank, tuple(eng.pp_range), toks))
    dist.barrier()
    dist.destroy_process_group()


def main():
    tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
    ids = tk.encode(PROMPT).ids

    os.environ['WOLF_NO_PIN'] = '1'
    torch.cuda.set_device(0)
    eng = BatchEngine()
    eng.warm()
    ref = greedy_ref(eng, ids)
    print('单卡参考:', repr(tk.decode(ref)), flush=True)
    del eng
    torch.cuda.empty_cache()

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 29533, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    got = {}
    while not q.empty():
        r, rng, toks = q.get()
        got[r] = (rng, toks)

    last = got[WORLD - 1][1]
    print('PP 切分  ', got[0][0], got[1][0], flush=True)
    print('PP 输出  ', repr(tk.decode(last)), flush=True)
    print('逐字一致 ', 'PASS' if last == ref else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
