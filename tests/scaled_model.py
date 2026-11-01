"""200 层合成模型：权重表观 200 GB（远超 60 G 内存），验证仍能出 token。

    python3 tests/scaled_model.py
"""
import os
import sys
import time

sys.path.insert(0, '.')
os.environ['WOLF_NO_PIN'] = '1'          # 200 GB 不能 pin，强制流式

import torch
from tokenizers import Tokenizer

from core.moe_batch import BatchEngine

MODEL = '.tmp/fake200'
PROMPT = '中国的首都是'


def main():
    tk = Tokenizer.from_file(os.path.join(MODEL, 'tokenizer.json'))
    t0 = time.perf_counter()
    eng = BatchEngine(model_dir=MODEL)
    eng.warm()
    print('warm %.1f s | 层数 %d' % (time.perf_counter() - t0,
                                   eng.cfg['num_hidden_layers']), flush=True)

    ids = tk.encode(PROMPT).ids
    cache = {}
    t1 = time.perf_counter()
    lg = eng.prefill(ids, cache, last_only=True)
    print('prefill %.1f s' % (time.perf_counter() - t1), flush=True)

    cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
    out = [cur]
    for _ in range(2):
        lg = eng.decode([cur], [cache])
        cur = int(lg.reshape(-1, lg.shape[-1])[0].argmax())
        out.append(cur)

    print('生成 %d token | 显存 %.2f GB' % (len(out),
                                          torch.cuda.memory_allocated() / 2 ** 30))
    print('输出:', repr(tk.decode(out)))


if __name__ == '__main__':
    main()
