"""语料流式读入：按字节块读 → tokenize → 定长样本，可断点续读。"""
import json
import os

import numpy as np
from tokenizers import Tokenizer

CHUNK = 1 << 20          # 每次读 1 MB 原文


class Loader:
    def __init__(self, corpus, tokenizer, seq_len=512, batch=4, offset=0):
        self.path = corpus
        self.tk = Tokenizer.from_file(tokenizer)
        self.seq_len = seq_len
        self.batch = batch
        self.pos = offset                    # 已消费的字节数
        self.buf = []                        # 上次切剩的 token

    def _fill(self, fh, need):
        """从文件补足 need 个 token；到文件尾返回 False。"""
        while len(self.buf) < need:
            fh.seek(self.pos)
            raw = fh.read(CHUNK)
            if not raw:
                return False
            self.pos += len(raw)
            self.buf.extend(self.tk.encode(raw.decode('utf-8', 'ignore')).ids)
        return True

    def state(self):
        return {'pos': self.pos, 'buf': self.buf[-self.seq_len:]}

    def save_state(self, path):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.state(), fh)

    def load_state(self, path):
        if os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                st = json.load(fh)
            self.pos, self.buf = st['pos'], st['buf']

    def __iter__(self):
        with open(self.path, 'rb') as fh:
            while True:
                if not self._fill(fh, self.batch * (self.seq_len + 1)):
                    break
                take = self.batch * (self.seq_len + 1)
                block = np.array(self.buf[:take], dtype=np.int32)
                self.buf = self.buf[take:]
                block = block.reshape(self.batch, self.seq_len + 1)
                yield block[:, :-1], block[:, 1:]        # 输入 / 目标


if __name__ == '__main__':
    import tempfile

    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, 'c.txt')
    with open(corpus, 'w', encoding='utf-8') as fh:
        fh.write('中国的首都是北京。' * 200)
    tk_dir = '/data/models/Qwen1.5-MoE-A2.7B/tokenizer.json'
    ld = Loader(corpus, tk_dir, seq_len=16, batch=2)
    n = 0
    for x, y in ld:
        assert x.shape == (2, 16) and y.shape == (2, 16)
        n += 1
    print('样本批数 %d，最后偏移 %d 字节' % (n, ld.pos))

    # 断点：存状态 → 新建 loader → 读状态，应从中断处继续
    ld2 = Loader(corpus, tk_dir, seq_len=16, batch=2)
    sp = os.path.join(tmp, 'state.json')
    ld.save_state(sp)
    ld2.load_state(sp)
    assert ld2.pos == ld.pos
    print('断点偏移一致 PASS')
