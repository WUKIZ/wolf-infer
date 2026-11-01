"""生成放大的合成 MoE 模型：结构合法、体积可远超内存（稀疏文件）。

    python3 tools/make_scaled_model.py --layers 200 --out /data/fake-moe
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np


def tensor_specs(layers, experts, d, inter, vocab):
    """按 Qwen1.5-MoE 的命名给出 (名字, 形状)。"""
    specs = [('model.embed_tokens.weight', (vocab, d)),
             ('model.norm.weight', (d,)),
             ('lm_head.weight', (vocab, d))]
    for i in range(layers):
        p = 'model.layers.%d.' % i
        specs += [
            (p + 'input_layernorm.weight', (d,)),
            (p + 'self_attn.q_proj.weight', (d, d)), (p + 'self_attn.q_proj.bias', (d,)),
            (p + 'self_attn.k_proj.weight', (d, d)), (p + 'self_attn.k_proj.bias', (d,)),
            (p + 'self_attn.v_proj.weight', (d, d)), (p + 'self_attn.v_proj.bias', (d,)),
            (p + 'self_attn.o_proj.weight', (d, d)),
            (p + 'post_attention_layernorm.weight', (d,)),
            (p + 'mlp.gate.weight', (experts, d)),
            (p + 'mlp.shared_expert.gate_proj.weight', (inter, d)),
            (p + 'mlp.shared_expert.up_proj.weight', (inter, d)),
            (p + 'mlp.shared_expert.down_proj.weight', (d, inter)),
            (p + 'mlp.shared_expert_gate.weight', (1, d)),
        ]
        for e in range(experts):
            q = p + 'mlp.experts.%d.' % e
            specs += [(q + 'gate_proj.weight', (inter, d)),
                      (q + 'up_proj.weight', (inter, d)),
                      (q + 'down_proj.weight', (d, inter))]
    return specs


def write_sparse(path, specs):
    """按 safetensors 规范写 header，数据区用 seek 造孔洞。"""
    header, offset = {}, 0
    for name, shape in specs:
        nbytes = int(np.prod(shape)) * 2                 # bf16
        header[name] = {'dtype': 'BF16', 'shape': list(shape),
                        'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(',', ':')).encode()
    pad = (8 - len(raw) % 8) % 8
    raw += b' ' * pad
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(raw)))
        fh.write(raw)
        fh.seek(8 + len(raw) + offset - 1)
        fh.write(b'\0')
    return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=200)
    ap.add_argument('--experts', type=int, default=60)
    ap.add_argument('--d', type=int, default=2048)
    ap.add_argument('--inter', type=int, default=1408)
    ap.add_argument('--vocab', type=int, default=151936)
    ap.add_argument('--base', default=os.environ.get('WOLF_BASE_MODEL', '<model>'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for name in ('config.json', 'tokenizer.json'):
        src = os.path.join(a.base, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, name))

    cfg_path = os.path.join(a.out, 'config.json')
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    cfg.update(num_hidden_layers=a.layers, num_experts=a.experts,
               hidden_size=a.d, intermediate_size=a.inter, vocab_size=a.vocab)
    json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    specs = tensor_specs(a.layers, a.experts, a.d, a.inter, a.vocab)
    fname = 'model-00001-of-00001.safetensors'
    total = write_sparse(os.path.join(a.out, fname), specs)
    index = {'metadata': {'total_size': total},
             'weight_map': {n: fname for n, _ in specs}}
    json.dump(index, open(os.path.join(a.out, 'model.safetensors.index.json'), 'w'), indent=1)

    print('%d 张量 | 表观体积 %.1f GB | 文件 %s' % (len(specs), total / 2 ** 30, fname))
    print('实际磁盘占用（稀疏）：', end='')
    os.system('du -sh %s' % a.out)


if __name__ == '__main__':
    main()
