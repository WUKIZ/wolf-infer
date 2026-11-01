"""两进程流水并行：各持 12 层，gloo 传激活。"""
import json, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, '/data/wolf/core')
from qwen_backend import rms
from qwen_moe import attention
from qwen_moe_shard import load_dense, paged_moe
from pager import Pager
from shard import Shard
from core.tmpdir import TMP

D = '/data/models/Qwen1.5-MoE-A2.7B'
S = TMP + '/moe_shards'
NEW = 12


def build_shards(cfg):
    d, h = cfg['hidden_size'], cfg['moe_intermediate_size']
    n = 3 * d * h * 2
    out = []
    for L in range(cfg['num_hidden_layers']):
        for e in range(cfg['num_experts']):
            out.append(Shard(len(out), os.path.join(S, 'L%02de%02d.bin' % (L, e)),
                             0, n, L, L + 1, expert=e))
    return out


def worker(rank, world, port):
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:%d' % port,
                            rank=rank, world_size=world)
    cfg = json.load(open(os.path.join(D, 'config.json')))
    head = torch.load(os.path.join(S, 'head.pt'), map_location='cpu')
    lo, hi = (0, 12) if rank == 0 else (12, 24)
    ts = {L: load_dense(os.path.join(S, 'L%02d.dense.pt' % L)) for L in range(lo, hi)}
    pager = Pager(300 * 1024 * 1024)
    by = {(s.layer_from, s.expert): s for s in build_shards(cfg)}
    emb = head['model.embed_tokens.weight'].to('cuda', torch.bfloat16)
    nrm = head['model.norm.weight'].to('cuda', torch.bfloat16)
    lm = head['lm_head.weight'].to('cuda', torch.bfloat16)
    from transformers import AutoTokenizer
    _tk = AutoTokenizer.from_pretrained(D)
    seq = _tk.encode('中国的首都是') if rank == 0 else None
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(D) if rank == 0 else None
    for _ in range(NEW):
        if rank == 0:
            x = emb[torch.tensor([seq], device='cuda')]
            for L in range(0, 12):
                x = attention(ts[L], x, cfg)
                x = paged_moe(L, ts[L], x, cfg, pager, by)
            dist.send(torch.tensor(list(x.shape), dtype=torch.int64), dst=1)
            dist.send(x.view(torch.int16).cpu().contiguous(), dst=1)
            tok = torch.zeros(1, dtype=torch.int64)
            dist.recv(tok, src=1)
            seq.append(int(tok))
        else:
            shp = torch.zeros(3, dtype=torch.int64)
            dist.recv(shp, src=0)
            x = torch.zeros(tuple(int(i) for i in shp), dtype=torch.int16)
            dist.recv(x, src=0)
            x = x.to('cuda').view(torch.bfloat16)
            for L in range(12, 24):
                x = attention(ts[L], x, cfg)
                x = paged_moe(L, ts[L], x, cfg, pager, by)
            x = rms(x, nrm, cfg['rms_norm_eps'])
            tok = int((x @ lm.T)[0, -1].argmax())
            dist.send(torch.tensor([tok], dtype=torch.int64), dst=0)
    if rank == 0:
        print(tk.decode(seq))
    print('rank', rank, 'resident MB', pager.resident_bytes() // 1048576,
          'evictions', pager.stats.evictions)
    pager.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(2, 29544), nprocs=2, join=True)
