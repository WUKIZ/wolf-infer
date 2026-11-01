"""训练核：参数分片（FSDP）+ 优化器 + 分片检查点。

多卡时每卡只持 1/N 参数与状态；10T 场景下 DP 维交给 FSDP，专家维再叠 EP。
"""
import json
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from core.tmpdir import TMP

META = 'meta.json'


def init(backend=None, rank=None, world=None):
    backend = backend or ('nccl' if torch.cuda.is_available() else 'gloo')
    rank = int(os.environ.get('RANK', 0)) if rank is None else rank
    world = int(os.environ.get('WORLD_SIZE', 1)) if world is None else world
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29555')
    dist.init_process_group(backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    return dist.get_rank(), dist.get_world_size()


def shard(model):
    """整模型分片（FULL_SHARD）；小层也一起切，避免多包一层。"""
    return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None)


def save(model, opt, path, step=0):
    """原子保存：写 path.tmp，成功后整目录替换 path。"""
    tmp = path + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    dcp.save({'model': model.state_dict(), 'opt': opt.state_dict()}, checkpoint_id=tmp)
    with open(os.path.join(tmp, META), 'w', encoding='utf-8') as fh:
        json.dump({'step': step}, fh)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def load(model, opt, path):
    """恢复到给定模型/优化器，返回 meta（含 step）。"""
    dcp.load({'model': model.state_dict(), 'opt': opt.state_dict()}, checkpoint_id=path)
    meta_path = os.path.join(path, META)
    if not os.path.exists(meta_path):
        return {'step': 0}
    with open(meta_path, encoding='utf-8') as fh:
        return json.load(fh)


if __name__ == '__main__':
    rank, world = init()
    torch.manual_seed(rank)
    net = shard(torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.GELU(),
        torch.nn.Linear(64, 64)).cuda())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(4, 64, device='cuda')

    net(x).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()

    path = os.path.join(TMP, 'ckpt_demo')
    save(net, opt, path, step=3)
    before = [p.detach().clone() for p in net.parameters()]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)                      # 故意破坏，验 load 真恢复
    meta = load(net, opt, path)
    same = all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    if rank == 0:
        print('step 恢复 %d | 参数还原 %s' % (meta['step'], 'PASS' if same else 'FAIL'))
    dist.destroy_process_group()
