"""测三个带宽/算力，决定每层专家走哪条路：PCIe 搬 / CPU 算 / 磁盘换。"""
import os, time
import torch
from core.tmpdir import TMP

MB = 64


def pcie_gbps():
    n = MB * 1024 * 1024 // 2
    src = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    dst = torch.empty(n, dtype=torch.bfloat16, device='cuda')
    for _ in range(2):
        dst.copy_(src)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    return 3 * MB / 1024 / ((time.perf_counter() - t) / 3)


def cpu_gflops():
    a = torch.randn(512, 512); b = torch.randn(512, 512)
    a @ b
    t = time.perf_counter()
    for _ in range(5):
        a @ b
    dt = (time.perf_counter() - t) / 5
    return 2 * 512 ** 3 / dt / 1e9


def disk_gbps():
    p = TMP + '/moe_shards/L00e00.bin'
    if not os.path.exists(p):
        return 0.0
    t = time.perf_counter()
    with open(p, 'rb') as f:
        f.read(MB * 1024 * 1024)
    return MB / 1024 / (time.perf_counter() - t)


def choose(bytes_per_exp, flop_per_exp):
    pc, cpu, dk = pcie_gbps(), cpu_gflops(), disk_gbps()
    ms_pcie = bytes_per_exp / (pc * 1e9) * 1e3
    ms_cpu = flop_per_exp / (cpu * 1e9) * 1e3
    ms_disk = bytes_per_exp / (dk * 1e9) * 1e3 if dk else 1e9
    best = min([('pcie', ms_pcie), ('cpu', ms_cpu), ('disk', ms_disk)], key=lambda x: x[1])
    print('pcie %.2f GB/s  cpu %.1f GFLOP/s  disk %.2f GB/s' % (pc, cpu, dk))
    print('per-expert ms: pcie %.2f  cpu %.2f  disk %.2f  -> %s'
          % (ms_pcie, ms_cpu, ms_disk, best[0]))


if __name__ == '__main__':
    choose(3 * 2048 * 1408 * 2, 3 * 2 * 5 * 2048 * 1408)
