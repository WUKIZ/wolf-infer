"""专家在 CPU 上算：只传激活（KB 级），不搬权重（MB 级）。

对应 FreeToken/ktransformers 的取舍：miss 的专家用本地内存计算，
PCIe 只传特征向量，省掉 16.5 MB 的权重搬运。
"""
import time
import torch
import torch.nn.functional as F

D, H = 2048, 1408          # Qwen1.5-MoE 的 hidden / moe_intermediate
EXPERT_MB = 3 * D * H * 2 / 1048576


def expert_cpu(x, wg, wu, wd):
    """x:(N,d) float32 在 CPU；wg,wu:(h,d)，wd:(d,h)。"""
    return (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T


def bench(n_list=(1, 4, 16, 64), threads=None):
    if threads:
        torch.set_num_threads(threads)
    wg = torch.randn(H, D) * 0.02
    wu = torch.randn(H, D) * 0.02
    wd = torch.randn(D, H) * 0.02
    print("expert %.1f MB  threads %d" % (EXPERT_MB, torch.get_num_threads()))
    for n in n_list:
        x = torch.randn(n, D)
        expert_cpu(x, wg, wu, wd)                      # 预热
        t0 = time.perf_counter()
        for _ in range(3):
            expert_cpu(x, wg, wu, wd)
        dt = (time.perf_counter() - t0) / 3
        print("  N=%-3d %.2f ms   (若走磁盘读 %.1f MB ≈ 16 ms)"
              % (n, dt * 1000, EXPERT_MB))


if __name__ == '__main__':
    bench()
