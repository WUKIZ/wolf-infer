"""DCU 冒烟：设备可用性、bf16、矩阵吞吐、分片单层前向在设备上的数值一致。"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import numpy as np
import torch
from types import SimpleNamespace
from local_backend import layer_keys, forward_layer as np_forward
from torch_backend import forward_layer as t_forward

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
    torch.cuda.synchronize()


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


a = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
try:
    _ = a @ a
    print("bf16 matmul ok")
except RuntimeError as exc:
    print("bf16 不可用:", exc)

x = torch.randn(2048, 2048, device=dev)
for _ in range(3):
    _ = x @ x
sync()
t0 = time.time()
n = 20
for _ in range(n):
    _ = x @ x
sync()
dt = (time.time() - t0) / n
print(f"matmul {dt * 1e3:.2f} ms, {2 * 2048 ** 3 / dt / 1e12:.2f} TFLOPS")

d, H, T = 128, 8, 6
rng = np.random.default_rng(0)
W = {k: rng.normal(0, 0.05, s).astype(np.float32) for k, s in layer_keys(d)}
for g in ("ln1_g", "ln2_g"):
    W[g][:] = 1.0
for b in ("ln1_b", "ln2_b"):
    W[b][:] = 0.0
xt = rng.normal(0, 1, (2, T, d)).astype(np.float32)
mask = np.triu(np.ones((T, T), bool), 1)
ref = np_forward(SimpleNamespace(cfg=SimpleNamespace(n_embd=d, n_head=H), params=W), 0, xt, mask)
got = t_forward({k: torch.tensor(v, device=dev) for k, v in W.items()},
                torch.tensor(xt, device=dev), torch.tensor(mask, device=dev),
                {"n_embd": d, "n_head": H}).cpu().numpy()
print("layer diff", float(np.abs(got - ref).max()))
print("dcu smoke ok")
