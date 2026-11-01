# wolf-infer

在 **16 GB 显存的 DCU** 上跑 **14.3B MoE**（Qwen1.5-MoE-A2.7B），纯 PyTorch，无自定义 kernel。
不是 vLLM 的封装：专家分页、批处理、KV 管理、批量 prefill 都是自己写的。

## 实测（全部为本仓库数据）

| 项目 | 数值 | 口径 |
|---|---|---|
| 吞吐 | **212.4 tok/s** | B=512，不同 prompt，greedy |
| 吞吐 | 21.6 / 71.1 / 130.3 tok/s | B=32 / 128 / 256 |
| 批量 prefill | **1.790 → 0.199 s/条（9.0x）** | 64 条等长 20 token |
| 长上下文 | 4096 token，prefill 18.24 s，10.2 GB | 单卡 |
| 外部对照 | HF(offload) 0.03 vs 本引擎 4.05 tok/s | B=1、greedy、4 token |

**口径警告**：早期 591.5 tok/s 是「同一句 prompt 复制 N 份」测的，专家并集退化成 4，
系统性高估 4–5 倍，已作废。上表全部为不同 prompt。

## 5 分钟看到第一个数

```bash
git clone https://github.com/WUKIZ/wolf-infer.git && cd wolf-infer
pip install "numpy<2" "transformers==4.46.3" tokenizers safetensors
export WOLF_MODEL=/path/to/Qwen1.5-MoE-A2.7B
python3 tests/bench.py --tag myrun
python3 tests/bench.py --compare snapshots/base5.json snapshots/myrun.json
```

第 4 步打印三个数：`通过 4/12 | 单条 2.25 | 多样性B32 13.3 | prefill 0.078 s/条`

第 5 步判三条：用例 PASS→FAIL、吞吐掉 >10%、prefill 变慢 >10%。

> 数字随硬件变化（DCU 单卡 16 GB）。一致就行，不必和仓库里的数一样。
## 支持范围
•
模型：Qwen1.5-MoE-A2.7B（24 层 / 60 专家 top-4 / 2.69B 激活）
•
硬件：DCU（DTK 2504x，torch 2.4.1+das），CUDA 未验证
•
并行：DP 实测 2.22x；PP 管道已验证未接引擎；EP 只买容量不买速度
## 局限
•
无自定义 kernel，专家分组 GEMM 用 PyTorch 反而更慢（已实测回滚）
•
未支持前缀缓存、投机解码、磁盘级 KV
•
只验证过一个模型
## 许可
Apache-2.0
## 质量评估与 top-k 政策（2026-10-31 定稿）

三套指标都测过，结论如下：

| 指标 | 结果 | 是否可用 |
|---|---|---|
| 前缀一致率 | top-3 0.34 / top-2 0.11–0.36 / top-1 0.05 | 不可用：受生成退化污染 |
| 关键词准确率（10 事实 + 4 算术） | 事实 top-4 5/10、top-3 7/10、top-2 4/10、top-1 3/10；算术四档全 0/4 | 部分可用：算术类该基座无能力 |
| 答案 NLL（20 条常识） | top-1 1.9207 / top-3 2.2347 / top-4 2.2485 / top-8 2.2882 / top-2 2.4697 | 不可用：对 top-k 非单调，且与生成质量背离 |

**结论**：没有廉价指标能判定降级影响，可靠结论需要人工标注的任务级准确率。

**政策**：

- `quality=high` → top-4（默认）
- `quality=fast` → top-3（吞吐 +16~19%，唯一无反证的降级档）
- top-2 不进策略，仅 `WOLF_TOPK=2` 显式启用，且只用于可被程序校验的输出
- top-1 / top-8 弃用

批内合并规则：同一批里多个请求取 `max(top)`，保证不会静默降到用户未要求的档位；响应回带 `top_k`。

## EP（专家并行）定位

EP 只买容量，不买速度：B=32 时 EP=2 为 26.24 tok/s，单卡为 85 tok/s。
原因：dense 部分完整复制 + CPU 核被两进程平分。通信本身不是瓶颈（NCCL 0.376 ms/次）。
适用场景：模型超出单卡显存时作为容量方案，不作性能方案。

## 测试环境口径

武汉容器：16 核 / 120 G（cgroup 视实例而定）/ DCU 15–16 GB × 2 / torch 2.4.1 + DTK。
东京（昆山）容器：16 核 / cgroup 55.8 G / DCU 15 GB × 2 / torch 2.4.1 + DTK 25.04.1。

三处必须标注，否则数字不可复现：
1. CPU 核数（同一份代码 7 核与 14 核差约 3%）
2. cgroup 内存（决定 pin 档位，24 G 全驻留在 55.8 G 容器装不下）
3. ROCm 显存不随进程退出释放，OOM 后必须重启容器

## 批量 prefill（2026-11-01）

问题：prefill 逐请求串行，512 条请求要 600 s，其中每层专家并集（上限 60）的固定开销
无法摊薄——这是服务端首 token 延迟的主因。

实现：等长请求打包进一次前向（`prefill_batch`），K/V 按请求切回各自 cache；
长度不齐的请求由调用方分桶。批量路径与逐条路径的末位 logits **maxdiff 0.000000**。

实测（64 条等长 20 token 请求）：

| 方式 | 耗时 | 每条 |
|---|---|---|
| 逐条 prefill | 114.54 s | 1.790 s |
| 批量 prefill | 12.77 s | 0.199 s |

**加速 9.0×**。按此外推，512 条请求的 prefill 从约 600 s 降到约 102 s。

另附 `last_only`：只对末位算 head 投影，长序列 prefill 3.6×（256 token：10.69 s→2.95 s），
但短请求只有 1.08×——head 成本与序列长度成正比，短请求的瓶颈是每层专家开销。

## 外部对照：HuggingFace transformers（2026-11-01）

环境：同机同卡（DCU 15.98 GB ×2，torch 2.4.1+dtk），transformers 4.46.3，greedy，B=1。

| 实现 | 装载 | 4 token 耗时 | 单请求吞吐 |
|---|---|---|---|
| HF `generate`（`device_map=auto`，强制 12 GiB 显存 + CPU offload） | 必须 offload | 155.67 s | **0.03 tok/s** |
| 本引擎（专家分页/常驻） | 原生加载 | 0.99 s | **4.05 tok/s** |

**约 135×**，口径相同（同模型、同卡、B=1、greedy、4 token）。

限定条件：HF 之所以 offload，是因为 bf16 权重 28 GB 超过单卡 15.98 GB；
若显存足够放下全部权重，该差距会大幅缩小。**此数字仅代表受限显存条件下的优势**，
不可推广为一般性能对比；聚合吞吐（B=512 的 212.4 tok/s）也不与 HF 单请求数字并列。

## 回归台（tests/bench.py，2026-11-01）

固定题库 12 条 + 三类负载，快照存 snapshots/<tag>.json。基线 snapshots/base4.json：

- 通过率 4/12（基线本身偏低，只看是否回退）
- 单条 2.14 tok/s（逐条独立上下文，含 prefill，每条 1.28 s）
- 多样性 B=32 13.5 tok/s（32 条不同 prompt，prefill 按等长分桶，组内仅 1-3 条）
- 批量 prefill 0.078 s/条（64 条等长 20 token，逐条对照 1.790）

判回归：python3 tests/bench.py --compare snapshots/base4.json snapshots/<new>.json
任一用例 PASS→FAIL、或 tok_s_single / tok_s_div32 掉 >10%、或 prefill_s_item 慢 >10%，即 FAIL。

口径：多样性 13.5 低于线上 21.6（这里 32 条长度全不同、分桶后组小）；单条 2.14 低于纯 decode 4.05（含每条约 1.28 s prefill）。均只在同档位同构造下横向比。

## 验收清单（每条可复现）

| 项目 | 证据脚本 | 关键数字 |
|---|---|---|
| 单机吞吐（多样性口径） | tests/prof_div.py | B=512 212.4 tok/s |
| 批量 prefill | tests/prefill_batch_gain.py | 9.0x，maxdiff 0.000000 |
| 长上下文 | tests/prof_long.py | 4096 token，prefill 18.24 s，10.2 GB |
| 确定性 | tests/parity_repeat.py | 两次逐 token 相同 |
| 外部对照 | tests/hf_baseline.py | HF 0.03 vs 本引擎 4.05 tok/s |
| 回归台 | tests/bench.py | snapshots/base4.json |
| 语义检索对照 | tests/bge_probe.py | LSA 13/20 = bge 13/20 |
| 搜索稳定性 | tests/search_stab.py | 12/12 命中，跨轮 0.86 |
| 可达性分析 | tools/reach.py | 110 入口 / 167 可达 / 16 不可达 |

口径红线：旧表（127.4/385.3/591.5/743.6）为同 prompt 复制，已作废不可宣传；
135x 只代表受限显存条件；13.5（回归台多样性 B32）与线上 21.6 不同构造，不可混用。

## 流水并行（core/pp.py，管道已验证）

验证范围：层切分 + 激活收发 + 两进程顺序，**不包含模型接入**。
测试：tests/pp_smoke.py（CPU + gloo，8 层均分给 2 进程，与全模型对拍）。

实测：层切分 rank0=(0,4) rank1=(4,8)，maxdiff 0.000e+00，exit=0。

未完成：把 moe_batch.decode 的层循环换成「本地层段 + pp.send/recv」，
        以及 GPU 上的真实激活传递（现在验的是 CPU float32 管道）。

## 流水并行（PP，已跑通）

两卡各持 12 层，层间只传一次激活（gloo，CPU 中转），末卡算 head 并把 token 回传首卡。

实测（`tests/pp_two.py`）：

```
单卡参考: ____\nA. 北
PP 切分   (0, 12) (12, 24)
PP 输出   ____\nA. 北
逐字一致  PASS
```

实现：`core/moe_batch.py` 的 `_pp_bounds / _pp_in / _pp_out`，默认 `(0, NL)` 时行为不变。
已知限制：gloo 不支持 CUDA 张量，激活需 `.cpu()` 中转；上 NCCL 可省掉这次拷贝。

## 专家并行（EP，容量线已验）

`core/ep.py`：专家按 `expert % world` 切分，每层只算本卡负责的专家，再 `all_reduce` 求和。
注意力、嵌入、共享专家每卡完整计算（这部分不随卡数减少）。

实测（`tests/ep_capacity.py`，两卡 / stream 模式 / nccl）：

```
rank0 显存 4.03 GB  __\nA. 北
rank1 显存 4.03 GB  __\nA. 北
两卡一致 PASS
```

**口径**：显存与单卡 stream 模式（3.49 GB）接近，因为该模式下专家本就不常驻显存；
EP 的容量收益在主机/磁盘侧——每卡只处理一半专家（720 / 1440）。
速度上 EP 不划算：B=32 实测 26.24 tok/s vs 单卡 85，差距来自 dense 复制与 14 核被两进程平分。

## 自适应档位（plan.py）

输入（主机内存、显存、卡数、参数量）→ 输出（模式、每层 pin 专家数、预期 tok/s）。

| 主机内存 | 模式 | pin/层 | 预测 tok/s | 实测 |
|---|---|---|---|---|
| 60.75 G | pinned | 13 | 3.33 | — |
| 80 G | pinned | 18 | 3.64 | — |
| 121.5 G | pinned | 60 | 6.28 | **6.32（-1%）** |
| 200 G | pinned | 60 | 6.28 | — |

**已修复的缺陷**：早期判档只看原始张量体积（28.6 G ≤ 51.6 G 就判 pinned），
漏了 `PIN_INFLATION = 3.4`（pinned 实占约为张量的 3.4 倍），导致 60 G 容器也被判为全量 pinned。
现在把膨胀系数纳入判档与每层开销，档位随内存正确降级。

注：另两条历史实测（1.16 / 4.05 tok/s）是用 `WOLF_NO_PIN=1` 强制 stream 测的，
与 plan 推荐的 pinned 档不同配置，不能直接算误差。

## 训练闭环（骨架已跑通）

| 文件 | 职责 | 自检 |
|---|---|---|
| `train/engine.py` | FSDP 分片 + 原子 checkpoint（含 step）+ 恢复 | 存→破坏→读回，参数逐位 PASS |
| `train/loader.py` | 语料流式读入 → 定长样本，字节偏移断点 | 断点偏移一致 PASS |
| `train/loop.py` | 梯度累积 + 裁剪 + 日志 | 3 步跑通，token 计数 192 正确 |
| `train/eval.py` | 验证 loss/ppl（按 token 加权）+ 贪心样例 | loss 11.98 / 160 token |

四个文件都可 `python3 train/xxx.py` 单独自检，不依赖真实语料。

**未完成**：还没在 48 亿字真实语料上跑过一轮；学习率调度、数据混比、分布式评测未接。

### 数据管线实测（真实语料）

`<data>/mix.txt`（314 MB），seq_len=512、batch=4：

```
20 批 x 4 x 512 = 40960 token，1.08 s -> 37917 token/s
```

训练时 GPU 消耗约几百 token/s，数据侧快两个数量级，不构成瓶颈。
`big.txt` 是 0 字节空文件，已被排除。


## 训练闭环（真模型 + 真语料，已跑通）

357.6M 的 GPT（`train/tiny.py`，词嵌入与输出头共享），词表 151936、d=1024、16 层。

```
step 1   loss 11.9327   (ln(151936)=11.93，初始化正确)
step 10  loss 11.8352   (单调下降)
10 步 24.6 s | 1662 token/s | 40960 token
```

数据侧 37.9k token/s，训练侧 1.66k token/s，数据不是瓶颈。

**未完成**：只跑了 10 步；学习率调度、断点续训（ckpt 已支持）、分布式评测未接。

## 无上限实验（进行中，已定位缺口）

用 `tools/make_scaled_model.py` 生成 **200 层 / 60 专家 / 表观 204 GB** 的合成模型
（稀疏文件，实际占 58 MB），与本仓库结构逐键对齐（每层 193 个张量，缺/多均为空）。

**结果：OOM，13.37 GiB / 15.98 GiB。**

根因不是模型跑不动，而是**专家驻留随层数线性增长**：

```
200 层 x 4 专家 x 3 矩阵 x 5.5 MB ≈ 13.2 GB   （与实测 13.37 GiB 吻合）
```

即当前实现是「模型变大，显存跟着涨」，还没做到「模型变大，显存不涨」。

**待做**：跨层专家 LRU 淘汰（`WOLF_EXPERT_BUDGET_MB`，默认 4096；超预算按最久未用淘汰）。
做完这一条，「204 GB 权重 / 16 GB 卡出 token」才成立。

### 训练实测（3000 步，单卡 DCU）

| 项 | 值 |
|---|---|
| 模型 | 357.6 M（词嵌入与输出头共享） |
| 数据 | mix.txt，seq 256 x batch 2 x 累积 8 = 4096 token/步 |
| 总步数 | 3000 |
| 总 token | 12,288,000 |
| 耗时 | 4822.8 s（1.61 s/步） |
| 吞吐 | 2548 token/s |
| loss | **11.9327 → 6.8862（-42%）** |
| ppl | 151936 → 978.72 |

首步 loss 精确等于 ln(151936)=11.93，说明初始化正确；3000 步内单调下降且无发散。
对照：该模型按 Chinchilla 需约 7 B token，本轮只训练了 12.3 M（0.18%），
定位是**验证训练闭环**，不是产出可用模型。

## 无上限实证（102.6 GB 权重 / 60 GB 内存 / 16 GB 卡）

用 `tools/make_scaled_model.py` 生成 100 层合成模型（稀疏文件，表观 102.6 GB，实际占 50 MB），
结构与真实 Qwen1.5-MoE 逐键对齐（每层 193 个张量，缺/多均为空）。

```
权重表观   102.6 GB   （远超 60 GB 主机内存，也远超 16 GB 显存）
warm        5.92 GB
prefill     8.29 GB
生成 3 token 成功，显存 8.30 GB
```

即：**权重总量大于内存与显存之和仍能推理**，靠的是专家按需换入 + LRU 淘汰。

**已知限制**：显存随层数线性增长约 24 MB/层（未定位到具体容器），
因此 200 层会 OOM；100 层可稳定运行。详见 `docs/TODO.md`。
