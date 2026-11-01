legacy_paged/

0.031 tok/s 时代的磁盘分页/分片实现，2026-11-01 从主干移出。

包含：core/pager·shard·shardfile·pipeline·qwen_shard·qwen_moe_shard
      qwen_tile·tile·llama_shard·sharded_infer·supervisor·backend
      tests/test_pipeline·bench_paged·test_compute·test_e2e
      tests/test_sharded_model·test_qwen_tile·bench_pcie_layer
      tests/test_mixed·test_from_disk · tools/build_shards·shard_ctl

保留原因：0.031 tok/s 是报告里的负结果起点，代码留档可复算。
不要引用回主干；现行引擎是 core/moe_batch.py。
