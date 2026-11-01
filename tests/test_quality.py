"""质量对比：bf16 分片 vs int8 分片，同一句话各生成 12 token。"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import torch
from transformers import AutoTokenizer
from qwen_backend import load, rms
from qwen_shard import ORDER, make_run, make_run_int8, write_shards, write_shards_int8
from supervisor import Supervisor

d = "/data/models/Qwen3-1.7B"
tk = AutoTokenizer.from_pretrained(d)
cfg, w = load(d)
w = {k: v.to("cuda", torch.bfloat16) for k, v in w.items()}
shapes = {k: tuple(w["model.layers.0." + k].shape) for k in ORDER}
emb, nrm = w["model.embed_tokens.weight"], w["model.norm.weight"]


def make_logits(spec_path, runner):
    def fn(seq):
        T = len(seq)
        x = emb[torch.tensor([seq], device="cuda")]
        mask = torch.triu(torch.ones(T, T, device="cuda", dtype=torch.bool), 1)
        sup = Supervisor(spec_path, ["cuda"], 2 * 130_000_000, runner)
        out = sup.forward({"x": x, "mask": mask, "layer": 0})
        sup.close()
        return (rms(out["x"], nrm, cfg["rms_norm_eps"]) @ emb.T)[0, -1]
    return fn


def gen(fn, seq, n=12):
    s = list(seq)
    for _ in range(n):
        s.append(int(fn(s).argmax()))
    return s


db, d8 = tempfile.mkdtemp(), tempfile.mkdtemp()
fn_b = make_logits(write_shards(w, cfg["num_hidden_layers"], db), make_run(cfg, shapes, torch.bfloat16))
fn_8 = make_logits(write_shards_int8(w, cfg["num_hidden_layers"], d8),
                   make_run_int8(cfg, shapes, torch.bfloat16, d8))
prompt = tk.encode("本地助手不麻的")
for name, fn in (("bf16", fn_b), ("int8", fn_8)):
    out = gen(fn, prompt)
    print(name, tk.decode(out))
print("quality ok")
