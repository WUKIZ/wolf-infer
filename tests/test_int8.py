import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import torch
from qwen_backend import load, rms
from qwen_shard import ORDER, make_run, make_run_int8, write_shards, write_shards_int8
from supervisor import Supervisor

d = "/data/models/Qwen3-1.7B"
cfg, w = load(d)
w = {k: v.to("cuda", torch.bfloat16) for k, v in w.items()}
ids = [40, 100, 200, 300]
T = len(ids)
shapes = {k: tuple(w["model.layers.0." + k].shape) for k in ORDER}


def run_logits(spec, runner):
    ids_t = torch.tensor([ids], device="cuda")
    x = w["model.embed_tokens.weight"][ids_t]
    mask = torch.triu(torch.ones(T, T, device="cuda", dtype=torch.bool), 1)
    sup = Supervisor(spec, ["cuda"], 2 * 130_000_000, runner)
    out = sup.forward({"x": x, "mask": mask, "layer": 0})
    sup.close()
    xn = rms(out["x"], w["model.norm.weight"], cfg["rms_norm_eps"])
    return xn @ w["model.embed_tokens.weight"].T


db, d8 = tempfile.mkdtemp(), tempfile.mkdtemp()
got_b = run_logits(write_shards(w, cfg["num_hidden_layers"], db), make_run(cfg, shapes, torch.bfloat16))
got_8 = run_logits(write_shards_int8(w, cfg["num_hidden_layers"], d8),
                   make_run_int8(cfg, shapes, torch.bfloat16, d8))
sb = sum(os.path.getsize(os.path.join(db, f)) for f in os.listdir(db))
s8 = sum(os.path.getsize(os.path.join(d8, f)) for f in os.listdir(d8))
print("logits diff int8 vs bf16", float((got_8 - got_b).abs().max()))
print(f"size bf16 {sb/1e9:.2f} G  int8 {s8/1e9:.2f} G  ratio {s8/sb:.2f}")
print("int8 ok")
