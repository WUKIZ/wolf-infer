"""一键重建 /tmp/moe_shards：head + 24 层 dense + 1440 专家片。"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
import torch
from safetensors import safe_open
from qwen_moe import load_moe
from qwen_moe_shard import write_experts, write_dense
from core.tmpdir import TMP

D = '/data/models/Qwen1.5-MoE-A2.7B'
OUT = os.environ.get('MOE_SHARDS', TMP + '/moe_shards')
os.makedirs(OUT, exist_ok=True)

idx = json.load(open(os.path.join(D, 'model.safetensors.index.json')))['weight_map']
fs = {}
def get(k):
    fn = idx[k]
    if fn not in fs:
        fs[fn] = safe_open(os.path.join(D, fn), framework='pt')
    return fs[fn].get_tensor(k)

head = {k: get(k) for k in ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']}
torch.save(head, os.path.join(OUT, 'head.pt'))

cfg, w = load_moe(D, device='cpu', dtype=torch.bfloat16)
dense, shards = write_experts(w, cfg, OUT)
write_dense(dense, OUT)
print('shards %d  dense %d  head 1  -> %s' % (len(shards), len(dense), OUT))
