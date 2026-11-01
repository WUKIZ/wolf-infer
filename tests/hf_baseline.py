"""外部对照：HuggingFace transformers 跑同一模型，测 prefill + decode 速度。"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = '/data/models/Qwen1.5-MoE-A2.7B'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0: '12GiB', 'cpu': '100GiB'})
print('加载 %.1f s，显存 %.1f GB' % (time.perf_counter() - t0,
                                 torch.cuda.memory_allocated() / 2 ** 30))
ids = tok('中国的首都是', return_tensors='pt').input_ids.to(model.device)
gen = 4
t0 = time.perf_counter()
out = model.generate(ids, max_new_tokens=gen, do_sample=False)
dt = time.perf_counter() - t0
print('HF %d token 用时 %.2f s → %.2f tok/s' % (gen, dt, gen / dt))
print('文本:', repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
