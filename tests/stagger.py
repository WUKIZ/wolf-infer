"""错峰并发：请求陆续进入/退出，验证连续批处理路径与结果正确性。"""
import json
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8080/generate'
PROMPTS = ['中国的首都是', '水的化学式是', '太阳从哪边升起', '一天有多少小时'] * 3
EXP = ['北京', 'H2O', '东', '24']
res = {}


def call(i, prompt):
    t0 = time.time()
    body = json.dumps({'prompt': prompt, 'max_tokens': 16}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    res[i] = (PROMPTS[i % 4], d['text'], d['top_k'], time.time() - t0)


t_all = time.time()
ths = [threading.Thread(target=call, args=(i, p)) for i, p in enumerate(PROMPTS)]
for t in ths:
    t.start()
    time.sleep(0.25)
for t in ths:
    t.join()

ok = sum(1 for i, (_, txt, _) in res.items()
         if EXP[i % 4] in next((ln for ln in txt.splitlines() if ln.strip()), ''))
print('完成 %d/%d  命中 %d/%d  墙钟 %.1f s'
      % (len(res), len(PROMPTS), ok, len(PROMPTS), time.time() - t_all))
for i in sorted(res):
    print('  %-8s %.1fs top_k=%d %r' % (res[i][0], res[i][3], res[i][2], res[i][1][:26]))
