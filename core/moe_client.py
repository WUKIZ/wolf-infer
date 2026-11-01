"""本地 MoE 推理服务客户端，对接 serve.py 的 HTTP 接口。

环境变量：WOLF_MOE_URL（默认 http://127.0.0.1:8080/generate）、WOLF_MOE_TIMEOUT（秒）
"""
import json
import os
import urllib.error
import urllib.request

URL = os.environ.get('WOLF_MOE_URL', 'http://127.0.0.1:8080/generate')
TIMEOUT = float(os.environ.get('WOLF_MOE_TIMEOUT', '30'))


def generate(prompt, max_tokens=32):
    payload = json.dumps({'prompt': prompt, 'max_tokens': max_tokens}).encode('utf-8')
    req = urllib.request.Request(URL, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError('本地 MoE 服务不可用：%s' % exc) from exc
    text = body.get('text')
    if not isinstance(text, str):
        raise RuntimeError('响应缺少 text 字段：%r' % body)
    return text


def alive():
    try:
        generate('测试', 1)
        return True
    except RuntimeError:
        return False
