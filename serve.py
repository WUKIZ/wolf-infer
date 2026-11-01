"""最小 HTTP 服务：连续批处理 + KV cache。POST /gen {prompt, n} -> {text, tok_s}"""
import json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
from core.moe_batch import BatchEngine
from tokenizers import Tokenizer
from moe_pcie import D_DIR


# quality → 专家数。批内取最大值，保证不会静默降到用户没要求的档位。
TOPK_LEVELS = {'high': 4, 'fast': 3}
_TOPK_ENV = os.environ.get('WOLF_TOPK')
if _TOPK_ENV:
    TOPK_LEVELS = dict.fromkeys(TOPK_LEVELS, int(_TOPK_ENV))


class Scheduler:
    def __init__(self, max_batch=8):
        self.eng = BatchEngine().warm()
        self.tk = Tokenizer.from_file(os.path.join(D_DIR, 'tokenizer.json'))
        self.max_batch = max_batch
        self.waiting = Queue()
        self.active = []

    def submit(self, req):
        self.waiting.put(req)

    def _admit(self):
        batch = []
        while len(self.active) + len(batch) < self.max_batch and not self.waiting.empty():
            batch.append(self.waiting.get_nowait())
        if not batch:
            return
        top = max([r['top'] for r in batch] + [a['top'] for a in self.active])
        self.eng.cfg['num_experts_per_tok'] = top
        groups = {}
        for r in batch:
            r['top'] = top
            ids = self.tk.encode(r['prompt']).ids
            groups.setdefault(len(ids), []).append((r, ids))
        for rs in groups.values():
            lg = self.eng.prefill_batch([(ids, r['cache']) for r, ids in rs], last_only=True)
            for k, (r, _) in enumerate(rs):
                r['tok'] = int(lg[k, -1].argmax())
                r['n'] = 0
                r['t0'] = time.perf_counter()
                self.active.append(r)

    def loop(self):
        while True:
            self._admit()
            if not self.active:
                time.sleep(0.002)
                continue
            toks = [r['tok'] for r in self.active]
            self.eng.cfg['num_experts_per_tok'] = max(r['top'] for r in self.active)
            lg = self.eng.decode(toks, [r['cache'] for r in self.active])
            for i, r in enumerate(list(self.active)):
                r['tok'] = int(lg[i].argmax())
                r['out'].append(r['tok'])
                r['n'] += 1
                if r['n'] >= r['want'] or r['tok'] in r['stop']:
                    dt = time.perf_counter() - r['t0']
                    r['result'] = {'text': self.tk.decode(r['out']),
                                   'tok_s': round(r['n'] / dt, 2)}
                    r['done'].set()
                    self.active.remove(r)


SCHED = Scheduler()


MAX_TOKENS = 512
MAX_PROMPT_CHARS = 4096
LIMIT_PER_SEC = int(os.environ.get('WOLF_RATE', '32'))


class RateLimiter:
    """每秒令牌桶：超过上限直接 429，不排队。"""

    def __init__(self, per_sec):
        self.per_sec = per_sec
        self._window = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def allow(self):
        now = time.monotonic()
        with self._lock:
            if now - self._window >= 1.0:
                self._window, self._count = now, 0
            if self._count >= self.per_sec:
                return False
            self._count += 1
            return True


LIMITER = RateLimiter(LIMIT_PER_SEC)


class Handler(BaseHTTPRequestHandler):
    server_version = 'wolf-moe/0.2'

    def log_message(self, fmt, *args):
        pass                    # 访问日志太吵，刻意静默

    def _json(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.split('?')[0] == '/health':
            self._json(200, {'ok': True, 'active': len(SCHED.active),
                             'waiting': SCHED.waiting.qsize()})
            return
        self._json(404, {'error': 'not found'})

    def do_POST(self):
        if self.path.split('?')[0] != '/generate':
            self._json(404, {'error': 'not found'})
            return
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            self._json(400, {'error': 'bad json'})
            return

        prompt = str(body.get('prompt', ''))[:MAX_PROMPT_CHARS]
        try:
            want = int(body.get('max_tokens', 32))
        except (TypeError, ValueError):
            want = 0
        if not prompt or want <= 0 or want > MAX_TOKENS:
            self._json(400, {'error': 'prompt/max_tokens invalid'})
            return
        quality = str(body.get('quality', 'high'))
        if quality not in TOPK_LEVELS:
            self._json(400, {'error': 'quality must be high or fast'})
            return
        if not LIMITER.allow():
            self._json(429, {'error': 'too many requests'})
            return

        req = {'prompt': prompt, 'want': want, 'cache': {}, 'out': [],
               'top': TOPK_LEVELS[quality],
               'stop': body.get('stop', []),
               'done': threading.Event(), 'id': uuid.uuid4().hex[:8]}
        SCHED.submit(req)
        if not req['done'].wait(120):
            self._json(504, {'error': 'timeout'})
            return
        result = dict(req['result'])
        result['top_k'] = req['top']
        self._json(200, result)

if __name__ == '__main__':
    threading.Thread(target=SCHED.loop, daemon=True).start()
    port = int(os.environ.get('PORT', 8080))
    print('serving on :%d (batch %d)' % (port, SCHED.max_batch), flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
