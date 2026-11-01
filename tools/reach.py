"""从真实入口做可达性分析，列出没有任何入口能到达的模块（只读，不删）。"""
import ast
import pathlib
import sys

ROOTS = ([str(x) for x in pathlib.Path('.').glob('*.py')] +
         [str(x) for x in pathlib.Path('tests').glob('*.py')] +
         [str(x) for x in pathlib.Path('tools').glob('*.py')])
SKIP = {'.tmp', '__pycache__', 'experiments', 'snapshots'}


def modname(p):
    return str(p)[:-3].replace('/', '.').replace('./', '')


FILES = {modname(p): p for p in pathlib.Path('.').rglob('*.py')
         if not any(x in p.parts for x in SKIP)}


def deps(cur):
    try:
        tree = ast.parse(FILES[cur].read_text(encoding='utf-8'))
    except SyntaxError:
        return set()
    pkg = cur.rsplit('.', 1)[0] if '.' in cur else ''
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            base = pkg + '.' + n.module if n.level else n.module
            out.add(base)
            for al in n.names:                     # from core import canon 这种
                out.add(base + '.' + al.name)
    return out


def resolve(name, seen):
    for cand in (name, 'core.' + name, name + '.__init__'):
        if cand in FILES and cand not in seen:
            return cand
    return None


seen, queue = set(), []
for r in ROOTS:
    m = r[:-3].replace('/', '.')
    if m in FILES:
        seen.add(m)
        queue.append(m)

while queue:
    cur = queue.pop()
    for d in deps(cur):
        hit = resolve(d, seen)
        if hit:
            seen.add(hit)
            queue.append(hit)

dead = sorted(set(FILES) - seen)
print('入口 %d 个 | 模块总数 %d | 可达 %d | 不可达 %d'
      % (len(ROOTS), len(FILES), len(seen), len(dead)))
print('--- 不可达 ---')
for m in dead:
    print('  %-32s %s' % (m, FILES[m]))
