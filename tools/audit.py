"""静态审计：未使用 import / 空函数 / 魔数 / 待办标记 / 重复函数体。"""
import ast
import collections
import hashlib
import pathlib
import re

ROOT = pathlib.Path('.')
SKIP = {0, 1, 2, -1, 0.0, 1.0, 0.5, 100}
report = collections.defaultdict(list)
bodies = collections.defaultdict(list)


def norm(tree):
    return hashlib.md5(ast.dump(tree).encode()).hexdigest()


FILES = sorted(list(pathlib.Path('core').rglob('*.py')) + list(pathlib.Path('tests').rglob('*.py')))
MAGIC_ONLY = ('moe_pcie', 'moe_kv', 'moe_batch', 'moe_graph', 'plan', 'serve')

for f in FILES:
    src = f.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        report['语法错误'].append('%s:%s %s' % (f, exc.lineno, exc.msg))
        continue

    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split('.')[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == '__future__':
                continue
            for a in node.names:
                if a.name != '*':
                    imported[a.asname or a.name] = node.lineno
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name, line in imported.items():
        if name not in names and src.count(name) <= 1:
            report['未使用 import'].append('%s:%d %s' % (f, line, name))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body
                    if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            stub = body and all(isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                 and b.value.value is Ellipsis for b in body)
            bare_pass = body and all(isinstance(b, ast.Pass) for b in body) and not stub
            if bare_pass:
                src_line = src.split('\n')[node.body[-1].lineno - 1]
                if not src_line.split('pass', 1)[-1].strip().startswith('#'):
                    report['空函数'].append('%s:%d %s' % (f, node.lineno, node.name))
            elif len(body) >= 4:
                bodies[norm(ast.Module(body=body, type_ignores=[]))].append(
                    '%s:%d %s' % (f, node.lineno, node.name))
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            big = isinstance(node.value, float) or abs(node.value) >= 1000
            if node.value not in SKIP and big and any(k in f.stem for k in MAGIC_ONLY):
                report['魔数'].append('%s:%d %r' % (f, node.lineno, node.value))

    for i, line in enumerate(src.split('\n'), 1):
        if re.search(r'\b(TODO|FIXME|XXX|HACK)\b', line):
            report['待办标记'].append('%s:%d %s' % (f, i, line.strip()[:56]))

for items in bodies.values():
    if len(items) > 1:
        report['重复函数体'].append(' | '.join(items))

for key in ('语法错误', '重复函数体', '空函数', '未使用 import', '待办标记', '魔数'):
    rows = report.get(key, [])
    print('== %s (%d)' % (key, len(rows)))
    for r in rows[:12]:
        print('   ' + r)
