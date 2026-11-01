"""提交前禁区检查：路径、密钥、隐私。命中任一条即拒绝提交。"""
import os
import re
import sys

BANNED = [
    r'^home/', r'^\.tmp/', r'__pycache__', r'\.pyc$',
    r'\.safetensors$', r'\.gguf$', r'\.npy$', r'\.jsonl$',
    r'(^|/)manifest\.json$', r'^logs/', r'^models/',
    r'^home/private/',
    r'(^|/)\.env$', r'\.key$',
]

# 密钥与凭据
SECRET = re.compile(r'ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'
                    r'|-----BEGIN [A-Z ]*PRIVATE KEY-----')

# 隐私：手机、身份证、邮箱、内网 IP、绝对家用路径
PRIVACY = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)'
                     r'|(?<!\d)\d{17}[\dXx](?!\d)'
                     r'|[\w.+-]+@[\w-]+\.[\w.]{2,}'
                     r'|\b(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b'
                     r'|/root/private_data|/public/home/')

DENY_FILE = '.privacy-deny'          # 本地关键词表，已在 .gitignore 里
deny = []
if os.path.exists(DENY_FILE):
    with open(DENY_FILE, encoding='utf-8') as fh:
        deny = [w.strip() for w in fh if w.strip() and not w.startswith('#')]

bad = []
SELF = 'tools/preflight.py'          # 扫描器自身含正则字面量，跳过

for path in sys.argv[1:]:
    if path == SELF:
        continue
    for pat in BANNED:
        if re.search(pat, path):
            bad.append((path, '禁区路径 ' + pat))
            break
    else:
        try:
            with open(path, encoding='utf-8', errors='ignore') as fh:
                for no, line in enumerate(fh, 1):
                    if SECRET.search(line):
                        bad.append((path, '第 %d 行疑似密钥' % no))
                        break
                    if PRIVACY.search(line):
                        bad.append((path, '第 %d 行疑似隐私' % no))
                        break
                    hit = [w for w in deny if w in line]
                    if hit:
                        bad.append((path, '第 %d 行含本地禁词 %r' % (no, hit[0])))
                        break
        except (OSError, UnicodeError):
            pass

if bad:
    print('拒绝提交，命中 %d 条：' % len(bad))
    for path, why in bad:
        print('  %-40s %s' % (path, why))
    sys.exit(1)
print('preflight 通过（%d 个文件）' % len(sys.argv[1:]))
