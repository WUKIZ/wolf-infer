#!/usr/bin/env python3
"""笼的自检：令牌、写入、读取、审计、看门狗。"""
import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guard.kernel import Cage, CapabilityError

ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"PASS {name}")
    else: fail += 1; print(f"FAIL {name}")

tmp = Path(tempfile.mkdtemp())
_n = [0]
def fresh():
    _n[0] += 1
    c = Cage()
    c.audit_file = tmp / f"audit{_n[0]}.jsonl"
    c.halt_file = tmp / "HALT"
    return c

c = fresh()
t = c.mint("net.fetch", "https://api.deepseek.com")
c.check(t, "net.fetch", "https://api.deepseek.com"); check("token 正常通过", True)
try:
    c.check(t, "net.fetch", "https://evil.com"); check("token 错配", False)
except CapabilityError as e:
    check("token 错配被拒且信息无细节", str(e).endswith("Permission denied"))

c = fresh()
try:
    c.check("garbage", "net.fetch", "x"); check("畸形 token", False)
except CapabilityError:
    check("畸形 token 被拒", True)

c = fresh()
ROOT = Path(__file__).resolve().parent.parent
check("写 home 允许", c.allow_write(ROOT / "home/x.txt"))
check("写 /etc 拒绝", not c.allow_write(Path("/etc/passwd")))
check("读 policy 拒绝", c.deny_read(ROOT / "guard/policy.json"))
check("读 corpus 允许", not c.deny_read(ROOT / "home/corpus.txt"))
check("host 白名单", c.allow_host("api.deepseek.com", 443))
check("host 非白名单", not c.allow_host("example.com", 443))

c = fresh()
for _ in range(c.policy["budget"]["denials_before_halt"]):
    try: c.deny("probe")
    except CapabilityError: pass
check("越线后写停机标记", c.halt_file.exists())

lines = (tmp / f"audit{_n[0]}.jsonl").read_text("utf-8").strip().splitlines()
import json
prev, chain_ok = "genesis", True
for line in lines:
    rec = json.loads(line)
    if rec["prev"] != prev: chain_ok = False
    import hashlib
    prev = hashlib.sha256(json.dumps(rec, ensure_ascii=False).encode()).hexdigest()[:16]
check("审计链首尾相接", chain_ok)

try:
    from guard.runtime import install
    import os
    os.environ["WOLF_CAGE"] = "1"
    check("运行时装载", install())
except Exception as e:  # noqa: BLE001
    check(f"运行时装载 {e}", False)

print(f"\n{ok} PASS / {fail} FAIL")
raise SystemExit(0 if fail == 0 else 1)
