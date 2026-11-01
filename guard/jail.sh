#!/bin/sh
# 第三层（OS 级）：bubblewrap 再套一层。Android proot 里没有 bwrap，VPS/PC 上用。
# 用法：CAGE_NET=1 ./guard/jail.sh ./bin/py main.py --status
set -eu
if ! command -v bwrap >/dev/null 2>&1; then
  echo "[jail] 没有 bwrap，跳过 OS 层（Python 层仍在）" >&2
  exec "$@"
fi
NET="--unshare-net"
[ "${CAGE_NET:-0}" = "1" ] && NET=""
exec bwrap --die-with-parent --ro-bind /usr /usr --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 --ro-bind /etc /etc --proc /proc --dev /dev \
  --bind "$PWD" "$PWD" --tmpfs /tmp $NET --chdir "$PWD" "$@"
