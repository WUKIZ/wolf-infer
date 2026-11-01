"""运行时执法：把 open/socket/Popen 换成会拒绝的版本。错误信息只有 EACCES。"""
from __future__ import annotations
import builtins, ipaddress, os, socket, subprocess
from pathlib import Path
from .kernel import CAGE

_installed = False
_orig_open = builtins.open
_orig_connect = socket.socket.connect
_orig_popen = subprocess.Popen
_allow_ips: set[str] = set()


def _resolve_allow() -> None:
    for host in CAGE.policy["net"]["hosts"]:
        try:
            for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
                _allow_ips.add(info[4][0])
        except OSError:
            continue


def install() -> bool:
    global _installed
    if _installed or os.environ.get("WOLF_CAGE") != "1":
        return False
    _resolve_allow()
    builtins.open = _open
    socket.socket.connect = _connect
    subprocess.Popen = _popen
    _installed = True
    CAGE.log("install", f"cage on, hosts={len(_allow_ips)}")
    return True


def _wrap(path) -> Path | None:
    if isinstance(path, int):
        return None
    try:
        return Path(os.fspath(path))
    except TypeError:
        return None


def _open(file, mode="r", *args, **kwargs):
    p = _wrap(file)
    if p is not None:
        if any(c in mode for c in "wax+"):
            if not CAGE.allow_write(p):
                CAGE.deny(f"write:{p}")
            if not CAGE.write_budget.take():
                CAGE.deny("write_budget")
        elif CAGE.deny_read(p):
            CAGE.deny(f"read:{p}")
    return _orig_open(file, mode, *args, **kwargs)


def _connect(self, address):
    if not isinstance(address, tuple) or len(address) < 2:
        return _orig_connect(self, address)
    host, port = str(address[0]), int(address[1])
    if CAGE.policy["net"]["deny_private"]:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and ip.is_loopback:
            return _orig_connect(self, address)     # 本机服务（webapp）放行
        if ip is not None and (ip.is_private or ip.is_link_local):
            CAGE.deny("private_ip")
    if host in CAGE.policy["net"]["hosts"]:
        pass
    else:
        try:
            addrs = {i[4][0] for i in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
        except OSError:
            CAGE.deny("dns")
        if not addrs or not addrs <= _allow_ips:
            CAGE.deny(f"host:{host}")
    if not CAGE.net_budget.take():
        CAGE.deny("net_budget")
    return _orig_connect(self, address)


def _popen(*args, **kwargs):
    CAGE.deny("spawn")


def active() -> bool:
    return _installed
