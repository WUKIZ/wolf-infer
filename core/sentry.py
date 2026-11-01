"""哨兵：看看谁在连我们。只观察、只记录，不反击、不伪装。

数据来自 /proc/net/tcp（和 tcp6），只统计连到本机监听端口的外部地址。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


def _ipv4(hexaddr: str) -> str:
    raw = bytes.fromhex(hexaddr)
    return ".".join(str(b) for b in raw[::-1])


def _split(addr: str) -> Tuple[str, int]:
    host, port = addr.split(":")
    return (host, int(port, 16))


def parse_tcp(lines: Iterable[str], port: int) -> List[Dict[str, object]]:
    """纯函数，方便测试。返回连到 port 的连接（含状态）。"""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            lhost, lport = _split(parts[1])
            rhost, rport = _split(parts[2])
        except ValueError:
            continue
        if lport != port:
            continue
        ip = _ipv4(rhost) if len(rhost) == 8 else "v6"
        out.append({"remote": ip, "remote_port": rport, "state": parts[3]})
    return out


def watch(port: int) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            text = open(fn).read().splitlines()[1:]
        except OSError:
            continue
        rows.extend(parse_tcp(text, port))
    peers = sorted({str(r["remote"]) for r in rows if r["remote"] not in ("127.0.0.1", "0.0.0.0")})
    established = [r for r in rows if r["state"] == "01"]
    return {"port": port, "total": len(rows), "established": len(established),
            "peers": peers[:20], "peer_count": len(peers)}


def alert(rows: Dict[str, object], threshold: int = 8) -> Optional[str]:
    """连接数超阈值就返回一句告警，否则 None。只用于记录，不自动反击。"""
    if int(rows.get("peer_count", 0)) >= threshold:
        return f"sentry: {rows['peer_count']} 个外部地址连着 {rows['port']}"
    return None
