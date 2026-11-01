"""观察层。看环境，算暴露度，在被发现前一刻回家。

暴露度 = 外部连接 + 端口扫描 + 文件被翻动 + 指纹查询 的加权和。
当暴露度穿越阈值，或者斜率（每秒新增）预示 lead_time 内会越线 → 立即回家。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from dataclasses import dataclass

TCP_ESTABLISHED = "01"
TCP_SYN_SENT = "02"
TCP_TIME_WAIT = "06"


def _hex_ip(hexip: str) -> str:
    # /proc/net/tcp 里是小端十六进制
    b = bytes.fromhex(hexip)
    return ".".join(str(x) for x in b[::-1])


def _read_tcp(path: str) -> List[Tuple[str, int, str, str]]:
    rows: List[Tuple[str, int, str, str]] = []
    try:
        lines = Path(path).read_text().splitlines()[1:]
    except OSError:
        return rows
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        local, remote, st = parts[1], parts[2], parts[3]
        try:
            lip, lport = local.split(":")
            rip, rport = remote.split(":")
            rows.append((_hex_ip(lip), int(lport, 16), _hex_ip(rip), st))
        except ValueError:
            continue
    return rows


TRACERS = {"strace", "ltrace", "gdb", "lldb", "tcpdump", "wireshark", "tshark",
           "frida", "r2", "radare2", "objdump", "perf", "inotifywait"}


class InotifyWatch:
    """直接向 libc 要 inotify —— Python 标准库没有这层。

    mtime 只看得到「谁写了」，看不到「谁在读」。敌人读我们的规则文件不会改 mtime，
    但会触发 IN_ACCESS。只认 IN_ACCESS：写是我们自己的心跳，读不是。
    """

    IN_ACCESS = 0x00000001
    IN_MODIFY = 0x00000002
    IN_ATTRIB = 0x00000004
    IN_CLOSE_WRITE = 0x00000008
    IN_OPEN = 0x00000020
    IN_Q_OVERFLOW = 0x00004000
    IN_IGNORED = 0x00008000

    MASK = IN_ACCESS | IN_OPEN | IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE
    EVENT = struct.Struct("iIII")      # wd, mask, cookie, name_len
    O_NONBLOCK = 0o4000
    O_CLOEXEC = 0o2000000

    def __init__(self, dirs: List[Path]) -> None:
        self.available = False
        self.fd = -1
        self._paths: Dict[int, Path] = {}
        try:
            libname = ctypes.util.find_library("c") or "libc.so.6"
            self._libc = ctypes.CDLL(libname, use_errno=True)
            self.fd = self._libc.inotify_init1(self.O_NONBLOCK | self.O_CLOEXEC)
        except (OSError, AttributeError):
            return
        if self.fd < 0:
            self.fd = -1
            return
        for d in dirs:
            try:
                wd = self._libc.inotify_add_watch(
                    self.fd, str(d).encode("utf-8"), self.MASK)
            except OSError:
                continue
            if wd >= 0:
                self._paths[wd] = Path(d)
        self.available = bool(self._paths)

    def poll(self) -> List[Tuple[Path, int]]:
        """把队列读空。非阻塞，没有事件就是空列表。"""
        if not self.available or self.fd < 0:
            return []
        out: List[Tuple[Path, int]] = []
        while True:
            try:
                buf = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                self.available = False
                break
            if not buf:
                break
            off = 0
            while off + self.EVENT.size <= len(buf):
                wd, mask, _cookie, name_len = self.EVENT.unpack_from(buf, off)
                off += self.EVENT.size
                raw = buf[off:off + name_len].split(b"\0", 1)[0]
                off += name_len
                base = self._paths.get(wd)
                if base is None or mask & self.IN_Q_OVERFLOW:
                    continue
                named = raw.decode("utf-8", "replace")
                out.append((base / named if named else base, mask))
        return out

    def close(self) -> None:
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = -1
        self.available = False



@dataclass
class Exposure:
    external_conns: int = 0
    scan_hits: int = 0
    file_touches: int = 0
    signature_hits: int = 0
    core_writes: int = 0      # core/ 被写（inotify，写是我们自己也不该发生的事）
    read_hits: int = 0        # 门槛文件被 read（inotify IN_ACCESS）
    fd_hits: int = 0          # 别的进程持有我们树里的 fd
    tracer_hits: int = 0      # strace / gdb / inotifywait 一类在场
    found: tuple = ()
    score: float = 0.0
    slope: float = 0.0  # 单位：分/秒

    def dict(self) -> Dict[str, object]:
        return {
            "external_conns": self.external_conns,
            "scan_hits": self.scan_hits,
            "file_touches": self.file_touches,
            "signature_hits": self.signature_hits,
            "core_writes": self.core_writes,
            "read_hits": self.read_hits,
            "fd_hits": self.fd_hits,
            "tracer_hits": self.tracer_hits,
            "found": list(self.found),
            "score": round(self.score, 2),
            "slope": round(self.slope, 2),
        }



class Observer:
    W_CONN = 12.0
    W_SCAN = 25.0
    W_TOUCH = 8.0
    W_CORE_WRITE = 25.0      # core/ 被写：要么是我们自己在 refreeze，要么不是我们
    W_THRESHOLD_READ = 0.0   # 读没法归因到进程，先只记数（见下）
    W_FD = 15.0              # 别的进程攥着我们树里的文件
    W_TRACER = 45.0          # tracer 在场，一项就过阈值

    # 斜率窗口与最小采样跨度：第一次 tick 的 dt 可能只有几微秒，
    # 拿它算 (12-0)/7e-6 = 1.6e6 分/秒，eta 直接归零 → 明明只有 12 分却立刻回家。
    MIN_DT_S = 0.05        # 小于这个的采样不算斜率，沿用上一次
    MIN_SPAN_S = 1.0       # 至少观测这么久，斜率先行规则才允许开火
    WINDOW_S = 3.0         # EWMA 的时间常数
    HISTORY_MAX = 240      # 有界历史，长跑不再涨内存

    def __init__(
        self,
        watch_paths: Optional[List[Path]] = None,
        threshold: float = 40.0,
        lead_time_s: float = 1.5,
        watch_dirs: Optional[List[Path]] = None,
        root: Optional[Path] = None,
    ) -> None:
        self.watch_paths = watch_paths or []
        self.threshold = threshold
        self.lead_time_s = lead_time_s
        self.root = Path(root) if root else None
        self.watch_dirs = [Path(d) for d in (watch_dirs or []) if Path(d).exists()]
        self._ino = InotifyWatch(self.watch_dirs) if self.watch_dirs else None
        self._ino_ok = bool(self._ino and self._ino.available)
        self._mtimes: Dict[str, float] = {}
        self._prev_score = 0.0
        self._prev_ts = time.monotonic()
        self._t0 = self._prev_ts
        self._base_score = 0.0
        self._slope = 0.0
        self._samples = 0
        self.history: Deque[Exposure] = deque(maxlen=self.HISTORY_MAX)

    # ---------- 采样 ----------
    def _connections(self) -> Tuple[int, int]:
        ext = 0
        scans = 0
        seen: set[int] = set()
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            for lip, lport, rip, st in _read_tcp(table):
                if rip.startswith("127.") or rip == "::1" or rip == "0.0.0.0":
                    continue
                if st == TCP_ESTABLISHED:
                    ext += 1
                elif st == TCP_SYN_SENT:
                    seen.add(lport)
        scans = max(0, len(seen) - 1)
        return ext, scans

    def _touches(self) -> int:
        n = 0
        for p in self.watch_paths:
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            key = str(p)
            if key in self._mtimes and m != self._mtimes[key]:
                n += 1
            self._mtimes[key] = m
        return n

    # ---------- inotify / 进程面 ----------
    def _ino_events(self) -> Tuple[int, int]:
        """返回 (core 被写, 门槛被读)。

        读没法归因到进程（inotify 不带 pid，fanotify 要 root），所以我们自己
        每轮读门槛也会算进去 —— 这就是 W_THRESHOLD_READ 默认 0 的原因：
        先只记数，别让它把自己的正常读当成敌人。
        """
        if not self._ino_ok or self._ino is None:
            return 0, 0
        core_w = th_read = 0
        for path, mask in self._ino.poll():
            parts = path.parts
            if mask & (InotifyWatch.IN_MODIFY | InotifyWatch.IN_ATTRIB
                       | InotifyWatch.IN_CLOSE_WRITE):
                if "core" in parts:
                    core_w += 1
            if mask & InotifyWatch.IN_ACCESS and "threshold" in parts:
                th_read += 1
        return core_w, th_read

    def _process_signals(self) -> Tuple[int, int]:
        """返回 (攥着我们树里 fd 的外部进程数, tracer 进程数)。"""
        me = os.getpid()
        fd_hits = tracer_hits = 0
        root = str(self.root) if self.root else None
        try:
            procs = [p for p in Path("/proc").iterdir() if p.name.isdigit()]
        except OSError:
            return 0, 0
        for p in procs:
            pid = int(p.name)
            if pid == me:
                continue
            try:
                comm = (p / "comm").read_text("utf-8", "replace").strip()
            except OSError:
                continue
            if comm in TRACERS:
                tracer_hits += 1
            if not root:
                continue
            try:
                for fd in (p / "fd").iterdir():
                    try:
                        target = os.readlink(fd)
                    except OSError:
                        continue
                    if target.startswith(root):
                        fd_hits += 1
                        break
            except OSError:
                continue
        return fd_hits, tracer_hits

    def close(self) -> None:
        if self._ino is not None:
            self._ino.close()
        self._ino_ok = False

    def probe(self) -> Exposure:
        ext, scans = self._connections()
        core_w, th_read = self._ino_events()
        fd_hits, tracer_hits = self._process_signals()
        exp = Exposure(
            external_conns=ext,
            scan_hits=scans,
            file_touches=self._touches(),
            core_writes=core_w,
            read_hits=th_read,
            fd_hits=fd_hits,
            tracer_hits=tracer_hits,
        )
        exp.score = (
            ext * self.W_CONN
            + scans * self.W_SCAN
            + exp.file_touches * self.W_TOUCH
            + core_w * self.W_CORE_WRITE
            + th_read * self.W_THRESHOLD_READ
            + fd_hits * self.W_FD
            + tracer_hits * self.W_TRACER
        )
        now = time.monotonic()
        dt = now - self._prev_ts
        if self._samples == 0:
            # 第一笔只当基线：建观察器到第一次 tick 之间的 dt 不可信
            self._prev_score = exp.score
            self._prev_ts = now
            self._samples = 1
        elif dt >= self.MIN_DT_S:
            instant = (exp.score - self._prev_score) / dt
            alpha = min(1.0, dt / self.WINDOW_S)
            self._slope = (1 - alpha) * self._slope + alpha * instant
            self._prev_score = exp.score
            self._prev_ts = now
            self._samples += 1
        # 两个趋势取大：EWMA 抗抖，跨度趋势对"真在爬"反应快。
        # 只用 EWMA 的话 1.2s 涨 24 分会被 alpha=0.4 压成 8 分/秒，eta 变成 2s 就不开火了。
        span = max(now - self._t0, 1e-6)
        trend = (exp.score - self._base_score) / span if span >= self.MIN_SPAN_S else 0.0
        exp.slope = max(self._slope, trend, 0.0)   # 只关心上涨；掉分不算斜率
        self.history.append(exp)
        return exp

    def rebaseline(self) -> None:
        """回家之后调用：把当前分当新基线，别把'回家那一刻的落差'读成斜率。"""
        self._prev_score = self.history[-1].score if self.history else 0.0
        self._base_score = self._prev_score
        self._prev_ts = time.monotonic()
        self._t0 = self._prev_ts
        self._slope = 0.0
        self._samples = 0

    # ---------- 判定 ----------
    def must_conceal(self) -> Tuple[bool, str]:
        if not self.history:
            return False, "no_sample"
        exp = self.history[-1]

        if exp.score >= self.threshold:
            return True, f"score={exp.score:.1f}>=thr"

        # 斜率先行：观测够久、分数确实在涨、且 lead_time 内会越线 → 现在回家
        observed = time.monotonic() - self._t0
        if self._samples < 2 or observed < self.MIN_SPAN_S or exp.score <= 0 or exp.slope <= 0:
            return False, f"quiet:{exp.score:.1f}"

        eta = (self.threshold - exp.score) / exp.slope
        if eta <= self.lead_time_s:
            return True, f"eta={eta:.2f}s<=lead"
        return False, f"quiet:{exp.score:.1f}"

    def tick(self) -> Tuple[bool, str]:
        self.probe()
        return self.must_conceal()
