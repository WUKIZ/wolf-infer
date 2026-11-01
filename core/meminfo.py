"""/proc/meminfo 读取：一处实现，调用方只取数值。"""
from pathlib import Path


def available_mb() -> float:
    """MemAvailable（MB）。读不到或格式异常返回 0.0，由调用方决定退化策略。"""
    try:
        text = Path('/proc/meminfo').read_text(encoding='utf-8')
    except OSError:
        return 0.0
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024.0
    return 0.0
