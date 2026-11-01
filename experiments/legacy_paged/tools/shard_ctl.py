"""分片工具：info / verify。用法：python3 tools/shard_ctl.py info|verify <目录>"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import shardfile


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, d = argv[1], argv[2]
    if cmd == "info":
        s = shardfile.load_spec(d)
        print(f"{s.name}: {s.n_layers} 层, {s.n_params} 参数, {len(s.shards)} 片")
        print(f"总字节 {s.total_bytes} ({s.total_bytes / 1e9:.2f} GB)")
        for x in s.shards[:5]:
            print(f"  L{x.layer_from}-{x.layer_to} {x.nbytes / 1e6:.1f} MB")
        return 0
    if cmd == "verify":
        bad = shardfile.verify(d)
        print("ok" if not bad else f"损坏: {bad}")
        return 0 if not bad else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
