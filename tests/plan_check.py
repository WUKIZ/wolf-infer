"""plan() 回测：用已知实测值检验自适应档位预测的误差。"""
import sys

sys.path.insert(0, '.')
from core.plan import plan

# (说明, 主机内存 G, 显存 G, 卡数, 实测 tok/s)
CASES = [
    ('60.75 G 主机 / 单卡 / 未预热', 60.75, 15.98, 1, 1.16),
    ('121.5 G 主机 / 单卡 / 全量 pin', 121.5, 15.98, 1, 6.32),
    ('120 G 主机 / 单卡 / 不 pin', 120.0, 15.98, 1, 4.05),
]


def main():
    print('%-32s %-10s %-10s %-8s %s' % ('场景', '预测', '实测', '误差', '档位'))
    errs = []
    for name, mem, vram, cards, measured in CASES:
        p = plan(mem, vram, cards, 14.31, 2.69)
        pred = p['expect_tok_s']
        err = (pred - measured) / measured * 100
        errs.append(abs(err))
        print('%-32s %-10.2f %-10.2f %+7.0f%%  %s'
              % (name, pred, measured, err, p['mode']))
    print('\n平均绝对误差 %.0f%% | 最大 %.0f%%' % (sum(errs) / len(errs), max(errs)))


if __name__ == '__main__':
    main()
