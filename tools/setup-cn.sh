#!/data/data/com.termux/files/usr/bin/bash
# Termux 中文字体安装与自检。幂等：已经装好的步骤会跳过，不覆盖你换过的终端字体。
# 用法：  bash setup-cn.sh [--terminal]
#   默认只配 fontconfig（给 matplotlib / GTK 之类用）
#   --terminal 额外把同一个字体设为 Termux 终端字体
set -eu

FONT_NAME="NotoSansMonoCJKsc-Regular.otf"
FONT_URL="https://github.com/notofonts/noto-cjk/raw/main/Sans/Mono/${FONT_NAME}"
FONT_DIR="${PREFIX:-/data/data/com.termux/files/usr}/share/fonts/TTF"
FONT_PATH="$FONT_DIR/$FONT_NAME"
WANT_TERMINAL=0
[ "${1:-}" = "--terminal" ] && WANT_TERMINAL=1

ok()  { printf '  \033[32m[ok]\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m[!!]\033[0m %s\n' "$1"; }
say() { printf '\n== %s\n' "$1"; }

say "1/5 依赖"
for pkg in fontconfig-utils curl; do
    if command -v "${pkg%%-*}" >/dev/null 2>&1 || dpkg -s "$pkg" >/dev/null 2>&1; then
        ok "$pkg 已在"
    else
        printf '  安装 %s ...\n' "$pkg"
        pkg install -y "$pkg" >/dev/null 2>&1 && ok "$pkg 已装" || bad "$pkg 安装失败"
    fi
done
command -v fc-list >/dev/null 2>&1 || { bad "fc-list 仍不存在，停"; exit 2; }

say "2/5 字体文件"
mkdir -p "$FONT_DIR"
if [ -s "$FONT_PATH" ] && [ "$(stat -c %s "$FONT_PATH")" -gt 10000000 ]; then
    ok "已存在 $(du -h "$FONT_PATH" | cut -f1)"
else
    printf '  下载 %s（约 15 MB）...\n' "$FONT_NAME"
    if curl -fL --retry 3 --connect-timeout 20 -o "$FONT_PATH.part" "$FONT_URL"; then
        mv "$FONT_PATH.part" "$FONT_PATH"
        ok "下载完成"
    else
        rm -f "$FONT_PATH.part"; bad "下载失败（换 Wi-Fi 或代理再试）"; exit 3
    fi
fi
hex=$(od -An -tx1 -N4 "$FONT_PATH" | tr -d ' \n')
case "$hex" in
    4f54544f|74746366|00010000) ok "文件头正常：$hex" ;;
    *) bad "文件头不对（$hex），可能是错误页，删掉重下"; rm -f "$FONT_PATH"; exit 4 ;;
esac

say "3/5 fontconfig 索引"
fc-cache -f "$FONT_DIR" >/dev/null 2>&1 || fc-cache -f >/dev/null 2>&1 || true
if fc-list :lang=zh 2>/dev/null | grep -qi "Noto Sans Mono CJK"; then
    ok "fc-list 能看到：$(fc-list :lang=zh | grep -i 'Noto Sans Mono CJK' | head -1 | cut -d: -f2-)"
else
    bad "fc-list :lang=zh 没列出它 —— 程序里可能仍显示方块"
fi

say "4/5 终端字体（可选）"
if [ "$WANT_TERMINAL" = "1" ]; then
    mkdir -p "$HOME/.termux"
    if [ -e "$HOME/.termux/font.ttf" ]; then
        ok "已有自定义终端字体，保持不动"
    else
        ln "$FONT_PATH" "$HOME/.termux/font.ttf" 2>/dev/null \
            || cp "$FONT_PATH" "$HOME/.termux/font.ttf"
        ok "已设置 ~/.termux/font.ttf"
    fi
    command -v termux-reload-settings >/dev/null 2>&1 && termux-reload-settings && ok "设置已重载"
else
    printf '  跳过（要设终端字体就加 --terminal）\n'
fi

say "5/5 自检"
printf '  shell : '; echo 孤狼中文测试
if command -v python >/dev/null 2>&1; then
    printf '  python: '; python -c "print('孤狼：中文正常')" || bad "python 输出失败"
    python - <<'PY' || true
import sys, os
sys.path.insert(0, os.path.expanduser("~/wolf/tools"))
try:
    from cnfont import setup
    info = setup(matplotlib_cjk=True)
    print("  cnfont:", info)
except Exception as exc:
    print("  cnfont: 跳过", type(exc).__name__, exc)
PY
else
    bad "没有 python，跳过"
fi

printf '\n完成。图表里用字体时：from cnfont import setup; setup(matplotlib_cjk=True)\n'
