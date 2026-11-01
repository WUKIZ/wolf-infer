"""临时目录：容器重启会清空 /tmp，统一走 WOLF_TMP。"""
import os

TMP = os.environ.get('WOLF_TMP') or '/tmp'
