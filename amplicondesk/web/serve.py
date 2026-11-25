"""AmpliconDesk 本地服务启动器。

用浏览器打开一个只在本机运行的界面，用来配置并执行分析。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

BANNER = r"""
    _                      _             ____            _
   / \   _ __ ___  _ __ | (_) ___ ___  |  _ \  ___  ___| | __
  / _ \ | '_ ` _ \| '_ \| | |/ __/ _ \ | | | |/ _ \/ __| |/ /
 / ___ \| | | | | | |_) | | | (_| (_) || |_| |  __/\__ \   <
/_/   \_\_| |_| |_| .__/|_|_|\___\___/ |____/ \___||___/_|\_\

  16S / 18S / ITS 扩增子 · 下游分析与出图
"""


def find_free_port(preferred: int = 8765, attempts: int = 30) -> int:
    """从首选端口开始找一个可用端口。"""
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"端口 {preferred}~{preferred + attempts} 都被占用，请手动指定 --port")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="启动 AmpliconDesk 本地分析服务")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765，被占用时自动顺延）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from .app import app

    port = find_free_port(args.port)
    url = f"http://{args.host}:{port}/"

    print(BANNER)
    print(f"  工作目录 : {root}")
    print(f"  结果目录 : {root / 'output'}")
    print(f"  访问地址 : {url}")
    print("  按 Ctrl+C 退出\n")

    if not args.no_browser:
        def open_later():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=open_later, daemon=True).start()

    try:
        app.run(host=args.host, port=port, debug=args.debug, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
