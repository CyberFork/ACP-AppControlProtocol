"""ACP 测试环境简单 HTTP 服务器。

用法：
    python acp/testenv/server.py
    或
    python acp/testenv/server.py --port 8765

访问：http://localhost:8765
"""

import http.server
import socketserver
import os
import sys
import argparse
from pathlib import Path


TESTENV_DIR = Path(__file__).parent.resolve()
DEFAULT_PORT = 8765


class TestEnvHandler(http.server.SimpleHTTPRequestHandler):
    """处理静态文件，从 testenv 目录根提供服务。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(TESTENV_DIR), **kwargs)

    def log_message(self, format, *args):  # noqa: A002
        # 颜色化日志输出
        method = args[0].split()[0] if args else ''
        colors = {
            'GET': '\033[32m',    # green
            'POST': '\033[34m',   # blue
            '404': '\033[31m',    # red
        }
        code = str(args[1]) if len(args) > 1 else ''
        color = colors.get(code, colors.get(method, ''))
        reset = '\033[0m'
        print(f"{color}[ACP] {self.address_string()} {format % args}{reset}")

    def end_headers(self):
        # 允许跨域，方便 ACP agent 调用
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='ACP 测试环境服务器')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'监听端口（默认 {DEFAULT_PORT}）')
    parser.add_argument('--host', default='', help='监听地址（默认 0.0.0.0）')
    args = parser.parse_args()

    os.chdir(TESTENV_DIR)

    pages = sorted((TESTENV_DIR / 'pages').glob('**/*.html'))
    print(f"\n\033[35m{'='*50}\033[0m")
    print(f"\033[1m ACP 测试环境服务器\033[0m")
    print(f"\033[35m{'='*50}\033[0m")
    print(f"\033[32m ✓ 根目录: {TESTENV_DIR}\033[0m")
    print(f"\033[32m ✓ 入口:   http://localhost:{args.port}/index.html\033[0m")
    print(f"\033[33m\n 测试页面:\033[0m")
    for p in pages:
        rel = p.relative_to(TESTENV_DIR)
        print(f"   http://localhost:{args.port}/{rel.as_posix()}")
    print(f"\033[35m{'='*50}\033[0m\n")

    with socketserver.TCPServer((args.host, args.port), TestEnvHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"监听 {args.host or '0.0.0.0'}:{args.port} ... 按 Ctrl+C 停止\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n\033[33m[ACP] 服务器已停止\033[0m')
            sys.exit(0)


if __name__ == '__main__':
    main()
