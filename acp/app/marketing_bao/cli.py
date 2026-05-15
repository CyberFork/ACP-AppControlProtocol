"""营销宝 CLI。"""

from __future__ import annotations

import argparse
import asyncio
import json

from acp.app.marketing_bao.agent import MarketingBaoAgent
from acp.app.marketing_bao.control_agent.admin_backend.config_service import ConfigService
from acp.app.marketing_bao.execution_agent.platforms.android.emulator_check import check_android_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="营销宝 MarketingBao")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_once = sub.add_parser("run-once", help="执行一轮营销宝闭环")
    run_once.add_argument("--runtime", help="runtime yaml 路径（可选）")
    sub.add_parser("check-android", help="M0: 检查 Android/ADB 环境")

    admin_validate = sub.add_parser("admin-validate", help="校验规划端后台配置")
    admin_validate.add_argument("--config-dir", default=None, help="配置目录，默认 marketing_bao/config")

    admin_export = sub.add_parser("admin-export", help="导出规划端后台配置 JSON")
    admin_export.add_argument("--config-dir", default=None, help="配置目录，默认 marketing_bao/config")

    admin_serve = sub.add_parser("admin-serve", help="启动规划端后台 API")
    admin_serve.add_argument("--host", default="127.0.0.1")
    admin_serve.add_argument("--port", type=int, default=8787)
    admin_serve.add_argument("--config-dir", default=None, help="配置目录，默认 marketing_bao/config")
    admin_serve.add_argument("--db", default="logs/marketing_bao/marketing_bao.sqlite3")
    return parser


async def _run_once(runtime_path: str | None = None) -> None:
    agent = MarketingBaoAgent(runtime_path=runtime_path)
    result = await agent.run_once()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _admin_validate(config_dir: str | None = None) -> int:
    service = ConfigService(config_dir) if config_dir else ConfigService()
    result = service.validate_stages(service.list_stages())
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def _admin_export(config_dir: str | None = None) -> int:
    service = ConfigService(config_dir) if config_dir else ConfigService()
    data = {
        "stages": [s.model_dump(mode="json") for s in service.list_stages()],
        "playbooks": [p.model_dump(mode="json") for p in service.list_playbooks()],
        "product": service.get_product(),
        "runtime": service.get_runtime().model_dump(mode="json"),
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _admin_serve(host: str, port: int, config_dir: str | None = None, db_path: str = "logs/marketing_bao/marketing_bao.sqlite3") -> int:
    try:
        import uvicorn
    except ImportError:
        print("admin-serve 需要安装可选依赖：pip install fastapi uvicorn")
        return 2
    from acp.app.marketing_bao.control_agent.admin_backend.app import create_app

    app = create_app(config_dir=config_dir or None, db_path=db_path)
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "check-android":
        print(json.dumps(check_android_environment(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "run-once":
        asyncio.run(_run_once(args.runtime))
        return 0
    if args.cmd == "admin-validate":
        return _admin_validate(args.config_dir)
    if args.cmd == "admin-export":
        return _admin_export(args.config_dir)
    if args.cmd == "admin-serve":
        return _admin_serve(args.host, args.port, args.config_dir, args.db)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
