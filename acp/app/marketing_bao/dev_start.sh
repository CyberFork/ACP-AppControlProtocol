#!/usr/bin/env bash
# 营销宝开发启动脚本
#
# 用法：
#   ./acp/app/marketing_bao/dev_start.sh                # 安装/检查依赖并启动规划端后台 API
#   ./acp/app/marketing_bao/dev_start.sh setup          # 安装/检查依赖 + 检查配置
#   ./acp/app/marketing_bao/dev_start.sh validate       # 校验规划端配置
#   ./acp/app/marketing_bao/dev_start.sh export         # 导出规划端配置 JSON
#   ./acp/app/marketing_bao/dev_start.sh run            # 执行一轮营销宝闭环（默认 dry_run）
#   ./acp/app/marketing_bao/dev_start.sh admin          # 启动规划端后台 API
#   ./acp/app/marketing_bao/dev_start.sh feishu-check   # 检查飞书环境变量
#   ./acp/app/marketing_bao/dev_start.sh android-check  # 检查 ADB/Android 环境
#   ./acp/app/marketing_bao/dev_start.sh test           # 编译 + 配置校验 + dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${CYAN}[MarketingBao]${NC} $1"; }
ok() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; }

PY="${PYTHON:-python3}"
HOST="${MARKETING_BAO_HOST:-127.0.0.1}"
PORT="${MARKETING_BAO_PORT:-8787}"
DB_PATH="${MARKETING_BAO_DB:-logs/marketing_bao/marketing_bao.sqlite3}"
RUNTIME_PATH="${MARKETING_BAO_RUNTIME:-}"

run_cli() {
  if [[ -n "$RUNTIME_PATH" && "$1" == "run-once" ]]; then
    "$PY" -m acp.app.marketing_bao.cli run-once --runtime "$RUNTIME_PATH"
  else
    "$PY" -m acp.app.marketing_bao.cli "$@"
  fi
}

check_python() {
  if ! command -v "$PY" >/dev/null 2>&1; then
    fail "未找到 Python：$PY"
    exit 1
  fi
  local ver
  ver="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local major minor
  major="$($PY -c 'import sys; print(sys.version_info.major)')"
  minor="$($PY -c 'import sys; print(sys.version_info.minor)')"
  if [[ "$major" -lt 3 || "$minor" -lt 10 ]]; then
    fail "Python 版本过低：$ver，需要 3.10+"
    exit 1
  fi
  ok "Python $ver"
}

install_python_packages() {
  info "安装缺失依赖: $*"
  "$PY" -m pip install "$@"
}

check_core_deps() {
  info "检查核心依赖..."
  local missing
  missing="$($PY - <<'PY'
import importlib.util
mods = [("pydantic", "pydantic"), ("yaml", "pyyaml"), ("httpx", "httpx")]
print(" ".join(pkg for mod, pkg in mods if importlib.util.find_spec(mod) is None))
PY
)"
  if [[ -n "$missing" ]]; then
    install_python_packages $missing
  fi
  ok "核心依赖 pydantic/pyyaml/httpx 已就绪"
}

check_admin_deps() {
  local missing
  missing="$($PY - <<'PY'
import importlib.util
mods = [("fastapi", "fastapi"), ("uvicorn", "uvicorn")]
print(" ".join(pkg for mod, pkg in mods if importlib.util.find_spec(mod) is None))
PY
)"
  if [[ -n "$missing" ]]; then
    install_python_packages $missing
  fi
  ok "FastAPI/uvicorn 已安装，可启动 admin 后台"
}

check_configs() {
  info "检查营销宝配置文件..."
  local files=(
    "acp/app/marketing_bao/config/stages.yaml"
    "acp/app/marketing_bao/config/playbooks.yaml"
    "acp/app/marketing_bao/config/product.yaml"
    "acp/app/marketing_bao/config/runtime.yaml"
    "acp/app/marketing_bao/config/feishu.yaml"
  )
  for f in "${files[@]}"; do
    [[ -f "$f" ]] && ok "$f" || warn "缺少 $f"
  done
  run_cli admin-validate >/tmp/marketing_bao_validate.json
  ok "阶段链配置校验通过"
}

feishu_check() {
  info "检查飞书真实发送所需环境变量..."
  local missing=0
  for k in FEISHU_APP_ID FEISHU_APP_SECRET FEISHU_DEFAULT_RECEIVE_ID FEISHU_DEFAULT_RECEIVE_ID_TYPE; do
    if [[ -n "${!k:-}" ]]; then
      if [[ "$k" == "FEISHU_APP_SECRET" ]]; then
        ok "$k=***"
      else
        ok "$k=${!k}"
      fi
    else
      warn "未设置 $k"
      missing=1
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    ok "飞书环境变量已齐全"
  else
    cat <<'EOF'

最小配置示例：
  export FEISHU_APP_ID="cli_xxx"
  export FEISHU_APP_SECRET="xxx"
  export FEISHU_DEFAULT_RECEIVE_ID="oc_xxx 或 ou_xxx"
  export FEISHU_DEFAULT_RECEIVE_ID_TYPE="chat_id 或 open_id"

建议第一轮使用：群聊 chat_id + FEISHU_DEFAULT_RECEIVE_ID_TYPE=chat_id
EOF
  fi
}

show_status() {
  echo -e ""
  echo -e "${CYAN}════════════════════════════════════════════${NC}"
  echo -e "  营销宝 MarketingBao Dev"
  echo -e "${CYAN}════════════════════════════════════════════${NC}"
  echo -e ""
  echo -e "  Repo:       $REPO_ROOT"
  echo -e "  App:        acp/app/marketing_bao"
  echo -e "  Admin API:  启动后访问 http://$HOST:$PORT/docs"
  echo -e "  DB:         $DB_PATH"
  echo -e "  Runtime:    ${RUNTIME_PATH:-acp/app/marketing_bao/config/runtime.yaml}"
  echo -e ""
  echo -e "  常用命令："
  echo -e "    ./acp/app/marketing_bao/dev_start.sh validate"
  echo -e "    ./acp/app/marketing_bao/dev_start.sh run"
  echo -e "    ./acp/app/marketing_bao/dev_start.sh        # 默认启动后台"
  echo -e "    ./acp/app/marketing_bao/dev_start.sh admin"
  echo -e "    ./acp/app/marketing_bao/dev_start.sh feishu-check"
  echo -e "    ./acp/app/marketing_bao/dev_start.sh test"
  echo -e ""
  echo -e "${CYAN}════════════════════════════════════════════${NC}"
}

usage() {
  sed -n '1,16p' "$0"
}

cmd="${1:-start}"
case "$cmd" in
  start|dev)
    check_python
    check_core_deps
    check_admin_deps
    check_configs
    show_status
    info "启动规划端后台 API..."
    run_cli admin-serve --host "$HOST" --port "$PORT" --db "$DB_PATH"
    ;;
  setup)
    check_python
    check_core_deps
    check_admin_deps
    check_configs
    show_status
    ;;
  validate)
    check_python
    run_cli admin-validate
    ;;
  export)
    check_python
    run_cli admin-export
    ;;
  run|run-once)
    check_python
    run_cli run-once
    ;;
  admin|admin-serve)
    check_python
    check_admin_deps
    run_cli admin-serve --host "$HOST" --port "$PORT" --db "$DB_PATH"
    ;;
  feishu-check)
    feishu_check
    ;;
  android-check)
    check_python
    run_cli check-android
    ;;
  test)
    check_python
    check_core_deps
    info "编译 marketing_bao..."
    "$PY" -m compileall -q acp/app/marketing_bao
    ok "编译通过"
    run_cli admin-validate
    info "执行 dry-run 闭环..."
    run_cli run-once
    ;;
  status)
    show_status
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    fail "未知命令：$cmd"
    usage
    exit 1
    ;;
esac
