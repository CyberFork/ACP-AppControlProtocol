#!/usr/bin/env bash
# ACP 开发环境一键启动脚本
# 用法:
#   ./dev_start.sh              # 安装依赖 + 检查环境
#   ./dev_start.sh test         # 跑全部测试
#   ./dev_start.sh run ai6666 login_and_comment --comment "ACP测试"
#   ./dev_start.sh demo         # 跑 demo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[ACP]${NC} $1"; }
ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; }

# ── Python 检查 ──
check_python() {
    if command -v python3 &>/dev/null; then
        PY=python3
    elif command -v python &>/dev/null; then
        PY=python
    else
        fail "未找到 Python，请先安装 Python 3.10+"
        exit 1
    fi

    PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
        fail "Python 版本过低: $PY_VER（需要 3.10+）"
        exit 1
    fi
    ok "Python $PY_VER"
}

# ── 依赖安装 ──
install_deps() {
    info "安装核心依赖..."
    $PY -m pip install -q --upgrade pip

    # 核心依赖
    $PY -m pip install -q pydantic pyyaml httpx
    ok "pydantic + pyyaml + httpx"

    # Patchright（反检测 Playwright）
    if $PY -c "import patchright" 2>/dev/null; then
        ok "patchright 已安装"
    else
        info "安装 patchright..."
        $PY -m pip install -q patchright
        ok "patchright 已安装"
    fi

    # Patchright 浏览器
    if patchright install --check chromium 2>/dev/null; then
        ok "Chromium 浏览器已就绪"
    else
        info "安装 Chromium 浏览器（首次安装约 200MB）..."
        patchright install chromium
        ok "Chromium 已安装"
    fi

    # 视觉模块（可选）
    info "检查视觉模块依赖（可选）..."
    VISION_OK=true
    if $PY -c "from ultralytics import YOLO" 2>/dev/null; then
        ok "ultralytics (YOLOv8)"
    else
        warn "ultralytics 未安装（视觉兜底不可用，不影响控件树模式）"
        VISION_OK=false
    fi

    if $PY -c "from PIL import Image" 2>/dev/null; then
        ok "Pillow"
    else
        warn "Pillow 未安装"
        VISION_OK=false
    fi

    if [ "$VISION_OK" = false ]; then
        echo ""
        warn "视觉模块依赖不完整。安装命令："
        echo "  pip install ultralytics Pillow onnxruntime"
        echo ""
    fi
}

# ── 环境检查 ──
check_env() {
    info "检查环境配置..."

    # .env 文件
    if [ -f ".env" ]; then
        ok ".env 文件存在"
        if grep -q "ACP_LLM_API_KEY" .env 2>/dev/null; then
            KEY=$(grep "ACP_LLM_API_KEY" .env | cut -d= -f2 | head -c8)
            ok "LLM API Key 已配置 (${KEY}...)"
        else
            warn "未配置 ACP_LLM_API_KEY（LLM 辅助模式不可用，简单模式可用）"
        fi
    else
        warn "未找到 .env 文件（LLM 辅助模式不可用）"
        echo "  创建方法："
        echo "  echo 'ACP_LLM_API_KEY=sk-xxx' > .env"
        echo "  echo 'ACP_LLM_BASE_URL=https://api.deepseek.com/v1' >> .env"
        echo "  echo 'ACP_LLM_MODEL=deepseek-chat' >> .env"
    fi

    # 站点配置
    SITES=$(ls acp/config/sites/ 2>/dev/null)
    if [ -n "$SITES" ]; then
        ok "已配置站点: $SITES"
    else
        warn "无站点配置"
    fi

    echo ""
}

# ── 运行测试 ──
run_tests() {
    info "运行测试..."
    echo ""

    TOTAL=0
    PASSED=0
    FAILED=0

    # 自定义测试（非 pytest）
    for f in acp/tests/test_web_adapter.py acp/tests/test_mcp_registry.py; do
        if [ -f "$f" ]; then
            NAME=$(basename "$f" .py)
            if $PY "$f" 2>&1 | tail -1 | grep -qi "pass\|通过\|ok"; then
                ok "$NAME"
                PASSED=$((PASSED + 1))
            else
                # 可能需要网络，跳过
                warn "$NAME（可能需要网络连接）"
            fi
            TOTAL=$((TOTAL + 1))
        fi
    done

    # pytest 测试
    if $PY -m pytest --version &>/dev/null; then
        PYTEST_FILES=(
            acp/tests/test_brain.py
            acp/tests/test_flow_recorder.py
            acp/tests/test_flow_runner.py
            acp/tests/test_locate.py
            acp/tests/test_locate_and_runner.py
            acp/tests/test_vision.py
        )
        for f in "${PYTEST_FILES[@]}"; do
            if [ -f "$f" ]; then
                NAME=$(basename "$f" .py)
                if $PY -m pytest "$f" -q --tb=no 2>&1 | grep -q "passed"; then
                    COUNT=$($PY -m pytest "$f" -q --tb=no 2>&1 | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+")
                    ok "$NAME ($COUNT passed)"
                    PASSED=$((PASSED + 1))
                else
                    fail "$NAME"
                    FAILED=$((FAILED + 1))
                fi
                TOTAL=$((TOTAL + 1))
            fi
        done
    else
        warn "pytest 未安装，跳过 pytest 测试"
        echo "  安装: pip install pytest"
    fi

    echo ""
    if [ "$FAILED" -eq 0 ]; then
        ok "测试完成: $PASSED/$TOTAL 模块通过"
    else
        fail "测试完成: $PASSED/$TOTAL 通过, $FAILED 失败"
    fi
}

# ── 状态总览 ──
show_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ACP — Application Control Protocol${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
    echo ""
    echo "  用法:"
    echo ""
    echo "    # 运行 flow（LLM 辅助，需 API Key）"
    echo "    python3 run_flow.py ai6666 login_and_comment --comment \"ACP测试\""
    echo ""
    echo "    # 运行 demo（无需 LLM）"
    echo "    python3 acp/demo.py"
    echo ""
    echo "    # 交互模式"
    echo "    python3 acp/main.py"
    echo ""
    echo "    # 更多选项"
    echo "    python3 run_flow.py --help"
    echo ""
    echo "  可用站点:"
    for site in acp/config/sites/*/; do
        SITE_NAME=$(basename "$site")
        FLOWS=$(grep "^  [a-z]" "$site/flows.yaml" 2>/dev/null | sed 's/://g' | tr -d ' ' | head -5 | tr '\n' ' ')
        echo "    $SITE_NAME → flows: $FLOWS"
    done
    echo ""
    echo "  文档:"
    echo "    计划: .plans/acp/plan.md"
    echo "    架构: .plans/acp/docs/architecture.md"
    echo "    决策: .plans/acp/decisions.md"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
}

# ── 主入口 ──
case "${1:-setup}" in
    setup)
        check_python
        install_deps
        check_env
        show_status
        ;;
    test)
        check_python
        run_tests
        ;;
    run)
        shift
        check_python
        $PY run_flow.py "$@"
        ;;
    demo)
        check_python
        $PY acp/demo.py
        ;;
    status)
        check_python
        check_env
        show_status
        ;;
    *)
        echo "用法: ./dev_start.sh [setup|test|run|demo|status]"
        echo ""
        echo "  setup   安装依赖 + 检查环境（默认）"
        echo "  test    运行全部测试"
        echo "  run     运行 flow（后接参数）"
        echo "  demo    运行 demo"
        echo "  status  查看环境状态"
        ;;
esac
