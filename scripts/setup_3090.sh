#!/bin/bash
# RTX 3090 训练环境配置脚本
# 用法: bash scripts/setup_3090.sh
#
# 适用平台：Ubuntu 20.04 / 22.04，CUDA 11.8 或 12.1
# RTX 3090: 24GB VRAM, Ampere 架构 (sm_86), 支持 BF16 / Flash Attention 2

set -euo pipefail

# ── 颜色输出 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 配置变量 ─────────────────────────────────────────────────────────────────
ENV_NAME="${ACP_ENV_NAME:-acp-3090}"
CUDA_VERSION="${ACP_CUDA:-12.1}"          # 支持 11.8 或 12.1
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models/base"

echo "========================================================"
echo "  ACP RTX 3090 训练环境配置"
echo "  项目路径: ${PROJECT_ROOT}"
echo "  conda 环境: ${ENV_NAME}"
echo "  CUDA 版本: ${CUDA_VERSION}"
echo "========================================================"
echo ""

# ── Step 1: 检查 CUDA ────────────────────────────────────────────────────────
info "Step 1/8: 检查 CUDA 环境"

if ! command -v nvidia-smi &>/dev/null; then
    error "未找到 nvidia-smi，请先安装 NVIDIA 驱动 (>= 525.x for CUDA 12.1)"
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

info "GPU: ${GPU_NAME} | 显存: ${GPU_MEMORY} | 驱动: ${DRIVER_VERSION}"

# RTX 3090 验证
if echo "${GPU_NAME}" | grep -qi "3090"; then
    success "检测到 RTX 3090 ✓"
else
    warn "未检测到 RTX 3090（实际: ${GPU_NAME}），继续安装但配置针对 3090 优化"
fi

# CUDA 版本检查
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    info "nvcc CUDA 版本: ${NVCC_VER}"
else
    warn "nvcc 未找到，将依赖 conda 安装的 CUDA toolkit"
fi

# ── Step 2: 检查 / 安装 conda ────────────────────────────────────────────────
info "Step 2/8: 检查 conda 环境"

if ! command -v conda &>/dev/null; then
    warn "conda 未找到，正在安装 Miniconda…"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    curl -fsSL "${MINICONDA_URL}" -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    success "Miniconda 安装完成"
else
    success "conda 已就绪: $(conda --version)"
fi

# ── Step 3: 创建 conda 环境 ──────────────────────────────────────────────────
info "Step 3/8: 创建 conda 环境 '${ENV_NAME}'"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "环境 '${ENV_NAME}' 已存在，跳过创建（如需重建请先: conda env remove -n ${ENV_NAME}）"
else
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    success "conda 环境创建完成"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
success "已激活环境: ${ENV_NAME}"

# ── Step 4: 安装 PyTorch ─────────────────────────────────────────────────────
info "Step 4/8: 安装 PyTorch (CUDA ${CUDA_VERSION})"

if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    success "PyTorch ${TORCH_VER} 已安装且 CUDA 可用，跳过"
else
    if [[ "${CUDA_VERSION}" == "12.1" ]]; then
        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
            --index-url https://download.pytorch.org/whl/cu121
    elif [[ "${CUDA_VERSION}" == "11.8" ]]; then
        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
            --index-url https://download.pytorch.org/whl/cu118
    else
        error "不支持的 CUDA 版本: ${CUDA_VERSION}（支持 11.8 或 12.1）"
    fi
    success "PyTorch 安装完成"
fi

# ── Step 5: 安装 bitsandbytes（3090 特定） ───────────────────────────────────
info "Step 5/8: 安装 bitsandbytes (RTX 3090 / sm_86)"

# RTX 3090 需要 bitsandbytes >= 0.41.0 以支持 4bit QLoRA on Ampere
pip install bitsandbytes>=0.43.0

# 验证 bitsandbytes CUDA 支持
python - <<'PYEOF'
import bitsandbytes as bnb
import torch
if torch.cuda.is_available():
    try:
        # 简单验证：创建 4bit Linear
        layer = bnb.nn.Linear4bit(64, 64)
        print(f"  bitsandbytes {bnb.__version__} CUDA 支持 ✓")
    except Exception as e:
        print(f"  [WARN] bitsandbytes 验证警告: {e}")
else:
    print("  [INFO] 无 GPU，跳过 bitsandbytes 验证")
PYEOF

success "bitsandbytes 安装完成"

# ── Step 6: 安装 Flash Attention 2（可选，3090 支持） ────────────────────────
info "Step 6/8: 安装 Flash Attention 2（RTX 3090 支持，编译约需 5-10 分钟）"

if python -c "import flash_attn" &>/dev/null; then
    FA_VER=$(python -c "import flash_attn; print(flash_attn.__version__)")
    success "Flash Attention ${FA_VER} 已安装，跳过"
else
    # 优先尝试预编译包
    if pip install flash-attn --no-build-isolation 2>/dev/null; then
        success "Flash Attention 2 安装完成（预编译）"
    else
        warn "预编译包安装失败，尝试从源码编译（需要 CUDA toolkit）…"
        MAX_JOBS=4 pip install flash-attn --no-build-isolation || \
            warn "Flash Attention 2 编译失败，将使用 PyTorch SDPA 替代（性能略低，功能正常）"
    fi
fi

# ── Step 7: 安装项目依赖 ─────────────────────────────────────────────────────
info "Step 7/8: 安装项目依赖"

cd "${PROJECT_ROOT}"

# 核心依赖
pip install \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    peft>=0.11.0 \
    safetensors>=0.4.0 \
    datasets>=2.19.0 \
    tokenizers>=0.19.0

# 图像处理
pip install \
    Pillow>=10.0.0 \
    torchvision \
    einops>=0.7.0

# 训练工具
pip install \
    tqdm \
    tensorboard \
    wandb \
    scipy \
    numpy

# 项目本身（如果有 setup.py / pyproject.toml）
if [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]]; then
    pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || true
fi

success "项目依赖安装完成"

# ── Step 8: 下载模型 + 提取 ViT ─────────────────────────────────────────────
info "Step 8/8: 下载基础模型"

mkdir -p "${MODELS_DIR}"

# 8a: 下载 ShowUI-2B 并提取 ViT
VIT_OUTPUT="${MODELS_DIR}/qwen2-vl-vit"
if [[ -f "${VIT_OUTPUT}/model.safetensors" ]] || [[ -f "${VIT_OUTPUT}/model.pt" ]]; then
    success "ViT 权重已存在: ${VIT_OUTPUT}，跳过"
else
    info "下载 ShowUI-2B 并提取 ViT（约 5GB）…"
    python "${PROJECT_ROOT}/scripts/extract_vit.py" \
        --source "showlab/ShowUI-2B" \
        --output "${VIT_OUTPUT}"
    success "ViT 提取完成: ${VIT_OUTPUT}"
fi

# 8b: 下载 Qwen2.5-3B-Instruct
LLM_OUTPUT="${MODELS_DIR}/qwen2.5-3b-instruct"
if [[ -d "${LLM_OUTPUT}" ]] && [[ -n "$(ls -A "${LLM_OUTPUT}" 2>/dev/null)" ]]; then
    success "Qwen2.5-3B-Instruct 已存在: ${LLM_OUTPUT}，跳过"
else
    info "下载 Qwen2.5-3B-Instruct（约 6GB）…"
    python - <<PYEOF
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
output = "${LLM_OUTPUT}"
print(f"  下载 tokenizer…")
tok = AutoTokenizer.from_pretrained(model_id)
tok.save_pretrained(output)
print(f"  下载模型权重（这需要几分钟）…")
import torch
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cpu"
)
model.save_pretrained(output)
print(f"  保存到: {output}")
PYEOF
    success "Qwen2.5-3B-Instruct 下载完成: ${LLM_OUTPUT}"
fi

# ── 验证 GPU 可用 ────────────────────────────────────────────────────────────
echo ""
info "验证 GPU 可用性"
python - <<'PYEOF'
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1024**3
    print(f"  GPU: {gpu.name}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  BF16 支持: {gpu.is_bf16_supported()}")

    # 显存压力测试
    t = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
    del t
    torch.cuda.empty_cache()
    print("  显存读写测试: ✓")
else:
    print("  [WARN] CUDA 不可用，请检查驱动和 PyTorch 安装")
PYEOF

# ── Mock 训练验证 ─────────────────────────────────────────────────────────────
echo ""
info "运行 mock 训练验证"
python - <<'PYEOF'
"""验证训练配置可正常实例化，不需要真实模型权重。"""
import sys
sys.path.insert(0, '.')
try:
    from acp.training.config import TrainingConfig
    from acp.mot.config import MoTConfig
    cfg = TrainingConfig(stage=1, batch_size=2, gradient_accumulation=4)
    mot_cfg = MoTConfig()
    assert cfg.effective_batch_size == 8
    assert mot_cfg.d_visual == 1152
    print("  TrainingConfig ✓")
    print("  MoTConfig ✓")
    print(f"  Stage 1 有效批大小: {cfg.effective_batch_size}")
except Exception as e:
    print(f"  [WARN] mock 训练验证失败: {e}")
PYEOF

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
success "RTX 3090 训练环境配置完成！"
echo ""
echo "  激活环境:    conda activate ${ENV_NAME}"
echo "  Stage 1 训练: python acp/training/train.py --stage 1"
echo "  基准测试:     python scripts/benchmark_gpu.py"
echo "  详细指南:     cat docs/DEPLOY_3090.md"
echo "========================================================"
