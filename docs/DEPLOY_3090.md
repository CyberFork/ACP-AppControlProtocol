# ACP MoT — RTX 3090 部署与训练指南

## 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3090 (24GB) | RTX 3090 × 2 |
| CPU | 8 核 | 16 核 |
| RAM | 32GB | 64GB |
| 磁盘 | 50GB SSD | 100GB NVMe |
| NVIDIA 驱动 | 525.x (CUDA 12.1) | 535.x+ |
| CUDA Toolkit | 11.8 | 12.1 |

RTX 3090 架构：Ampere (sm_86)，支持 BF16、Flash Attention 2、INT8/INT4 量化。

---

## 一键环境配置

```bash
# 克隆并配置
git clone <repo>
cd ACP
bash scripts/setup_3090.sh

# 指定 CUDA 版本（默认 12.1）
ACP_CUDA=11.8 bash scripts/setup_3090.sh

# 指定 conda 环境名
ACP_ENV_NAME=my-acp bash scripts/setup_3090.sh
```

脚本会自动完成：
1. 检查 CUDA / 驱动版本
2. 创建 conda 环境（Python 3.10）
3. 安装 PyTorch + bitsandbytes + Flash Attention 2
4. 安装项目依赖
5. 下载 ShowUI-2B → 提取 ViT 权重
6. 下载 Qwen2.5-3B-Instruct
7. 验证 GPU 可用性 + mock 训练测试

---

## 目录结构

```
models/
└── base/
    ├── qwen2-vl-vit/
    │   ├── config.json
    │   ├── model.safetensors   # ViT 权重 (~2.2GB)
    │   └── preprocessor.json
    └── qwen2.5-3b-instruct/
        ├── config.json
        ├── model.safetensors   # LLM 权重 (~6GB, fp16)
        └── tokenizer.json

models/adapters/               # 训练输出
├── stage1_yyyymmdd/
├── stage2_yyyymmdd/
└── stage3_yyyymmdd/
```

---

## 显存预算表

> 基于 RTX 3090 24GB VRAM 估算，使用 float16 + INT4 量化

### 推理阶段

| 组件 | 显存 | 说明 |
|------|------|------|
| ViT (fp16) | ~2.2 GB | Qwen2-VL ViT，1.1B 参数 |
| LLM INT4 (4bit) | ~1.8 GB | Qwen2.5-3B，bitsandbytes |
| 胶水层 (fp16) | ~0.3 GB | Perceiver + CrossAttn × 4 + ActionHead |
| 激活值 / KV cache | ~0.5 GB | 512 token，batch=1 |
| **总计** | **~4.8 GB** | 推理余量充足 |

### 训练阶段

| Stage | 批大小 | 梯度累积 | 有效批 | 峰值显存 | 说明 |
|-------|--------|---------|--------|---------|------|
| Stage 1: V-L 对齐 | 4 | 8 | 32 | ~10-12 GB | 只训练 Perceiver |
| Stage 2: UI 融合 | 2 | 16 | 32 | ~16-18 GB | Perceiver + CrossAttn × 4 |
| Stage 3: 端到端 | 1 | 32 | 32 | ~20-22 GB | 全部胶水层 + ActionHead |

> **注意**：Stage 3 显存接近上限，务必开启 gradient checkpointing 和 fp16。

---

## 训练命令速查

### 环境激活

```bash
conda activate acp-3090
cd /path/to/ACP
```

### Stage 1: V-L 对齐（Perceiver 只读）

```bash
python acp/training/train.py \
    --stage 1 \
    --data data/stage1_vl_pairs.jsonl \
    --batch-size 4 \
    --grad-accum 8 \
    --epochs 3 \
    --lr 1e-3 \
    --output models/adapters/stage1/
```

### Stage 2: UI 融合（Perceiver + CrossAttn）

```bash
python acp/training/train.py \
    --stage 2 \
    --data data/stage2_ui.jsonl \
    --checkpoint models/adapters/stage1/checkpoint-final.pt \
    --batch-size 2 \
    --grad-accum 16 \
    --epochs 5 \
    --lr 2e-4 \
    --output models/adapters/stage2/
```

### Stage 3: 端到端（全部胶水层）

```bash
python acp/training/train.py \
    --stage 3 \
    --data data/stage3_traces.jsonl \
    --checkpoint models/adapters/stage2/checkpoint-final.pt \
    --batch-size 1 \
    --grad-accum 32 \
    --epochs 3 \
    --lr 1e-4 \
    --fp16 \
    --gradient-checkpointing \
    --output models/adapters/stage3/
```

### ViT 权重提取

```bash
# 首次运行（需下载 ShowUI-2B ~5GB）
python scripts/extract_vit.py \
    --source showlab/ShowUI-2B \
    --output models/base/qwen2-vl-vit/

# 从本地路径
python scripts/extract_vit.py \
    --source /data/models/ShowUI-2B \
    --output models/base/qwen2-vl-vit/

# 跳过验证（CI 环境）
python scripts/extract_vit.py --skip-verify
```

### GPU 基准测试

```bash
# 完整测试（ViT 延迟 + QLoRA 显存）
python scripts/benchmark_gpu.py

# 仅 ViT 延迟测试
python scripts/benchmark_gpu.py --vit-only

# 仅 QLoRA 显存测试
python scripts/benchmark_gpu.py --qlora-only

# 输出 JSON 报告
python scripts/benchmark_gpu.py --output benchmark_3090.json
```

### 推理部署

```bash
# 使用训练好的 adapter 推理
python acp/main.py \
    --adapter models/adapters/stage3/checkpoint-final.pt \
    --device cuda

# MCP Server 模式
python -m acp.mcp \
    --adapter models/adapters/stage3/checkpoint-final.pt \
    --port 8765
```

---

## 常见问题

### bitsandbytes 报错

**症状：**
```
CUDA Setup failed despite CUDA being available. Please run the following command to get more information...
```

**解决：**
```bash
# 重装指定版本
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.43.1

# 确认 CUDA 版本匹配
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

**RTX 3090 需要 bitsandbytes >= 0.41.0**（Ampere sm_86 支持从该版本起稳定）。

---

### OOM（显存不足）

**Stage 3 OOM 解决方案（按优先级）：**

```python
# 1. 开启 gradient checkpointing（必选，节省 30-40% 激活显存）
config = TrainingConfig(gradient_checkpointing=True)

# 2. 减小 batch size，增加 gradient accumulation（有效批不变）
config = TrainingConfig(batch_size=1, gradient_accumulation=32)

# 3. 开启 fp16（默认已开启）
config = TrainingConfig(fp16=True)

# 4. 强制 ViT/LLM 在 CPU 上（推理时卸载，训练时速度会下降）
# 注意：这会显著降低训练速度，作为最后手段
```

**显存占用分析命令：**
```bash
# 查看实时显存
watch -n 1 nvidia-smi

# Python 内查看
python -c "import torch; print(torch.cuda.memory_summary())"
```

---

### 梯度累积配置

有效批大小 = `batch_size × gradient_accumulation`。

```python
from acp.training.config import TrainingConfig

# RTX 3090 推荐配置（有效批=32）
stage1 = TrainingConfig(stage=1, batch_size=4,  gradient_accumulation=8)   # 峰值 ~12GB
stage2 = TrainingConfig(stage=2, batch_size=2,  gradient_accumulation=16)  # 峰值 ~18GB
stage3 = TrainingConfig(stage=3, batch_size=1,  gradient_accumulation=32)  # 峰值 ~22GB

print(stage1.effective_batch_size)  # 32
```

---

### Flash Attention 2 编译失败

**症状：** `pip install flash-attn` 报 CUDA 编译错误

**解决：**

```bash
# 方案 1：用 PyTorch SDPA 替代（性能相差不大，3090 上约慢 15%）
# 在代码中不显式依赖 flash_attn 即可自动使用

# 方案 2：安装预编译包
pip install flash-attn --no-build-isolation

# 方案 3：指定 CUDA 架构编译
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=2 \
    pip install flash-attn --no-build-isolation

# 确认 sm_86 是否支持
python -c "import torch; print(torch.cuda.get_device_capability())"  # 应输出 (8, 6)
```

---

### Qwen2-VL transformers 版本问题

**症状：** `ImportError: cannot import name 'Qwen2VLForConditionalGeneration'`

**解决：**
```bash
pip install transformers>=4.45.0
# 如需最新版
pip install git+https://github.com/huggingface/transformers.git
```

---

### HuggingFace 下载慢 / 超时

```bash
# 使用镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 指定缓存目录
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache

# 使用 huggingface-cli 断点续传
pip install huggingface_hub
huggingface-cli download showlab/ShowUI-2B --local-dir models/base/ShowUI-2B
```

---

## 3090 性能参考

以下数据来自 RTX 3090 24GB 实测（float16）：

| 测试项 | 结果 |
|--------|------|
| ViT forward (256 patches) | ~15-25ms |
| ViT forward (1024 patches) | ~50-80ms |
| Stage 1 训练吞吐 | ~8-12 样本/秒 |
| Stage 2 训练吞吐 | ~4-6 样本/秒 |
| Stage 3 训练吞吐 | ~2-3 样本/秒 |
| 端到端推理延迟 | ~200-400ms/步 |

> 使用 Flash Attention 2 可提升约 20-30%。
