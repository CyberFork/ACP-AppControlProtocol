# ACP Demo

ACP miniDemo 实验，验证不同感知+决策方案在本地 testenv 场景的可行性。

## Demo A：OmniParser v2 + Qwen2.5-3B（本地 Apple Silicon，≤4GB 显存）

### 架构

```
screenshot ──► OmniParser v2 (MPS) ──► 元素列表 [bbox + label]
                                               │
                               render prompt ──┘
                                    │
                               Qwen2.5-3B (Ollama, Q4_K_M)
                                    │
                               action JSON
                                    │
                          WebAdapter 坐标点击 / 键入
                                    │
                               check → loop
```

**显存预算**：OmniParser ~1GB (MPS) + Ollama Qwen2.5-3B ~2GB = ~3GB，留 1GB 余量

### 前置依赖

1. **Ollama**：
   ```bash
   brew install ollama
   ollama serve         # 后台运行
   ollama pull qwen2.5:3b
   ```

2. **Python 依赖**：
   ```bash
   pip install transformers==4.49.0 huggingface_hub accelerate
   pip install easyocr supervision opencv-python-headless torchvision
   pip install einops timm
   ```

3. **testenv server**（可选，用 `--file` 则不需要）：
   ```bash
   python acp/testenv/server.py
   ```

### 一键运行

```bash
# file:// 模式（不需要 server）
python scripts/run_mini_a.py --file \
    --instruction "关闭弹窗，然后用用户名 demo 密码 123456 登录"

# server 模式
python acp/testenv/server.py &
python scripts/run_mini_a.py \
    --instruction "关闭弹窗，然后用用户名 demo 密码 123456 登录"

# 统计成功率（跑 5 次）
python scripts/run_mini_a.py --file --runs 5
```

### 日志

每次运行在 `logs/mini_a/<run_id>/` 下生成：
- `step_00.png`, `step_01.png`, ... — 每步截图
- `run_summary.json` — 完整步骤日志（含元素 JSON、LLM I/O、action、结果）

### 验收标准

- popup-login.html "关闭弹窗+登录" 成功率 ≥ 80% / 5 次
- 总 step 数 ≤ 5
- 单步耗时 ≤ 10s（OmniParser MPS 1-2s + Ollama 本地 3-5s）

---

## Demo B：UI-TARS-7B（3090，vLLM）

> 待 SSH 配置后实施。

---

## 模块说明

| 文件 | 职责 |
|------|------|
| `mini_a/perception.py` | OmniParser v2 封装，截图→元素列表 |
| `mini_a/llm_backend.py` | Ollama HTTP API 封装，elements→Action |
| `mini_a/prompt_template.py` | Prompt 模板（system + few-shot） |
| `mini_a/loop.py` | V-L-A 主循环，协调感知/决策/执行 |
| `../../scripts/run_mini_a.py` | CLI 入口，支持单次/多次运行 |
