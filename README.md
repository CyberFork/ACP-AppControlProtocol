# ACP — Application Control Protocol

**泛用应用控制协议框架，让 AI 看懂并操作任意 App**

A universal application control protocol framework that enables AI to understand and operate any app.

---

## 架构 / Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Brain Layer                     │
│   IntentParser → TaskPlanner → Executor → Feedback  │
│                  PTG (Policy Tree Graph)             │
└──────────────────────┬──────────────────────────────┘
                       │  MCP 三层控制
                       ▼
┌─────────────────────────────────────────────────────┐
│                    MCP Layer                        │
│  Tier-1: Dedicated API  (最快，直接调用应用接口)      │
│  Tier-2: Control Tree   (无障碍树/DOM 精准操控)       │
│  Tier-3: Visual Fallback (视觉兜底，YOLOv8n 检测)    │
└──────────────────────┬──────────────────────────────┘
                       │  MoT V-L-A 融合
                       ▼
┌─────────────────────────────────────────────────────┐
│                    MoT Layer                        │
│  Vision  →  PerceiverResampler  →  GatedCrossAttn   │
│  Language →  LLM Backbone       →  ActionHead       │
│                   ↓ QLoRA 三阶段训练                  │
│           align → fuse → end-to-end                 │
└─────────────────────────────────────────────────────┘
```

---

## 快速开始 / Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # 填入 API keys

# 3. 启动开发环境
./dev_start.sh

# 4. 运行 Flow
python3 run_flow.py --flow acp/config/sites/baidu/flows.yaml
```

---

## 目录结构 / Project Structure

```
ACP/
├── acp/
│   ├── brain/          # 大脑模块（意图 → 规划 → 执行 → 反馈）
│   ├── mcp/            # MCP 三层控制协议
│   ├── adapters/       # 平台适配器（Web / iOS / Android）
│   ├── vision/         # 视觉模块（YOLOv8n + MobileSAM）
│   ├── mot/            # MoT 融合层（V-L-A 架构）
│   ├── training/       # QLoRA 三阶段训练管道
│   ├── schema/         # 数据模型（Pydantic）
│   ├── config/         # 应用配置 + 站点 YAML
│   ├── tests/          # 测试套件（273+ tests）
│   └── main.py
├── models/
│   ├── base/           # 基座模型权重（gitignore）
│   └── adapters/       # LoRA 适配器（gitignore）
├── requirements.txt    # 统一依赖
├── run_flow.py         # Flow CLI 入口
├── dev_start.sh        # 开发环境启动脚本
└── ACP.md              # 需求规格文档
```

---

## 技术栈 / Tech Stack

| 层级 | 技术 |
|------|------|
| Web 自动化 | Patchright (anti-detect Playwright) |
| 视觉检测 | YOLOv8n (Ultralytics) + MobileSAM |
| MoT 骨干 | Transformers + PEFT (QLoRA) |
| 数据建模 | Pydantic v2 |
| 配置管理 | PyYAML |
| 异步 HTTP | httpx |
| 训练加速 | bitsandbytes + accelerate |

---

## License

MIT © 2025 ACP Contributors
