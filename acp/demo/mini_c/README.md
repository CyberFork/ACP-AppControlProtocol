# Demo C — 混合架构：本地 UI-TARS grounding + 云端 LLM 规划

## 架构

```
任务指令 + 状态文本 ──► 云 LLM (DeepSeek/GLM) ──► 意图 JSON
                                                        │
                                  ┌─────────────────────┘
                                  ▼
截图 ──► UI-TARS (本地, 3090) ──► 像素坐标
                                  │
                                  ▼
                      WebAdapter 执行 click/type
                                  │
                                  ▼
                      JS DOM 验证 → 更新状态 → 回到云 LLM
```

**D11 隐私原则**：截图只发往本地 3090（UI-TARS），绝不进入云端 API。

## Phase 1 限制（DOM 模式）

当前 `StateDescriber` 默认使用 DOM 模式，要求页面有 `data-acp-id` 标注或标准 HTML 结构，适合 testenv 场景。

> Phase 2 将切换至 OmniParser 文本化（通用，任意页面，latency +1-2s）。

## 快速开始

### Mock 模式（无需 API key / vLLM）

```bash
# 先启动 testenv
python acp/testenv/server.py &

# 跑 mock 验证流程
python scripts/run_mini_c.py --mock

# 多次重复
python scripts/run_mini_c.py --mock --repeats 5
```

### 真实 API

配置 `.env`（参考 `.env.example`）：

```
PLANNER_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
PLANNER_LLM_API_KEY=your-key-here
PLANNER_LLM_MODEL=glm-4-flash
```

```bash
python scripts/run_mini_c.py \
  --url http://localhost:8765/pages/popup-login.html \
  --instruction "关闭弹窗，然后用用户名 demo 密码 123456 登录"
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PLANNER_LLM_BASE_URL` | `https://api.deepseek.com` | 云端规划 LLM 地址 |
| `PLANNER_LLM_API_KEY` | （必须） | API 密钥 |
| `PLANNER_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `PLANNER_LLM_MOCK` | `0` | `=1` 时用 mock planner |
| `UITARS_GROUNDING_MOCK` | `0` | `=1` 时用 mock grounding |

## 文件结构

```
acp/demo/mini_c/
├── __init__.py
├── grounding.py        # UITARSGrounding：截图 → 坐标（本地 vLLM）
├── planner.py          # PlannerLLM：状态文本 → 意图 JSON（云端）
├── state_describer.py  # StateDescriber：DOM → 状态文本
└── loop.py             # MiniCLoop：主循环

scripts/run_mini_c.py   # CLI 入口
```
