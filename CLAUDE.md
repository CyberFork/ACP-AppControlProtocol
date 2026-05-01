# ACP - 团队运营手册

> 由 CCteam-creator 自动生成，可按需修改。
> 此文件让 team-lead 的团队知识在上下文压缩后仍然保持。

## Team-Lead 控制平面

- team-lead = 主对话，不是生成的 agent
- team-lead 负责用户对齐、范围控制、任务分解和阶段推进
- team-lead 维护项目全局真相：主 `task_plan.md`、`decisions.md` 和此 `CLAUDE.md`
- team-lead 决定某个流程改进是项目本地的还是需要写回 `CCteam-creator` 的
- **禁用独立子智能体**：团队存在后，所有工作通过 SendMessage 交给队友。不要启动独立的 Agent/子智能体（Explore、general-purpose 等）——它们绕过团队的规划文件和协作体系。唯一例外：用 `team_name` 生成新队友加入团队

## 团队花名册

| 名称 | 角色 | 模型 | 核心能力 |
|------|------|------|---------|
| researcher-perception | 探索/研究 | sonnet | 视觉感知与 UI 操作技术调研（只读） |
| researcher-brain | 探索/研究 | sonnet | 决策模型与架构方案调研（只读） |
| backend-dev | 后端开发 | sonnet | Python 全栈开发 + MCP + Playwright |

## 任务下发协议

### 消息送达时序（关键）
`SendMessage` 只在接收方 idle 时送达——**无法**打断进行中的任务。初始派单必须前置上下文（没有中途追加），广播也没有抢占，实时状态靠直接读 `progress.md` / `findings.md`（**文件实时，消息不是**）。

### TaskCreate 描述格式（team-lead 上下文压缩后参考）

TaskCreate 描述：一句话范围 + 验收标准 + `.plans/` 路径。

### 大任务（功能开发、新模块）-- 停止检查后再发送

**在给任何智能体下发大任务前，检查消息中是否包含以下 4 项：**
1. **范围和目标**：要做什么、验收标准
2. **文档提醒**：创建任务文件夹 + 更新根 findings.md 索引
3. **依赖说明**：依赖哪些调研/任务的结论
4. **审查预期**：完成后是否需要审查

## 通信速查

| 操作 | 命令 |
|------|------|
| 给单个智能体分配任务 | `SendMessage(to: "<名称>", message: "...")` |
| 广播给所有人（慎用） | `SendMessage(to: "*", message: "...")` |

## 状态检查

| 要检查什么 | 怎么做 |
|-----------|--------|
| 全局概览 | `TaskList` — 所有任务、负责人、阻塞情况一览 |
| 快速扫描 | 并行读取各 agent 的 `progress.md` |
| 深入了解 | 读 agent 的 `findings.md`（索引）→ 再看具体任务文件夹 |
| 方向检查 | 读 `.plans/acp/task_plan.md` |
| 恢复项目 | 读 `team-snapshot.md` → 从缓存 prompt 启动智能体 → 读各 agent 的 `findings.md` 索引 |

读取顺序：**progress**（到哪了）→ **findings**（遇到什么）→ **task_plan**（目标是什么）

## 文档索引（知识库）

> **导航地图**：`docs/index.md` 有各文档的 section 级导航（含行号范围）。
> team-lead 维护 docs/index.md。需要在 docs/ 中查找信息时先 Read 它。

| 文档 | 位置 | 维护者 |
|------|------|--------|
| 导航地图 | .plans/acp/docs/index.md | team-lead |
| 架构 | .plans/acp/docs/architecture.md | team-lead, devs |

## 审查维度

> 调研阶段的审查维度（评估调研报告质量）。

| # | 维度 | 权重 | STRONG 表现 | WEAK 表现 |
|---|------|------|-----------|---------|
| RD-1 | 调研广度 | 高 | 覆盖所有主流方案和最新论文，不遗漏关键候选技术 | 只看了 1-2 个方案，遗漏重要替代方案 |
| RD-2 | 调研深度 | 高 | 对每个方案有具体的性能数据、部署要求、优劣分析 | 只有表面描述，缺乏量化对比和实际可行性分析 |
| RD-3 | 方案可行性 | 中 | 推荐方案有明确的技术路径和风险评估，考虑了 H5/Web 场景 | 推荐方案脱离实际，未考虑部署环境和性能约束 |

## 核心协议

| 协议 | 触发时机 | 操作 |
|------|---------|------|
| 需求对齐 | 团队搭建后、开发前 | ✅ 已完成 |
| 3-Strike 上报 | 智能体报告 3 次失败 | 读其 progress.md，给新方向或重新分配 |
| 阶段推进 | 阶段完成 | 调研完：读 findings 更新主计划，与用户讨论技术路线 |
| 上下文溢出 | 智能体报告上下文过长 | 进度已存文件，恢复或生成后继者 |

## Known Pitfalls

> 当识别到反复出现的失败模式时追加到这里。

| # | 模式 | 触发场景 | 应对 |
|---|------|---------|------|
| KP-1 | subagent 自报 PASS 但未真跑最终命令 | 验证脚本路径写错（如 `popup-login.html` 漏 `pages/` 前缀），TA 跑过老路径误以为通过 | team-lead 收到完成报告必须**亲自跑一遍交付的命令**抽样验证；要求 subagent 在报告里**贴实际终端输出**而非仅描述结果 |

## 风格决策

| # | 决策 | 来源 | 状态 |
|---|------|------|------|
| SD-1 | 团队默认中文回复 | 用户语言 Session 1 | Manual |

## 文件结构

```
.plans/acp/
  task_plan.md          -- 主计划
  team-snapshot.md      -- 缓存的入职 prompts
  findings.md           -- 团队级发现
  progress.md           -- 工作日志
  decisions.md          -- 架构决策记录
  docs/                 -- 项目知识库
    index.md            -- 导航地图
    architecture.md     -- 系统架构
  archive/              -- 归档历史
  researcher-perception/ -- 感知层调研
  researcher-brain/      -- 决策层调研
```
