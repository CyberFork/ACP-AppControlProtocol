# ACP 测试环境 (testenv)

基于 UI 交互分类学设计的私有测试页面集，供 ACP agent 开发和验证使用。

## 启动服务器

```bash
python acp/testenv/server.py
# 默认监听 http://localhost:8765
```

## 页面导航

| 编号 | 文件 | 用途 |
|------|------|------|
| T1 | pages/t1_form.html | 综合表单（登录/注册/验证） |
| T2 | pages/t2_dashboard.html | 数据表格 / Dashboard |
| T3 | pages/t3_list_modal.html | 列表 + 弹窗 |
| T4 | pages/t4_ecommerce.html | 电商购物流程 |
| T5 | pages/t5_navigation.html | 导航 + 菜单 |
| T6 | pages/t6_file_drag.html | 文件拖拽操作 |
| T7 | pages/t7_media_map.html | 媒体 + 地图 + 设置 |
| T8 | pages/t8_async_error.html | 异步状态 + 错误处理 |
| **T9** | **pages/popup-login.html** | **弹窗拦截 + 登录（miniDemo A 场景）** |
| **T10** | **pages/cross-app/notes-app.html** | **笔记应用：复制内容到剪贴板** |
| **T11** | **pages/cross-app/chat-app.html** | **聊天应用：从剪贴板粘贴并发送** |

## miniDemo 场景说明

### Demo A 场景（T9）：弹窗关闭 → 登录

**目标指令：** `关闭弹窗，然后用用户名 demo、密码 123456 登录`

**流程：**
1. 打开 `http://localhost:8765/pages/popup-login.html`
2. 页面加载后自动弹出"今日特惠"模态框
3. agent 识别并点击 `data-acp-id="modal-close"` 关闭弹窗
4. 填写 `data-acp-id="login-username"` 和 `data-acp-id="login-password"`
5. 点击 `data-acp-id="login-submit"` 提交
6. 验证 `data-acp-id="login-success"` 可见

### Demo 跨 App 场景（T10 → T11）：复制粘贴

**目标指令：** `从笔记应用复制第一条笔记，粘贴到聊天应用发送`

**流程：**
1. 打开 `http://localhost:8765/pages/cross-app/notes-app.html`
2. 点击 `data-acp-id="note-1-copy"` 复制内容
3. 切换到 `http://localhost:8765/pages/cross-app/chat-app.html`
4. 点击 `data-acp-id="paste-btn"` 粘贴
5. 点击 `data-acp-id="send-btn"` 发送

**注意：** Playwright headless 模式下 clipboard API 需要授权：
```python
context = await browser.new_context(
    permissions=["clipboard-read", "clipboard-write"]
)
```

## data-acp-id 约定

所有可交互元素带 `data-acp-id` 属性，格式：`{page}-{element}`，例如：
- `modal-close`：弹窗关闭按钮
- `login-username`：用户名输入框
- `note-1-copy`：第一条笔记的复制按钮
- `chat-input`：聊天输入框
