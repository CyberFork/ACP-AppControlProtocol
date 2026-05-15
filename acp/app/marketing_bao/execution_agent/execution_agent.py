"""营销宝 ExecutionAgent：将 ControlAgent 的业务任务落到具体 App。"""

from __future__ import annotations

from acp.app.marketing_bao.execution_agent.protocol import AppDriver
from acp.app.marketing_bao.schemas import ActionTask, ExecutionResult, ExecutionStatus, TaskIntent


class ExecutionAgent:
    def __init__(self, driver: AppDriver) -> None:
        self.driver = driver

    async def execute(self, task: ActionTask) -> ExecutionResult:
        try:
            if task.intent == TaskIntent.SEND_MESSAGE:
                if not task.contact_id:
                    return ExecutionResult(task_id=task.task_id, status=ExecutionStatus.FAILED, error="缺少 contact_id")
                text = str(task.payload.get("text", ""))
                result = await self.driver.send_message(task.contact_id, text)
                result.task_id = task.task_id
                return result

            if task.intent == TaskIntent.READ_MESSAGES:
                if not task.contact_id:
                    return ExecutionResult(task_id=task.task_id, status=ExecutionStatus.FAILED, error="缺少 contact_id")
                messages = await self.driver.read_messages(task.contact_id, int(task.payload.get("limit", 20)))
                return ExecutionResult(task_id=task.task_id, status=ExecutionStatus.SUCCESS, messages=messages)

            if task.intent == TaskIntent.SYNC_DIALOGS:
                dialogs = await self.driver.list_dialogs(int(task.payload.get("limit", 20)))
                return ExecutionResult(
                    task_id=task.task_id,
                    status=ExecutionStatus.SUCCESS,
                    observations={"dialogs": [d.model_dump() for d in dialogs]},
                )

            if task.intent == TaskIntent.SEARCH_USER:
                users = await self.driver.search_user(str(task.payload.get("keyword", "")))
                return ExecutionResult(
                    task_id=task.task_id,
                    status=ExecutionStatus.SUCCESS,
                    observations={"users": [u.model_dump() for u in users]},
                )

            if task.intent == TaskIntent.ADD_FRIEND:
                result = await self.driver.add_friend(
                    str(task.payload.get("user_id", "")),
                    str(task.payload.get("message", "")),
                )
                result.task_id = task.task_id
                return result

            return ExecutionResult(task_id=task.task_id, status=ExecutionStatus.SKIPPED, error=f"未知任务: {task.intent}")
        except Exception as exc:
            return ExecutionResult(task_id=task.task_id, status=ExecutionStatus.FAILED, error=str(exc))
