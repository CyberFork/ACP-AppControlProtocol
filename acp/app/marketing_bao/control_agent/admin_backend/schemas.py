"""营销宝规划端后台管理 API schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from acp.app.marketing_bao.schemas import ActionTask, PlaybookRule, RuntimeConfig, StageDefinition


class AdminConfigBundle(BaseModel):
    stages: list[StageDefinition] = Field(default_factory=list)
    playbooks: list[PlaybookRule] = Field(default_factory=list)
    product: dict[str, Any] = Field(default_factory=dict)
    runtime: RuntimeConfig | None = None


class SaveResult(BaseModel):
    ok: bool = True
    path: str
    backup_path: str | None = None


class UpsertStageRequest(BaseModel):
    stage: StageDefinition


class UpsertPlaybookRequest(BaseModel):
    playbook: PlaybookRule


class ReorderStagesRequest(BaseModel):
    stage_ids: list[str]


class SaveStagesRequest(BaseModel):
    stages: list[StageDefinition]


class SavePlaybooksRequest(BaseModel):
    playbooks: list[PlaybookRule]


class SaveProductRequest(BaseModel):
    product: dict[str, Any]


class SaveRuntimeRequest(BaseModel):
    runtime: RuntimeConfig
    driver: str | None = None


class SessionListItem(BaseModel):
    contact_id: str
    app: str
    display_name: str = ""
    current_stage: str
    sub_status: str = ""
    status: str
    intent_score: float = 0.0
    memory_summary: str = ""
    updated_at: str


class PreviewNextActionRequest(BaseModel):
    persist_observation: bool = False


class PreviewNextActionResult(BaseModel):
    contact_id: str
    stage_before: str
    stage_after: str
    memory_summary: str = ""
    task: ActionTask | None = None
    messages_used: int = 0


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    message: str
    target: str | None = None


class ValidationResult(BaseModel):
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
