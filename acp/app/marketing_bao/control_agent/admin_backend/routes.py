"""FastAPI routes for 营销宝规划端后台。"""

from __future__ import annotations

from pathlib import Path

from acp.app.marketing_bao.config_loader import CONFIG_DIR
from acp.app.marketing_bao.control_agent.admin_backend.config_service import ConfigService
from acp.app.marketing_bao.control_agent.admin_backend.preview_service import PreviewService
from acp.app.marketing_bao.control_agent.admin_backend.schemas import (
    AdminConfigBundle,
    PreviewNextActionRequest,
    ReorderStagesRequest,
    SavePlaybooksRequest,
    SaveProductRequest,
    SaveRuntimeRequest,
    SaveStagesRequest,
    UpsertPlaybookRequest,
    UpsertStageRequest,
)
from acp.app.marketing_bao.control_agent.admin_backend.session_service import SessionService


def create_router(config_dir: str | Path | None = CONFIG_DIR, db_path: str | Path = "logs/marketing_bao/marketing_bao.sqlite3"):
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError as exc:  # pragma: no cover - only happens when optional dep missing
        raise RuntimeError("admin_backend 需要安装 FastAPI：pip install fastapi uvicorn") from exc

    router = APIRouter(prefix="/admin", tags=["marketing-bao-admin"])
    resolved_config_dir = config_dir or CONFIG_DIR
    config_service = ConfigService(resolved_config_dir)
    session_service = SessionService(db_path)
    preview_service = PreviewService(resolved_config_dir, db_path)

    @router.get("/health")
    def health():
        return {"ok": True, "service": "marketing_bao.control_agent.admin_backend"}

    @router.get("/config", response_model=AdminConfigBundle)
    def get_config():
        return AdminConfigBundle(
            stages=config_service.list_stages(),
            playbooks=config_service.list_playbooks(),
            product=config_service.get_product(),
            runtime=config_service.get_runtime(),
        )

    @router.get("/stages")
    def list_stages():
        return config_service.list_stages()

    @router.put("/stages")
    def save_stages(req: SaveStagesRequest):
        try:
            return config_service.save_stages(req.stages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/stages")
    def upsert_stage(req: UpsertStageRequest):
        try:
            return config_service.upsert_stage(req.stage)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/stages/{stage_id}")
    def delete_stage(stage_id: str):
        try:
            return config_service.delete_stage(stage_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/stages/reorder")
    def reorder_stages(req: ReorderStagesRequest):
        try:
            return config_service.reorder_stages(req.stage_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/stages/validate")
    def validate_stages():
        return config_service.validate_stages(config_service.list_stages())

    @router.get("/playbooks")
    def list_playbooks():
        return config_service.list_playbooks()

    @router.put("/playbooks")
    def save_playbooks(req: SavePlaybooksRequest):
        return config_service.save_playbooks(req.playbooks)

    @router.post("/playbooks")
    def upsert_playbook(req: UpsertPlaybookRequest):
        return config_service.upsert_playbook(req.playbook)

    @router.get("/product")
    def get_product():
        return config_service.get_product()

    @router.put("/product")
    def save_product(req: SaveProductRequest):
        return config_service.save_product(req.product)

    @router.get("/runtime")
    def get_runtime():
        return config_service.get_runtime()

    @router.put("/runtime")
    def save_runtime(req: SaveRuntimeRequest):
        return config_service.save_runtime(req.runtime, driver=req.driver)

    @router.get("/sessions")
    def list_sessions(limit: int = 100):
        return session_service.list_sessions(limit=limit)

    @router.get("/sessions/{contact_id}")
    def get_session(contact_id: str):
        session = session_service.get_session(contact_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"contact_id 不存在: {contact_id}")
        return session

    @router.get("/sessions/{contact_id}/messages")
    def list_messages(contact_id: str, limit: int = 50):
        return session_service.list_messages(contact_id, limit=limit)

    @router.post("/sessions/{contact_id}/preview-next-action")
    def preview_next_action(contact_id: str, req: PreviewNextActionRequest | None = None):
        try:
            return preview_service.preview_next_action(
                contact_id,
                persist_observation=bool(req.persist_observation) if req else False,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
