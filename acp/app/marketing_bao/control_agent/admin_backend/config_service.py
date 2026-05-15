"""营销宝后台配置服务。

职责：
- 管理用户可编辑的阶段链、playbook、产品配置和 runtime。
- 文件仍以 YAML 作为真源，便于版本管理和人工编辑。
- 写入时自动生成 .bak 备份，降低误操作风险。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from acp.app.marketing_bao.control_agent.admin_backend.schemas import SaveResult, ValidationIssue, ValidationResult
from acp.app.marketing_bao.config_loader import CONFIG_DIR
from acp.app.marketing_bao.schemas import PlaybookRule, RuntimeConfig, StageDefinition, now_iso


class ConfigService:
    def __init__(self, config_dir: str | Path = CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------

    @property
    def stages_path(self) -> Path:
        return self.config_dir / "stages.yaml"

    @property
    def playbooks_path(self) -> Path:
        return self.config_dir / "playbooks.yaml"

    @property
    def product_path(self) -> Path:
        return self.config_dir / "product.yaml"

    @property
    def runtime_path(self) -> Path:
        return self.config_dir / "runtime.yaml"

    # ---------- low-level yaml ----------

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> SaveResult:
        backup_path = None
        if path.exists():
            stamp = now_iso().replace(":", "").replace("+", "_")
            backup = path.with_suffix(path.suffix + f".{stamp}.bak")
            shutil.copy2(path, backup)
            backup_path = str(backup)
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return SaveResult(path=str(path), backup_path=backup_path)

    # ---------- stages ----------

    def list_stages(self) -> list[StageDefinition]:
        data = self._read_yaml(self.stages_path)
        return [StageDefinition(**item) for item in data.get("stages", [])]

    def save_stages(self, stages: list[StageDefinition]) -> SaveResult:
        result = self.validate_stages(stages)
        if not result.ok:
            msg = "; ".join(i.message for i in result.issues if i.level == "error")
            raise ValueError(f"阶段配置校验失败: {msg}")
        return self._write_yaml(self.stages_path, {"stages": [s.model_dump(mode="json") for s in stages]})

    def upsert_stage(self, stage: StageDefinition) -> SaveResult:
        stages = self.list_stages()
        for idx, existing in enumerate(stages):
            if existing.id == stage.id:
                stages[idx] = stage
                break
        else:
            stages.append(stage)
        return self.save_stages(stages)

    def delete_stage(self, stage_id: str) -> SaveResult:
        stages = [s for s in self.list_stages() if s.id != stage_id]
        return self.save_stages(stages)

    def reorder_stages(self, stage_ids: list[str]) -> SaveResult:
        by_id = {s.id: s for s in self.list_stages()}
        missing = [sid for sid in stage_ids if sid not in by_id]
        extra = [sid for sid in by_id if sid not in stage_ids]
        if missing or extra:
            raise ValueError(f"阶段 ID 不匹配 missing={missing} extra={extra}")
        return self.save_stages([by_id[sid] for sid in stage_ids])

    def validate_stages(self, stages: list[StageDefinition]) -> ValidationResult:
        issues: list[ValidationIssue] = []
        seen: set[str] = set()
        for idx, stage in enumerate(stages):
            target = f"stages[{idx}]"
            if not stage.id.strip():
                issues.append(ValidationIssue(level="error", target=target, message="阶段 id 不能为空"))
            if stage.id in seen:
                issues.append(ValidationIssue(level="error", target=target, message=f"阶段 id 重复: {stage.id}"))
            seen.add(stage.id)
            if not stage.name.strip():
                issues.append(ValidationIssue(level="warning", target=target, message=f"阶段 {stage.id} 缺少 name"))
            if not stage.allowed_actions:
                issues.append(ValidationIssue(level="warning", target=target, message=f"阶段 {stage.id} 未配置 allowed_actions"))
        return ValidationResult(ok=not any(i.level == "error" for i in issues), issues=issues)

    # ---------- playbooks ----------

    def list_playbooks(self) -> list[PlaybookRule]:
        data = self._read_yaml(self.playbooks_path)
        return [PlaybookRule(**item) for item in data.get("playbooks", [])]

    def save_playbooks(self, playbooks: list[PlaybookRule]) -> SaveResult:
        return self._write_yaml(self.playbooks_path, {"playbooks": [p.model_dump(mode="json") for p in playbooks]})

    def upsert_playbook(self, playbook: PlaybookRule) -> SaveResult:
        playbooks = self.list_playbooks()
        for idx, existing in enumerate(playbooks):
            if existing.stage_id == playbook.stage_id:
                playbooks[idx] = playbook
                break
        else:
            playbooks.append(playbook)
        return self.save_playbooks(playbooks)

    # ---------- product/runtime ----------

    def get_product(self) -> dict[str, Any]:
        return self._read_yaml(self.product_path).get("product", {})

    def save_product(self, product: dict[str, Any]) -> SaveResult:
        return self._write_yaml(self.product_path, {"product": product})

    def get_runtime(self) -> RuntimeConfig:
        data = self._read_yaml(self.runtime_path).get("runtime", {})
        raw = dict(data)
        raw.pop("driver", None)
        return RuntimeConfig(**raw)

    def save_runtime(self, runtime: RuntimeConfig, driver: str | None = None) -> SaveResult:
        data = runtime.model_dump(mode="json")
        if driver:
            data["driver"] = driver
        else:
            old = self._read_yaml(self.runtime_path).get("runtime", {})
            if "driver" in old:
                data["driver"] = old["driver"]
        return self._write_yaml(self.runtime_path, {"runtime": data})
