"""FlowRecorder 单元测试

测试覆盖：
  - 基础录制和保存
  - 敏感信息脱敏（credentials 反查 + 密码字段语义检测）
  - 追加模式（不覆盖已有流程）
  - 从 FlowRunner 日志录制
  - 空步骤边界情况
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from acp.brain.flow_recorder import (
    FlowRecorder,
    RecordedStep,
    _find_in_credentials,
    _is_sensitive_target,
    _sanitize_value,
)


# ---------------------------------------------------------------------------
# 工具函数测试
# ---------------------------------------------------------------------------


def test_is_sensitive_target_password():
    assert _is_sensitive_target("密码输入框") is True
    assert _is_sensitive_target("password field") is True
    assert _is_sensitive_target("邮箱输入框") is False
    assert _is_sensitive_target("登录按钮") is False


def test_find_in_credentials_flat():
    creds = {"auth": {"email": "user@example.com", "password": "s3cr3t"}}
    assert _find_in_credentials("user@example.com", creds, "") == "auth.email"
    assert _find_in_credentials("s3cr3t", creds, "") == "auth.password"
    assert _find_in_credentials("notfound", creds, "") is None


def test_find_in_credentials_nested():
    creds = {"account": {"api": {"key": "abc123"}}}
    assert _find_in_credentials("abc123", creds, "") == "account.api.key"


def test_sanitize_value_credential_match():
    creds = {"auth": {"email": "user@example.com", "password": "mypassword"}}
    # 邮箱被替换为变量
    result = _sanitize_value("user@example.com", "邮箱输入框", creds)
    assert result == "${auth.email}"
    # 密码被替换为变量
    result = _sanitize_value("mypassword", "密码输入框", creds)
    assert result == "${auth.password}"


def test_sanitize_value_password_semantic():
    # 没有 credentials 匹配，但 target 是密码字段 + 值够长
    result = _sanitize_value("unknownpass123", "密码输入框", {})
    assert result == "${auth.password}"


def test_sanitize_value_short_password_not_masked():
    # 短于 6 个字符的密码字段不替换（可能是误判）
    result = _sanitize_value("abc", "密码输入框", {})
    assert result == "abc"


def test_sanitize_value_normal():
    result = _sanitize_value("some normal text", "搜索框", {})
    assert result == "some normal text"


# ---------------------------------------------------------------------------
# RecordedStep 测试
# ---------------------------------------------------------------------------


def test_recorded_step_to_dict_minimal():
    step = RecordedStep(action="click", target="登录按钮")
    d = step.to_dict()
    assert d["action"] == "click"
    assert d["target"] == "登录按钮"
    assert "value" not in d
    assert "url" not in d


def test_recorded_step_to_dict_full():
    step = RecordedStep(
        action="type",
        target="邮箱输入框",
        value="${auth.email}",
        wait=2,
    )
    d = step.to_dict()
    assert d["action"] == "type"
    assert d["value"] == "${auth.email}"
    assert d["wait"] == 2


def test_recorded_step_default_wait_not_in_dict():
    step = RecordedStep(action="click", target="按钮", wait=1)
    d = step.to_dict()
    # wait=1 是默认值，不输出到 YAML
    assert "wait" not in d


# ---------------------------------------------------------------------------
# FlowRecorder 集成测试
# ---------------------------------------------------------------------------


@pytest.fixture
def site_dir(tmp_path: Path) -> Path:
    """创建一个临时站点目录，带 credentials.yaml 和空 flows.yaml。"""
    credentials = {
        "auth": {
            "email": "user@example.com",
            "password": "mySecretPass123",
        }
    }
    (tmp_path / "credentials.yaml").write_text(
        yaml.dump(credentials, allow_unicode=True), encoding="utf-8"
    )
    existing_flows = {
        "flows": {
            "existing_flow": {
                "description": "已有流程",
                "steps": [{"action": "navigate", "url": "https://example.com"}],
            }
        }
    }
    (tmp_path / "flows.yaml").write_text(
        yaml.dump(existing_flows, allow_unicode=True), encoding="utf-8"
    )
    return tmp_path


def test_basic_record_and_save(site_dir: Path):
    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_step("navigate", url="https://example.com")
    recorder.record_step("click", target="登录按钮")
    recorder.record_step("type", target="邮箱输入框", value="user@example.com")
    assert recorder.step_count == 3
    ok = recorder.save("test_flow", description="测试流程")
    assert ok is True

    # 验证写入 flows.yaml
    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    assert "test_flow" in data["flows"]
    steps = data["flows"]["test_flow"]["steps"]
    assert len(steps) == 3
    assert steps[0]["action"] == "navigate"
    assert steps[2]["value"] == "${auth.email}"  # 已脱敏


def test_password_sanitization(site_dir: Path):
    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_step("type", target="密码输入框", value="mySecretPass123")
    recorder.save("pwd_test")
    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    step = data["flows"]["pwd_test"]["steps"][0]
    # 密码已被替换为 credentials 对应变量
    assert step["value"] == "${auth.password}"


def test_password_semantic_fallback(site_dir: Path):
    """没有在 credentials 匹配的密码也应被语义替换。"""
    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_step("type", target="密码框", value="unknownSecret123")
    recorder.save("semantic_test")
    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    step = data["flows"]["semantic_test"]["steps"][0]
    assert step["value"] == "${auth.password}"


def test_existing_flows_preserved(site_dir: Path):
    """保存新流程不应影响已有流程。"""
    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_step("click", target="按钮")
    recorder.save("new_flow")

    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    # 原有流程仍在
    assert "existing_flow" in data["flows"]
    assert "new_flow" in data["flows"]


def test_overwrite_same_name_flow(site_dir: Path):
    """同名流程被覆盖（不影响其他流程）。"""
    recorder1 = FlowRecorder(site_dir=str(site_dir))
    recorder1.record_step("navigate", url="https://v1.com")
    recorder1.save("my_flow")

    recorder2 = FlowRecorder(site_dir=str(site_dir))
    recorder2.record_step("navigate", url="https://v2.com")
    recorder2.record_step("click", target="按钮")
    recorder2.save("my_flow")

    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    steps = data["flows"]["my_flow"]["steps"]
    assert len(steps) == 2
    assert steps[0]["url"] == "https://v2.com"


def test_save_empty_steps_returns_false(site_dir: Path):
    recorder = FlowRecorder(site_dir=str(site_dir))
    result = recorder.save("empty_flow")
    assert result is False


def test_clear(site_dir: Path):
    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_step("click", target="按钮")
    assert recorder.step_count == 1
    recorder.clear()
    assert recorder.step_count == 0


def test_record_from_flow_runner_log(site_dir: Path):
    """从 FlowRunner 日志批量录制（只录制成功步骤）。"""
    flow_steps = [
        {"action": "navigate", "url": "https://example.com", "wait": 2},
        {"action": "click", "target": "登录按钮"},
        {"action": "type", "target": "邮箱输入框", "value": "user@example.com"},
        {"action": "type", "target": "密码输入框", "value": "mySecretPass123"},
        {"action": "verify", "expected": "登录成功"},
    ]
    log = [
        {"step": 1, "action": "navigate", "success": True},
        {"step": 2, "action": "click", "success": True},
        {"step": 3, "action": "type", "success": True},
        {"step": 4, "action": "type", "success": False},  # 失败步骤不录
        {"step": 5, "action": "verify", "success": True},
    ]

    recorder = FlowRecorder(site_dir=str(site_dir))
    recorder.record_from_flow_runner_log(log=log, flow_steps=flow_steps)

    # 只有 4 个成功步骤
    assert recorder.step_count == 4
    recorder.save("from_log")

    data = yaml.safe_load((site_dir / "flows.yaml").read_text())
    steps = data["flows"]["from_log"]["steps"]
    assert len(steps) == 4
    # 邮箱已脱敏
    type_step = next(s for s in steps if s.get("target") == "邮箱输入框")
    assert type_step["value"] == "${auth.email}"


def test_no_credentials_file():
    """没有 credentials.yaml 时，仍能正常运行。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = FlowRecorder(site_dir=tmpdir)
        recorder.record_step("navigate", url="https://example.com")
        recorder.record_step("type", target="普通输入框", value="normal text")
        recorder.save("no_creds_flow")

        data = yaml.safe_load((Path(tmpdir) / "flows.yaml").read_text())
        steps = data["flows"]["no_creds_flow"]["steps"]
        assert steps[1]["value"] == "normal text"  # 非密码字段不脱敏
