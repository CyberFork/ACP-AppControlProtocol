"""
Demo C 单测：隐私 assertion + planner 解析 + grounding mock。
"""

from __future__ import annotations

import pytest

from acp.demo.mini_c.planner import PlannerLLM, _assert_no_image, _parse_intent
from acp.demo.mini_c.grounding import UITARSGrounding


# ---------------------------------------------------------------------------
# 隐私 assertion 测试（核心）
# ---------------------------------------------------------------------------

def test_assert_no_image_passes_text_only():
    """纯文本 messages 不报错。"""
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "What is on the screen?"},
    ]
    _assert_no_image(messages)  # 不应 raise


def test_assert_no_image_raises_on_image_url():
    """包含 image_url 的 messages 必须 raise AssertionError。"""
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
            ],
        },
    ]
    with pytest.raises(AssertionError, match="D11 隐私违规"):
        _assert_no_image(messages)


def test_assert_no_image_raises_on_base64_string():
    """字符串内容中含 base64 图像特征应 raise。"""
    # PNG base64 header
    messages = [
        {"role": "user", "content": "iVBOR" + "A" * 600},
    ]
    with pytest.raises(AssertionError, match="D11 隐私违规"):
        _assert_no_image(messages)


def test_assert_no_image_passes_list_text_only():
    """list 格式但只有 text type 不报错。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "page state: login form visible"},
            ],
        },
    ]
    _assert_no_image(messages)  # 不应 raise


# ---------------------------------------------------------------------------
# Planner 输出解析测试
# ---------------------------------------------------------------------------

def test_parse_intent_clean_json():
    raw = '{"intent": "click", "target_description": "X 关闭按钮", "text": "", "rationale": "关弹窗", "is_done": false}'
    intent = _parse_intent(raw)
    assert intent.intent == "click"
    assert intent.target_description == "X 关闭按钮"
    assert intent.is_done is False


def test_parse_intent_with_markdown():
    raw = "```json\n{\"intent\": \"type\", \"target_description\": \"用户名输入框\", \"text\": \"demo\", \"rationale\": \"填用户名\", \"is_done\": false}\n```"
    intent = _parse_intent(raw)
    assert intent.intent == "type"
    assert intent.text == "demo"


def test_parse_intent_done():
    raw = '{"intent": "done", "target_description": "", "rationale": "已完成登录", "is_done": true}'
    intent = _parse_intent(raw)
    assert intent.intent == "done"
    assert intent.is_done is True


def test_parse_intent_regex_fallback():
    # 格式不标准时用 regex 兜底
    raw = 'The next action is: {"intent": "click", "target_description": "登录按钮", "is_done": false} done.'
    intent = _parse_intent(raw)
    assert intent.intent == "click"


def test_parse_intent_unparseable_returns_fail():
    raw = "I cannot determine what to do next."
    intent = _parse_intent(raw)
    assert intent.intent == "fail"
    assert intent.is_done is True


# ---------------------------------------------------------------------------
# Mock Planner 序列测试
# ---------------------------------------------------------------------------

def test_mock_planner_sequence():
    planner = PlannerLLM(mock=True)
    # 第 1 步：关弹窗
    intent = planner.plan("关闭弹窗并登录", "state", [])
    assert intent.intent == "click"
    assert "X" in intent.target_description or "关闭" in intent.target_description or "弹窗" in intent.target_description

    # 跑完所有步骤
    for _ in range(6):
        intent = planner.plan("", "", [])

    # 超出序列后返回 done
    intent = planner.plan("", "", [])
    assert intent.is_done is True


def test_mock_planner_reset():
    planner = PlannerLLM(mock=True)
    planner.plan("", "", [])
    planner.plan("", "", [])
    planner.reset_mock()
    intent = planner.plan("", "", [])
    # 重置后从第 1 步开始
    assert "X" in intent.target_description or "关闭" in intent.target_description or "弹窗" in intent.target_description


# ---------------------------------------------------------------------------
# Mock Grounding 测试
# ---------------------------------------------------------------------------

def test_mock_grounding_keyword_match():
    grounder = UITARSGrounding(mock=True)
    coord = grounder.locate(b"", "弹窗右上角的 X 关闭按钮")
    assert coord is not None
    assert coord[0] > 1000  # X 按钮应该在右侧


def test_mock_grounding_login_keyword():
    grounder = UITARSGrounding(mock=True)
    coord = grounder.locate(b"", "登录按钮")
    assert coord is not None
    assert coord[1] > 400   # 登录按钮在页面下半部


def test_mock_grounding_fallback():
    grounder = UITARSGrounding(mock=True)
    coord = grounder.locate(b"", "一个完全未知的元素描述ABC123")
    assert coord == (640, 400)  # fallback 到中心


def test_mock_grounding_health_check():
    grounder = UITARSGrounding(mock=True)
    assert grounder.health_check() is True
