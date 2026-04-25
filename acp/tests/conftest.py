"""
pytest 配置与共享 Fixture

为 test_web_adapter.py 等集成测试提供 WebAdapter fixture。
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from acp.adapters.web_adapter import WebAdapter


@pytest_asyncio.fixture(scope="module")
async def adapter():
    """模块级 WebAdapter fixture：启动无头浏览器，导航到 example.com，测试结束后关闭。"""
    async with WebAdapter(headless=True) as web_adapter:
        await web_adapter.navigate("https://example.com")
        yield web_adapter
