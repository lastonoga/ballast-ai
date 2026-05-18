from pydantic_ai_stateflow.ruff.stateflow_rules import check_source


def _violations(code: str) -> list[str]:
    return [v.rule_id for v in check_source(code)]


def test_STATEFLOW001_datetime_now_in_workflow_body() -> None:  # noqa: N802
    code = """
from datetime import datetime
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return datetime.now()
"""
    assert "STATEFLOW001" in _violations(code)


def test_STATEFLOW001_clean_when_datetime_in_step() -> None:  # noqa: N802
    code = """
from datetime import datetime
from dbos import DBOS

@DBOS.step()
async def ok():
    return datetime.now()
"""
    assert "STATEFLOW001" not in _violations(code)


def test_STATEFLOW002_time_time_in_workflow() -> None:  # noqa: N802
    code = """
import time
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return time.time()
"""
    assert "STATEFLOW002" in _violations(code)


def test_STATEFLOW003_httpx_call_in_workflow() -> None:  # noqa: N802
    code = """
import httpx
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return await httpx.get("https://example.com")
"""
    assert "STATEFLOW003" in _violations(code)


def test_STATEFLOW004_random_in_workflow() -> None:  # noqa: N802
    code = """
import random
from dbos import DBOS

@DBOS.workflow()
async def bad():
    return random.random()
"""
    assert "STATEFLOW004" in _violations(code)


def test_STATEFLOW005_asyncio_sleep_in_workflow() -> None:  # noqa: N802
    code = """
import asyncio
from dbos import DBOS

@DBOS.workflow()
async def bad():
    await asyncio.sleep(1)
"""
    assert "STATEFLOW005" in _violations(code)


def test_clean_workflow_with_no_violations() -> None:
    code = """
from dbos import DBOS

@DBOS.workflow()
async def good():
    x = 1 + 1
    return x
"""
    assert _violations(code) == []
