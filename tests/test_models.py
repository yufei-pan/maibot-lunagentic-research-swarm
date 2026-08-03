from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lunagentic_research_swarm.models import (
    BranchRuntime,
    FormalizedTask,
    TaskStatus,
    new_task_id,
)
from lunagentic_research_swarm.errors import LRSError


def test_task_id_and_formalized_task_are_stable() -> None:
    """若 ID 格式或 UTF-8 哈希改变，本测试应失败。"""

    task_id = new_task_id()
    formalized = FormalizedTask.create("目标：逐字节保持。\n")

    assert task_id.startswith("lrs_")
    assert len(task_id) == 36
    assert formalized.text.encode("utf-8") == (
        b"\xe7\x9b\xae\xe6\xa0\x87\xef\xbc\x9a\xe9\x80\x90\xe5\xad\x97\xe8\x8a\x82"
        b"\xe4\xbf\x9d\xe6\x8c\x81\xe3\x80\x82\n"
    )
    assert FormalizedTask.create(formalized.text).sha256 == formalized.sha256
    assert TaskStatus.RUNNING.value == "RUNNING"


def test_formalized_task_rejects_blank_text() -> None:
    """若空白任务可进入调度，本测试应失败。"""

    with pytest.raises(ValueError, match="正式任务描述不能为空"):
        FormalizedTask.create("  ")


def test_lrs_error_has_a_stable_result_envelope() -> None:
    """若错误载荷丢失 code、message 或 metadata，本测试应失败。"""

    error = LRSError("invalid_state", "状态不允许", {"status": "PAUSED"})

    assert error.to_result() == {
        "success": False,
        "error": {"code": "invalid_state", "message": "状态不允许", "metadata": {"status": "PAUSED"}},
    }


def test_formalized_task_cannot_be_mutated() -> None:
    """若正式任务正文可被摘要或运行期逻辑覆盖，本测试应失败。"""

    formalized = FormalizedTask.create("保留的原始任务")

    with pytest.raises(FrozenInstanceError):
        formalized.text = "被覆盖"  # type: ignore[misc]


def test_branch_prompt_reinserts_original_task_text() -> None:
    """若压缩消息能替代正式任务原文，本测试应失败。"""

    formalized = FormalizedTask.create("原始正式任务\n")
    runtime = BranchRuntime(
        branch_id="br_1",
        task=formalized,
        catalog_fingerprint="catalog_1",
        generation=2,
        messages=[{"role": "user", "content": "压缩后的替代文本"}],
        credits=10.0,
        depth=0,
    )

    prompt_messages = runtime.build_prompt_messages()

    assert prompt_messages[0] == {"role": "user", "content": "原始正式任务\n"}
    assert "压缩后的替代文本" in [message["content"] for message in prompt_messages]
