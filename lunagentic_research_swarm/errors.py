"""LRS 对外暴露的稳定错误载荷。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TASK_NOT_FOUND = "task_not_found"
INVALID_STATE = "invalid_state"
INVALID_SELECTOR = "invalid_selector"
ROOT_AGENT_UNAVAILABLE = "root_agent_unavailable"
SUMMARIZER_UNAVAILABLE = "summarizer_unavailable"
AGENT_UNAVAILABLE = "agent_unavailable"
PROCEDURE_UNAVAILABLE = "procedure_unavailable"
PROTOCOL_INVALID = "protocol_invalid"
STORAGE_COMMIT_FAILED = "storage_commit_failed"
TASK_FINISHED_INSUFFICIENT_FUNDS = "task_finished_insufficient_funds"
EMBEDDING_GENERATION_MISMATCH = "embedding_generation_mismatch"
VECTOR_INDEX_REBUILDING = "vector_index_rebuilding"
VECTOR_REBUILD_FAILED = "vector_rebuild_failed"
VECTOR_INDEX_UNAVAILABLE = "vector_index_unavailable"


class LRSError(Exception):
    """可安全返回给调用方的领域错误。"""

    def __init__(self, code: str, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = dict(metadata) if metadata is not None else None

    def to_result(self) -> dict[str, Any]:
        """返回命令与工具共用的失败信封。"""

        return {
            "success": False,
            "error": {"code": self.code, "message": self.message, "metadata": self.metadata},
        }
