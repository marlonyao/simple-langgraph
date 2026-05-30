"""
Simple LangGraph — Checkpointer 模块

提供状态持久化能力：
- BaseCheckpointSaver：抽象基类，定义 save / load / list_history 接口
- MemorySaver：内存实现，用字典存储，支持时间旅行（历史回溯）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCheckpointSaver(ABC):
    """Checkpointer 抽象基类"""

    @abstractmethod
    def save(
        self,
        thread_id: str,
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        保存一个 checkpoint。

        返回值：checkpoint_id（唯一标识这次保存）
        """
        ...

    @abstractmethod
    def load(self, thread_id: str) -> Optional[dict[str, Any]]:
        """
        加载最新的 checkpoint。

        返回值：{"state": ..., "metadata": ..., "checkpoint_id": ...} 或 None
        """
        ...

    @abstractmethod
    def list_history(self, thread_id: str) -> list[dict[str, Any]]:
        """
        列出所有历史 checkpoint，最新在前。

        返回值：[{"state": ..., "metadata": ..., "checkpoint_id": ...}, ...]
        """
        ...


class MemorySaver(BaseCheckpointSaver):
    """
    内存 Checkpointer。

    数据结构：{thread_id: [checkpoint1, checkpoint2, ...]}
    每次 save 追加到列表末尾，load 返回最后一个，list_history 返回反转后的列表。
    """

    def __init__(self) -> None:
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._counter: int = 0

    def save(
        self,
        thread_id: str,
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        self._counter += 1
        checkpoint = {
            "state": dict(state),
            "metadata": metadata or {},
            "checkpoint_id": self._counter,
        }
        self._storage.setdefault(thread_id, []).append(checkpoint)
        return self._counter

    def load(self, thread_id: str) -> Optional[dict[str, Any]]:
        entries = self._storage.get(thread_id)
        if not entries:
            return None
        return entries[-1]

    def list_history(self, thread_id: str) -> list[dict[str, Any]]:
        entries = self._storage.get(thread_id, [])
        return list(reversed(entries))
