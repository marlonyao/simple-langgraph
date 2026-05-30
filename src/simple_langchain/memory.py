"""
Simple LangChain — Memory（对话记忆）

核心概念：
- BaseMemory：抽象基类
- ConversationBufferMemory：完整对话历史
- ConversationBufferWindowMemory：滑动窗口，保留最近 k 轮
- ConversationSummaryMemory：用 LLM 压缩历史为摘要
"""

from abc import ABC, abstractmethod
from typing import Any

from simple_langchain.llms import BaseLLM


class BaseMemory(ABC):
    """记忆基类"""

    @abstractmethod
    def save_context(self, human_input: str, ai_output: str) -> None:
        """保存一轮对话"""
        ...

    @abstractmethod
    def load_memory(self) -> list[dict[str, str]]:
        """加载记忆为消息列表"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        ...


class ConversationBufferMemory(BaseMemory):
    """
    完整对话历史。

    把所有对话原样保存，不做任何压缩。
    简单但会随着对话变长而占用越来越多 token。
    """

    def __init__(self):
        self._messages: list[dict[str, str]] = []

    def save_context(self, human_input: str, ai_output: str) -> None:
        self._messages.append({"role": "human", "content": human_input})
        self._messages.append({"role": "ai", "content": ai_output})

    def load_memory(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class ConversationBufferWindowMemory(BaseMemory):
    """
    滑动窗口记忆。

    只保留最近 k 轮对话（1 轮 = human + ai = 2 条消息）。
    """

    def __init__(self, k: int = 5):
        self._k = k
        self._messages: list[dict[str, str]] = []

    def save_context(self, human_input: str, ai_output: str) -> None:
        self._messages.append({"role": "human", "content": human_input})
        self._messages.append({"role": "ai", "content": ai_output})

    def load_memory(self) -> list[dict[str, str]]:
        # 只保留最近 k 轮（k × 2 条消息）
        max_messages = self._k * 2
        return self._messages[-max_messages:]

    def clear(self) -> None:
        self._messages.clear()


class ConversationSummaryMemory(BaseMemory):
    """
    摘要式记忆。

    每次保存新对话时，用 LLM 把历史摘要 + 新对话压缩成新的摘要。
    永远只有一条 system 消息，内容是对话摘要。
    """

    def __init__(self, llm: BaseLLM):
        self._llm = llm
        self._summary: str = ""

    def save_context(self, human_input: str, ai_output: str) -> None:
        if self._summary:
            prompt = (
                f"以下是之前的对话摘要：\n{self._summary}\n\n"
                f"新的对话：\n用户：{human_input}\n助手：{ai_output}\n\n"
                f"请生成一个包含以上所有信息的简洁摘要："
            )
        else:
            prompt = (
                f"请用简洁的一句话总结以下对话：\n"
                f"用户：{human_input}\n助手：{ai_output}"
            )
        self._summary = self._llm.invoke(prompt)

    def load_memory(self) -> list[dict[str, str]]:
        if not self._summary:
            return []
        return [{"role": "system", "content": self._summary}]

    def clear(self) -> None:
        self._summary = ""
