"""
Simple LangChain — LLM 抽象层

核心概念：
- BaseLLM：抽象基类，定义 invoke / generate 接口
- FakeListLLM：测试用假模型，按预设列表返回结果
- Generation / LLMResult：调用的输入输出数据结构
- 回调机制：on_llm_start / on_llm_end / on_llm_error
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Generation:
    """单次 LLM 生成的结果"""
    text: str
    prompt: str = ""


@dataclass
class LLMResult:
    """LLM 调用的完整结果，包含多个 Generation"""
    generations: list[Generation] = field(default_factory=list)

    def __iter__(self):
        return iter(self.generations)


class BaseLLM(ABC):
    """
    LLM 抽象基类。

    子类只需实现 _invoke(prompt) -> str，
    invoke() 和 generate() 由基类提供，包含回调逻辑。
    """

    def __init__(self, callbacks: dict[str, Callable] | None = None):
        self._callbacks = callbacks or {}

    @abstractmethod
    def _invoke(self, prompt: str) -> str:
        """子类实现：给定 prompt，返回生成的文本"""
        ...

    def invoke(self, prompt: str) -> str:
        """
        调用 LLM 生成回复。

        触发回调：on_llm_start → _invoke → on_llm_end / on_llm_error
        """
        # on_llm_start
        if "on_llm_start" in self._callbacks:
            self._callbacks["on_llm_start"](prompt)

        try:
            result = self._invoke(prompt)
        except Exception as e:
            # on_llm_error
            if "on_llm_error" in self._callbacks:
                self._callbacks["on_llm_error"](e)
            raise

        # on_llm_end
        if "on_llm_end" in self._callbacks:
            self._callbacks["on_llm_end"](result)

        return result

    def generate(self, prompts: list[str]) -> LLMResult:
        """
        批量调用：给定多个 prompt，返回 LLMResult。

        内部循环调用 invoke()，每次都会触发回调。
        """
        generations = []
        for prompt in prompts:
            text = self.invoke(prompt)
            generations.append(Generation(text=text, prompt=prompt))
        return LLMResult(generations=generations)


class FakeListLLM(BaseLLM):
    """
    测试用假 LLM。

    按预设列表循环返回固定响应，不调用任何真实 API。
    """

    def __init__(
        self,
        responses: list[str],
        callbacks: dict[str, Callable] | None = None,
    ):
        super().__init__(callbacks=callbacks)
        self._responses = responses
        self._index = 0

    def _invoke(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("FakeListLLM has empty responses list")
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response
