"""
Simple LangChain — LLM 抽象层

核心概念：
- Usage：token 使用量统计
- Generation：单次生成结果（含 usage）
- LLMResult：批量调用结果（含聚合 usage）
- BaseLLM：抽象基类，定义 invoke / invoke_full / generate 接口
- FakeListLLM：测试用假模型，按预设列表返回结果
- 回调机制：on_llm_start / on_llm_end（传 Generation）/ on_llm_error
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from simple_langchain.runnable import Runnable


@dataclass
class Usage:
    """Token 使用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Generation:
    """单次 LLM 生成的结果"""
    text: str
    prompt: str = ""
    usage: Usage = field(default_factory=Usage)


@dataclass
class LLMResult:
    """LLM 调用的完整结果，包含多个 Generation"""
    generations: list[Generation] = field(default_factory=list)

    def __iter__(self):
        return iter(self.generations)

    @property
    def total_usage(self) -> Usage:
        """聚合所有 generation 的 usage"""
        result = Usage()
        for gen in self.generations:
            result = result + gen.usage
        return result


class BaseLLM(Runnable, ABC):
    """
    LLM 抽象基类。

    继承 Runnable，子类只需实现 _invoke(prompt) -> str，
    invoke()、invoke_full()、generate() 由基类提供，包含回调和 usage 逻辑。

    作为 Runnable：invoke(str) → str（直接生成文本）
    """

    def __init__(self, callbacks: dict[str, Callable] | None = None):
        self._callbacks = callbacks or {}

    @abstractmethod
    def _invoke(self, prompt: str) -> str:
        """子类实现：给定 prompt，返回生成的文本"""
        ...

    def _estimate_tokens(self, text: str) -> int:
        """
        粗略估算 token 数。

        真实场景由 API 服务端计算。这里用简单的空格分词模拟。
        子类可以重写这个方法用更精确的 tokenizer。
        """
        if not text:
            return 0
        return len(text.split())

    def invoke(self, prompt: str) -> str:
        """
        调用 LLM 生成回复，返回文本字符串。

        触发回调：on_llm_start → _invoke → on_llm_end / on_llm_error
        """
        return self.invoke_full(prompt).text

    def invoke_full(self, prompt: str) -> Generation:
        """
        调用 LLM 生成回复，返回完整的 Generation（含 usage）。

        触发回调：on_llm_start → _invoke → on_llm_end / on_llm_error
        on_llm_end 收到的是 Generation 对象（含 usage）。
        """
        # on_llm_start
        if "on_llm_start" in self._callbacks:
            self._callbacks["on_llm_start"](prompt)

        try:
            result_text = self._invoke(prompt)
        except Exception as e:
            # on_llm_error
            if "on_llm_error" in self._callbacks:
                self._callbacks["on_llm_error"](e)
            raise

        # 构建 Generation（含 usage）
        generation = Generation(
            text=result_text,
            prompt=prompt,
            usage=Usage(
                prompt_tokens=self._estimate_tokens(prompt),
                completion_tokens=self._estimate_tokens(result_text),
            ),
        )

        # on_llm_end：传 Generation 对象
        if "on_llm_end" in self._callbacks:
            self._callbacks["on_llm_end"](generation)

        return generation

    def generate(self, prompts: list[str]) -> LLMResult:
        """
        批量调用：给定多个 prompt，返回 LLMResult。

        内部循环调用 invoke_full()，每次都会触发回调。
        """
        generations = []
        for prompt in prompts:
            gen = self.invoke_full(prompt)
            generations.append(gen)
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


class FakeToolCallLLM:
    """
    测试用假 LLM：返回结构化 tool_call 格式。

    和 FakeListLLM 不同，这个不继承 BaseLLM，
    因为它的 invoke 返回 dict 而不是 str。

    返回格式：
    - {"tool_calls": [{"name": "...", "arguments": {...}}]}  → 工具调用
    - {"content": "..."}  → 最终文本回答
    """

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._index = 0

    def invoke(self, prompt: str) -> dict:
        if not self._responses:
            raise RuntimeError("FakeToolCallLLM has empty responses list")
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response
