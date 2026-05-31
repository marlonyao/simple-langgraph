"""
Simple LangChain — Runnable 协议（LCEL 基础）

核心概念：
- Runnable：所有可运行组件的基类（invoke / batch / __or__）
- RunnableLambda：把普通函数包装成 Runnable
- RunnableSequence：管道组合（a | b | c）

对应 LangChain LCEL（LangChain Expression Language）的核心抽象。
"""

from __future__ import annotations
from typing import Any, Callable


class Runnable:
    """
    所有可运行组件的基类。

    核心方法：
    - invoke(input) → 输出
    - batch(inputs) → 输出列表
    - __or__(other) → 管道组合
    """

    def invoke(self, input: Any) -> Any:
        """执行单个输入，返回输出。子类必须实现。"""
        raise NotImplementedError

    def batch(self, inputs: list[Any]) -> list[Any]:
        """批量执行：对每个输入调用 invoke"""
        return [self.invoke(input) for input in inputs]

    def __or__(self, other: Runnable) -> RunnableSequence:
        """管道操作：self | other → RunnableSequence"""
        if not isinstance(other, Runnable):
            raise TypeError(
                f"Cannot pipe with {type(other).__name__}; "
                f"expected a Runnable"
            )
        return RunnableSequence(first=self, second=other)


class RunnableLambda(Runnable):
    """
    把普通函数包装成 Runnable。

    用法：RunnableLambda(lambda x: x + 1)
    """

    def __init__(self, func: Callable):
        self._func = func

    def invoke(self, input: Any) -> Any:
        return self._func(input)


class RunnableSequence(Runnable):
    """
    管道组合：将两个 Runnable 串联。

    invoke 时：先把 input 传给 first，输出传给 second。
    支持继续管道：(a | b) | c
    """

    def __init__(self, first: Runnable, second: Runnable):
        self._first = first
        self._second = second

    def invoke(self, input: Any) -> Any:
        mid = self._first.invoke(input)
        return self._second.invoke(mid)
