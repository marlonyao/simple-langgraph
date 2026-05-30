"""
LC-M2.1 增强：Token 使用量统计

TDD RED 阶段：测试 Usage 数据结构、Generation 挂载、LLMResult 聚合、invoke_full、回调收集。
"""

import pytest


# ============================================================
# Usage 数据结构
# ============================================================

class TestUsage:

    def test_create_usage(self):
        from simple_langchain.llms import Usage
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    def test_total_tokens(self):
        from simple_langchain.llms import Usage
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15

    def test_default_zero(self):
        from simple_langchain.llms import Usage
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_add_two_usages(self):
        from simple_langchain.llms import Usage
        a = Usage(prompt_tokens=10, completion_tokens=5)
        b = Usage(prompt_tokens=8, completion_tokens=3)
        total = a + b
        assert total.prompt_tokens == 18
        assert total.completion_tokens == 8
        assert total.total_tokens == 26


# ============================================================
# Generation 含 Usage
# ============================================================

class TestGenerationWithUsage:

    def test_generation_has_usage(self):
        from simple_langchain.llms import Generation, Usage
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        gen = Generation(text="你好", prompt="hello", usage=usage)
        assert gen.usage.prompt_tokens == 10
        assert gen.usage.completion_tokens == 5

    def test_generation_default_usage(self):
        from simple_langchain.llms import Generation
        gen = Generation(text="hello")
        assert gen.usage.total_tokens == 0


# ============================================================
# LLMResult 聚合
# ============================================================

class TestLLMResultAggregation:

    def test_total_usage_from_generations(self):
        from simple_langchain.llms import LLMResult, Generation, Usage
        result = LLMResult(generations=[
            Generation(text="a", usage=Usage(prompt_tokens=10, completion_tokens=2)),
            Generation(text="b", usage=Usage(prompt_tokens=5, completion_tokens=3)),
        ])
        total = result.total_usage
        assert total.prompt_tokens == 15
        assert total.completion_tokens == 5
        assert total.total_tokens == 20

    def test_empty_result_usage(self):
        from simple_langchain.llms import LLMResult
        result = LLMResult(generations=[])
        assert result.total_usage.total_tokens == 0


# ============================================================
# invoke_full
# ============================================================

class TestInvokeFull:

    def test_invoke_full_returns_generation(self):
        from simple_langchain.llms import FakeListLLM, Generation
        llm = FakeListLLM(responses=["你好世界"])
        gen = llm.invoke_full("hello")
        assert isinstance(gen, Generation)
        assert gen.text == "你好世界"

    def test_invoke_full_has_usage(self):
        from simple_langchain.llms import FakeListLLM
        llm = FakeListLLM(responses=["你好世界"])
        gen = llm.invoke_full("hello world test")
        assert gen.usage.prompt_tokens > 0
        assert gen.usage.completion_tokens > 0

    def test_invoke_full_usage_accumulates_in_generate(self):
        from simple_langchain.llms import FakeListLLM
        llm = FakeListLLM(responses=["a", "b"])
        result = llm.generate(["prompt one", "prompt two"])
        total = result.total_usage
        assert total.prompt_tokens > 0
        assert total.completion_tokens > 0


# ============================================================
# 回调收集
# ============================================================

class TestCallbackCollection:

    def test_on_llm_end_receives_generation(self):
        from simple_langchain.llms import FakeListLLM, Generation
        collected = []
        llm = FakeListLLM(
            responses=["hello"],
            callbacks={"on_llm_end": lambda gen: collected.append(gen)},
        )
        llm.invoke("hi")
        assert len(collected) == 1
        assert isinstance(collected[0], Generation)
        assert collected[0].text == "hello"

    def test_on_llm_end_generation_has_usage(self):
        from simple_langchain.llms import FakeListLLM
        collected = []
        llm = FakeListLLM(
            responses=["你好"],
            callbacks={"on_llm_end": lambda gen: collected.append(gen)},
        )
        llm.invoke("test prompt")
        assert collected[0].usage.total_tokens > 0

    def test_multiple_invokes_accumulate_usage(self):
        from simple_langchain.llms import FakeListLLM
        collected = []
        llm = FakeListLLM(
            responses=["a", "b", "c"],
            callbacks={"on_llm_end": lambda gen: collected.append(gen)},
        )
        llm.invoke("one")
        llm.invoke("two two")
        llm.invoke("three three three")
        total = sum(g.usage.total_tokens for g in collected)
        assert total > 0
        # "two two" 比 "one" 的 prompt_tokens 多
        assert collected[1].usage.prompt_tokens > collected[0].usage.prompt_tokens
