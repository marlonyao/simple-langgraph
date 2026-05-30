"""
Milestone 2 测试：LLM 抽象层

TDD RED 阶段：测试 BaseLLM、FakeListLLM、回调机制。
"""

import pytest


# ============================================================
# BaseLLM 抽象基类
# ============================================================

class TestBaseLLM:

    def test_cannot_instantiate_base_llm(self):
        """BaseLLM 是抽象类，不能直接实例化"""
        from simple_langchain.llms import BaseLLM

        with pytest.raises(TypeError):
            BaseLLM()

    def test_subclass_must_implement_invoke(self):
        """子类必须实现 _invoke"""
        from simple_langchain.llms import BaseLLM

        class IncompleteLLM(BaseLLM):
            pass

        with pytest.raises(TypeError):
            IncompleteLLM()


# ============================================================
# FakeListLLM
# ============================================================

class TestFakeListLLM:

    def test_invoke_returns_first_response(self):
        """第一次调用返回列表第一个"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["你好", "世界"])
        result = llm.invoke("随便什么输入")
        assert result == "你好"

    def test_invoke_cycles_through_responses(self):
        """循环使用响应列表"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["A", "B", "C"])
        assert llm.invoke("q1") == "A"
        assert llm.invoke("q2") == "B"
        assert llm.invoke("q3") == "C"
        assert llm.invoke("q4") == "A"  # 循环回来

    def test_invoke_single_response(self):
        """只有一个响应，永远返回它"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["固定回复"])
        assert llm.invoke("1") == "固定回复"
        assert llm.invoke("2") == "固定回复"

    def test_generate_returns_generation_object(self):
        """generate 返回包含 Generation 对象的结果"""
        from simple_langchain.llms import FakeListLLM, Generation, LLMResult

        llm = FakeListLLM(responses=["hello"])
        result = llm.generate(["prompt1"])
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert isinstance(result.generations[0], Generation)
        assert result.generations[0].text == "hello"

    def test_generate_multiple_prompts(self):
        """generate 支持多个 prompt"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["A", "B", "C"])
        result = llm.generate(["p1", "p2"])
        assert len(result.generations) == 2
        assert result.generations[0].text == "A"
        assert result.generations[1].text == "B"

    def test_generate_preserves_llm_output(self):
        """Generation 保留元信息"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["response"])
        result = llm.generate(["prompt"])
        gen = result.generations[0]
        assert gen.text == "response"
        assert gen.prompt == "prompt"


# ============================================================
# 回调机制
# ============================================================

class TestCallbacks:

    def test_on_llm_start_callback(self):
        """invoke 前触发 on_llm_start"""
        from simple_langchain.llms import FakeListLLM

        calls = []
        llm = FakeListLLM(
            responses=["hi"],
            callbacks={"on_llm_start": lambda prompt: calls.append(("start", prompt))},
        )
        llm.invoke("test prompt")
        assert len(calls) == 1
        assert calls[0] == ("start", "test prompt")

    def test_on_llm_end_callback(self):
        """invoke 后触发 on_llm_end"""
        from simple_langchain.llms import FakeListLLM

        calls = []
        llm = FakeListLLM(
            responses=["hi"],
            callbacks={"on_llm_end": lambda response: calls.append(("end", response))},
        )
        llm.invoke("test")
        assert len(calls) == 1
        assert calls[0] == ("end", "hi")

    def test_on_llm_error_callback(self):
        """出错时触发 on_llm_error"""
        from simple_langchain.llms import FakeListLLM

        calls = []
        llm = FakeListLLM(
            responses=[],
            callbacks={"on_llm_error": lambda error: calls.append(("error", str(error)))},
        )
        with pytest.raises(Exception):
            llm.invoke("test")
        assert len(calls) == 1
        assert "empty" in calls[0][1].lower()

    def test_callbacks_fire_in_order(self):
        """回调按 start → end 的顺序触发"""
        from simple_langchain.llms import FakeListLLM

        order = []
        llm = FakeListLLM(
            responses=["ok"],
            callbacks={
                "on_llm_start": lambda p: order.append("start"),
                "on_llm_end": lambda r: order.append("end"),
            },
        )
        llm.invoke("test")
        assert order == ["start", "end"]


# ============================================================
# LLMResult / Generation 数据结构
# ============================================================

class TestDataStructures:

    def test_generation_fields(self):
        """Generation 有 text 和 prompt"""
        from simple_langchain.llms import Generation

        gen = Generation(text="hello", prompt="hi")
        assert gen.text == "hello"
        assert gen.prompt == "hi"

    def test_llm_result_fields(self):
        """LLMResult 包含 generations 列表"""
        from simple_langchain.llms import LLMResult, Generation

        gens = [Generation(text="a", prompt="p1"), Generation(text="b", prompt="p2")]
        result = LLMResult(generations=gens)
        assert len(result.generations) == 2

    def test_llm_result_iterable(self):
        """LLMResult 可以迭代"""
        from simple_langchain.llms import LLMResult, Generation

        result = LLMResult(generations=[
            Generation(text="a", prompt="p1"),
            Generation(text="b", prompt="p2"),
        ])
        texts = [g.text for g in result]
        assert texts == ["a", "b"]
