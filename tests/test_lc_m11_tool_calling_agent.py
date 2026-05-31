"""
Milestone 11 测试：Tool Calling Agent（结构化工具调用）

TDD RED 阶段：LLM 返回 JSON 格式的 tool_calls，Agent 直接执行，
取代 ReAct 文本正则匹配。
"""

import pytest


# ============================================================
# LLM tool_calls 格式
# ============================================================

class TestToolCallFormat:

    def test_llm_returns_tool_call(self):
        """LLM 返回结构化 tool_call（JSON dict）"""
        from simple_langchain.llms import FakeToolCallLLM

        # FakeToolCallLLM：模拟 LLM 返回 tool_calls
        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "search", "arguments": {"query": "Python"}}]},
            {"content": "Python 是一种编程语言"},
        ])
        result = llm.invoke("什么是 Python")
        assert isinstance(result, dict)
        assert "tool_calls" in result


# ============================================================
# ToolCallingAgent
# ============================================================

class TestToolCallingAgent:

    def test_single_tool_call(self):
        """Agent 执行一次工具调用后返回"""
        from simple_langchain.agents import ToolCallingAgent, tool
        from simple_langchain.llms import FakeToolCallLLM

        @tool
        def search(query: str) -> str:
            """搜索信息"""
            return f"搜索结果: {query}"

        # LLM 第一轮返回 tool_call，第二轮返回最终回答
        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "search", "arguments": {"query": "Python"}}]},
            {"content": "Python 是一种流行的编程语言"},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[search])
        result = agent.invoke("什么是 Python")
        assert "Python" in result

    def test_multi_tool_calls(self):
        """Agent 多次工具调用"""
        from simple_langchain.agents import ToolCallingAgent, tool
        from simple_langchain.llms import FakeToolCallLLM

        @tool
        def add(a: int, b: int) -> int:
            """加法"""
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            """乘法"""
            return a * b

        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}}]},
            {"tool_calls": [{"name": "multiply", "arguments": {"a": 5, "b": 4}}]},
            {"content": "2+3=5, 5*4=20"},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[add, multiply])
        result = agent.invoke("先加后乘")
        assert "20" in result

    def test_no_tool_call_direct_answer(self):
        """LLM 直接回答，不调用工具"""
        from simple_langchain.agents import ToolCallingAgent
        from simple_langchain.llms import FakeToolCallLLM

        llm = FakeToolCallLLM(responses=[
            {"content": "你好！我是 AI 助手。"},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[])
        result = agent.invoke("你好")
        assert result == "你好！我是 AI 助手。"

    def test_tool_not_found(self):
        """LLM 调用不存在的工具"""
        from simple_langchain.agents import ToolCallingAgent, tool
        from simple_langchain.llms import FakeToolCallLLM

        @tool
        def search(query: str) -> str:
            """搜索"""
            return "ok"

        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "nonexistent", "arguments": {"x": 1}}]},
            {"content": "抱歉，没有那个工具"},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[search])
        result = agent.invoke("测试")
        assert "抱歉" in result

    def test_max_iterations(self):
        """超过最大迭代次数"""
        from simple_langchain.agents import ToolCallingAgent, tool
        from simple_langchain.llms import FakeToolCallLLM

        @tool
        def loop_tool(x: str) -> str:
            """循环"""
            return "looping"

        # LLM 永远返回 tool_call
        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "loop_tool", "arguments": {"x": "again"}}]},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[loop_tool], max_iterations=3)
        with pytest.raises(RuntimeError, match="maximum"):
            agent.invoke("循环测试")

    def test_tool_call_with_multi_params(self):
        """@tool 装饰的多参数工具在 agent 中正常工作"""
        from simple_langchain.agents import ToolCallingAgent, tool
        from simple_langchain.llms import FakeToolCallLLM

        @tool
        def weather(city: str, unit: str) -> str:
            """查天气"""
            return f"{city} 晴天 25°{unit}"

        llm = FakeToolCallLLM(responses=[
            {"tool_calls": [{"name": "weather", "arguments": {"city": "北京", "unit": "C"}}]},
            {"content": "北京今天晴天 25°C"},
        ])

        agent = ToolCallingAgent(llm=llm, tools=[weather])
        result = agent.invoke("北京天气怎么样")
        assert "25°C" in result
