"""
Milestone 7 测试：Agent + Tool（代理 + 工具调用）

TDD RED 阶段：测试工具定义、ReAct Agent、AgentExecutor。
"""

import pytest


# ============================================================
# Tool
# ============================================================

class TestTool:

    def test_create_tool(self):
        from simple_langchain.agents import Tool

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(name="add", description="两数相加", func=add)
        assert tool.name == "add"
        assert tool.description == "两数相加"

    def test_tool_run(self):
        from simple_langchain.agents import Tool

        tool = Tool(name="echo", description="回显输入", func=lambda x: f"echo: {x}")
        result = tool.run("hello")
        assert result == "echo: hello"

    def test_tool_run_with_dict_input(self):
        from simple_langchain.agents import Tool

        tool = Tool(
            name="weather",
            description="查天气",
            func=lambda city: f"{city}晴天",
        )
        result = tool.run("北京")
        assert "北京" in result
        assert "晴天" in result

    def test_tool_with_args_schema_accepts_dict(self):
        """Tool 带 args_schema 时，run() 接受 dict 参数"""
        from simple_langchain.agents import Tool
        from pydantic import BaseModel

        class CalculatorInput(BaseModel):
            a: int
            b: int

        def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(
            name="multiply",
            description="两数相乘",
            func=multiply,
            args_schema=CalculatorInput,
        )
        result = tool.run({"a": 3, "b": 4})
        assert result == "12"

    def test_tool_with_args_schema_single_arg(self):
        """单参数工具仍然支持字符串输入"""
        from simple_langchain.agents import Tool

        tool = Tool(
            name="echo",
            description="回显",
            func=lambda x: f"echo: {x}",
        )
        assert tool.run("hello") == "echo: hello"

    def test_tool_args_schema_converts_and_validates(self):
        """args_schema 自动做类型转换"""
        from simple_langchain.agents import Tool
        from pydantic import BaseModel

        class SearchInput(BaseModel):
            query: str
            limit: int

        def search(query: str, limit: int) -> str:
            return f"搜索 '{query}'，返回 {limit} 条结果"

        tool = Tool(
            name="search",
            description="搜索",
            func=search,
            args_schema=SearchInput,
        )
        # limit 传字符串也能自动转 int
        result = tool.run({"query": "Python", "limit": "5"})
        assert "Python" in result
        assert "5 条" in result


# ============================================================
# AgentExecutor
# ============================================================

class TestAgentExecutor:

    def test_simple_tool_call(self):
        """Agent 调用一个工具然后返回"""
        from simple_langchain.agents import Tool, AgentExecutor
        from simple_langchain.llms import FakeListLLM

        # FakeListLLM 的响应模拟 LLM 的 ReAct 格式
        # 第一轮：选择工具 + 输入
        # 第二轮：最终回答
        llm = FakeListLLM(responses=[
            'Thought: 我需要用计算器\nAction: calculator\nAction Input: 2+3',
            'Thought: 我已经得到答案\nFinal Answer: 2+3=5',
        ])
        tools = [
            Tool(name="calculator", description="计算数学表达式", func=lambda x: str(eval(x))),
        ]

        agent = AgentExecutor(llm=llm, tools=tools)
        result = agent.invoke("2+3等于多少？")
        assert result == "2+3=5"

    def test_multi_step_agent(self):
        """Agent 多步调用工具"""
        from simple_langchain.agents import Tool, AgentExecutor
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[
            'Action: search\nAction Input: 北京天气',
            'Action: translate\nAction Input: sunny',
            'Final Answer: 北京今天是晴天',
        ])
        tools = [
            Tool(name="search", description="搜索信息", func=lambda q: "sunny"),
            Tool(name="translate", description="翻译", func=lambda w: {"sunny": "晴天"}[w]),
        ]

        agent = AgentExecutor(llm=llm, tools=tools)
        result = agent.invoke("北京今天天气怎么样？")
        assert result == "北京今天是晴天"

    def test_agent_no_tool_needed(self):
        """Agent 直接回答，不需要工具"""
        from simple_langchain.agents import AgentExecutor
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[
            'Final Answer: 你好！我是AI助手。',
        ])

        agent = AgentExecutor(llm=llm, tools=[])
        result = agent.invoke("你好")
        assert result == "你好！我是AI助手。"

    def test_max_iterations_protection(self):
        """超过最大迭代次数时报错"""
        from simple_langchain.agents import AgentExecutor, Tool
        from simple_langchain.llms import FakeListLLM

        # LLM 永远不输出 Final Answer，导致无限循环
        llm = FakeListLLM(responses=[
            'Action: search\nAction Input: something',
        ])
        tools = [
            Tool(name="search", description="搜索", func=lambda x: "result"),
        ]

        agent = AgentExecutor(llm=llm, tools=tools, max_iterations=3)
        with pytest.raises(RuntimeError, match="maximum"):
            agent.invoke("无限循环测试")

    def test_invalid_tool_name(self):
        """Agent 选了不存在的工具"""
        from simple_langchain.agents import AgentExecutor, Tool
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[
            'Action: nonexistent\nAction Input: test',
            'Final Answer: 抱歉，没有这个工具',
        ])
        tools = [
            Tool(name="search", description="搜索", func=lambda x: "ok"),
        ]

        agent = AgentExecutor(llm=llm, tools=tools)
        result = agent.invoke("测试")
        assert result == "抱歉，没有这个工具"

    def test_agent_receives_observation(self):
        """工具执行结果作为观察反馈给 Agent"""
        from simple_langchain.agents import Tool, AgentExecutor
        from simple_langchain.llms import FakeListLLM

        call_log = []

        def mock_llm_invoke(prompt):
            call_log.append(prompt)
            if len(call_log) == 1:
                return 'Action: calc\nAction Input: 10*10'
            return 'Final Answer: 100'

        class SpyLLM:
            """记录每次调用收到的 prompt"""
            def invoke(self, prompt):
                return mock_llm_invoke(prompt)

        tools = [
            Tool(name="calc", description="计算", func=lambda x: str(eval(x))),
        ]

        agent = AgentExecutor(llm=SpyLLM(), tools=tools)
        agent.invoke("10*10=?")

        # 第二次调用应该包含工具的观察结果
        assert "100" in call_log[1]

    def test_agent_with_scratchpad(self):
        """Agent 维护完整的思考过程"""
        from simple_langchain.agents import Tool, AgentExecutor
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[
            'Action: lookup\nAction Input: capital of France',
            'Action: lookup\nAction Input: population of Paris',
            'Final Answer: 法国的首都是巴黎，人口约200万',
        ])
        tools = [
            Tool(name="lookup", description="查找", func=lambda q: {
                "capital of France": "Paris",
                "population of Paris": "2 million",
            }.get(q, "unknown")),
        ]

        agent = AgentExecutor(llm=llm, tools=tools)
        result = agent.invoke("法国首都是哪？人口多少？")
        assert "巴黎" in result
