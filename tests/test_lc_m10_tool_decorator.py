"""
Milestone 10 测试：@tool 装饰器

TDD RED 阶段：从函数签名自动推导 name / description / args_schema。
"""

import pytest


# ============================================================
# @tool 基本用法
# ============================================================

class TestToolDecorator:

    def test_tool_from_function(self):
        """@tool 把函数变成 Tool"""
        from simple_langchain.agents import tool, Tool

        @tool
        def search(query: str) -> str:
            """搜索信息"""
            return f"结果: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert search.description == "搜索信息"

    def test_tool_run_with_str(self):
        """@tool 函数仍然可以 run（单参数 str）"""
        from simple_langchain.agents import tool

        @tool
        def echo(text: str) -> str:
            """回显输入"""
            return f"echo: {text}"

        assert echo.run("hello") == "echo: hello"

    def test_tool_multi_param(self):
        """@tool 多参数函数，自动推导 args_schema"""
        from simple_langchain.agents import tool

        @tool
        def add(a: int, b: int) -> int:
            """两数相加"""
            return a + b

        assert add.name == "add"
        assert add.args_schema is not None
        result = add.run({"a": 3, "b": 4})
        assert result == "7"

    def test_tool_custom_name(self):
        """@tool 支持自定义名称"""
        from simple_langchain.agents import tool

        @tool(name="web_search")
        def search(query: str) -> str:
            """搜索网页"""
            return f"found: {query}"

        assert search.name == "web_search"

    def test_tool_no_docstring(self):
        """没有 docstring 时 description 为空"""
        from simple_langchain.agents import tool

        @tool
        def no_docs(x: str) -> str:
            return x

        assert no_docs.description == ""

    def test_tool_preserves_function_behavior(self):
        """@tool 装饰后底层函数仍然可用"""
        from simple_langchain.agents import tool

        @tool
        def double(x: int) -> int:
            """翻倍"""
            return x * 2

        # 通过 func 属性访问原始函数
        assert double.func(5) == 10

    def test_tool_args_schema_has_types(self):
        """args_schema 包含正确的字段类型"""
        from simple_langchain.agents import tool

        @tool
        def greet(name: str, age: int) -> str:
            """问候"""
            return f"{name} is {age}"

        schema = greet.args_schema
        fields = schema.model_fields
        assert "name" in fields
        assert "age" in fields
        # 验证类型注解
        assert fields["name"].annotation == str
        assert fields["age"].annotation == int

    def test_tool_single_str_param_no_schema(self):
        """只有单个 str 参数时不生成 args_schema（走老路径）"""
        from simple_langchain.agents import tool

        @tool
        def simple(text: str) -> str:
            """简单"""
            return text

        # 单参数 str 的工具，args_schema 为 None
        assert simple.args_schema is None
        assert simple.run("hello") == "hello"
