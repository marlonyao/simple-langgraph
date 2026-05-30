"""
Milestone 1 测试：Prompt Template

TDD RED 阶段：测试 PromptTemplate 和 ChatPromptTemplate。
"""

import pytest


# ============================================================
# PromptTemplate 基础
# ============================================================

class TestPromptTemplate:

    def test_format_with_single_variable(self):
        """单个变量的模板"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("你好，{name}！")
        result = template.format(name="世界")
        assert result == "你好，世界！"

    def test_format_with_multiple_variables(self):
        """多个变量的模板"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template(
            "请用{language}写一个关于{topic}的{style}文章。"
        )
        result = template.format(language="Python", topic="排序算法", style="教程")
        assert "Python" in result
        assert "排序算法" in result
        assert "教程" in result

    def test_input_variables_auto_detected(self):
        """自动检测模板中的变量名"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("{a} and {b} and {c}")
        assert template.input_variables == ["a", "b", "c"]

    def test_format_missing_variable_raises(self):
        """缺少变量时报错"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("{name} is {age} years old")
        with pytest.raises(KeyError):
            template.format(name="Alice")  # 缺少 age

    def test_format_extra_variables_ignored(self):
        """多余变量被忽略"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("Hello {name}!")
        result = template.format(name="World", extra="ignored")
        assert result == "Hello World!"

    def test_no_variables_template(self):
        """没有变量的模板"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("这是一个静态提示词。")
        assert template.input_variables == []
        assert template.format() == "这是一个静态提示词。"

    def test_direct_construction(self):
        """直接构造 PromptTemplate"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate(
            template="Hello {name}!",
            input_variables=["name"],
        )
        assert template.format(name="World") == "Hello World!"

    def test_format_preserves_braces(self):
        """双花括号 {{ }} 应被保留为单花括号（转义）"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template(
            "输出 JSON: {{\"key\": \"{value}\"}}"
        )
        result = template.format(value="hello")
        assert result == '输出 JSON: {"key": "hello"}'

    def test_template_property(self):
        """可以读取原始模板字符串"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("Hello {name}!")
        assert template.template == "Hello {name}!"


# ============================================================
# ChatPromptTemplate 消息角色
# ============================================================

class TestChatPromptTemplate:

    def test_create_with_system_and_human(self):
        """system + human 消息模板"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}助手。"),
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(role="翻译", question="把你好翻译成英文")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个翻译助手。"
        assert messages[1]["role"] == "human"
        assert messages[1]["content"] == "把你好翻译成英文"

    def test_input_variables_collected(self):
        """收集所有消息模板中的变量"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是{role}助手"),
            ("human", "{question}"),
            ("ai", "{previous_answer}"),
        ])
        assert set(prompt.input_variables) == {"role", "question", "previous_answer"}

    def test_format_with_three_roles(self):
        """system + human + ai 三种角色"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是{role}。"),
            ("human", "{user_input}"),
            ("ai", "好的，让我想想..."),
        ])
        messages = prompt.format_messages(role="数学老师", user_input="1+1=?")

        assert len(messages) == 3
        assert messages[2]["role"] == "ai"
        assert messages[2]["content"] == "好的，让我想想..."

    def test_missing_variable_raises(self):
        """缺少变量报错"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是{role}助手"),
            ("human", "{question}"),
        ])
        with pytest.raises(KeyError):
            prompt.format_messages(role="翻译")  # 缺少 question

    def test_format_messages_returns_dicts(self):
        """返回的每条消息是 dict 格式"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("human", "你好"),
        ])
        messages = prompt.format_messages()
        msg = messages[0]
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg

    def test_from_messages_with_empty_list(self):
        """空消息列表"""
        from simple_langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([])
        messages = prompt.format_messages()
        assert messages == []


# ============================================================
# PromptTemplate + partial（偏应用）
# ============================================================

class TestPromptPartial:

    def test_partial_fills_some_variables(self):
        """partial 预填充部分变量"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("{greeting}, {name}!")
        partial = template.partial(greeting="你好")
        result = partial.format(name="世界")
        assert result == "你好, 世界!"

    def test_partial_chain(self):
        """partial 可以链式调用"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("{a} {b} {c}")
        partial = template.partial(a="1").partial(b="2")
        result = partial.format(c="3")
        assert result == "1 2 3"

    def test_partial_returns_new_template(self):
        """partial 不修改原模板"""
        from simple_langchain.prompts import PromptTemplate

        template = PromptTemplate.from_template("{a} {b}")
        partial = template.partial(a="1")
        assert template.input_variables == ["a", "b"]
        assert "a" not in partial.input_variables
        assert partial.input_variables == ["b"]
