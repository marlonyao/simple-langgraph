"""
Milestone 5 测试：Memory（对话记忆）

TDD RED 阶段：测试对话记忆机制。
"""

import pytest


# ============================================================
# ConversationBufferMemory
# ============================================================

class TestConversationBufferMemory:

    def test_save_and_load_context(self):
        """保存和加载对话历史"""
        from simple_langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory()
        memory.save_context("你好", "你好！有什么可以帮你？")
        history = memory.load_memory()
        assert len(history) == 2
        assert history[0]["role"] == "human"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "ai"
        assert history[1]["content"] == "你好！有什么可以帮你？"

    def test_multiple_turns(self):
        """多轮对话"""
        from simple_langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory()
        memory.save_context("hi", "hello")
        memory.save_context("how are you?", "fine")
        history = memory.load_memory()
        assert len(history) == 4  # 2 turns × 2 messages

    def test_clear_memory(self):
        """清空记忆"""
        from simple_langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory()
        memory.save_context("q1", "a1")
        memory.clear()
        assert memory.load_memory() == []

    def test_empty_memory(self):
        """空记忆"""
        from simple_langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory()
        assert memory.load_memory() == []


# ============================================================
# ConversationBufferWindowMemory
# ============================================================

class TestConversationBufferWindowMemory:

    def test_keeps_last_k_turns(self):
        """只保留最近 k 轮"""
        from simple_langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=2)
        memory.save_context("q1", "a1")
        memory.save_context("q2", "a2")
        memory.save_context("q3", "a3")
        history = memory.load_memory()
        # 只保留最近 2 轮 = 4 条消息
        assert len(history) == 4
        assert history[0]["content"] == "q2"
        assert history[1]["content"] == "a2"
        assert history[2]["content"] == "q3"
        assert history[3]["content"] == "a3"

    def test_k_equals_1(self):
        """k=1 只保留最近 1 轮"""
        from simple_langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=1)
        memory.save_context("q1", "a1")
        memory.save_context("q2", "a2")
        memory.save_context("q3", "a3")
        history = memory.load_memory()
        assert len(history) == 2
        assert history[0]["content"] == "q3"
        assert history[1]["content"] == "a3"

    def test_window_larger_than_history(self):
        """窗口大于历史时全部保留"""
        from simple_langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=10)
        memory.save_context("q1", "a1")
        memory.save_context("q2", "a2")
        history = memory.load_memory()
        assert len(history) == 4


# ============================================================
# ConversationSummaryMemory
# ============================================================

class TestConversationSummaryMemory:

    def test_summarize_on_save(self):
        """保存时用 LLM 生成摘要"""
        from simple_langchain.memory import ConversationSummaryMemory
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["用户问了天气，助手回答了晴天"])
        memory = ConversationSummaryMemory(llm=llm)
        memory.save_context("今天天气怎么样？", "今天晴天")
        summary = memory.load_memory()
        assert len(summary) == 1
        assert summary[0]["role"] == "system"
        assert "天气" in summary[0]["content"]

    def test_summary_accumulates(self):
        """摘要逐步积累"""
        from simple_langchain.memory import ConversationSummaryMemory
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[
            "用户问了天气",
            "之前问了天气，现在又问了午饭",
        ])
        memory = ConversationSummaryMemory(llm=llm)
        memory.save_context("天气？", "晴天")
        memory.save_context("午饭吃啥？", "面条")
        summary = memory.load_memory()
        assert "午饭" in summary[0]["content"]

    def test_empty_summary(self):
        """空记忆"""
        from simple_langchain.memory import ConversationSummaryMemory
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=[])
        memory = ConversationSummaryMemory(llm=llm)
        assert memory.load_memory() == []


# ============================================================
# Memory + LLMChain 集成
# ============================================================

class TestMemoryWithChain:

    def test_chain_with_memory(self):
        """LLMChain 自动注入历史"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain
        from simple_langchain.memory import ConversationBufferMemory

        prompt = PromptTemplate.from_template(
            "历史：{history}\n用户：{input}\n助手："
        )
        llm = FakeListLLM(responses=["你好！", "你刚才说了什么？"])
        memory = ConversationBufferMemory()

        chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

        # 第一次调用
        r1 = chain.invoke({"input": "你好"})
        assert r1 == "你好！"

        # 第二次调用：历史应该被注入
        r2 = chain.invoke({"input": "我刚才说了什么？"})
        assert r2 == "你刚才说了什么？"

        # 检查 memory 有历史
        history = memory.load_memory()
        assert len(history) == 4  # 2 turns

    def test_chain_memory_auto_save(self):
        """chain 调用后自动保存到 memory"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain
        from simple_langchain.memory import ConversationBufferMemory

        prompt = PromptTemplate.from_template("{input}")
        llm = FakeListLLM(responses=["ok"])
        memory = ConversationBufferMemory()
        chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

        chain.invoke({"input": "hello"})
        history = memory.load_memory()
        assert len(history) == 2
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "ok"
