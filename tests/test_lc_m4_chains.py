"""
Milestone 4 测试：Chain（链式调用）

TDD RED 阶段：测试 LLMChain、SequentialChain、batch。
"""

import pytest


# ============================================================
# LLMChain
# ============================================================

class TestLLMChain:

    def test_basic_chain_invoke(self):
        """prompt → llm → parser 完整链"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import StrOutputParser
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("告诉我关于{topic}的事")
        llm = FakeListLLM(responses=["AI是一个有趣的话题"])
        parser = StrOutputParser()

        chain = LLMChain(prompt=prompt, llm=llm, output_parser=parser)
        result = chain.invoke({"topic": "AI"})
        assert result == "AI是一个有趣的话题"

    def test_chain_without_parser(self):
        """没有 parser 时直接返回 LLM 输出"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("你好{name}")
        llm = FakeListLLM(responses=["你好！"])
        chain = LLMChain(prompt=prompt, llm=llm)
        result = chain.invoke({"name": "世界"})
        assert result == "你好！"

    def test_chain_with_json_parser(self):
        """链中使用 JSON 解析器"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import JsonOutputParser
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("列出{category}中的3种动物")
        llm = FakeListLLM(responses=['{"animals": ["猫", "狗", "鸟"]}'])
        parser = JsonOutputParser()

        chain = LLMChain(prompt=prompt, llm=llm, output_parser=parser)
        result = chain.invoke({"category": "宠物"})
        assert result == {"animals": ["猫", "狗", "鸟"]}

    def test_chain_input_keys(self):
        """chain 知道自己需要哪些输入"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("{a}和{b}")
        llm = FakeListLLM(responses=["ok"])
        chain = LLMChain(prompt=prompt, llm=llm)
        assert set(chain.input_keys) == {"a", "b"}

    def test_chain_invoke_missing_input_raises(self):
        """缺少输入时报错"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("{name}你好")
        llm = FakeListLLM(responses=["hi"])
        chain = LLMChain(prompt=prompt, llm=llm)
        with pytest.raises(KeyError):
            chain.invoke({})

    def test_chain_invoke_returns_string_by_default(self):
        """默认返回字符串"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("say {word}")
        llm = FakeListLLM(responses=["hello"])
        chain = LLMChain(prompt=prompt, llm=llm)
        result = chain.invoke({"word": "hi"})
        assert isinstance(result, str)


# ============================================================
# SequentialChain
# ============================================================

class TestSequentialChain:

    def test_two_chains_sequential(self):
        """两个 chain 串行"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain, SequentialChain

        # Chain 1: 生成主题
        prompt1 = PromptTemplate.from_template("给我一个关于{domain}的主题")
        llm1 = FakeListLLM(responses=["机器学习"])
        chain1 = LLMChain(prompt=prompt1, llm=llm1, output_key="topic")

        # Chain 2: 用主题写文章
        prompt2 = PromptTemplate.from_template("写一篇关于{topic}的摘要")
        llm2 = FakeListLLM(responses=["机器学习是AI的核心"])
        chain2 = LLMChain(prompt=prompt2, llm=llm2, output_key="summary")

        seq = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["domain"],
        )
        result = seq.invoke({"domain": "科技"})
        assert "summary" in result
        assert result["summary"] == "机器学习是AI的核心"

    def test_sequential_passes_all_keys(self):
        """前一个 chain 的输出作为后一个 chain 的输入"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain, SequentialChain

        prompt1 = PromptTemplate.from_template("翻译{word}到英文")
        llm1 = FakeListLLM(responses=["hello"])
        chain1 = LLMChain(prompt=prompt1, llm=llm1, output_key="english")

        prompt2 = PromptTemplate.from_template("{english}的大写是什么")
        llm2 = FakeListLLM(responses=["HELLO"])
        chain2 = LLMChain(prompt=prompt2, llm=llm2, output_key="upper")

        seq = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["word"],
        )
        result = seq.invoke({"word": "你好"})
        assert result["english"] == "hello"
        assert result["upper"] == "HELLO"

    def test_sequential_input_keys(self):
        """SequentialChain 知道自己需要哪些输入"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain, SequentialChain

        prompt1 = PromptTemplate.from_template("处理{input1}")
        llm1 = FakeListLLM(responses=["a"])
        chain1 = LLMChain(prompt=prompt1, llm=llm1, output_key="step1")

        prompt2 = PromptTemplate.from_template("处理{step1}")
        llm2 = FakeListLLM(responses=["b"])
        chain2 = LLMChain(prompt=prompt2, llm=llm2, output_key="step2")

        seq = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["input1"],
        )
        assert seq.input_keys == ["input1"]


# ============================================================
# batch
# ============================================================

class TestBatch:

    def test_batch_multiple_inputs(self):
        """batch 批量调用"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("翻译{word}")
        llm = FakeListLLM(responses=["apple", "banana"])
        chain = LLMChain(prompt=prompt, llm=llm)

        results = chain.batch([
            {"word": "苹果"},
            {"word": "香蕉"},
        ])
        assert len(results) == 2
        assert results[0] == "apple"
        assert results[1] == "banana"

    def test_batch_empty_list(self):
        """空列表返回空结果"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("test")
        llm = FakeListLLM(responses=["ok"])
        chain = LLMChain(prompt=prompt, llm=llm)
        assert chain.batch([]) == []

    def test_batch_single_input(self):
        """单个输入的 batch"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.chains import LLMChain

        prompt = PromptTemplate.from_template("hello {name}")
        llm = FakeListLLM(responses=["hi!"])
        chain = LLMChain(prompt=prompt, llm=llm)
        results = chain.batch([{"name": "world"}])
        assert results == ["hi!"]
