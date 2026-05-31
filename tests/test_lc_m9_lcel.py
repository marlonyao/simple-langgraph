"""
Milestone 9 测试：LCEL 管道（prompt | llm | parser）

TDD RED 阶段：PromptTemplate / BaseLLM / BaseOutputParser 实现 Runnable，
让 prompt | llm | parser 语法跑通。
"""

import pytest


# ============================================================
# PromptTemplate 作为 Runnable
# ============================================================

class TestPromptTemplateRunnable:

    def test_prompt_is_runnable(self):
        """PromptTemplate 继承 Runnable"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.runnable import Runnable

        prompt = PromptTemplate.from_template("hello {name}")
        assert isinstance(prompt, Runnable)

    def test_prompt_invoke(self):
        """prompt.invoke(dict) 返回格式化后的字符串"""
        from simple_langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template("你好 {name}")
        assert prompt.invoke({"name": "世界"}) == "你好 世界"

    def test_prompt_batch(self):
        """prompt.batch 多个输入"""
        from simple_langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template("hi {name}")
        results = prompt.batch([{"name": "A"}, {"name": "B"}])
        assert results == ["hi A", "hi B"]

    def test_prompt_pipe_to_llm(self):
        """prompt | llm 管道"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM

        prompt = PromptTemplate.from_template("说 {word}")
        llm = FakeListLLM(responses=["hello!"])

        chain = prompt | llm
        result = chain.invoke({"word": "hi"})
        assert result == "hello!"


# ============================================================
# BaseLLM 作为 Runnable
# ============================================================

class TestLLMRunnable:

    def test_llm_is_runnable(self):
        """BaseLLM 继承 Runnable"""
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.runnable import Runnable

        llm = FakeListLLM(responses=["ok"])
        assert isinstance(llm, Runnable)

    def test_llm_invoke_is_backward_compat(self):
        """LLM 的 invoke(str) 仍然正常工作"""
        from simple_langchain.llms import FakeListLLM

        llm = FakeListLLM(responses=["world"])
        assert llm.invoke("hello") == "world"


# ============================================================
# BaseOutputParser 作为 Runnable
# ============================================================

class TestParserRunnable:

    def test_parser_is_runnable(self):
        """Parser 继承 Runnable"""
        from simple_langchain.parsers import JsonOutputParser
        from simple_langchain.runnable import Runnable

        parser = JsonOutputParser()
        assert isinstance(parser, Runnable)

    def test_parser_invoke(self):
        """parser.invoke(str) 调用 parse"""
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        assert parser.invoke('{"a": 1}') == {"a": 1}


# ============================================================
# 完整 LCEL 管道
# ============================================================

class TestLCELPipeline:

    def test_prompt_llm_parser_chain(self):
        """prompt | llm | parser 完整管道"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import JsonOutputParser

        prompt = PromptTemplate.from_template("列出 {n} 个颜色")
        llm = FakeListLLM(responses=['["red", "green", "blue"]'])
        parser = JsonOutputParser()

        chain = prompt | llm | parser
        result = chain.invoke({"n": "3"})
        assert result == ["red", "green", "blue"]

    def test_prompt_llm_pipe(self):
        """prompt | llm 两步管道"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM

        prompt = PromptTemplate.from_template("翻译 {word}")
        llm = FakeListLLM(responses=["apple"])

        chain = prompt | llm
        assert chain.invoke({"word": "苹果"}) == "apple"

    def test_llm_parser_pipe(self):
        """llm | parser 两步管道"""
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import JsonOutputParser

        llm = FakeListLLM(responses=['{"count": 5}'])
        parser = JsonOutputParser()

        chain = llm | parser
        assert chain.invoke("数一下") == {"count": 5}

    def test_lcel_batch(self):
        """LCEL 管道支持 batch"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM

        prompt = PromptTemplate.from_template("翻译 {word}")
        llm = FakeListLLM(responses=["apple", "banana"])

        chain = prompt | llm
        results = chain.batch([
            {"word": "苹果"},
            {"word": "香蕉"},
        ])
        assert results == ["apple", "banana"]

    def test_lcel_with_list_parser(self):
        """LCEL 管道 + CommaSeparatedListOutputParser"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        prompt = PromptTemplate.from_template("列出 {item}")
        llm = FakeListLLM(responses=["苹果, 香蕉, 橘子"])
        parser = CommaSeparatedListOutputParser()

        chain = prompt | llm | parser
        result = chain.invoke({"item": "水果"})
        assert result == ["苹果", "香蕉", "橘子"]

    def test_lcel_with_pydantic_parser(self):
        """LCEL 管道 + PydanticOutputParser"""
        from simple_langchain.prompts import PromptTemplate
        from simple_langchain.llms import FakeListLLM
        from simple_langchain.parsers import PydanticOutputParser
        from pydantic import BaseModel

        class Movie(BaseModel):
            title: str
            year: int

        parser = PydanticOutputParser(pydantic_object=Movie)
        prompt = PromptTemplate.from_template(
            "描述电影 {name}\n{format_instructions}"
        )
        llm = FakeListLLM(responses=['{"title": "Inception", "year": 2010}'])

        chain = prompt | llm | parser
        # LCEL 管道：用户需要手动传 format_instructions
        result = chain.invoke({
            "name": "盗梦空间",
            "format_instructions": parser.get_format_instructions(),
        })
        assert result.title == "Inception"
        assert result.year == 2010
