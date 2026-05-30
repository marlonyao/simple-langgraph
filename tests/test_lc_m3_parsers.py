"""
Milestone 3 测试：Output Parser（输出解析）

TDD RED 阶段：测试各种输出解析器。
"""

import pytest


# ============================================================
# StrOutputParser
# ============================================================

class TestStrOutputParser:

    def test_parse_returns_input_unchanged(self):
        from simple_langchain.parsers import StrOutputParser

        parser = StrOutputParser()
        assert parser.parse("hello world") == "hello world"

    def test_parse_with_empty_string(self):
        from simple_langchain.parsers import StrOutputParser

        parser = StrOutputParser()
        assert parser.parse("") == ""

    def test_get_format_instructions(self):
        """StrOutputParser 不需要格式指令"""
        from simple_langchain.parsers import StrOutputParser

        parser = StrOutputParser()
        assert parser.get_format_instructions() == ""


# ============================================================
# JsonOutputParser
# ============================================================

class TestJsonOutputParser:

    def test_parse_valid_json(self):
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}

    def test_parse_json_with_markdown_code_block(self):
        """LLM 经常返回 ```json ... ``` 包裹的 JSON"""
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        result = parser.parse('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_invalid_json_raises(self):
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        with pytest.raises(ValueError, match="json"):
            parser.parse("not json at all")

    def test_get_format_instructions(self):
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions

    def test_parse_json_array(self):
        from simple_langchain.parsers import JsonOutputParser

        parser = JsonOutputParser()
        result = parser.parse('[1, 2, 3]')
        assert result == [1, 2, 3]


# ============================================================
# CommaSeparatedListOutputParser
# ============================================================

class TestCommaSeparatedListParser:

    def test_parse_basic_list(self):
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        result = parser.parse("苹果, 香蕉, 橘子")
        assert result == ["苹果", "香蕉", "橘子"]

    def test_parse_strips_whitespace(self):
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        result = parser.parse("a,  b , c")
        assert result == ["a", "b", "c"]

    def test_parse_single_item(self):
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        result = parser.parse("唯一的一项")
        assert result == ["唯一的一项"]

    def test_get_format_instructions(self):
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        instructions = parser.get_format_instructions()
        assert "逗号" in instructions or "comma" in instructions.lower()

    def test_parse_empty_string_returns_empty_list(self):
        from simple_langchain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        result = parser.parse("")
        assert result == []


# ============================================================
# PydanticOutputParser
# ============================================================

class TestPydanticOutputParser:

    def test_parse_valid_pydantic_json(self):
        from simple_langchain.parsers import PydanticOutputParser

        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        parser = PydanticOutputParser(pydantic_object=Person)
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_with_markdown_wrapper(self):
        from simple_langchain.parsers import PydanticOutputParser

        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Item(BaseModel):
            title: str

        parser = PydanticOutputParser(pydantic_object=Item)
        result = parser.parse('```json\n{"title": "hello"}\n```')
        assert result.title == "hello"

    def test_parse_invalid_data_raises(self):
        from simple_langchain.parsers import PydanticOutputParser

        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Strict(BaseModel):
            count: int

        parser = PydanticOutputParser(pydantic_object=Strict)
        with pytest.raises(ValueError):
            parser.parse('{"count": "not a number"}')

    def test_get_format_instructions_contains_schema(self):
        from simple_langchain.parsers import PydanticOutputParser

        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Movie(BaseModel):
            title: str
            year: int

        parser = PydanticOutputParser(pydantic_object=Movie)
        instructions = parser.get_format_instructions()
        assert "title" in instructions
        assert "year" in instructions

    def test_parse_missing_field_raises(self):
        from simple_langchain.parsers import PydanticOutputParser

        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Required(BaseModel):
            name: str
            email: str

        parser = PydanticOutputParser(pydantic_object=Required)
        with pytest.raises(ValueError):
            parser.parse('{"name": "Alice"}')  # 缺 email
