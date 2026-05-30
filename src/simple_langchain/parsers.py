"""
Simple LangChain — Output Parser（输出解析器）

核心概念：
- StrOutputParser：原样返回
- JsonOutputParser：解析 JSON，支持 ```json 代码块
- CommaSeparatedListOutputParser：逗号分隔列表
- PydanticOutputParser：JSON → Pydantic 模型验证
"""

import json
import re
from typing import Any


class BaseOutputParser:
    """所有解析器的基类"""

    def parse(self, output: str) -> Any:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        """返回给 LLM 的格式说明（可嵌入 prompt）"""
        return ""


class StrOutputParser(BaseOutputParser):
    """原样返回，不做任何解析"""

    def parse(self, output: str) -> str:
        return output


class JsonOutputParser(BaseOutputParser):
    """
    解析 JSON 输出。

    支持：
    - 纯 JSON 字符串
    - ```json ... ``` 代码块包裹的 JSON
    """

    def parse(self, output: str) -> Any:
        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = output.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse json: {e}") from e

    def get_format_instructions(self) -> str:
        return "请以 JSON 格式输出结果。"


class CommaSeparatedListOutputParser(BaseOutputParser):
    """解析逗号分隔的列表"""

    def parse(self, output: str) -> list[str]:
        if not output.strip():
            return []
        return [item.strip() for item in output.split(",")]

    def get_format_instructions(self) -> str:
        return "请以逗号分隔的列表形式输出结果。"


class PydanticOutputParser(BaseOutputParser):
    """
    解析 JSON 并用 Pydantic 模型验证。

    和 JsonOutputParser 类似，但增加了类型验证和格式说明。
    """

    def __init__(self, pydantic_object: type):
        self._pydantic_object = pydantic_object

    def parse(self, output: str) -> Any:
        # 复用 JSON 解析逻辑
        json_parser = JsonOutputParser()
        data = json_parser.parse(output)

        try:
            return self._pydantic_object.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to validate Pydantic model: {e}") from e

    def get_format_instructions(self) -> str:
        schema = self._pydantic_object.model_json_schema()
        fields_desc = []
        for name, info in schema.get("properties", {}).items():
            field_type = info.get("type", "any")
            fields_desc.append(f"  - {name}: {field_type}")
        fields_str = "\n".join(fields_desc)
        return f"请以 JSON 格式输出，包含以下字段：\n{fields_str}"
